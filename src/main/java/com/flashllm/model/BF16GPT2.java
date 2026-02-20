package com.flashllm.model;

import com.flashllm.kernel.TensorUtils;
import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.config.GPT2Config;

import java.io.IOException;

/**
 * BF16 GPT-2 model for training.
 *
 * BF16 has the same dynamic range as FP32 (8-bit exponent), so:
 *   - No LossScaler needed
 *   - No overflow detection needed
 *   - No gradient unscaling needed
 *   - Softmax + CrossEntropy still FP32 (precision, not range)
 *
 * All operations use flash 0.1.10 GPU kernels — zero CPU fallback.
 *
 * Weight layout follows llm.c convention (PyTorch Conv1D):
 *   qkvw:     (3C, C)  — forward: out = inp @ w^T
 *   attprojw: (C, C)
 *   fcw:      (4C, C)
 *   fcprojw:  (C, 4C)
 *   wte:      (V, C)
 *
 * @author flash-llm
 * @since 3.0.0
 */
public class BF16GPT2 implements AutoCloseable {

    private final GPT2Config config;
    private final int B;
    private final int T;

    // Backend
    private final FlashBackend backend;
    private final CudaDevice device;
    private final CudaLlmKernels kernels;
    private final CudaBlasExt blasExt;

    // FP32 Master Weights
    private CudaTensor masterWte, masterWpe, masterLnfw, masterLnfb;
    private CudaTensor[] masterLn1w, masterLn1b;
    private CudaTensor[] masterQkvw, masterQkvb;
    private CudaTensor[] masterAttnProjw, masterAttnProjb;
    private CudaTensor[] masterLn2w, masterLn2b;
    private CudaTensor[] masterFcw, masterFcb;
    private CudaTensor[] masterFcProjw, masterFcProjb;

    // BF16 Weights (for compute)
    private CudaTensor bf16Wte, bf16Wpe, bf16Lnfw, bf16Lnfb;
    private CudaTensor[] bf16Ln1w, bf16Ln1b;
    private CudaTensor[] bf16Qkvw, bf16Qkvb;
    private CudaTensor[] bf16AttnProjw, bf16AttnProjb;
    private CudaTensor[] bf16Ln2w, bf16Ln2b;
    private CudaTensor[] bf16Fcw, bf16Fcb;
    private CudaTensor[] bf16FcProjw, bf16FcProjb;

    // Gradients — BF16 (BF16 dynamic range = FP32, no overflow risk)
    private CudaTensor gradWte, gradWpe;
    private CudaTensor gradLnfw, gradLnfb;
    private CudaTensor[] gradLn1w, gradLn1b;
    private CudaTensor[] gradLn2w, gradLn2b;
    private CudaTensor[] gradQkvw, gradQkvb;
    private CudaTensor[] gradAttnProjw, gradAttnProjb;
    private CudaTensor[] gradFcw, gradFcb;
    private CudaTensor[] gradFcProjw, gradFcProjb;

    // BF16 Activations
    private CudaTensor encoded;
    private CudaTensor[] ln1, qkv, attnOut, residual1;
    private CudaTensor[] ln2, fch, fchGelu, residual2;
    private CudaTensor lnf, logits, probs;
    private CudaTensor dlogits, dlnf;
    private CudaTensor[] dresidual;

    // FP32 activations (LayerNorm stats, flash attention)
    private CudaTensor[] ln1Mean, ln1Rstd, ln2Mean, ln2Rstd, attLse;
    private CudaTensor lnfMean, lnfRstd;

    // AdamW State (FP32)
    private CudaTensor adamMWte, adamVWte, adamMWpe, adamVWpe;
    private CudaTensor adamMLnfw, adamVLnfw, adamMLnfb, adamVLnfb;
    private CudaTensor[] adamMLn1w, adamVLn1w, adamMLn1b, adamVLn1b;
    private CudaTensor[] adamMQkvw, adamVQkvw, adamMQkvb, adamVQkvb;
    private CudaTensor[] adamMAttnProjw, adamVAttnProjw, adamMAttnProjb, adamVAttnProjb;
    private CudaTensor[] adamMLn2w, adamVLn2w, adamMLn2b, adamVLn2b;
    private CudaTensor[] adamMFcw, adamVFcw, adamMFcb, adamVFcb;
    private CudaTensor[] adamMFcProjw, adamVFcProjw, adamMFcProjb, adamVFcProjb;

    // State
    private boolean closed = false;
    private boolean weightsLoaded = false;
    private float meanLoss = 0.0f;

    public BF16GPT2(GPT2Config config, int B, int T) {
        this.config = config;
        this.B = B;
        this.T = T;
        this.backend = FlashBackend.getInstance();
        this.device = backend.getDevice();
        this.kernels = backend.getLlmKernels();
        this.blasExt = backend.getBlasExt();

        int L = config.numLayers;
        int C = config.channels;
        int V = config.vocabSize;
        int BT = B * T;

        System.out.println("Creating BF16 GPT-2:");
        System.out.println("  Layers: " + L + ", Channels: " + C + ", Heads: " + config.numHeads);
        System.out.println("  Batch: " + B + ", SeqLen: " + T);

        initializeArrays(L);
        allocateActivations(L, C, V, BT);

        System.out.println("  Model created (BF16 — no LossScaler needed)!");
    }

    private void initializeArrays(int L) {
        masterLn1w = new CudaTensor[L]; masterLn1b = new CudaTensor[L];
        masterQkvw = new CudaTensor[L]; masterQkvb = new CudaTensor[L];
        masterAttnProjw = new CudaTensor[L]; masterAttnProjb = new CudaTensor[L];
        masterLn2w = new CudaTensor[L]; masterLn2b = new CudaTensor[L];
        masterFcw = new CudaTensor[L]; masterFcb = new CudaTensor[L];
        masterFcProjw = new CudaTensor[L]; masterFcProjb = new CudaTensor[L];

        bf16Ln1w = new CudaTensor[L]; bf16Ln1b = new CudaTensor[L];
        bf16Qkvw = new CudaTensor[L]; bf16Qkvb = new CudaTensor[L];
        bf16AttnProjw = new CudaTensor[L]; bf16AttnProjb = new CudaTensor[L];
        bf16Ln2w = new CudaTensor[L]; bf16Ln2b = new CudaTensor[L];
        bf16Fcw = new CudaTensor[L]; bf16Fcb = new CudaTensor[L];
        bf16FcProjw = new CudaTensor[L]; bf16FcProjb = new CudaTensor[L];

        gradLn1w = new CudaTensor[L]; gradLn1b = new CudaTensor[L];
        gradLn2w = new CudaTensor[L]; gradLn2b = new CudaTensor[L];
        gradQkvw = new CudaTensor[L]; gradQkvb = new CudaTensor[L];
        gradAttnProjw = new CudaTensor[L]; gradAttnProjb = new CudaTensor[L];
        gradFcw = new CudaTensor[L]; gradFcb = new CudaTensor[L];
        gradFcProjw = new CudaTensor[L]; gradFcProjb = new CudaTensor[L];

        ln1 = new CudaTensor[L]; ln1Mean = new CudaTensor[L]; ln1Rstd = new CudaTensor[L];
        qkv = new CudaTensor[L]; attnOut = new CudaTensor[L]; attLse = new CudaTensor[L];
        residual1 = new CudaTensor[L];
        ln2 = new CudaTensor[L]; ln2Mean = new CudaTensor[L]; ln2Rstd = new CudaTensor[L];
        fch = new CudaTensor[L]; fchGelu = new CudaTensor[L]; residual2 = new CudaTensor[L];
        dresidual = new CudaTensor[L];
    }

    private void allocateActivations(int L, int C, int V, int BT) {
        encoded = alloc(BT * C, Precision.BF16);

        for (int l = 0; l < L; l++) {
            ln1[l] = alloc(BT * C, Precision.BF16);
            ln1Mean[l] = alloc(BT, Precision.FP32);
            ln1Rstd[l] = alloc(BT, Precision.FP32);
            qkv[l] = alloc((long) BT * 3 * C, Precision.BF16);
            attnOut[l] = alloc(BT * C, Precision.BF16);
            attLse[l] = alloc((long) B * config.numHeads * T, Precision.FP32);
            residual1[l] = alloc(BT * C, Precision.BF16);
            ln2[l] = alloc(BT * C, Precision.BF16);
            ln2Mean[l] = alloc(BT, Precision.FP32);
            ln2Rstd[l] = alloc(BT, Precision.FP32);
            fch[l] = alloc((long) BT * 4 * C, Precision.BF16);
            fchGelu[l] = alloc((long) BT * 4 * C, Precision.BF16);
            residual2[l] = alloc(BT * C, Precision.BF16);
            dresidual[l] = alloc(BT * C, Precision.BF16);
        }

        lnf = alloc(BT * C, Precision.BF16);
        lnfMean = alloc(BT, Precision.FP32);
        lnfRstd = alloc(BT, Precision.FP32);
        logits = alloc((long) BT * V, Precision.BF16);
        probs = alloc((long) BT * V, Precision.BF16);
        dlogits = alloc((long) BT * V, Precision.BF16);
        dlnf = alloc(BT * C, Precision.BF16);
    }

    private CudaTensor alloc(long n, Precision p) {
        return CudaTensor.allocate(device, n, p);
    }

    // ========================================================================
    // Weight Loading
    // ========================================================================

    public void loadWeights(String path) throws IOException {
        System.out.println("Loading weights from: " + path);

        GPT2WeightLoader loader = new GPT2WeightLoader();
        loader.load(path);

        int L = config.numLayers;
        int C = config.channels;

        // FP32 master weights
        masterWte = CudaTensor.fromFloat(device, loader.getWte(), Precision.FP32);
        masterWpe = CudaTensor.fromFloat(device, loader.getWpe(), Precision.FP32);
        masterLnfw = CudaTensor.fromFloat(device, loader.getLnfw(), Precision.FP32);
        masterLnfb = CudaTensor.fromFloat(device, loader.getLnfb(), Precision.FP32);

        // BF16 weights
        bf16Wte = alloc(masterWte.getElementCount(), Precision.BF16);
        bf16Wpe = alloc(masterWpe.getElementCount(), Precision.BF16);
        bf16Lnfw = alloc(C, Precision.BF16);
        bf16Lnfb = alloc(C, Precision.BF16);

        // Gradients — BF16
        gradWte = alloc(masterWte.getElementCount(), Precision.BF16);
        gradWpe = alloc(masterWpe.getElementCount(), Precision.BF16);
        gradLnfw = alloc(C, Precision.FP32);  // LN grads stay FP32 (accumulated by LN backward kernel)
        gradLnfb = alloc(C, Precision.FP32);

        for (int l = 0; l < L; l++) {
            masterLn1w[l] = CudaTensor.fromFloat(device, loader.getLn1w(l), Precision.FP32);
            masterLn1b[l] = CudaTensor.fromFloat(device, loader.getLn1b(l), Precision.FP32);
            masterQkvw[l] = CudaTensor.fromFloat(device, loader.getQkvw(l), Precision.FP32);
            masterQkvb[l] = CudaTensor.fromFloat(device, loader.getQkvb(l), Precision.FP32);
            masterAttnProjw[l] = CudaTensor.fromFloat(device, loader.getAttprojw(l), Precision.FP32);
            masterAttnProjb[l] = CudaTensor.fromFloat(device, loader.getAttprojb(l), Precision.FP32);
            masterLn2w[l] = CudaTensor.fromFloat(device, loader.getLn2w(l), Precision.FP32);
            masterLn2b[l] = CudaTensor.fromFloat(device, loader.getLn2b(l), Precision.FP32);
            masterFcw[l] = CudaTensor.fromFloat(device, loader.getFcw(l), Precision.FP32);
            masterFcb[l] = CudaTensor.fromFloat(device, loader.getFcb(l), Precision.FP32);
            masterFcProjw[l] = CudaTensor.fromFloat(device, loader.getFcprojw(l), Precision.FP32);
            masterFcProjb[l] = CudaTensor.fromFloat(device, loader.getFcprojb(l), Precision.FP32);

            bf16Ln1w[l] = alloc(C, Precision.BF16);
            bf16Ln1b[l] = alloc(C, Precision.BF16);
            bf16Qkvw[l] = alloc((long) 3 * C * C, Precision.BF16);
            bf16Qkvb[l] = alloc(3 * C, Precision.BF16);
            bf16AttnProjw[l] = alloc((long) C * C, Precision.BF16);
            bf16AttnProjb[l] = alloc(C, Precision.BF16);
            bf16Ln2w[l] = alloc(C, Precision.BF16);
            bf16Ln2b[l] = alloc(C, Precision.BF16);
            bf16Fcw[l] = alloc((long) 4 * C * C, Precision.BF16);
            bf16Fcb[l] = alloc(4 * C, Precision.BF16);
            bf16FcProjw[l] = alloc((long) 4 * C * C, Precision.BF16);
            bf16FcProjb[l] = alloc(C, Precision.BF16);

            // Gradients — BF16 for weight grads, FP32 for LN grads
            gradLn1w[l] = alloc(C, Precision.FP32);
            gradLn1b[l] = alloc(C, Precision.FP32);
            gradLn2w[l] = alloc(C, Precision.FP32);
            gradLn2b[l] = alloc(C, Precision.FP32);
            gradQkvw[l] = alloc((long) 3 * C * C, Precision.BF16);
            gradQkvb[l] = alloc(3 * C, Precision.BF16);
            gradAttnProjw[l] = alloc((long) C * C, Precision.BF16);
            gradAttnProjb[l] = alloc(C, Precision.BF16);
            gradFcw[l] = alloc((long) 4 * C * C, Precision.BF16);
            gradFcb[l] = alloc(4 * C, Precision.BF16);
            gradFcProjw[l] = alloc((long) C * 4 * C, Precision.BF16);
            gradFcProjb[l] = alloc(C, Precision.BF16);
        }

        syncWeightsToBf16();
        device.synchronize();
        weightsLoaded = true;
        System.out.println("Weights loaded (BF16)!");
    }

    private void syncWeightsToBf16() {
        kernels.convertF32ToBf16(masterWte, bf16Wte, (int) masterWte.getElementCount());
        kernels.convertF32ToBf16(masterWpe, bf16Wpe, (int) masterWpe.getElementCount());
        kernels.convertF32ToBf16(masterLnfw, bf16Lnfw, (int) masterLnfw.getElementCount());
        kernels.convertF32ToBf16(masterLnfb, bf16Lnfb, (int) masterLnfb.getElementCount());

        for (int l = 0; l < config.numLayers; l++) {
            kernels.convertF32ToBf16(masterLn1w[l], bf16Ln1w[l], (int) masterLn1w[l].getElementCount());
            kernels.convertF32ToBf16(masterLn1b[l], bf16Ln1b[l], (int) masterLn1b[l].getElementCount());
            kernels.convertF32ToBf16(masterQkvw[l], bf16Qkvw[l], (int) masterQkvw[l].getElementCount());
            kernels.convertF32ToBf16(masterQkvb[l], bf16Qkvb[l], (int) masterQkvb[l].getElementCount());
            kernels.convertF32ToBf16(masterAttnProjw[l], bf16AttnProjw[l], (int) masterAttnProjw[l].getElementCount());
            kernels.convertF32ToBf16(masterAttnProjb[l], bf16AttnProjb[l], (int) masterAttnProjb[l].getElementCount());
            kernels.convertF32ToBf16(masterLn2w[l], bf16Ln2w[l], (int) masterLn2w[l].getElementCount());
            kernels.convertF32ToBf16(masterLn2b[l], bf16Ln2b[l], (int) masterLn2b[l].getElementCount());
            kernels.convertF32ToBf16(masterFcw[l], bf16Fcw[l], (int) masterFcw[l].getElementCount());
            kernels.convertF32ToBf16(masterFcb[l], bf16Fcb[l], (int) masterFcb[l].getElementCount());
            kernels.convertF32ToBf16(masterFcProjw[l], bf16FcProjw[l], (int) masterFcProjw[l].getElementCount());
            kernels.convertF32ToBf16(masterFcProjb[l], bf16FcProjb[l], (int) masterFcProjb[l].getElementCount());
        }
    }

    // ========================================================================
    // Forward Pass
    // ========================================================================

    public float forward(int[] tokens, int[] targets) {
        ensureNotClosed();
        if (!weightsLoaded) throw new IllegalStateException("Weights not loaded");

        int C = config.channels;
        int V = config.vocabSize;
        int L = config.numLayers;
        int NH = config.numHeads;
        int HS = C / NH;
        int BT = B * T;
        int BNH = B * NH;

        // 1. Embedding (all GPU)
        kernels.embeddingForwardBf16(encoded, tokens, bf16Wte, B, T, C);
        addPositionEmbeddings(encoded, bf16Wpe, B, T, C);

        // 2. Transformer blocks
        CudaTensor residual = encoded;
        for (int l = 0; l < L; l++) {
            // LN1
            kernels.layerNormForwardBf16(ln1[l], ln1Mean[l], ln1Rstd[l],
                    residual, bf16Ln1w[l], bf16Ln1b[l], BT, C);

            // QKV projection: qkv = ln1 @ qkvw^T + qkvb
            blasExt.gemmEx(false, true, BT, 3 * C, C,
                    1.0, ln1[l], bf16Qkvw[l], 0.0, qkv[l]);
            kernels.broadcastAddBf16(qkv[l], bf16Qkvb[l], BT, 3 * C);

            // Attention (flash attention requires FP32)
            try (CudaTensor qBf16 = alloc((long) BNH * T * HS, Precision.BF16);
                 CudaTensor kBf16 = alloc((long) BNH * T * HS, Precision.BF16);
                 CudaTensor vBf16 = alloc((long) BNH * T * HS, Precision.BF16);
                 CudaTensor qF32 = alloc((long) BNH * T * HS, Precision.FP32);
                 CudaTensor kF32 = alloc((long) BNH * T * HS, Precision.FP32);
                 CudaTensor vF32 = alloc((long) BNH * T * HS, Precision.FP32);
                 CudaTensor attnOutF32 = alloc((long) BNH * T * HS, Precision.FP32);
                 CudaTensor attnProjBf16 = alloc(BT * C, Precision.BF16)) {

                // QKV reshape: (BT, 3C) → Q,K,V each (BNH, T, HS)
                kernels.reshapeQKVBf16(qBf16, kBf16, vBf16, qkv[l], B, T, NH, HS);

                // Convert to FP32 for flash attention
                kernels.convertBf16ToF32(qBf16, qF32, BNH * T * HS);
                kernels.convertBf16ToF32(kBf16, kF32, BNH * T * HS);
                kernels.convertBf16ToF32(vBf16, vF32, BNH * T * HS);

                // Flash attention (FP32 internal)
                kernels.flashAttentionForward(attnOutF32, attLse[l], qF32, kF32, vF32, BNH, T, HS, true);

                // Reshape: (BNH, T, HS) → (BT, C) in FP32, then convert to BF16
                try (CudaTensor attnReshF32 = alloc(BT * C, Precision.FP32)) {
                    reshapeAttentionOutputF32(attnOutF32, attnReshF32, B, T, NH, HS);
                    kernels.convertF32ToBf16(attnReshF32, attnOut[l], BT * C);
                }

                // Attention projection: out = attnOut @ attprojw^T + attprojb
                blasExt.gemmEx(false, true, BT, C, C,
                        1.0, attnOut[l], bf16AttnProjw[l], 0.0, attnProjBf16);
                kernels.broadcastAddBf16(attnProjBf16, bf16AttnProjb[l], BT, C);

                // Residual1 = residual + attnProj
                CudaOps.add(device, residual, attnProjBf16, residual1[l]);
            }

            // LN2
            kernels.layerNormForwardBf16(ln2[l], ln2Mean[l], ln2Rstd[l],
                    residual1[l], bf16Ln2w[l], bf16Ln2b[l], BT, C);

            // MLP: fc → gelu → fcproj
            blasExt.gemmEx(false, true, BT, 4 * C, C,
                    1.0, ln2[l], bf16Fcw[l], 0.0, fch[l]);
            kernels.broadcastAddBf16(fch[l], bf16Fcb[l], BT, 4 * C);
            kernels.geluForwardBf16(fchGelu[l], fch[l], BT * 4 * C);

            try (CudaTensor mlpProj = alloc(BT * C, Precision.BF16)) {
                blasExt.gemmEx(false, true, BT, C, 4 * C,
                        1.0, fchGelu[l], bf16FcProjw[l], 0.0, mlpProj);
                kernels.broadcastAddBf16(mlpProj, bf16FcProjb[l], BT, C);

                // Residual2 = residual1 + mlpProj
                CudaOps.add(device, residual1[l], mlpProj, residual2[l]);
            }

            residual = residual2[l];
        }

        // 3. Final LayerNorm
        kernels.layerNormForwardBf16(lnf, lnfMean, lnfRstd,
                residual, bf16Lnfw, bf16Lnfb, BT, C);

        // 4. Output projection: logits = lnf @ wte^T
        blasExt.gemmEx(false, true, BT, V, C,
                1.0, lnf, bf16Wte, 0.0, logits);

        // 5. Softmax + CrossEntropy in FP32 (precision requirement, not range)
        if (targets != null) {
            float[] logitsF = logits.toFloatArray();
            float[] probsF = new float[BT * V];

            for (int i = 0; i < BT; i++) {
                int offset = i * V;
                float maxVal = Float.NEGATIVE_INFINITY;
                for (int v = 0; v < V; v++) {
                    maxVal = Math.max(maxVal, logitsF[offset + v]);
                }
                float expSum = 0.0f;
                for (int v = 0; v < V; v++) {
                    probsF[offset + v] = (float) Math.exp(logitsF[offset + v] - maxVal);
                    expSum += probsF[offset + v];
                }
                float invSum = 1.0f / expSum;
                for (int v = 0; v < V; v++) {
                    probsF[offset + v] *= invSum;
                }
            }

            TensorUtils.copyFromHost(device, probsF, probs);

            float loss = 0.0f;
            for (int i = 0; i < BT; i++) {
                float p = probsF[i * V + targets[i]];
                loss -= (float) Math.log(Math.max(p, 1e-10f));
            }
            meanLoss = loss / BT;
            return meanLoss;
        }
        return 0.0f;
    }

    // ========================================================================
    // Backward Pass — No loss scaling needed for BF16
    // ========================================================================

    public void backward(int[] tokens, int[] targets) {
        ensureNotClosed();
        int C = config.channels;
        int V = config.vocabSize;
        int L = config.numLayers;
        int NH = config.numHeads;
        int HS = C / NH;
        int BT = B * T;
        int BNH = B * NH;
        float scale = 1.0f / BT;

        // dlogits = (probs - onehot) / BT  — FP32 compute, store as BF16
        {
            float[] probsF = probs.toFloatArray();
            float[] dlogitsF = new float[BT * V];
            for (int i = 0; i < BT; i++) {
                int offset = i * V;
                for (int v = 0; v < V; v++) {
                    dlogitsF[offset + v] = probsF[offset + v] * scale;
                    if (v == targets[i]) {
                        dlogitsF[offset + v] -= scale;
                    }
                }
            }
            TensorUtils.copyFromHost(device, dlogitsF, dlogits);
        }

        // dlnf = dlogits @ wte  (BF16 matmul)
        blasExt.gemmEx(false, false, BT, C, V,
                1.0, dlogits, bf16Wte, 0.0, dlnf);

        // dwte += dlogits^T @ lnf  (BF16 → BF16 grad)
        accumulateWeightGrad(gradWte, dlogits, lnf, V, C, BT);

        // Final LN backward
        kernels.layerNormBackwardBf16(dresidual[L - 1], gradLnfw, gradLnfb,
                dlnf, residual2[L - 1], bf16Lnfw, lnfMean, lnfRstd, BT, C);

        // === Transformer blocks backward ===
        for (int l = L - 1; l >= 0; l--) {
            CudaTensor residual = (l == 0) ? encoded : residual2[l - 1];

            try (CudaTensor dfcproj = alloc(BT * C, Precision.BF16);
                 CudaTensor dfchGelu = alloc((long) BT * 4 * C, Precision.BF16);
                 CudaTensor dfch = alloc((long) BT * 4 * C, Precision.BF16);
                 CudaTensor dln2 = alloc(BT * C, Precision.BF16);
                 CudaTensor dresidual2 = alloc(BT * C, Precision.BF16);
                 CudaTensor dattproj = alloc(BT * C, Precision.BF16);
                 CudaTensor dln1 = alloc(BT * C, Precision.BF16)) {

                // 1. Residual backward: dresidual splits to MLP and attention paths
                copyTensor(dresidual[l], dresidual2);
                copyTensor(dresidual[l], dfcproj);

                // 2. FC proj backward (activation grad)
                blasExt.gemmEx(false, false, BT, 4 * C, C,
                        1.0, dfcproj, bf16FcProjw[l], 0.0, dfchGelu);

                // 3. GELU backward
                kernels.geluBackwardBf16(dfch, fch[l], dfchGelu, BT * 4 * C);

                // 4. FC backward (activation grad)
                blasExt.gemmEx(false, false, BT, C, 4 * C,
                        1.0, dfch, bf16Fcw[l], 0.0, dln2);

                // Weight grads (BF16 accumulate — safe with BF16 range)
                accumulateWeightGrad(gradFcProjw[l], dfcproj, fchGelu[l], C, 4 * C, BT);
                accumulateBiasGrad(gradFcProjb[l], dfcproj, BT, C);
                accumulateWeightGrad(gradFcw[l], dfch, ln2[l], 4 * C, C, BT);
                accumulateBiasGrad(gradFcb[l], dfch, BT, 4 * C);

                // 5. LN2 backward
                kernels.layerNormBackwardBf16(dresidual2, gradLn2w[l], gradLn2b[l],
                        dln2, residual1[l], bf16Ln2w[l], ln2Mean[l], ln2Rstd[l], BT, C);

                // 6. Residual backward
                copyTensor(dresidual2, dresidual[l]);
                copyTensor(dresidual2, dattproj);

                // 7. Attn proj backward
                try (CudaTensor dattnResh = alloc(BT * C, Precision.BF16)) {
                    blasExt.gemmEx(false, false, BT, C, C,
                            1.0, dattproj, bf16AttnProjw[l], 0.0, dattnResh);

                    accumulateWeightGrad(gradAttnProjw[l], dattproj, attnOut[l], C, C, BT);
                    accumulateBiasGrad(gradAttnProjb[l], dattproj, BT, C);

                    // 8. Flash attention backward (FP32 internal)
                    try (CudaTensor dattnReshF32 = alloc(BT * C, Precision.FP32);
                         CudaTensor qkvF32 = alloc((long) BT * 3 * C, Precision.FP32);
                         CudaTensor qF32 = alloc((long) BNH * T * HS, Precision.FP32);
                         CudaTensor kF32 = alloc((long) BNH * T * HS, Precision.FP32);
                         CudaTensor vF32 = alloc((long) BNH * T * HS, Precision.FP32);
                         CudaTensor dqF32 = alloc((long) BNH * T * HS, Precision.FP32);
                         CudaTensor dkF32 = alloc((long) BNH * T * HS, Precision.FP32);
                         CudaTensor dvF32 = alloc((long) BNH * T * HS, Precision.FP32);
                         CudaTensor attnOutF32 = alloc((long) BNH * T * HS, Precision.FP32);
                         CudaTensor dAttnOutF32 = alloc((long) BNH * T * HS, Precision.FP32);
                         CudaTensor dqkvF32 = alloc((long) BT * 3 * C, Precision.FP32);
                         CudaTensor dqkvBf16 = alloc((long) BT * 3 * C, Precision.BF16)) {

                        // dattnResh BF16 → FP32 → reshape → flash attn backward
                        kernels.convertBf16ToF32(dattnResh, dattnReshF32, BT * C);
                        reshapeAttentionOutputBackwardF32(dattnReshF32, dAttnOutF32, B, T, NH, HS);

                        kernels.convertBf16ToF32(qkv[l], qkvF32, BT * 3 * C);
                        reshapeQKVF32(qkvF32, qF32, kF32, vF32, B, T, NH, HS);

                        kernels.flashAttentionForward(attnOutF32, attLse[l], qF32, kF32, vF32, BNH, T, HS, true);
                        kernels.flashAttentionBackward(dqF32, dkF32, dvF32,
                                qF32, kF32, vF32, attnOutF32, dAttnOutF32, attLse[l], BNH, T, HS, true);

                        reshapeQKVBackwardF32(dqF32, dkF32, dvF32, dqkvF32, B, T, NH, HS);
                        kernels.convertF32ToBf16(dqkvF32, dqkvBf16, BT * 3 * C);

                        // 9. QKV backward
                        blasExt.gemmEx(false, false, BT, C, 3 * C,
                                1.0, dqkvBf16, bf16Qkvw[l], 0.0, dln1);

                        accumulateWeightGrad(gradQkvw[l], dqkvBf16, ln1[l], 3 * C, C, BT);
                        accumulateBiasGrad(gradQkvb[l], dqkvBf16, BT, 3 * C);
                    }
                }

                // 10. LN1 backward
                kernels.layerNormBackwardBf16(dresidual[l], gradLn1w[l], gradLn1b[l],
                        dln1, residual, bf16Ln1w[l], ln1Mean[l], ln1Rstd[l], BT, C);

                if (l > 0) {
                    copyTensor(dresidual[l], dresidual[l - 1]);
                }
            }
        }

        // Embedding backward
        try (CudaTensor gradWteBf16 = alloc(masterWte.getElementCount(), Precision.BF16)) {
            CudaOps.fill(device, gradWteBf16, 0.0);
            kernels.embeddingBackwardBf16(gradWteBf16, tokens, dresidual[0], B, T, C);
            // Accumulate to gradWte (BF16)
            CudaOps.add(device, gradWte, gradWteBf16, gradWte);
        }

        // Position embedding backward
        addPositionEmbeddingsBackward(gradWpe, dresidual[0], B, T, C);
    }

    // ========================================================================
    // Weight Gradient Accumulation (all GPU, BF16)
    // ========================================================================

    /**
     * dw[OC, IC] += dout^T[OC, BT] @ inp[BT, IC]
     * All BF16, no precision conversion needed.
     */
    private void accumulateWeightGrad(CudaTensor gradBf16, CudaTensor doutBf16, CudaTensor inpBf16,
                                       int OC, int IC, int BT) {
        long n = (long) OC * IC;
        try (CudaTensor temp = alloc(n, Precision.BF16)) {
            blasExt.gemmEx(true, false, OC, IC, BT,
                    1.0, doutBf16, inpBf16, 0.0, temp);
            CudaOps.add(device, gradBf16, temp, gradBf16);
        }
    }

    /**
     * Bias gradient: biasGrad[C] += sum_rows(dout[BT, C])
     * Uses broadcastAdd in reverse — column-wise reduction via CPU for now.
     * TODO: Replace with biasGradReduceBf16ToFp32 kernel when available.
     */
    private void accumulateBiasGrad(CudaTensor biasGrad, CudaTensor dout, int rows, int cols) {
        // biasGrad is BF16 for weight biases, FP32 for LN biases
        // For simplicity, do CPU reduction
        float[] biasData = biasGrad.toFloatArray();
        float[] doutData = dout.toFloatArray();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                biasData[c] += doutData[r * cols + c];
            }
        }
        TensorUtils.copyFromHost(device, biasData, biasGrad);
    }

    // ========================================================================
    // Position Embedding Helpers (CPU fallback for B > 1 tiling)
    // ========================================================================

    private void addPositionEmbeddings(CudaTensor encoded, CudaTensor wpe, int B, int T, int C) {
        if (B == 1) {
            // Simple case: broadcastAdd works directly
            kernels.broadcastAddBf16(encoded, wpe, T, C);
        } else {
            // wpe is [T, C], need to add wpe[t] to encoded[b*T+t] for all b
            // CPU fallback for now
            float[] encData = encoded.toFloatArray();
            float[] wpeData = wpe.toFloatArray();
            for (int b = 0; b < B; b++) {
                for (int t = 0; t < T; t++) {
                    int encOffset = (b * T + t) * C;
                    int wpeOffset = t * C;
                    for (int c = 0; c < C; c++) {
                        encData[encOffset + c] += wpeData[wpeOffset + c];
                    }
                }
            }
            TensorUtils.copyFromHost(device, encData, encoded);
        }
    }

    private void addPositionEmbeddingsBackward(CudaTensor dwpe, CudaTensor dout, int B, int T, int C) {
        float[] dwpeData = dwpe.toFloatArray();
        float[] doutData = dout.toFloatArray();
        for (int t = 0; t < T; t++) {
            int dwpeOffset = t * C;
            for (int b = 0; b < B; b++) {
                int doutOffset = (b * T + t) * C;
                for (int c = 0; c < C; c++) {
                    dwpeData[dwpeOffset + c] += doutData[doutOffset + c];
                }
            }
        }
        TensorUtils.copyFromHost(device, dwpeData, dwpe);
    }

    // ========================================================================
    // Reshape Helpers (FP32 — for flash attention interface)
    // ========================================================================

    private void reshapeQKVF32(CudaTensor qkv, CudaTensor q, CudaTensor k, CudaTensor v,
                                int B, int T, int NH, int HS) {
        int C = NH * HS;
        float[] qkvData = qkv.toFloatArray();
        float[] qData = new float[B * NH * T * HS];
        float[] kData = new float[B * NH * T * HS];
        float[] vData = new float[B * NH * T * HS];

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int nh = 0; nh < NH; nh++) {
                    for (int hs = 0; hs < HS; hs++) {
                        int qkvIdx = (b * T + t) * 3 * C + nh * HS + hs;
                        int outIdx = ((b * NH + nh) * T + t) * HS + hs;
                        qData[outIdx] = qkvData[qkvIdx];
                        kData[outIdx] = qkvData[qkvIdx + C];
                        vData[outIdx] = qkvData[qkvIdx + 2 * C];
                    }
                }
            }
        }
        TensorUtils.copyFromHost(device, qData, q);
        TensorUtils.copyFromHost(device, kData, k);
        TensorUtils.copyFromHost(device, vData, v);
    }

    private void reshapeAttentionOutputF32(CudaTensor attnOut, CudaTensor out,
                                            int B, int T, int NH, int HS) {
        int C = NH * HS;
        float[] attnData = attnOut.toFloatArray();
        float[] outData = new float[B * T * C];

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int nh = 0; nh < NH; nh++) {
                    for (int hs = 0; hs < HS; hs++) {
                        int attnIdx = ((b * NH + nh) * T + t) * HS + hs;
                        int outIdx = (b * T + t) * C + nh * HS + hs;
                        outData[outIdx] = attnData[attnIdx];
                    }
                }
            }
        }
        TensorUtils.copyFromHost(device, outData, out);
    }

    private void reshapeAttentionOutputBackwardF32(CudaTensor dout, CudaTensor dAttnOut,
                                                    int B, int T, int NH, int HS) {
        int C = NH * HS;
        float[] doutData = dout.toFloatArray();
        float[] dAttnData = new float[B * NH * T * HS];

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int nh = 0; nh < NH; nh++) {
                    for (int hs = 0; hs < HS; hs++) {
                        int doutIdx = (b * T + t) * C + nh * HS + hs;
                        int dAttnIdx = ((b * NH + nh) * T + t) * HS + hs;
                        dAttnData[dAttnIdx] = doutData[doutIdx];
                    }
                }
            }
        }
        TensorUtils.copyFromHost(device, dAttnData, dAttnOut);
    }

    private void reshapeQKVBackwardF32(CudaTensor dq, CudaTensor dk, CudaTensor dv,
                                        CudaTensor dqkv, int B, int T, int NH, int HS) {
        int C = NH * HS;
        float[] dqData = dq.toFloatArray();
        float[] dkData = dk.toFloatArray();
        float[] dvData = dv.toFloatArray();
        float[] dqkvData = new float[B * T * 3 * C];

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int nh = 0; nh < NH; nh++) {
                    for (int hs = 0; hs < HS; hs++) {
                        int gradIdx = ((b * NH + nh) * T + t) * HS + hs;
                        int dqkvIdx = (b * T + t) * 3 * C + nh * HS + hs;
                        dqkvData[dqkvIdx] = dqData[gradIdx];
                        dqkvData[dqkvIdx + C] = dkData[gradIdx];
                        dqkvData[dqkvIdx + 2 * C] = dvData[gradIdx];
                    }
                }
            }
        }
        TensorUtils.copyFromHost(device, dqkvData, dqkv);
    }

    // ========================================================================
    // Tensor Utilities
    // ========================================================================

    private void copyTensor(CudaTensor src, CudaTensor dst) {
        float[] data = src.toFloatArray();
        TensorUtils.copyFromHost(device, data, dst);
    }

    // ========================================================================
    // Zero Gradients
    // ========================================================================

    public void zeroGradients() {
        CudaOps.fill(device, gradWte, 0.0);
        CudaOps.fill(device, gradWpe, 0.0);
        CudaOps.fill(device, gradLnfw, 0.0);
        CudaOps.fill(device, gradLnfb, 0.0);

        for (int l = 0; l < config.numLayers; l++) {
            CudaOps.fill(device, gradLn1w[l], 0.0);
            CudaOps.fill(device, gradLn1b[l], 0.0);
            CudaOps.fill(device, gradLn2w[l], 0.0);
            CudaOps.fill(device, gradLn2b[l], 0.0);
            CudaOps.fill(device, gradQkvw[l], 0.0);
            CudaOps.fill(device, gradQkvb[l], 0.0);
            CudaOps.fill(device, gradAttnProjw[l], 0.0);
            CudaOps.fill(device, gradAttnProjb[l], 0.0);
            CudaOps.fill(device, gradFcw[l], 0.0);
            CudaOps.fill(device, gradFcb[l], 0.0);
            CudaOps.fill(device, gradFcProjw[l], 0.0);
            CudaOps.fill(device, gradFcProjb[l], 0.0);
        }
    }

    // ========================================================================
    // Weight Update — Direct BF16 AdamW
    // ========================================================================

    public void updateWeights(float lr, float beta1, float beta2,
                               float eps, float weightDecay, int step) {
        if (adamMWte == null) initAdamState();
        int C = config.channels;

        adamUpdate(masterWte, bf16Wte, adamMWte, adamVWte, gradWte,
                lr, beta1, beta2, eps, weightDecay, step, (int) masterWte.getElementCount());
        adamUpdate(masterWpe, bf16Wpe, adamMWpe, adamVWpe, gradWpe,
                lr, beta1, beta2, eps, weightDecay, step, (int) masterWpe.getElementCount());

        // LN weights: no weight decay, grads are FP32
        adamUpdateF32Grad(masterLnfw, bf16Lnfw, adamMLnfw, adamVLnfw, gradLnfw,
                lr, beta1, beta2, eps, 0.0f, step, C);
        adamUpdateF32Grad(masterLnfb, bf16Lnfb, adamMLnfb, adamVLnfb, gradLnfb,
                lr, beta1, beta2, eps, 0.0f, step, C);

        for (int l = 0; l < config.numLayers; l++) {
            adamUpdateF32Grad(masterLn1w[l], bf16Ln1w[l], adamMLn1w[l], adamVLn1w[l], gradLn1w[l],
                    lr, beta1, beta2, eps, 0.0f, step, C);
            adamUpdateF32Grad(masterLn1b[l], bf16Ln1b[l], adamMLn1b[l], adamVLn1b[l], gradLn1b[l],
                    lr, beta1, beta2, eps, 0.0f, step, C);

            adamUpdate(masterQkvw[l], bf16Qkvw[l], adamMQkvw[l], adamVQkvw[l], gradQkvw[l],
                    lr, beta1, beta2, eps, weightDecay, step, (int) masterQkvw[l].getElementCount());
            adamUpdate(masterQkvb[l], bf16Qkvb[l], adamMQkvb[l], adamVQkvb[l], gradQkvb[l],
                    lr, beta1, beta2, eps, 0.0f, step, (int) masterQkvb[l].getElementCount());

            adamUpdate(masterAttnProjw[l], bf16AttnProjw[l], adamMAttnProjw[l], adamVAttnProjw[l], gradAttnProjw[l],
                    lr, beta1, beta2, eps, weightDecay, step, (int) masterAttnProjw[l].getElementCount());
            adamUpdate(masterAttnProjb[l], bf16AttnProjb[l], adamMAttnProjb[l], adamVAttnProjb[l], gradAttnProjb[l],
                    lr, beta1, beta2, eps, 0.0f, step, (int) masterAttnProjb[l].getElementCount());

            adamUpdateF32Grad(masterLn2w[l], bf16Ln2w[l], adamMLn2w[l], adamVLn2w[l], gradLn2w[l],
                    lr, beta1, beta2, eps, 0.0f, step, C);
            adamUpdateF32Grad(masterLn2b[l], bf16Ln2b[l], adamMLn2b[l], adamVLn2b[l], gradLn2b[l],
                    lr, beta1, beta2, eps, 0.0f, step, C);

            adamUpdate(masterFcw[l], bf16Fcw[l], adamMFcw[l], adamVFcw[l], gradFcw[l],
                    lr, beta1, beta2, eps, weightDecay, step, (int) masterFcw[l].getElementCount());
            adamUpdate(masterFcb[l], bf16Fcb[l], adamMFcb[l], adamVFcb[l], gradFcb[l],
                    lr, beta1, beta2, eps, 0.0f, step, (int) masterFcb[l].getElementCount());

            adamUpdate(masterFcProjw[l], bf16FcProjw[l], adamMFcProjw[l], adamVFcProjw[l], gradFcProjw[l],
                    lr, beta1, beta2, eps, weightDecay, step, (int) masterFcProjw[l].getElementCount());
            adamUpdate(masterFcProjb[l], bf16FcProjb[l], adamMFcProjb[l], adamVFcProjb[l], gradFcProjb[l],
                    lr, beta1, beta2, eps, 0.0f, step, (int) masterFcProjb[l].getElementCount());
        }
    }

    // BF16 grads → adamwUpdateBf16 directly
    private void adamUpdate(CudaTensor master, CudaTensor bf16,
                             CudaTensor adamM, CudaTensor adamV,
                             CudaTensor gradBf16,
                             float lr, float beta1, float beta2,
                             float eps, float weightDecay, int step, int n) {
        kernels.adamwUpdateBf16(master, bf16, adamM, adamV, gradBf16,
                lr, beta1, beta2, eps, weightDecay, step, n);
    }

    // FP32 LN grads → convert to BF16 → adamwUpdateBf16
    private void adamUpdateF32Grad(CudaTensor master, CudaTensor bf16,
                                    CudaTensor adamM, CudaTensor adamV,
                                    CudaTensor gradFp32,
                                    float lr, float beta1, float beta2,
                                    float eps, float weightDecay, int step, int n) {
        try (CudaTensor gradBf16 = alloc(n, Precision.BF16)) {
            kernels.convertF32ToBf16(gradFp32, gradBf16, n);
            kernels.adamwUpdateBf16(master, bf16, adamM, adamV, gradBf16,
                    lr, beta1, beta2, eps, weightDecay, step, n);
        }
    }

    private void initAdamState() {
        adamMWte = alloc(masterWte.getElementCount(), Precision.FP32);
        adamVWte = alloc(masterWte.getElementCount(), Precision.FP32);
        adamMWpe = alloc(masterWpe.getElementCount(), Precision.FP32);
        adamVWpe = alloc(masterWpe.getElementCount(), Precision.FP32);
        adamMLnfw = alloc(masterLnfw.getElementCount(), Precision.FP32);
        adamVLnfw = alloc(masterLnfw.getElementCount(), Precision.FP32);
        adamMLnfb = alloc(masterLnfb.getElementCount(), Precision.FP32);
        adamVLnfb = alloc(masterLnfb.getElementCount(), Precision.FP32);

        int L = config.numLayers;
        int C = config.channels;
        adamMLn1w = new CudaTensor[L]; adamVLn1w = new CudaTensor[L];
        adamMLn1b = new CudaTensor[L]; adamVLn1b = new CudaTensor[L];
        adamMQkvw = new CudaTensor[L]; adamVQkvw = new CudaTensor[L];
        adamMQkvb = new CudaTensor[L]; adamVQkvb = new CudaTensor[L];
        adamMAttnProjw = new CudaTensor[L]; adamVAttnProjw = new CudaTensor[L];
        adamMAttnProjb = new CudaTensor[L]; adamVAttnProjb = new CudaTensor[L];
        adamMLn2w = new CudaTensor[L]; adamVLn2w = new CudaTensor[L];
        adamMLn2b = new CudaTensor[L]; adamVLn2b = new CudaTensor[L];
        adamMFcw = new CudaTensor[L]; adamVFcw = new CudaTensor[L];
        adamMFcb = new CudaTensor[L]; adamVFcb = new CudaTensor[L];
        adamMFcProjw = new CudaTensor[L]; adamVFcProjw = new CudaTensor[L];
        adamMFcProjb = new CudaTensor[L]; adamVFcProjb = new CudaTensor[L];

        for (int l = 0; l < L; l++) {
            adamMLn1w[l] = alloc(C, Precision.FP32); adamVLn1w[l] = alloc(C, Precision.FP32);
            adamMLn1b[l] = alloc(C, Precision.FP32); adamVLn1b[l] = alloc(C, Precision.FP32);
            adamMQkvw[l] = alloc((long) 3*C*C, Precision.FP32); adamVQkvw[l] = alloc((long) 3*C*C, Precision.FP32);
            adamMQkvb[l] = alloc(3*C, Precision.FP32); adamVQkvb[l] = alloc(3*C, Precision.FP32);
            adamMAttnProjw[l] = alloc((long) C*C, Precision.FP32); adamVAttnProjw[l] = alloc((long) C*C, Precision.FP32);
            adamMAttnProjb[l] = alloc(C, Precision.FP32); adamVAttnProjb[l] = alloc(C, Precision.FP32);
            adamMLn2w[l] = alloc(C, Precision.FP32); adamVLn2w[l] = alloc(C, Precision.FP32);
            adamMLn2b[l] = alloc(C, Precision.FP32); adamVLn2b[l] = alloc(C, Precision.FP32);
            adamMFcw[l] = alloc((long) 4*C*C, Precision.FP32); adamVFcw[l] = alloc((long) 4*C*C, Precision.FP32);
            adamMFcb[l] = alloc(4*C, Precision.FP32); adamVFcb[l] = alloc(4*C, Precision.FP32);
            adamMFcProjw[l] = alloc((long) 4*C*C, Precision.FP32); adamVFcProjw[l] = alloc((long) 4*C*C, Precision.FP32);
            adamMFcProjb[l] = alloc(C, Precision.FP32); adamVFcProjb[l] = alloc(C, Precision.FP32);
        }
    }

    // ========================================================================
    // Getters
    // ========================================================================

    public float getMeanLoss() { return meanLoss; }

    // ========================================================================
    // Close
    // ========================================================================

    private void ensureNotClosed() {
        if (closed) throw new IllegalStateException("BF16GPT2 is closed");
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            System.out.println("BF16GPT2 closed");
        }
    }
}
