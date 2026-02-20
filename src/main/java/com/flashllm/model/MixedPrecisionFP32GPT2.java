package com.flashllm.model;

import com.flashllm.kernel.TensorUtils;
import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.config.GPT2Config;

import java.io.IOException;

/**
 * Mixed-Precision GPT-2 model with FP16 training.
 * 
 * Key design: forward/backward compute in FP16, but all gradient STORAGE is FP32
 * to avoid overflow during accumulation. FP32 grads are converted to FP16 just
 * before AdamW update.
 *
 * @author flash-llm
 * @since 2.3.0
 */
public class MixedPrecisionFP32GPT2 implements AutoCloseable {

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

    // FP16 Weights (used in forward/backward compute)
    private CudaTensor fp16Wte, fp16Wpe, fp16Lnfw, fp16Lnfb;
    private CudaTensor[] fp16Ln1w, fp16Ln1b;
    private CudaTensor[] fp16Qkvw, fp16Qkvb;
    private CudaTensor[] fp16AttnProjw, fp16AttnProjb;
    private CudaTensor[] fp16Ln2w, fp16Ln2b;
    private CudaTensor[] fp16Fcw, fp16Fcb;
    private CudaTensor[] fp16FcProjw, fp16FcProjb;

    // *** ALL Gradients stored in FP32 to avoid overflow ***
    private CudaTensor gradWte, gradWpe;       // FP32
    private CudaTensor gradLnfw, gradLnfb;     // FP32
    private CudaTensor[] gradLn1w, gradLn1b;   // FP32
    private CudaTensor[] gradLn2w, gradLn2b;   // FP32
    private CudaTensor[] gradQkvw, gradQkvb;   // FP32
    private CudaTensor[] gradAttnProjw, gradAttnProjb; // FP32
    private CudaTensor[] gradFcw, gradFcb;     // FP32
    private CudaTensor[] gradFcProjw, gradFcProjb; // FP32

    // FP16 Activations
    private CudaTensor encoded;
    private CudaTensor[] ln1, ln1Mean, ln1Rstd;
    private CudaTensor[] qkv, attnOut, residual1;
    private CudaTensor[] attLse;  // flash attention log-sum-exp (FP32)
    private CudaTensor[] ln2, ln2Mean, ln2Rstd;
    private CudaTensor[] fch, fchGelu, residual2;
    private CudaTensor lnf, lnfMean, lnfRstd;
    private CudaTensor logits, probs;
    private CudaTensor dlogits, dlnf;
    private CudaTensor[] dresidual;

    // AdamW State (FP32)
    private CudaTensor adamMWte, adamVWte;
    private CudaTensor adamMWpe, adamVWpe;
    private CudaTensor adamMLnfw, adamVLnfw;
    private CudaTensor adamMLnfb, adamVLnfb;
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
    private boolean backwardOverflow = false;

    public MixedPrecisionFP32GPT2(GPT2Config config, int B, int T) {
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

        System.out.println("Creating Mixed-Precision GPT-2:");
        System.out.println("  Layers: " + L + ", Channels: " + C + ", Heads: " + config.numHeads);
        System.out.println("  Batch: " + B + ", SeqLen: " + T);

        initializeArrays(L);
        allocateActivations(L, C, V, BT);

        System.out.println("  Model created!");
    }

    private void initializeArrays(int L) {
        masterLn1w = new CudaTensor[L]; masterLn1b = new CudaTensor[L];
        masterQkvw = new CudaTensor[L]; masterQkvb = new CudaTensor[L];
        masterAttnProjw = new CudaTensor[L]; masterAttnProjb = new CudaTensor[L];
        masterLn2w = new CudaTensor[L]; masterLn2b = new CudaTensor[L];
        masterFcw = new CudaTensor[L]; masterFcb = new CudaTensor[L];
        masterFcProjw = new CudaTensor[L]; masterFcProjb = new CudaTensor[L];

        fp16Ln1w = new CudaTensor[L]; fp16Ln1b = new CudaTensor[L];
        fp16Qkvw = new CudaTensor[L]; fp16Qkvb = new CudaTensor[L];
        fp16AttnProjw = new CudaTensor[L]; fp16AttnProjb = new CudaTensor[L];
        fp16Ln2w = new CudaTensor[L]; fp16Ln2b = new CudaTensor[L];
        fp16Fcw = new CudaTensor[L]; fp16Fcb = new CudaTensor[L];
        fp16FcProjw = new CudaTensor[L]; fp16FcProjb = new CudaTensor[L];

        gradLn1w = new CudaTensor[L]; gradLn1b = new CudaTensor[L];
        gradLn2w = new CudaTensor[L]; gradLn2b = new CudaTensor[L];
        gradQkvw = new CudaTensor[L]; gradQkvb = new CudaTensor[L];
        gradAttnProjw = new CudaTensor[L]; gradAttnProjb = new CudaTensor[L];
        gradFcw = new CudaTensor[L]; gradFcb = new CudaTensor[L];
        gradFcProjw = new CudaTensor[L]; gradFcProjb = new CudaTensor[L];

        ln1 = new CudaTensor[L]; ln1Mean = new CudaTensor[L]; ln1Rstd = new CudaTensor[L];
        qkv = new CudaTensor[L]; attnOut = new CudaTensor[L]; residual1 = new CudaTensor[L];
        attLse = new CudaTensor[L];
        ln2 = new CudaTensor[L]; ln2Mean = new CudaTensor[L]; ln2Rstd = new CudaTensor[L];
        fch = new CudaTensor[L]; fchGelu = new CudaTensor[L]; residual2 = new CudaTensor[L];
        dresidual = new CudaTensor[L];
    }

    private void allocateActivations(int L, int C, int V, int BT) {
        encoded = CudaTensor.allocate(device, (long) BT * C, Precision.FP16);

        for (int l = 0; l < L; l++) {
            ln1[l] = CudaTensor.allocate(device, (long) BT * C, Precision.FP16);
            ln1Mean[l] = CudaTensor.allocate(device, BT, Precision.FP32);
            ln1Rstd[l] = CudaTensor.allocate(device, BT, Precision.FP32);
            qkv[l] = CudaTensor.allocate(device, (long) BT * 3 * C, Precision.FP16);
            attnOut[l] = CudaTensor.allocate(device, (long) BT * C, Precision.FP16);
            attLse[l] = CudaTensor.allocate(device, (long) B * config.numHeads * T, Precision.FP32);
            residual1[l] = CudaTensor.allocate(device, (long) BT * C, Precision.FP16);
            ln2[l] = CudaTensor.allocate(device, (long) BT * C, Precision.FP16);
            ln2Mean[l] = CudaTensor.allocate(device, BT, Precision.FP32);
            ln2Rstd[l] = CudaTensor.allocate(device, BT, Precision.FP32);
            fch[l] = CudaTensor.allocate(device, (long) BT * 4 * C, Precision.FP16);
            fchGelu[l] = CudaTensor.allocate(device, (long) BT * 4 * C, Precision.FP16);
            residual2[l] = CudaTensor.allocate(device, (long) BT * C, Precision.FP16);
            dresidual[l] = CudaTensor.allocate(device, (long) BT * C, Precision.FP16);
        }

        lnf = CudaTensor.allocate(device, (long) BT * C, Precision.FP16);
        lnfMean = CudaTensor.allocate(device, BT, Precision.FP32);
        lnfRstd = CudaTensor.allocate(device, BT, Precision.FP32);
        logits = CudaTensor.allocate(device, (long) BT * V, Precision.FP16);
        probs = CudaTensor.allocate(device, (long) BT * V, Precision.FP16);
        dlogits = CudaTensor.allocate(device, (long) BT * V, Precision.FP16);
        dlnf = CudaTensor.allocate(device, (long) BT * C, Precision.FP16);
    }

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

        // FP16 weights
        fp16Wte = CudaTensor.allocate(device, masterWte.getElementCount(), Precision.FP16);
        fp16Wpe = CudaTensor.allocate(device, masterWpe.getElementCount(), Precision.FP16);
        fp16Lnfw = CudaTensor.allocate(device, C, Precision.FP16);
        fp16Lnfb = CudaTensor.allocate(device, C, Precision.FP16);

        // *** ALL gradients FP32 ***
        gradWte = CudaTensor.allocate(device, masterWte.getElementCount(), Precision.FP32);
        gradWpe = CudaTensor.allocate(device, masterWpe.getElementCount(), Precision.FP32);
        gradLnfw = CudaTensor.allocate(device, C, Precision.FP32);
        gradLnfb = CudaTensor.allocate(device, C, Precision.FP32);

        for (int l = 0; l < L; l++) {
            // Master (FP32)
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

            // FP16 weights
            fp16Ln1w[l] = CudaTensor.allocate(device, C, Precision.FP16);
            fp16Ln1b[l] = CudaTensor.allocate(device, C, Precision.FP16);
            fp16Qkvw[l] = CudaTensor.allocate(device, (long) 3 * C * C, Precision.FP16);
            fp16Qkvb[l] = CudaTensor.allocate(device, 3 * C, Precision.FP16);
            fp16AttnProjw[l] = CudaTensor.allocate(device, (long) C * C, Precision.FP16);
            fp16AttnProjb[l] = CudaTensor.allocate(device, C, Precision.FP16);
            fp16Ln2w[l] = CudaTensor.allocate(device, C, Precision.FP16);
            fp16Ln2b[l] = CudaTensor.allocate(device, C, Precision.FP16);
            fp16Fcw[l] = CudaTensor.allocate(device, (long) 4 * C * C, Precision.FP16);
            fp16Fcb[l] = CudaTensor.allocate(device, 4 * C, Precision.FP16);
            fp16FcProjw[l] = CudaTensor.allocate(device, (long) 4 * C * C, Precision.FP16);
            fp16FcProjb[l] = CudaTensor.allocate(device, C, Precision.FP16);

            // *** ALL gradients FP32 ***
            gradLn1w[l] = CudaTensor.allocate(device, C, Precision.FP32);
            gradLn1b[l] = CudaTensor.allocate(device, C, Precision.FP32);
            gradLn2w[l] = CudaTensor.allocate(device, C, Precision.FP32);
            gradLn2b[l] = CudaTensor.allocate(device, C, Precision.FP32);
            gradQkvw[l] = CudaTensor.allocate(device, (long) 3 * C * C, Precision.FP32);
            gradQkvb[l] = CudaTensor.allocate(device, 3 * C, Precision.FP32);
            gradAttnProjw[l] = CudaTensor.allocate(device, (long) C * C, Precision.FP32);
            gradAttnProjb[l] = CudaTensor.allocate(device, C, Precision.FP32);
            gradFcw[l] = CudaTensor.allocate(device, (long) 4 * C * C, Precision.FP32);
            gradFcb[l] = CudaTensor.allocate(device, 4 * C, Precision.FP32);
            gradFcProjw[l] = CudaTensor.allocate(device, (long) C * 4 * C, Precision.FP32);
            gradFcProjb[l] = CudaTensor.allocate(device, C, Precision.FP32);
        }

        syncWeightsToFp16();
        device.synchronize();
        weightsLoaded = true;
        System.out.println("Weights loaded!");
    }

    public void syncWeightsToFp16() {
        kernels.convertF32ToF16(masterWte, fp16Wte, (int) masterWte.getElementCount());
        kernels.convertF32ToF16(masterWpe, fp16Wpe, (int) masterWpe.getElementCount());
        kernels.convertF32ToF16(masterLnfw, fp16Lnfw, (int) masterLnfw.getElementCount());
        kernels.convertF32ToF16(masterLnfb, fp16Lnfb, (int) masterLnfb.getElementCount());

        for (int l = 0; l < config.numLayers; l++) {
            kernels.convertF32ToF16(masterLn1w[l], fp16Ln1w[l], (int) masterLn1w[l].getElementCount());
            kernels.convertF32ToF16(masterLn1b[l], fp16Ln1b[l], (int) masterLn1b[l].getElementCount());
            kernels.convertF32ToF16(masterQkvw[l], fp16Qkvw[l], (int) masterQkvw[l].getElementCount());
            kernels.convertF32ToF16(masterQkvb[l], fp16Qkvb[l], (int) masterQkvb[l].getElementCount());
            kernels.convertF32ToF16(masterAttnProjw[l], fp16AttnProjw[l], (int) masterAttnProjw[l].getElementCount());
            kernels.convertF32ToF16(masterAttnProjb[l], fp16AttnProjb[l], (int) masterAttnProjb[l].getElementCount());
            kernels.convertF32ToF16(masterLn2w[l], fp16Ln2w[l], (int) masterLn2w[l].getElementCount());
            kernels.convertF32ToF16(masterLn2b[l], fp16Ln2b[l], (int) masterLn2b[l].getElementCount());
            kernels.convertF32ToF16(masterFcw[l], fp16Fcw[l], (int) masterFcw[l].getElementCount());
            kernels.convertF32ToF16(masterFcb[l], fp16Fcb[l], (int) masterFcb[l].getElementCount());
            kernels.convertF32ToF16(masterFcProjw[l], fp16FcProjw[l], (int) masterFcProjw[l].getElementCount());
            kernels.convertF32ToF16(masterFcProjb[l], fp16FcProjb[l], (int) masterFcProjb[l].getElementCount());
        }

        device.synchronize();
        System.out.println("=== ALL syncWeightsToFp16 COMPLETE ===");
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
        int BT = B * T;

        // 1. Embedding
        kernels.embeddingForwardFp16(encoded, tokens, fp16Wte, B, T, C);
        addPositionEmbeddingsFp16(encoded, fp16Wpe, B, T, C);

        // 2. Transformer blocks
        CudaTensor residual = encoded;
        for (int l = 0; l < L; l++) {
            // LN1
            kernels.layerNormForwardFp16(ln1[l], ln1Mean[l], ln1Rstd[l],
                    residual, fp16Ln1w[l], fp16Ln1b[l], BT, C);

            // QKV
            blasExt.gemmEx(false, true, BT, 3 * C, C,
                    1.0, ln1[l], fp16Qkvw[l], 0.0, qkv[l]);
            addBiasFp16(qkv[l], fp16Qkvb[l], BT, 3 * C);

            // Attention
            {
                int NH = config.numHeads;
                int HS = C / NH;
                int BNH = B * NH;

                try (CudaTensor qkvF32 = CudaTensor.allocate(device, (long) BT * 3 * C, Precision.FP32);
                     CudaTensor qF32 = CudaTensor.allocate(device, (long) BNH * T * HS, Precision.FP32);
                     CudaTensor kF32 = CudaTensor.allocate(device, (long) BNH * T * HS, Precision.FP32);
                     CudaTensor vF32 = CudaTensor.allocate(device, (long) BNH * T * HS, Precision.FP32);
                     CudaTensor attnOutF32 = CudaTensor.allocate(device, (long) BNH * T * HS, Precision.FP32);
                     CudaTensor attnReshF32 = CudaTensor.allocate(device, (long) BT * C, Precision.FP32);
                     CudaTensor attnProjFp16 = CudaTensor.allocate(device, (long) BT * C, Precision.FP16)) {

                    kernels.convertF16ToF32(qkv[l], qkvF32, BT * 3 * C);
                    reshapeQKV(qkvF32, qF32, kF32, vF32, B, T, NH, HS);
                    kernels.flashAttentionForward(attnOutF32, attLse[l], qF32, kF32, vF32, BNH, T, HS, true);
                    reshapeAttentionOutput(attnOutF32, attnReshF32, B, T, NH, HS);
                    kernels.convertF32ToF16(attnReshF32, attnOut[l], BT * C);

                    blasExt.gemmEx(false, true, BT, C, C,
                            1.0, attnOut[l], fp16AttnProjw[l], 0.0, attnProjFp16);
                    addBiasFp16(attnProjFp16, fp16AttnProjb[l], BT, C);
                    CudaOps.add(device, residual, attnProjFp16, residual1[l]);
                }
            }

            // LN2
            kernels.layerNormForwardFp16(ln2[l], ln2Mean[l], ln2Rstd[l],
                    residual1[l], fp16Ln2w[l], fp16Ln2b[l], BT, C);

            // MLP
            blasExt.gemmEx(false, true, BT, 4 * C, C,
                    1.0, ln2[l], fp16Fcw[l], 0.0, fch[l]);
            addBiasFp16(fch[l], fp16Fcb[l], BT, 4 * C);
            kernels.geluForwardFp16(fchGelu[l], fch[l], BT * 4 * C);

            try (CudaTensor mlpProj = CudaTensor.allocate(device, (long) BT * C, Precision.FP16)) {
                blasExt.gemmEx(false, true, BT, C, 4 * C,
                        1.0, fchGelu[l], fp16FcProjw[l], 0.0, mlpProj);
                addBiasFp16(mlpProj, fp16FcProjb[l], BT, C);
                CudaOps.add(device, residual1[l], mlpProj, residual2[l]);
            }

            residual = residual2[l];
        }

        // 3. Final LN
        kernels.layerNormForwardFp16(lnf, lnfMean, lnfRstd,
                residual, fp16Lnfw, fp16Lnfb, BT, C);

        // 4. Output projection
        blasExt.gemmEx(false, true, BT, V, C,
                1.0, lnf, fp16Wte, 0.0, logits);

        // 5. Loss
        if (targets != null) {
            kernels.softmaxForwardFp16(probs, logits, BT, V);
            meanLoss = computeLoss(targets);
            return meanLoss;
        }
        return 0.0f;
    }

    // ========================================================================
    // Backward Pass
    // ========================================================================

    public void backward(int[] tokens, int[] targets, float lossScale) {
        backwardOverflow = false;
        ensureNotClosed();
        int C = config.channels;
        int V = config.vocabSize;
        int L = config.numLayers;
        int NH = config.numHeads;
        int HS = C / NH;
        int BT = B * T;
        int BNH = B * NH;
        float scale = lossScale / BT;

        // dlogits = (probs - onehot) * scale
        kernels.crossEntropySoftmaxBackwardFp16(dlogits, probs, targets, scale, B, T, V);

        // dlnf = dlogits @ wte
        blasExt.gemmEx(false, false, BT, C, V,
                1.0, dlogits, fp16Wte, 0.0, dlnf);

        // dwte += lnf^T @ dlogits  (FP16 compute â†’ accumulate to FP32 via CPU)
        accumulateWeightGrad(gradWte, dlogits, lnf, V, C, BT);

        // Final LN backward
        kernels.layerNormBackwardFp16(dresidual[L - 1], gradLnfw, gradLnfb,
                dlnf, residual2[L - 1], fp16Lnfw, lnfMean, lnfRstd, BT, C);

        // === Transformer blocks backward ===
        for (int l = L - 1; l >= 0; l--) {
            CudaTensor residual = (l == 0) ? encoded : residual2[l - 1];

            try (CudaTensor dfcproj = CudaTensor.allocate(device, (long) BT * C, Precision.FP16);
                 CudaTensor dfchGelu = CudaTensor.allocate(device, (long) BT * 4 * C, Precision.FP16);
                 CudaTensor dfch = CudaTensor.allocate(device, (long) BT * 4 * C, Precision.FP16);
                 CudaTensor dln2 = CudaTensor.allocate(device, (long) BT * C, Precision.FP16);
                 CudaTensor dresidual2 = CudaTensor.allocate(device, (long) BT * C, Precision.FP16);
                 CudaTensor dattproj = CudaTensor.allocate(device, (long) BT * C, Precision.FP16);
                 CudaTensor dln1 = CudaTensor.allocate(device, (long) BT * C, Precision.FP16)) {

                // 1. Residual backward
                copyTensor(dresidual[l], dresidual2);
                copyTensor(dresidual[l], dfcproj);

                // 2. FC proj input gradient (activation grad only, no weight grad yet)
                blasExt.gemmEx(false, false, BT, 4 * C, C,
                        1.0, dfcproj, fp16FcProjw[l], 0.0, dfchGelu);

                // 3. GELU backward
                kernels.geluBackwardFp16(dfch, fch[l], dfchGelu, BT * 4 * C);

                // 4. FC input gradient
                blasExt.gemmEx(false, false, BT, C, 4 * C,
                        1.0, dfch, fp16Fcw[l], 0.0, dln2);

                // *** Overflow check on dln2 (most likely to overflow) ***
                if (checkFp16Overflow(dln2)) break;

                // NOW safe to accumulate weight gradients for steps 2 & 4
                // FC proj weight grad (FP32 accumulate via CPU)
                accumulateWeightGrad(gradFcProjw[l], dfcproj, fchGelu[l], C, 4 * C, BT);
                addBiasGradFp32(gradFcProjb[l], dfcproj, BT, C);

                // FC weight grad (FP32 accumulate via CPU)
                accumulateWeightGrad(gradFcw[l], dfch, ln2[l], 4 * C, C, BT);
                addBiasGradFp32(gradFcb[l], dfch, BT, 4 * C);

                // 5. LN2 backward
                kernels.layerNormBackwardFp16(dresidual2, gradLn2w[l], gradLn2b[l],
                        dln2, residual1[l], fp16Ln2w[l], ln2Mean[l], ln2Rstd[l], BT, C);

                if (checkFp16Overflow(dresidual2)) break;

                // 6. Residual backward
                copyTensor(dresidual2, dresidual[l]);
                copyTensor(dresidual2, dattproj);

                // 7. Attn proj backward
                try (CudaTensor dattnResh = CudaTensor.allocate(device, (long) BT * C, Precision.FP16)) {
                    blasExt.gemmEx(false, false, BT, C, C,
                            1.0, dattproj, fp16AttnProjw[l], 0.0, dattnResh);

                    // Attn proj weight grad (FP32 accumulate via CPU)
                    accumulateWeightGrad(gradAttnProjw[l], dattproj, attnOut[l], C, C, BT);
                    addBiasGradFp32(gradAttnProjb[l], dattproj, BT, C);

                    // 8. Flash attention backward
                    try (CudaTensor dattnReshF32 = CudaTensor.allocate(device, (long) BT * C, Precision.FP32);
                         CudaTensor qkvF32 = CudaTensor.allocate(device, (long) BT * 3 * C, Precision.FP32);
                         CudaTensor qF32 = CudaTensor.allocate(device, (long) BNH * T * HS, Precision.FP32);
                         CudaTensor kF32 = CudaTensor.allocate(device, (long) BNH * T * HS, Precision.FP32);
                         CudaTensor vF32 = CudaTensor.allocate(device, (long) BNH * T * HS, Precision.FP32);
                         CudaTensor dqF32 = CudaTensor.allocate(device, (long) BNH * T * HS, Precision.FP32);
                         CudaTensor dkF32 = CudaTensor.allocate(device, (long) BNH * T * HS, Precision.FP32);
                         CudaTensor dvF32 = CudaTensor.allocate(device, (long) BNH * T * HS, Precision.FP32);
                         CudaTensor attnOutF32 = CudaTensor.allocate(device, (long) BNH * T * HS, Precision.FP32);
                         CudaTensor dAttnOutF32 = CudaTensor.allocate(device, (long) BNH * T * HS, Precision.FP32);
                         CudaTensor dqkvF32 = CudaTensor.allocate(device, (long) BT * 3 * C, Precision.FP32);
                         CudaTensor dqkvFp16 = CudaTensor.allocate(device, (long) BT * 3 * C, Precision.FP16)) {

                        kernels.convertF16ToF32(dattnResh, dattnReshF32, BT * C);
                        reshapeAttentionOutputBackward(dattnReshF32, dAttnOutF32, B, T, NH, HS);

                        kernels.convertF16ToF32(qkv[l], qkvF32, BT * 3 * C);
                        reshapeQKV(qkvF32, qF32, kF32, vF32, B, T, NH, HS);

                        kernels.flashAttentionForward(attnOutF32, attLse[l], qF32, kF32, vF32, BNH, T, HS, true);
                        kernels.flashAttentionBackward(dqF32, dkF32, dvF32,
                                qF32, kF32, vF32, attnOutF32, dAttnOutF32, attLse[l], BNH, T, HS, true);

                        reshapeQKVBackward(dqF32, dkF32, dvF32, dqkvF32, B, T, NH, HS);
                        kernels.convertF32ToF16(dqkvF32, dqkvFp16, BT * 3 * C);

                        // 9. QKV backward
                        blasExt.gemmEx(false, false, BT, C, 3 * C,
                                1.0, dqkvFp16, fp16Qkvw[l], 0.0, dln1);

                        // QKV weight grad (FP32 accumulate via CPU)
                        accumulateWeightGrad(gradQkvw[l], dqkvFp16, ln1[l], 3 * C, C, BT);
                        addBiasGradFp32(gradQkvb[l], dqkvFp16, BT, 3 * C);
                    }
                }

                // 10. LN1 backward
                kernels.layerNormBackwardFp16(dresidual[l], gradLn1w[l], gradLn1b[l],
                        dln1, residual, fp16Ln1w[l], ln1Mean[l], ln1Rstd[l], BT, C);

                if (checkFp16Overflow(dresidual[l])) break;

                if (l > 0) {
                    copyTensor(dresidual[l], dresidual[l - 1]);
                }
            }
            if (backwardOverflow) break;
        }

        if (!backwardOverflow) {
            // Embedding backward needs FP16 grad, then accumulate to FP32
            try (CudaTensor gradWteFp16 = CudaTensor.allocate(device, masterWte.getElementCount(), Precision.FP16)) {
                CudaOps.fill(device, gradWteFp16, 0.0);
                kernels.embeddingBackwardFp16(gradWteFp16, tokens, dresidual[0], B, T, C);
                // Accumulate FP16 â†’ FP32
                float[] fp16Data = gradWteFp16.toFloatArray();
                float[] fp32Data = gradWte.toFloatArray();
                for (int i = 0; i < fp16Data.length; i++) {
                    fp32Data[i] += fp16Data[i];
                }
                TensorUtils.copyFromHost(device, fp32Data, gradWte);
            }
            addPositionEmbeddingsBackwardFp32(gradWpe, dresidual[0], B, T, C);
        }
    }

    // ========================================================================
    // Weight gradient accumulation (FP16 tensor â†’ FP32 grad via CPU)
    // C_grad[M, N] += A^T[M, BT] @ B[BT, N]  (in row-major terms)
    // A is FP16 [BT, M], B is FP16 [BT, N], grad is FP32 [M, N]
    // ========================================================================
    private void accumulateWeightGrad(CudaTensor gradFp32, CudaTensor aFp16, CudaTensor bFp16,
                                       int M, int N, int BT) {
        float[] aData = aFp16.toFloatArray(); // [BT * M]
        float[] bData = bFp16.toFloatArray(); // [BT * N]
        float[] gData = gradFp32.toFloatArray(); // [M * N]

        // grad[m, n] += sum_over_bt( a[bt, m] * b[bt, n] )
        for (int bt = 0; bt < BT; bt++) {
            for (int m = 0; m < M; m++) {
                float aVal = aData[bt * M + m];
                int gOffset = m * N;
                int bOffset = bt * N;
                for (int n = 0; n < N; n++) {
                    gData[gOffset + n] += aVal * bData[bOffset + n];
                }
            }
        }

        TensorUtils.copyFromHost(device, gData, gradFp32);
    }

    // Bias gradient: FP32 accumulation from FP16 dout
    private void addBiasGradFp32(CudaTensor biasGradFp32, CudaTensor doutFp16, int rows, int cols) {
        float[] biasData = biasGradFp32.toFloatArray(); // FP32
        float[] doutData = doutFp16.toFloatArray();      // FP16 â†’ float
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                biasData[c] += doutData[r * cols + c];
            }
        }
        TensorUtils.copyFromHost(device, biasData, biasGradFp32); // stays FP32
    }

    // Embedding backward into FP32 grad
    private void addPositionEmbeddingsBackwardFp32(CudaTensor dwpe, CudaTensor dout,
                                                    int B, int T, int C) {
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

    private boolean checkFp16Overflow(CudaTensor t) {
        float[] data = t.toFloatArray();
        for (float v : data) {
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                backwardOverflow = true;
                return true;
            }
        }
        return false;
    }

    // ========================================================================
    // Gradient Overflow Detection
    // ========================================================================

    public boolean hasGradientOverflow() {
        if (backwardOverflow) return true;
        // Check FP32 weight grads for NaN/Inf
        if (checkTensorOverflow(gradWte)) return true;
        if (checkTensorOverflow(gradWpe)) return true;
        for (int l = 0; l < config.numLayers; l++) {
            if (checkTensorOverflow(gradQkvw[l])) return true;
            if (checkTensorOverflow(gradFcw[l])) return true;
            if (checkTensorOverflow(gradFcProjw[l])) return true;
            if (checkTensorOverflow(gradAttnProjw[l])) return true;
        }
        return false;
    }

    public boolean isBackwardOverflow() { return backwardOverflow; }

    private boolean checkTensorOverflow(CudaTensor t) {
        float[] data = t.toFloatArray();
        for (float v : data) {
            if (Float.isNaN(v) || Float.isInfinite(v)) return true;
        }
        return false;
    }

    // ========================================================================
    // Weight Update (all grads are FP32, convert to FP16 for adamwUpdateFp16)
    // ========================================================================

    public void updateMasterWeights(float lr, float beta1, float beta2,
                                    float eps, float weightDecay, int step) {
        if (adamMWte == null) initAdamState();
        int C = config.channels;

        adamUpdateFp32Grad(masterWte, fp16Wte, adamMWte, adamVWte, gradWte,
                lr, beta1, beta2, eps, weightDecay, step, (int) masterWte.getElementCount());
        adamUpdateFp32Grad(masterWpe, fp16Wpe, adamMWpe, adamVWpe, gradWpe,
                lr, beta1, beta2, eps, weightDecay, step, (int) masterWpe.getElementCount());
        adamUpdateFp32Grad(masterLnfw, fp16Lnfw, adamMLnfw, adamVLnfw, gradLnfw,
                lr, beta1, beta2, eps, 0.0f, step, C);
        adamUpdateFp32Grad(masterLnfb, fp16Lnfb, adamMLnfb, adamVLnfb, gradLnfb,
                lr, beta1, beta2, eps, 0.0f, step, C);

        for (int l = 0; l < config.numLayers; l++) {
            adamUpdateFp32Grad(masterLn1w[l], fp16Ln1w[l], adamMLn1w[l], adamVLn1w[l], gradLn1w[l],
                    lr, beta1, beta2, eps, 0.0f, step, C);
            adamUpdateFp32Grad(masterLn1b[l], fp16Ln1b[l], adamMLn1b[l], adamVLn1b[l], gradLn1b[l],
                    lr, beta1, beta2, eps, 0.0f, step, C);

            adamUpdateFp32Grad(masterQkvw[l], fp16Qkvw[l], adamMQkvw[l], adamVQkvw[l], gradQkvw[l],
                    lr, beta1, beta2, eps, weightDecay, step, (int) masterQkvw[l].getElementCount());
            adamUpdateFp32Grad(masterQkvb[l], fp16Qkvb[l], adamMQkvb[l], adamVQkvb[l], gradQkvb[l],
                    lr, beta1, beta2, eps, 0.0f, step, (int) masterQkvb[l].getElementCount());

            adamUpdateFp32Grad(masterAttnProjw[l], fp16AttnProjw[l], adamMAttnProjw[l], adamVAttnProjw[l], gradAttnProjw[l],
                    lr, beta1, beta2, eps, weightDecay, step, (int) masterAttnProjw[l].getElementCount());
            adamUpdateFp32Grad(masterAttnProjb[l], fp16AttnProjb[l], adamMAttnProjb[l], adamVAttnProjb[l], gradAttnProjb[l],
                    lr, beta1, beta2, eps, 0.0f, step, (int) masterAttnProjb[l].getElementCount());

            adamUpdateFp32Grad(masterLn2w[l], fp16Ln2w[l], adamMLn2w[l], adamVLn2w[l], gradLn2w[l],
                    lr, beta1, beta2, eps, 0.0f, step, C);
            adamUpdateFp32Grad(masterLn2b[l], fp16Ln2b[l], adamMLn2b[l], adamVLn2b[l], gradLn2b[l],
                    lr, beta1, beta2, eps, 0.0f, step, C);

            adamUpdateFp32Grad(masterFcw[l], fp16Fcw[l], adamMFcw[l], adamVFcw[l], gradFcw[l],
                    lr, beta1, beta2, eps, weightDecay, step, (int) masterFcw[l].getElementCount());
            adamUpdateFp32Grad(masterFcb[l], fp16Fcb[l], adamMFcb[l], adamVFcb[l], gradFcb[l],
                    lr, beta1, beta2, eps, 0.0f, step, (int) masterFcb[l].getElementCount());

            adamUpdateFp32Grad(masterFcProjw[l], fp16FcProjw[l], adamMFcProjw[l], adamVFcProjw[l], gradFcProjw[l],
                    lr, beta1, beta2, eps, weightDecay, step, (int) masterFcProjw[l].getElementCount());
            adamUpdateFp32Grad(masterFcProjb[l], fp16FcProjb[l], adamMFcProjb[l], adamVFcProjb[l], gradFcProjb[l],
                    lr, beta1, beta2, eps, 0.0f, step, (int) masterFcProjb[l].getElementCount());
        }
    }

    // Convert FP32 grad â†’ FP16, then call adamwUpdateFp16
    private void adamUpdateFp32Grad(CudaTensor master, CudaTensor fp16,
                                     CudaTensor adamM, CudaTensor adamV,
                                     CudaTensor gradFp32,
                                     float lr, float beta1, float beta2,
                                     float eps, float weightDecay, int step, int n) {
        try (CudaTensor gradFp16 = CudaTensor.allocate(device, n, Precision.FP16)) {
            kernels.convertF32ToF16(gradFp32, gradFp16, n);
            kernels.adamwUpdateFp16(master, fp16, adamM, adamV, gradFp16,
                    lr, beta1, beta2, eps, weightDecay, step, n);
        }
    }

    private void initAdamState() {
        adamMWte = CudaTensor.allocate(device, masterWte.getElementCount(), Precision.FP32);
        adamVWte = CudaTensor.allocate(device, masterWte.getElementCount(), Precision.FP32);
        adamMWpe = CudaTensor.allocate(device, masterWpe.getElementCount(), Precision.FP32);
        adamVWpe = CudaTensor.allocate(device, masterWpe.getElementCount(), Precision.FP32);
        adamMLnfw = CudaTensor.allocate(device, masterLnfw.getElementCount(), Precision.FP32);
        adamVLnfw = CudaTensor.allocate(device, masterLnfw.getElementCount(), Precision.FP32);
        adamMLnfb = CudaTensor.allocate(device, masterLnfb.getElementCount(), Precision.FP32);
        adamVLnfb = CudaTensor.allocate(device, masterLnfb.getElementCount(), Precision.FP32);

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
            adamMLn1w[l] = CudaTensor.allocate(device, C, Precision.FP32);
            adamVLn1w[l] = CudaTensor.allocate(device, C, Precision.FP32);
            adamMLn1b[l] = CudaTensor.allocate(device, C, Precision.FP32);
            adamVLn1b[l] = CudaTensor.allocate(device, C, Precision.FP32);
            adamMQkvw[l] = CudaTensor.allocate(device, (long) 3 * C * C, Precision.FP32);
            adamVQkvw[l] = CudaTensor.allocate(device, (long) 3 * C * C, Precision.FP32);
            adamMQkvb[l] = CudaTensor.allocate(device, 3 * C, Precision.FP32);
            adamVQkvb[l] = CudaTensor.allocate(device, 3 * C, Precision.FP32);
            adamMAttnProjw[l] = CudaTensor.allocate(device, (long) C * C, Precision.FP32);
            adamVAttnProjw[l] = CudaTensor.allocate(device, (long) C * C, Precision.FP32);
            adamMAttnProjb[l] = CudaTensor.allocate(device, C, Precision.FP32);
            adamVAttnProjb[l] = CudaTensor.allocate(device, C, Precision.FP32);
            adamMLn2w[l] = CudaTensor.allocate(device, C, Precision.FP32);
            adamVLn2w[l] = CudaTensor.allocate(device, C, Precision.FP32);
            adamMLn2b[l] = CudaTensor.allocate(device, C, Precision.FP32);
            adamVLn2b[l] = CudaTensor.allocate(device, C, Precision.FP32);
            adamMFcw[l] = CudaTensor.allocate(device, (long) 4 * C * C, Precision.FP32);
            adamVFcw[l] = CudaTensor.allocate(device, (long) 4 * C * C, Precision.FP32);
            adamMFcb[l] = CudaTensor.allocate(device, 4 * C, Precision.FP32);
            adamVFcb[l] = CudaTensor.allocate(device, 4 * C, Precision.FP32);
            adamMFcProjw[l] = CudaTensor.allocate(device, (long) C * 4 * C, Precision.FP32);
            adamVFcProjw[l] = CudaTensor.allocate(device, (long) C * 4 * C, Precision.FP32);
            adamMFcProjb[l] = CudaTensor.allocate(device, C, Precision.FP32);
            adamVFcProjb[l] = CudaTensor.allocate(device, C, Precision.FP32);
        }
    }

    // ========================================================================
    // Zero / Unscale Gradients
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

    public void unscaleGradients(float scale) {
        CudaOps.scale(device, gradWte, gradWte, scale);
        CudaOps.scale(device, gradWpe, gradWpe, scale);
        CudaOps.scale(device, gradLnfw, gradLnfw, scale);
        CudaOps.scale(device, gradLnfb, gradLnfb, scale);
        for (int l = 0; l < config.numLayers; l++) {
            CudaOps.scale(device, gradLn1w[l], gradLn1w[l], scale);
            CudaOps.scale(device, gradLn1b[l], gradLn1b[l], scale);
            CudaOps.scale(device, gradLn2w[l], gradLn2w[l], scale);
            CudaOps.scale(device, gradLn2b[l], gradLn2b[l], scale);
            CudaOps.scale(device, gradQkvw[l], gradQkvw[l], scale);
            CudaOps.scale(device, gradQkvb[l], gradQkvb[l], scale);
            CudaOps.scale(device, gradAttnProjw[l], gradAttnProjw[l], scale);
            CudaOps.scale(device, gradAttnProjb[l], gradAttnProjb[l], scale);
            CudaOps.scale(device, gradFcw[l], gradFcw[l], scale);
            CudaOps.scale(device, gradFcb[l], gradFcb[l], scale);
            CudaOps.scale(device, gradFcProjw[l], gradFcProjw[l], scale);
            CudaOps.scale(device, gradFcProjb[l], gradFcProjb[l], scale);
        }
    }

    // ========================================================================
    // Getters
    // ========================================================================

    public float getMeanLoss() { return meanLoss; }
    public GPT2Config getConfig() { return config; }
    public boolean isWeightsLoaded() { return weightsLoaded; }

    private void ensureNotClosed() {
        if (closed) throw new IllegalStateException("Model closed");
    }

    // ========================================================================
    // Close
    // ========================================================================

    @Override
    public void close() {
        if (closed) return;
        closed = true;

        closeTensor(masterWte); closeTensor(masterWpe);
        closeTensor(masterLnfw); closeTensor(masterLnfb);
        closeTensor(fp16Wte); closeTensor(fp16Wpe);
        closeTensor(fp16Lnfw); closeTensor(fp16Lnfb);
        closeTensor(gradWte); closeTensor(gradWpe);
        closeTensor(gradLnfw); closeTensor(gradLnfb);
        closeTensor(encoded); closeTensor(lnf);
        closeTensor(lnfMean); closeTensor(lnfRstd);
        closeTensor(logits); closeTensor(probs);
        closeTensor(dlogits); closeTensor(dlnf);
        closeTensor(adamMWte); closeTensor(adamVWte);
        closeTensor(adamMWpe); closeTensor(adamVWpe);
        closeTensor(adamMLnfw); closeTensor(adamVLnfw);
        closeTensor(adamMLnfb); closeTensor(adamVLnfb);

        for (int l = 0; l < config.numLayers; l++) {
            closeTensor(masterLn1w[l]); closeTensor(masterLn1b[l]);
            closeTensor(masterQkvw[l]); closeTensor(masterQkvb[l]);
            closeTensor(masterAttnProjw[l]); closeTensor(masterAttnProjb[l]);
            closeTensor(masterLn2w[l]); closeTensor(masterLn2b[l]);
            closeTensor(masterFcw[l]); closeTensor(masterFcb[l]);
            closeTensor(masterFcProjw[l]); closeTensor(masterFcProjb[l]);

            closeTensor(fp16Ln1w[l]); closeTensor(fp16Ln1b[l]);
            closeTensor(fp16Qkvw[l]); closeTensor(fp16Qkvb[l]);
            closeTensor(fp16AttnProjw[l]); closeTensor(fp16AttnProjb[l]);
            closeTensor(fp16Ln2w[l]); closeTensor(fp16Ln2b[l]);
            closeTensor(fp16Fcw[l]); closeTensor(fp16Fcb[l]);
            closeTensor(fp16FcProjw[l]); closeTensor(fp16FcProjb[l]);

            closeTensor(gradLn1w[l]); closeTensor(gradLn1b[l]);
            closeTensor(gradLn2w[l]); closeTensor(gradLn2b[l]);
            closeTensor(gradQkvw[l]); closeTensor(gradQkvb[l]);
            closeTensor(gradAttnProjw[l]); closeTensor(gradAttnProjb[l]);
            closeTensor(gradFcw[l]); closeTensor(gradFcb[l]);
            closeTensor(gradFcProjw[l]); closeTensor(gradFcProjb[l]);

            closeTensor(ln1[l]); closeTensor(ln1Mean[l]); closeTensor(ln1Rstd[l]);
            closeTensor(qkv[l]); closeTensor(attnOut[l]); closeTensor(attLse[l]); closeTensor(residual1[l]);
            closeTensor(ln2[l]); closeTensor(ln2Mean[l]); closeTensor(ln2Rstd[l]);
            closeTensor(fch[l]); closeTensor(fchGelu[l]); closeTensor(residual2[l]);
            closeTensor(dresidual[l]);

            // Adam state
            closeTensor(adamMLn1w[l]); closeTensor(adamVLn1w[l]);
            closeTensor(adamMLn1b[l]); closeTensor(adamVLn1b[l]);
            closeTensor(adamMQkvw[l]); closeTensor(adamVQkvw[l]);
            closeTensor(adamMQkvb[l]); closeTensor(adamVQkvb[l]);
            closeTensor(adamMAttnProjw[l]); closeTensor(adamVAttnProjw[l]);
            closeTensor(adamMAttnProjb[l]); closeTensor(adamVAttnProjb[l]);
            closeTensor(adamMLn2w[l]); closeTensor(adamVLn2w[l]);
            closeTensor(adamMLn2b[l]); closeTensor(adamVLn2b[l]);
            closeTensor(adamMFcw[l]); closeTensor(adamVFcw[l]);
            closeTensor(adamMFcb[l]); closeTensor(adamVFcb[l]);
            closeTensor(adamMFcProjw[l]); closeTensor(adamVFcProjw[l]);
            closeTensor(adamMFcProjb[l]); closeTensor(adamVFcProjb[l]);
        }

        System.out.println("MixedPrecisionGPT2 closed");
    }

    private void closeTensor(CudaTensor t) {
        if (t != null && !t.isClosed()) t.close();
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    private void addBiasFp16(CudaTensor tensor, CudaTensor bias, int rows, int cols) {
        float[] data = tensor.toFloatArray();
        float[] biasData = bias.toFloatArray();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                data[r * cols + c] += biasData[c];
            }
        }
        TensorUtils.copyFromHost(device, data, tensor);
    }

    private void addPositionEmbeddingsFp16(CudaTensor out, CudaTensor wpe, int B, int T, int C) {
        float[] outData = out.toFloatArray();
        float[] wpeData = wpe.toFloatArray();
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                int outOffset = (b * T + t) * C;
                int wpeOffset = t * C;
                for (int c = 0; c < C; c++) {
                    outData[outOffset + c] += wpeData[wpeOffset + c];
                }
            }
        }
        TensorUtils.copyFromHost(device, outData, out);
    }

    private float computeLoss(int[] targets) {
        float[] p = probs.toFloatArray();
        int V = config.vocabSize;
        int BT = B * T;
        float loss = 0.0f;
        for (int i = 0; i < BT; i++) {
            loss -= (float) Math.log(Math.max(p[i * V + targets[i]], 1e-10f));
        }
        return loss / BT;
    }

    private void copyTensor(CudaTensor src, CudaTensor dst) {
        float[] data = src.toFloatArray();
        TensorUtils.copyFromHost(device, data, dst);
    }

    private void reshapeQKV(CudaTensor qkv, CudaTensor q, CudaTensor k, CudaTensor v,
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

    private void reshapeAttentionOutput(CudaTensor attnOut, CudaTensor out,
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

    private void reshapeAttentionOutputBackward(CudaTensor dout, CudaTensor dAttnOut,
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

    private void reshapeQKVBackward(CudaTensor dq, CudaTensor dk, CudaTensor dv,
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
}
