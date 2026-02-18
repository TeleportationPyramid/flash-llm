package com.flashllm.kernel;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;

/**
 * Attention kernel - Multi-Head Self-Attention.
 *
 * <p>Corresponds to llm.c's attention_forward/backward.</p>
 *
 * <p>Uses Flash Attention for memory-efficient computation.</p>
 *
 * <h2>Forward:</h2>
 * <pre>
 * 1. QKV = inp @ qkvw^T + qkvb        // (B, T, 3C), qkvw is (3C, C)
 * 2. Q, K, V = split(QKV)             // each (B, T, C)
 * 3. Q, K, V = reshape to (B*NH, T, HS)
 * 4. out = FlashAttention(Q, K, V)    // (B*NH, T, HS)
 * 5. out = reshape to (B, T, C)
 * 6. out = out @ attprojw^T + attprojb  // (B, T, C), attprojw is (C, C)
 * </pre>
 *
 * <p>Note: llm.c stores weights as (OC, IC), so we use forwardTransposed.</p>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public final class Attention {

    private Attention() {}

    /**
     * Attention forward pass using Flash Attention.
     *
     * @param out       output tensor (B*T, C)
     * @param qkv       QKV tensor for backward (B*T, 3C) - saved
     * @param lse       log-sum-exp from flash attention (B*NH, T) - saved for backward
     * @param inp       input tensor (B*T, C)
     * @param qkvw      QKV weight (3C, C) - llm.c layout
     * @param qkvb      QKV bias (3C,)
     * @param attprojw  attention output projection weight (C, C) - llm.c layout
     * @param attprojb  attention output projection bias (C,)
     * @param B         batch size
     * @param T         sequence length
     * @param C         channels (embedding dimension)
     * @param NH        number of attention heads
     */
    public static void forward(
            CudaTensor out,
            CudaTensor qkv,
            CudaTensor lse,
            CudaTensor inp,
            CudaTensor qkvw,
            CudaTensor qkvb,
            CudaTensor attprojw,
            CudaTensor attprojb,
            int B, int T, int C, int NH
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaLlmKernels llm = backend.getLlmKernels();
        CudaDevice device = backend.getDevice();

        int HS = C / NH;  // head size
        int BT = B * T;
        int BNH = B * NH;

        // Step 1: QKV projection
        // qkv = inp @ qkvw^T + qkvb
        // inp: (BT, C), qkvw: (3C, C), qkv: (BT, 3C)
        Matmul.forwardTransposed(qkv, inp, qkvw, qkvb, BT, 3 * C, C);

        // Step 2-5: Flash Attention and reshape
        try (CudaTensor q = backend.allocateF32(BNH * T * HS);
             CudaTensor k = backend.allocateF32(BNH * T * HS);
             CudaTensor v = backend.allocateF32(BNH * T * HS);
             CudaTensor attnOut = backend.allocateF32(BNH * T * HS);
             CudaTensor attnReshaped = backend.allocateF32(BT * C)) {

            // Reshape QKV: (B, T, 3, NH, HS) -> Q, K, V each (B, NH, T, HS)
            reshapeQKV(device, qkv, q, k, v, B, T, NH, HS);

            // Apply Flash Attention (causal = true for GPT)
            llm.flashAttentionForward(attnOut, lse, q, k, v, BNH, T, HS, true);

            // Reshape attention output: (B, NH, T, HS) -> (B, T, C)
            reshapeAttentionOutput(device, attnOut, attnReshaped, B, T, NH, HS);

            // Step 6: Output projection
            // out = attnReshaped @ attprojw^T + attprojb
            // attnReshaped: (BT, C), attprojw: (C, C), out: (BT, C)
            Matmul.forwardTransposed(out, attnReshaped, attprojw, attprojb, BT, C, C);
        }
    }

    /**
     * Attention backward pass.
     *
     * @param dinp       gradient for input (B*T, C) - output
     * @param dqkvw      gradient for QKV weight (3C, C) - accumulated
     * @param dqkvb      gradient for QKV bias (3C,) - accumulated
     * @param dattprojw  gradient for projection weight (C, C) - accumulated
     * @param dattprojb  gradient for projection bias (C,) - accumulated
     * @param dout       gradient from upstream (B*T, C)
     * @param inp        original input (B*T, C)
     * @param qkv        QKV from forward (B*T, 3C)
     * @param lse        log-sum-exp from forward (B*NH, T)
     * @param qkvw       QKV weight (3C, C) - llm.c layout
     * @param attprojw   projection weight (C, C) - llm.c layout
     * @param B          batch size
     * @param T          sequence length
     * @param C          channels
     * @param NH         number of heads
     */
    public static void backward(
            CudaTensor dinp,
            CudaTensor dqkvw,
            CudaTensor dqkvb,
            CudaTensor dattprojw,
            CudaTensor dattprojb,
            CudaTensor dout,
            CudaTensor inp,
            CudaTensor qkv,
            CudaTensor lse,
            CudaTensor qkvw,
            CudaTensor attprojw,
            int B, int T, int C, int NH
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaLlmKernels llm = backend.getLlmKernels();
        CudaDevice device = backend.getDevice();

        int HS = C / NH;
        int BT = B * T;
        int BNH = B * NH;

        // Allocate temporary tensors
        try (CudaTensor dqkv = backend.allocateF32(BT * 3 * C);
             CudaTensor q = backend.allocateF32(BNH * T * HS);
             CudaTensor k = backend.allocateF32(BNH * T * HS);
             CudaTensor v = backend.allocateF32(BNH * T * HS);
             CudaTensor dq = backend.allocateF32(BNH * T * HS);
             CudaTensor dk = backend.allocateF32(BNH * T * HS);
             CudaTensor dv = backend.allocateF32(BNH * T * HS);
             CudaTensor attnOut = backend.allocateF32(BNH * T * HS);
             CudaTensor dAttnOut = backend.allocateF32(BNH * T * HS);
             CudaTensor dAttnReshaped = backend.allocateF32(BT * C);
             CudaTensor attnReshaped = backend.allocateF32(BT * C)) {

            // Reshape QKV from saved forward pass
            reshapeQKV(device, qkv, q, k, v, B, T, NH, HS);

            // Recompute attention output for backward (needed for Flash Attention backward)
            llm.flashAttentionForward(attnOut, lse, q, k, v, BNH, T, HS, true);

            // Reshape attention output to (B*T, C) for projection backward
            reshapeAttentionOutput(device, attnOut, attnReshaped, B, T, NH, HS);

            // Step 1: Backward through output projection (using transposed weight layout)
            // dAttnReshaped = dout @ attprojw (no transpose, weight is (C, C))
            // dattprojw += dout^T @ attnReshaped (to get (C, C) layout)
            // dattprojb += sum(dout)
            Matmul.backwardTransposed(dAttnReshaped, dattprojw, dattprojb, dout, attnReshaped, attprojw, BT, C, C);

            // Reshape dAttnReshaped to (B, NH, T, HS)
            reshapeAttentionOutputBackward(device, dAttnReshaped, dAttnOut, B, T, NH, HS);

            // Step 2: Flash Attention backward
            llm.flashAttentionBackward(dq, dk, dv, q, k, v, attnOut, dAttnOut, lse, BNH, T, HS, true);

            // Step 3: Reshape dQ, dK, dV back to dQKV (B*T, 3*C)
            reshapeQKVBackward(device, dq, dk, dv, dqkv, B, T, NH, HS);

            // Step 4: Backward through QKV projection (using transposed weight layout)
            // dinp = dqkv @ qkvw (no transpose, weight is (3C, C))
            // dqkvw += dqkv^T @ inp (to get (3C, C) layout)
            // dqkvb += sum(dqkv)
            Matmul.backwardTransposed(dinp, dqkvw, dqkvb, dqkv, inp, qkvw, BT, 3 * C, C);
        }
    }

    // ========================================================================
    // Reshape Helpers
    // ========================================================================

    /**
     * Reshapes QKV from (B*T, 3*C) to Q, K, V each (B*NH, T, HS).
     *
     * Input layout: [b, t, (q0..qC-1, k0..kC-1, v0..vC-1)]
     * Output layout: Q[b*nh, t, hs], K[b*nh, t, hs], V[b*nh, t, hs]
     */
    private static void reshapeQKV(
            CudaDevice device,
            CudaTensor qkv,
            CudaTensor q, CudaTensor k, CudaTensor v,
            int B, int T, int NH, int HS
    ) {
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

                        qData[outIdx] = qkvData[qkvIdx];           // Q
                        kData[outIdx] = qkvData[qkvIdx + C];       // K (offset by C)
                        vData[outIdx] = qkvData[qkvIdx + 2 * C];   // V (offset by 2C)
                    }
                }
            }
        }

        TensorUtils.copyFromHost(device, qData, q);
        TensorUtils.copyFromHost(device, kData, k);
        TensorUtils.copyFromHost(device, vData, v);
    }

    /**
     * Reshapes attention output from (B*NH, T, HS) to (B*T, C).
     */
    private static void reshapeAttentionOutput(
            CudaDevice device,
            CudaTensor attnOut,
            CudaTensor out,
            int B, int T, int NH, int HS
    ) {
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

    /**
     * Reshapes gradient from (B*T, C) to (B*NH, T, HS).
     */
    private static void reshapeAttentionOutputBackward(
            CudaDevice device,
            CudaTensor dout,
            CudaTensor dAttnOut,
            int B, int T, int NH, int HS
    ) {
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

    /**
     * Reshapes dQ, dK, dV from (B*NH, T, HS) back to dQKV (B*T, 3*C).
     */
    private static void reshapeQKVBackward(
            CudaDevice device,
            CudaTensor dq, CudaTensor dk, CudaTensor dv,
            CudaTensor dqkv,
            int B, int T, int NH, int HS
    ) {
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
