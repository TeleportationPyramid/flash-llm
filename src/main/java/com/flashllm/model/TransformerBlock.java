package com.flashllm.model;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.kernel.*;

/**
 * Transformer Block - Single layer of the transformer.
 *
 * <p>Corresponds to llm.c's transformer block within gpt2_forward/backward.</p>
 *
 * <h2>Architecture:</h2>
 * <pre>
 * x = x + attention(layernorm1(x))
 * x = x + mlp(layernorm2(x))
 * </pre>
 *
 * <p>Note: llm.c stores weights as (OC, IC), so we use forwardTransposed/backwardTransposed.</p>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public final class TransformerBlock {

    private final int layerIdx;
    private final int B;  // batch size
    private final int T;  // sequence length
    private final int C;  // channels
    private final int NH; // number of heads
    private final int HS; // head size = C / NH

    /**
     * Create a transformer block.
     *
     * @param layerIdx layer index (0-based)
     * @param B batch size
     * @param T sequence length
     * @param C channels (embedding dimension)
     * @param NH number of attention heads
     */
    public TransformerBlock(int layerIdx, int B, int T, int C, int NH) {
        this.layerIdx = layerIdx;
        this.B = B;
        this.T = T;
        this.C = C;
        this.NH = NH;
        this.HS = C / NH;
    }

    /**
     * Forward pass through the transformer block.
     *
     * <pre>
     * residual1 = x
     * x = layernorm1(x)
     * x = attention(x)
     * x = residual1 + x
     * 
     * residual2 = x
     * x = layernorm2(x)
     * x = mlp(x)
     * x = residual2 + x
     * </pre>
     *
     * @param out output tensor (B*T, C)
     * @param inp input tensor (B*T, C)
     * @param ln1w LayerNorm1 weight (C,)
     * @param ln1b LayerNorm1 bias (C,)
     * @param qkvw QKV projection weight (3*C, C) - llm.c layout
     * @param qkvb QKV projection bias (3*C,)
     * @param attprojw Attention output projection weight (C, C) - llm.c layout
     * @param attprojb Attention output projection bias (C,)
     * @param ln2w LayerNorm2 weight (C,)
     * @param ln2b LayerNorm2 bias (C,)
     * @param fcw MLP fc weight (4*C, C) - llm.c layout
     * @param fcb MLP fc bias (4*C,)
     * @param fcprojw MLP projection weight (C, 4*C) - llm.c layout
     * @param fcprojb MLP projection bias (C,)
     * @param ln1 LayerNorm1 output (B*T, C) - saved for backward
     * @param ln1Mean LayerNorm1 mean (B*T,) - saved for backward
     * @param ln1Rstd LayerNorm1 rstd (B*T,) - saved for backward
     * @param qkv QKV tensor (B*T, 3*C) - saved for backward
     * @param atty Attention output before projection (B*T, C) - saved for backward
     * @param attLse Log-sum-exp from flash attention (B*NH, T) - saved for backward
     * @param attnOut Attention block output (B*T, C) - saved for backward
     * @param ln2 LayerNorm2 output (B*T, C) - saved for backward
     * @param ln2Mean LayerNorm2 mean (B*T,) - saved for backward
     * @param ln2Rstd LayerNorm2 rstd (B*T,) - saved for backward
     * @param fch MLP hidden (B*T, 4*C) - saved for backward
     * @param fchGelu GELU output (B*T, 4*C) - saved for backward
     */
    public void forward(
            // Output
            CudaTensor out,
            // Input
            CudaTensor inp,
            // Layer 1: Attention
            CudaTensor ln1w, CudaTensor ln1b,
            CudaTensor qkvw, CudaTensor qkvb,
            CudaTensor attprojw, CudaTensor attprojb,
            // Layer 2: MLP
            CudaTensor ln2w, CudaTensor ln2b,
            CudaTensor fcw, CudaTensor fcb,
            CudaTensor fcprojw, CudaTensor fcprojb,
            // Saved activations for backward
            CudaTensor ln1, CudaTensor ln1Mean, CudaTensor ln1Rstd,
            CudaTensor qkv, CudaTensor atty, CudaTensor attLse, CudaTensor attnOut,
            CudaTensor ln2, CudaTensor ln2Mean, CudaTensor ln2Rstd,
            CudaTensor fch, CudaTensor fchGelu
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaDevice device = backend.getDevice();
        int BT = B * T;

        // ============================================================
        // Attention Block: x = x + attention(layernorm1(x))
        // ============================================================

        // LayerNorm1
        LayerNorm.forward(ln1, ln1Mean, ln1Rstd, inp, ln1w, ln1b, BT, C);

        // Multi-Head Attention
        Attention.forward(attnOut, qkv, attLse, ln1, qkvw, qkvb, attprojw, attprojb, B, T, C, NH);

        // Residual1: out = inp + attnOut
        Residual.forward(out, inp, attnOut, BT * C);

        // ============================================================
        // MLP Block: x = x + mlp(layernorm2(x))
        // ============================================================

        // LayerNorm2
        LayerNorm.forward(ln2, ln2Mean, ln2Rstd, out, ln2w, ln2b, BT, C);

        // MLP: fc -> gelu -> fc_proj
        // fch = ln2 @ fcw^T + fcb
        // fcw is (4*C, C), so we use forwardTransposed
        Matmul.forwardTransposed(fch, ln2, fcw, fcb, BT, 4 * C, C);

        // fchGelu = gelu(fch)
        Gelu.forward(fchGelu, fch, BT * 4 * C);

        // mlpOut = fchGelu @ fcprojw^T + fcprojb
        // fcprojw is (C, 4*C), so we use forwardTransposed
        try (CudaTensor mlpOut = backend.allocateF32(BT * C)) {
            Matmul.forwardTransposed(mlpOut, fchGelu, fcprojw, fcprojb, BT, C, 4 * C);

            // Residual2: out = out + mlpOut
            Residual.forwardInplace(out, mlpOut, BT * C);
        }
    }

    /**
     * Backward pass through the transformer block.
     *
     * @param dinp gradient for input (B*T, C) - output
     * @param dout gradient from upstream (B*T, C)
     * @param inp original input (B*T, C)
     * @param ln1w LayerNorm1 weight
     * @param qkvw QKV weight (3*C, C) - llm.c layout
     * @param attprojw Attention projection weight (C, C) - llm.c layout
     * @param ln2w LayerNorm2 weight
     * @param fcw MLP fc weight (4*C, C) - llm.c layout
     * @param fcprojw MLP projection weight (C, 4*C) - llm.c layout
     * @param ln1 saved LayerNorm1 output
     * @param ln1Mean saved mean
     * @param ln1Rstd saved rstd
     * @param qkv saved QKV
     * @param atty saved attention output
     * @param attLse saved log-sum-exp
     * @param attnOut saved attention block output
     * @param ln2 saved LayerNorm2 output
     * @param ln2Mean saved mean
     * @param ln2Rstd saved rstd
     * @param fch saved MLP hidden
     * @param fchGelu saved GELU output
     * @param dln1w gradient for LayerNorm1 weight - accumulated
     * @param dln1b gradient for LayerNorm1 bias - accumulated
     * @param dqkvw gradient for QKV weight - accumulated
     * @param dqkvb gradient for QKV bias - accumulated
     * @param dattprojw gradient for attention projection weight - accumulated
     * @param dattprojb gradient for attention projection bias - accumulated
     * @param dln2w gradient for LayerNorm2 weight - accumulated
     * @param dln2b gradient for LayerNorm2 bias - accumulated
     * @param dfcw gradient for MLP fc weight - accumulated
     * @param dfcb gradient for MLP fc bias - accumulated
     * @param dfcprojw gradient for MLP projection weight - accumulated
     * @param dfcprojb gradient for MLP projection bias - accumulated
     */
    public void backward(
            // Output gradient
            CudaTensor dinp,
            // Input gradient
            CudaTensor dout,
            // Original inputs
            CudaTensor inp,
            // Weights
            CudaTensor ln1w, CudaTensor qkvw, CudaTensor attprojw,
            CudaTensor ln2w, CudaTensor fcw, CudaTensor fcprojw,
            // Saved activations
            CudaTensor ln1, CudaTensor ln1Mean, CudaTensor ln1Rstd,
            CudaTensor qkv, CudaTensor atty, CudaTensor attLse, CudaTensor attnOut,
            CudaTensor ln2, CudaTensor ln2Mean, CudaTensor ln2Rstd,
            CudaTensor fch, CudaTensor fchGelu,
            // Weight gradients
            CudaTensor dln1w, CudaTensor dln1b,
            CudaTensor dqkvw, CudaTensor dqkvb,
            CudaTensor dattprojw, CudaTensor dattprojb,
            CudaTensor dln2w, CudaTensor dln2b,
            CudaTensor dfcw, CudaTensor dfcb,
            CudaTensor dfcprojw, CudaTensor dfcprojb
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaDevice device = backend.getDevice();
        int BT = B * T;

        // ============================================================
        // Backward through MLP Block
        // ============================================================

        // Residual2 backward: dout flows to both branches
        // dresidual2 = dout, dmlpOut = dout

        // MLP projection backward: dmlpOut -> dfchGelu, dfcprojw, dfcprojb
        try (CudaTensor dfchGelu = backend.allocateF32(BT * 4 * C);
             CudaTensor dfch = backend.allocateF32(BT * 4 * C);
             CudaTensor dln2 = backend.allocateF32(BT * C);
             CudaTensor dresidual2 = backend.allocateF32(BT * C)) {

            // Copy dout to dresidual2 (will accumulate)
            TensorUtils.copy(device, dout, dresidual2);

            // Backward through fc_proj: dout -> dfchGelu
            // fcprojw is (C, 4*C), use backwardTransposed
            Matmul.backwardTransposed(dfchGelu, dfcprojw, dfcprojb, dout, fchGelu, fcprojw, BT, C, 4 * C);

            // Backward through GELU
            Gelu.backward(dfch, fch, dfchGelu, BT * 4 * C);

            // Backward through fc: dfch -> dln2
            // fcw is (4*C, C), use backwardTransposed
            Matmul.backwardTransposed(dln2, dfcw, dfcb, dfch, ln2, fcw, BT, 4 * C, C);

            // Backward through LayerNorm2
            // Note: attnOut was the input to ln2, which is the output of residual1
            // We need to compute dattnOut from dresidual2 (which already has dout)
            try (CudaTensor dattnOut = backend.allocateF32(BT * C)) {
                // LayerNorm2 backward: dln2 -> dattnOut (gradient for ln2 input)
                LayerNorm.backward(dattnOut, dln2w, dln2b, dln2, attnOut, ln2w, ln2Mean, ln2Rstd, BT, C);

                // Add residual gradient: dattnOut += dresidual2
                Residual.backwardAccumulate(dattnOut, dresidual2, BT * C);

                // ============================================================
                // Backward through Attention Block
                // ============================================================

                try (CudaTensor dln1 = backend.allocateF32(BT * C)) {
                    // Attention backward
                    Attention.backward(dln1, dqkvw, dqkvb, dattprojw, dattprojb,
                            dattnOut, ln1, qkv, attLse, qkvw, attprojw, B, T, C, NH);

                    // LayerNorm1 backward
                    LayerNorm.backward(dinp, dln1w, dln1b, dln1, inp, ln1w, ln1Mean, ln1Rstd, BT, C);

                    // Add residual gradient: dinp += dattnOut
                    Residual.backwardAccumulate(dinp, dattnOut, BT * C);
                }
            }
        }
    }

    public int getLayerIdx() {
        return layerIdx;
    }
}
