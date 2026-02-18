package com.flashllm.kernel;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;

/**
 * LayerNorm kernel - Layer Normalization.
 *
 * <p>Corresponds to llm.c's layernorm_forward/backward.</p>
 *
 * <h2>Forward:</h2>
 * <pre>
 * mean = mean(inp, axis=-1)
 * var = var(inp, axis=-1)
 * rstd = 1 / sqrt(var + eps)
 * out = (inp - mean) * rstd * weight + bias
 * </pre>
 *
 * <h2>Backward:</h2>
 * <pre>
 * Computes dinp, dweight, dbias from dout
 * </pre>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public final class LayerNorm {

    private static final float EPS = 1e-5f;

    private LayerNorm() {}

    /**
     * LayerNorm forward pass.
     *
     * @param out    output tensor (N, C)
     * @param mean   mean per sample (N,) - saved for backward
     * @param rstd   reciprocal std per sample (N,) - saved for backward
     * @param inp    input tensor (N, C)
     * @param weight gamma parameter (C,)
     * @param bias   beta parameter (C,)
     * @param N      number of samples (B * T)
     * @param C      feature dimension (channels)
     */
    public static void forward(
            CudaTensor out,
            CudaTensor mean,
            CudaTensor rstd,
            CudaTensor inp,
            CudaTensor weight,
            CudaTensor bias,
            int N, int C
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaLlmKernels llm = backend.getLlmKernels();

        llm.layerNormForward(inp, weight, bias, out, mean, rstd, N, C);
    }

    /**
     * LayerNorm forward pass (inference mode, no mean/rstd saved).
     *
     * @param out    output tensor (N, C)
     * @param inp    input tensor (N, C)
     * @param weight gamma parameter (C,)
     * @param bias   beta parameter (C,)
     * @param N      number of samples
     * @param C      feature dimension
     */
    public static void forward(
            CudaTensor out,
            CudaTensor inp,
            CudaTensor weight,
            CudaTensor bias,
            int N, int C
    ) {
        FlashBackend backend = FlashBackend.getInstance();

        // Allocate temporary tensors for mean and rstd
        try (CudaTensor mean = backend.allocateF32(N);
             CudaTensor rstd = backend.allocateF32(N)) {
            forward(out, mean, rstd, inp, weight, bias, N, C);
        }
    }

    /**
     * LayerNorm backward pass.
     *
     * @param dinp    gradient for input (N, C) - output
     * @param dweight gradient for weight (C,) - accumulated
     * @param dbias   gradient for bias (C,) - accumulated
     * @param dout    gradient from upstream (N, C)
     * @param inp     original input (N, C) - from forward
     * @param weight  gamma parameter (C,)
     * @param mean    mean per sample (N,) - from forward
     * @param rstd    reciprocal std per sample (N,) - from forward
     * @param N       number of samples
     * @param C       feature dimension
     */
    public static void backward(
            CudaTensor dinp,
            CudaTensor dweight,
            CudaTensor dbias,
            CudaTensor dout,
            CudaTensor inp,
            CudaTensor weight,
            CudaTensor mean,
            CudaTensor rstd,
            int N, int C
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaLlmKernels llm = backend.getLlmKernels();

        llm.layerNormBackward(dout, inp, weight, mean, rstd, dinp, dweight, dbias, N, C);
    }
}
