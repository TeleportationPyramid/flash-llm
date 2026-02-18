package com.flashllm.kernel;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;

/**
 * GELU kernel - Gaussian Error Linear Unit activation.
 *
 * <p>Corresponds to llm.c's gelu_forward/backward.</p>
 *
 * <h2>Forward:</h2>
 * <pre>out = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))</pre>
 *
 * <h2>Backward:</h2>
 * <pre>dinp = dout * gelu'(inp)</pre>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public final class Gelu {

    private Gelu() {}

    /**
     * GELU forward pass.
     *
     * @param out output tensor (N,)
     * @param inp input tensor (N,)
     * @param N   number of elements
     */
    public static void forward(CudaTensor out, CudaTensor inp, int N) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaLlmKernels llm = backend.getLlmKernels();

        llm.geluForward(out, inp, N);
    }

    /**
     * GELU forward pass (inplace version).
     *
     * @param x tensor to apply GELU inplace (N,)
     * @param N number of elements
     */
    public static void forwardInplace(CudaTensor x, int N) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaLlmKernels llm = backend.getLlmKernels();

        // Use x as both input and output
        llm.geluForward(x, x, N);
    }

    /**
     * GELU backward pass.
     *
     * @param dinp gradient for input (N,) - output
     * @param inp  original input from forward (N,)
     * @param dout gradient from upstream (N,)
     * @param N    number of elements
     */
    public static void backward(CudaTensor dinp, CudaTensor inp, CudaTensor dout, int N) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaLlmKernels llm = backend.getLlmKernels();

        llm.geluBackward(dinp, inp, dout, N);
    }

    /**
     * GELU backward pass (inplace version).
     *
     * <p>Computes: dinp = dout * gelu'(inp), storing result in dinp.</p>
     *
     * @param dinp gradient for input (N,) - will be overwritten with result
     * @param inp  original input from forward (N,)
     * @param dout gradient from upstream (N,)
     * @param N    number of elements
     */
    public static void backwardInplace(CudaTensor dinp, CudaTensor inp, CudaTensor dout, int N) {
        backward(dinp, inp, dout, N);
    }
}
