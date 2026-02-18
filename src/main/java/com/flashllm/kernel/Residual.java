package com.flashllm.kernel;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;

/**
 * Residual kernel - Residual connections (element-wise addition).
 *
 * <p>Corresponds to llm.c's residual_forward/backward.</p>
 *
 * <h2>Forward:</h2>
 * <pre>out = a + b</pre>
 *
 * <h2>Backward:</h2>
 * <pre>
 * da = dout
 * db = dout
 * </pre>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public final class Residual {

    private Residual() {}

    /**
     * Residual forward pass.
     *
     * <p>Computes element-wise addition: out = a + b</p>
     *
     * @param out output tensor (N,)
     * @param a   first input tensor (N,)
     * @param b   second input tensor (N,)
     * @param N   number of elements
     */
    public static void forward(CudaTensor out, CudaTensor a, CudaTensor b, long N) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaDevice device = backend.getDevice();

        // out = a + b
        CudaOps.add(device, a, b, out);
    }

    /**
     * Residual forward pass (inplace version).
     *
     * <p>Computes: a = a + b</p>
     *
     * @param a first input tensor, will be overwritten with result (N,)
     * @param b second input tensor (N,)
     * @param N number of elements
     */
    public static void forwardInplace(CudaTensor a, CudaTensor b, long N) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaDevice device = backend.getDevice();

        // a = a + b
        CudaOps.add(device, a, b, a);
    }

    /**
     * Residual backward pass.
     *
     * <p>For residual connection out = a + b:</p>
     * <ul>
     *   <li>da = dout</li>
     *   <li>db = dout</li>
     * </ul>
     *
     * <p>Since gradients are just copies, this is typically handled
     * by accumulating dout into both da and db.</p>
     *
     * @param da   gradient for a (N,) - accumulated
     * @param db   gradient for b (N,) - accumulated
     * @param dout gradient from upstream (N,)
     * @param N    number of elements
     */
    public static void backward(CudaTensor da, CudaTensor db, CudaTensor dout, long N) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaDevice device = backend.getDevice();

        // da += dout
        CudaOps.add(device, da, dout, da);

        // db += dout
        CudaOps.add(device, db, dout, db);
    }

    /**
     * Accumulates gradient into a single tensor.
     *
     * <p>Useful when one branch doesn't need separate gradient tracking.</p>
     *
     * @param da   gradient to accumulate into (N,)
     * @param dout gradient from upstream (N,)
     * @param N    number of elements
     */
    public static void backwardAccumulate(CudaTensor da, CudaTensor dout, long N) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaDevice device = backend.getDevice();

        // da += dout
        CudaOps.add(device, da, dout, da);
    }

    /**
     * Copies gradient without accumulation.
     *
     * <p>Useful for the first backward pass where we overwrite instead of accumulate.</p>
     *
     * @param da   gradient output (N,) - overwritten
     * @param dout gradient from upstream (N,)
     * @param N    number of elements
     */
    public static void backwardCopy(CudaTensor da, CudaTensor dout, long N) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaDevice device = backend.getDevice();

        // da = dout
        CudaOps.fill(device, da, 0.0);
        CudaOps.axpy(device, 1.0, dout, da);
    }
}
