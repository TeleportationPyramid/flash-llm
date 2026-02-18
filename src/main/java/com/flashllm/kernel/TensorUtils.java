package com.flashllm.kernel;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;

/**
 * Utility functions for tensor operations.
 *
 * <p>Provides helper methods not directly available in Flash CudaOps.</p>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public final class TensorUtils {

    private TensorUtils() {}

    /**
     * Copies data from source tensor to destination tensor.
     *
     * <p>Both tensors must have the same size and precision.</p>
     *
     * @param device the CUDA device
     * @param src source tensor
     * @param dst destination tensor
     */
    public static void copy(CudaDevice device, CudaTensor src, CudaTensor dst) {
        if (src.getElementCount() != dst.getElementCount()) {
            throw new IllegalArgumentException("Size mismatch: src=" + src.getElementCount() + 
                ", dst=" + dst.getElementCount());
        }
        if (src.getPrecision() != dst.getPrecision()) {
            throw new IllegalArgumentException("Precision mismatch");
        }

        // Copy via host memory
        float[] data = src.toFloatArray();
        copyFromHost(device, data, dst);
    }

    /**
     * Copies float array to tensor.
     *
     * @param device the CUDA device
     * @param data source float array
     * @param dst destination tensor
     */
    public static void copyFromHost(CudaDevice device, float[] data, CudaTensor dst) {
        // Create temp tensor and use axpy to copy
        // dst = 0 + 1.0 * temp = temp
        try (CudaTensor temp = CudaTensor.fromFloat(device, data, dst.getPrecision())) {
            CudaOps.fill(device, dst, 0.0);
            CudaOps.axpy(device, 1.0, temp, dst);
        }
    }

    /**
     * Copies tensor data to float array.
     *
     * @param src source tensor
     * @return float array with tensor data
     */
    public static float[] copyToHost(CudaTensor src) {
        return src.toFloatArray();
    }

    /**
     * Fills tensor with zeros.
     *
     * @param device the CUDA device
     * @param tensor tensor to zero
     */
    public static void zero(CudaDevice device, CudaTensor tensor) {
        CudaOps.fill(device, tensor, 0.0);
    }

    /**
     * Scales tensor in-place: x = alpha * x
     *
     * @param device the CUDA device
     * @param tensor tensor to scale
     * @param alpha scale factor
     */
    public static void scaleInplace(CudaDevice device, CudaTensor tensor, double alpha) {
        CudaOps.scale(device, tensor, tensor, alpha);
    }

    /**
     * Adds two tensors: c = a + b
     *
     * @param device the CUDA device
     * @param a first tensor
     * @param b second tensor
     * @param c output tensor
     */
    public static void add(CudaDevice device, CudaTensor a, CudaTensor b, CudaTensor c) {
        CudaOps.add(device, a, b, c);
    }

    /**
     * In-place add: y = y + alpha * x
     *
     * @param device the CUDA device
     * @param alpha scale factor
     * @param x tensor to add
     * @param y tensor to accumulate into
     */
    public static void axpy(CudaDevice device, double alpha, CudaTensor x, CudaTensor y) {
        CudaOps.axpy(device, alpha, x, y);
    }

    // ========================================================================
    // GPU Precision Conversion
    // ========================================================================

    /**
     * Converts FP32 tensor to FP16 on GPU.
     *
     * <p>This is much faster than CPU conversion for large tensors.</p>
     *
     * @param f32 input FP32 tensor
     * @param f16 output FP16 tensor (must be pre-allocated)
     */
    public static void convertF32ToF16(CudaTensor f32, CudaTensor f16) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaLlmKernels llm = backend.getLlmKernels();

        llm.convertF32ToF16(f32, f16, (int) f32.getElementCount());
    }

    /**
     * Converts FP16 tensor to FP32 on GPU.
     *
     * <p>This is much faster than CPU conversion for large tensors.</p>
     *
     * @param f16 input FP16 tensor
     * @param f32 output FP32 tensor (must be pre-allocated)
     */
    public static void convertF16ToF32(CudaTensor f16, CudaTensor f32) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaLlmKernels llm = backend.getLlmKernels();

        llm.convertF16ToF32(f16, f32, (int) f16.getElementCount());
    }

    /**
     * Creates an FP16 copy of an FP32 tensor.
     *
     * @param f32 input FP32 tensor
     * @return new FP16 tensor with converted data
     */
    public static CudaTensor toF16(CudaTensor f32) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaTensor f16 = backend.allocateF16(f32.getElementCount());
        convertF32ToF16(f32, f16);
        return f16;
    }

    /**
     * Creates an FP32 copy of an FP16 tensor.
     *
     * @param f16 input FP16 tensor
     * @return new FP32 tensor with converted data
     */
    public static CudaTensor toF32(CudaTensor f16) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaTensor f32 = backend.allocateF32(f16.getElementCount());
        convertF16ToF32(f16, f32);
        return f32;
    }
}
