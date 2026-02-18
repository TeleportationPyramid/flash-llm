package com.flashllm.training;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.kernel.TensorUtils;

import java.util.*;

/**
 * Gradient clipping utilities for stable training.
 * 
 * <p>Supports:</p>
 * <ul>
 *   <li><b>Global norm clipping</b> - Scale all gradients if total norm exceeds threshold</li>
 *   <li><b>Value clipping</b> - Clamp individual gradient values</li>
 * </ul>
 * 
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Clip gradients by global norm (recommended for transformers)
 * float gradNorm = GradientClipper.clipByGlobalNorm(gradients, maxNorm: 1.0f);
 * System.out.println("Grad norm: " + gradNorm);
 * 
 * // Or clip by value
 * GradientClipper.clipByValue(gradients, minVal: -1.0f, maxVal: 1.0f);
 * }</pre>
 * 
 * <p>Corresponds to torch.nn.utils.clip_grad_norm_ in PyTorch.</p>
 */
public final class GradientClipper {

    private GradientClipper() {}

    // ========================================================================
    // Global Norm Clipping
    // ========================================================================

    /**
     * Clip gradients by global norm.
     * 
     * <p>If the total L2 norm of all gradients exceeds maxNorm,
     * scale all gradients by (maxNorm / totalNorm).</p>
     * 
     * @param gradients list of gradient tensors (on GPU)
     * @param maxNorm maximum allowed global norm (e.g., 1.0)
     * @param device CUDA device for tensor operations
     * @return the computed global norm (before clipping)
     */
    public static float clipByGlobalNorm(List<CudaTensor> gradients, float maxNorm, CudaDevice device) {
        // Compute global norm
        double sumSq = 0.0;
        for (CudaTensor grad : gradients) {
            float[] data = grad.toFloatArray();
            for (float g : data) {
                sumSq += (double) g * g;
            }
        }
        float globalNorm = (float) Math.sqrt(sumSq);
        
        // Scale if needed
        if (globalNorm > maxNorm) {
            float scale = maxNorm / globalNorm;
            for (CudaTensor grad : gradients) {
                scaleGradient(grad, scale, device);
            }
        }
        
        return globalNorm;
    }

    /**
     * Clip gradients by global norm (array version).
     */
    public static float clipByGlobalNorm(CudaTensor[] gradients, float maxNorm, CudaDevice device) {
        return clipByGlobalNorm(Arrays.asList(gradients), maxNorm, device);
    }

    /**
     * Clip gradients by global norm (map version).
     * 
     * @param gradients map of gradient name to tensor
     * @param maxNorm maximum allowed global norm
     * @param device CUDA device for tensor operations
     * @return the computed global norm (before clipping)
     */
    public static float clipByGlobalNorm(Map<String, CudaTensor> gradients, float maxNorm, CudaDevice device) {
        return clipByGlobalNorm(new ArrayList<>(gradients.values()), maxNorm, device);
    }

    // ========================================================================
    // Value Clipping
    // ========================================================================

    /**
     * Clip gradients by value (clamp).
     * 
     * <p>Clamp all gradient values to [minVal, maxVal].</p>
     * 
     * @param gradients list of gradient tensors
     * @param minVal minimum value
     * @param maxVal maximum value
     * @param device CUDA device for tensor operations
     */
    public static void clipByValue(List<CudaTensor> gradients, float minVal, float maxVal, CudaDevice device) {
        for (CudaTensor grad : gradients) {
            clampGradient(grad, minVal, maxVal, device);
        }
    }

    /**
     * Clip gradients by value (symmetric).
     * 
     * <p>Clamp all gradient values to [-clipValue, clipValue].</p>
     * 
     * @param gradients list of gradient tensors
     * @param clipValue absolute maximum value
     * @param device CUDA device for tensor operations
     */
    public static void clipByValue(List<CudaTensor> gradients, float clipValue, CudaDevice device) {
        clipByValue(gradients, -clipValue, clipValue, device);
    }

    // ========================================================================
    // Norm Computation
    // ========================================================================

    /**
     * Compute L2 norm of a single gradient tensor.
     */
    public static float computeNorm(CudaTensor gradient) {
        float[] data = gradient.toFloatArray();
        double sumSq = 0.0;
        for (float g : data) {
            sumSq += (double) g * g;
        }
        return (float) Math.sqrt(sumSq);
    }

    /**
     * Compute global L2 norm of all gradients.
     */
    public static float computeGlobalNorm(List<CudaTensor> gradients) {
        double sumSq = 0.0;
        for (CudaTensor grad : gradients) {
            float[] data = grad.toFloatArray();
            for (float g : data) {
                sumSq += (double) g * g;
            }
        }
        return (float) Math.sqrt(sumSq);
    }

    /**
     * Compute gradient statistics for debugging.
     */
    public static GradientStats computeStats(List<CudaTensor> gradients) {
        double sumSq = 0.0;
        float minVal = Float.MAX_VALUE;
        float maxVal = Float.MIN_VALUE;
        float maxAbs = 0;
        long totalElements = 0;
        
        for (CudaTensor grad : gradients) {
            float[] data = grad.toFloatArray();
            for (float g : data) {
                sumSq += (double) g * g;
                minVal = Math.min(minVal, g);
                maxVal = Math.max(maxVal, g);
                maxAbs = Math.max(maxAbs, Math.abs(g));
                totalElements++;
            }
        }
        
        return new GradientStats(
            (float) Math.sqrt(sumSq),
            minVal,
            maxVal,
            maxAbs,
            totalElements
        );
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    private static void scaleGradient(CudaTensor grad, float scale, CudaDevice device) {
        float[] data = grad.toFloatArray();
        for (int i = 0; i < data.length; i++) {
            data[i] *= scale;
        }
        TensorUtils.copyFromHost(device, data, grad);
    }

    private static void clampGradient(CudaTensor grad, float minVal, float maxVal, CudaDevice device) {
        float[] data = grad.toFloatArray();
        for (int i = 0; i < data.length; i++) {
            data[i] = Math.max(minVal, Math.min(maxVal, data[i]));
        }
        TensorUtils.copyFromHost(device, data, grad);
    }

    // ========================================================================
    // Statistics class
    // ========================================================================

    /**
     * Gradient statistics for monitoring.
     */
    public static class GradientStats {
        public final float globalNorm;
        public final float minValue;
        public final float maxValue;
        public final float maxAbsValue;
        public final long totalElements;
        
        public GradientStats(float globalNorm, float minValue, float maxValue,
                            float maxAbsValue, long totalElements) {
            this.globalNorm = globalNorm;
            this.minValue = minValue;
            this.maxValue = maxValue;
            this.maxAbsValue = maxAbsValue;
            this.totalElements = totalElements;
        }
        
        @Override
        public String toString() {
            return String.format(
                "GradientStats(norm=%.4f, min=%.4f, max=%.4f, maxAbs=%.4f, elements=%d)",
                globalNorm, minValue, maxValue, maxAbsValue, totalElements
            );
        }
        
        /**
         * Check if gradients might be exploding.
         */
        public boolean isExploding(float threshold) {
            return globalNorm > threshold || Float.isNaN(globalNorm) || Float.isInfinite(globalNorm);
        }
        
        /**
         * Check if gradients might be vanishing.
         */
        public boolean isVanishing(float threshold) {
            return globalNorm < threshold;
        }
    }
}
