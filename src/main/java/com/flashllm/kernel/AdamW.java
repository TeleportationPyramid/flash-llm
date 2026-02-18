package com.flashllm.kernel;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;

/**
 * AdamW kernel - AdamW optimizer update.
 *
 * <p>Corresponds to llm.c's adamw_update.</p>
 *
 * <h2>Update Rule:</h2>
 * <pre>
 * m = beta1 * m + (1 - beta1) * grad
 * v = beta2 * v + (1 - beta2) * grad²
 * m_hat = m / (1 - beta1^t)
 * v_hat = v / (1 - beta2^t)
 * param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)
 * </pre>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public final class AdamW {

    private AdamW() {}

    /**
     * AdamW optimizer update step.
     *
     * <p>Updates parameters in-place using the AdamW algorithm.</p>
     *
     * @param params      model parameters (N,) - updated in-place
     * @param grads       gradients (N,)
     * @param m           first moment estimates (N,) - updated in-place
     * @param v           second moment estimates (N,) - updated in-place
     * @param lr          learning rate
     * @param beta1       first moment decay rate (typically 0.9)
     * @param beta2       second moment decay rate (typically 0.999)
     * @param eps         epsilon for numerical stability (typically 1e-8)
     * @param weightDecay weight decay coefficient
     * @param t           current timestep (1-indexed)
     * @param N           number of parameters
     */
    public static void update(
            CudaTensor params,
            CudaTensor grads,
            CudaTensor m,
            CudaTensor v,
            float lr,
            float beta1,
            float beta2,
            float eps,
            float weightDecay,
            int t,
            long N
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaLlmKernels llm = backend.getLlmKernels();

        llm.adamwUpdate(params, grads, m, v, lr, beta1, beta2, eps, weightDecay, t, (int) N);
    }

    /**
     * AdamW update with default hyperparameters.
     *
     * <p>Uses: beta1=0.9, beta2=0.999, eps=1e-8</p>
     *
     * @param params      model parameters (N,)
     * @param grads       gradients (N,)
     * @param m           first moment (N,)
     * @param v           second moment (N,)
     * @param lr          learning rate
     * @param weightDecay weight decay
     * @param t           timestep
     * @param N           number of parameters
     */
    public static void update(
            CudaTensor params,
            CudaTensor grads,
            CudaTensor m,
            CudaTensor v,
            float lr,
            float weightDecay,
            int t,
            long N
    ) {
        update(params, grads, m, v, lr, 0.9f, 0.999f, 1e-8f, weightDecay, t, N);
    }

    /**
     * AdamW update with learning rate warmup and cosine decay.
     *
     * <p>Computes effective learning rate:</p>
     * <pre>
     * if step < warmupSteps:
     *     lr_eff = lr * step / warmupSteps
     * else:
     *     progress = (step - warmupSteps) / (totalSteps - warmupSteps)
     *     lr_eff = minLr + 0.5 * (lr - minLr) * (1 + cos(π * progress))
     * </pre>
     *
     * @param params      model parameters
     * @param grads       gradients
     * @param m           first moment
     * @param v           second moment
     * @param lr          peak learning rate
     * @param minLr       minimum learning rate
     * @param beta1       first moment decay
     * @param beta2       second moment decay
     * @param eps         epsilon
     * @param weightDecay weight decay
     * @param step        current step (0-indexed)
     * @param warmupSteps warmup steps
     * @param totalSteps  total training steps
     * @param N           number of parameters
     */
    public static void updateWithSchedule(
            CudaTensor params,
            CudaTensor grads,
            CudaTensor m,
            CudaTensor v,
            float lr,
            float minLr,
            float beta1,
            float beta2,
            float eps,
            float weightDecay,
            int step,
            int warmupSteps,
            int totalSteps,
            long N
    ) {
        float effectiveLr = computeLearningRate(lr, minLr, step, warmupSteps, totalSteps);
        update(params, grads, m, v, effectiveLr, beta1, beta2, eps, weightDecay, step + 1, N);
    }

    /**
     * Computes learning rate with warmup and cosine decay.
     *
     * @param lr          peak learning rate
     * @param minLr       minimum learning rate
     * @param step        current step (0-indexed)
     * @param warmupSteps warmup steps
     * @param totalSteps  total steps
     * @return effective learning rate
     */
    public static float computeLearningRate(
            float lr,
            float minLr,
            int step,
            int warmupSteps,
            int totalSteps
    ) {
        if (step < warmupSteps) {
            // Linear warmup
            return lr * (step + 1) / warmupSteps;
        } else {
            // Cosine decay
            int decaySteps = totalSteps - warmupSteps;
            if (decaySteps <= 0) {
                return minLr;
            }
            float progress = (float) (step - warmupSteps) / decaySteps;
            progress = Math.min(progress, 1.0f);
            return minLr + 0.5f * (lr - minLr) * (1.0f + (float) Math.cos(Math.PI * progress));
        }
    }

    /**
     * Zeros all gradients in a parameter tensor.
     *
     * @param grads gradient tensor to zero
     */
    public static void zeroGrad(CudaTensor grads) {
        FlashBackend backend = FlashBackend.getInstance();
        backend.zeroFill(grads);
    }

    /**
     * Computes the L2 norm of gradients (for gradient clipping).
     *
     * @param grads gradient tensor
     * @return L2 norm
     */
    public static float gradNorm(CudaTensor grads) {
        float[] gradData = grads.toFloatArray();
        double sumSq = 0;
        for (float g : gradData) {
            sumSq += (double) g * g;
        }
        return (float) Math.sqrt(sumSq);
    }

    /**
     * Clips gradients by global norm.
     *
     * <p>If ||grads|| > maxNorm, scales grads by maxNorm / ||grads||</p>
     *
     * @param grads   gradient tensor - modified in-place
     * @param maxNorm maximum allowed norm
     * @return actual gradient norm (before clipping)
     */
    public static float clipGradNorm(CudaTensor grads, float maxNorm) {
        float norm = gradNorm(grads);

        if (norm > maxNorm) {
            float scale = maxNorm / norm;

            FlashBackend backend = FlashBackend.getInstance();
            CudaDevice device = backend.getDevice();

            // Scale gradients
            float[] gradData = grads.toFloatArray();
            for (int i = 0; i < gradData.length; i++) {
                gradData[i] *= scale;
            }

            // Copy back to GPU
            TensorUtils.copyFromHost(device, gradData, grads);
        }

        return norm;
    }
}
