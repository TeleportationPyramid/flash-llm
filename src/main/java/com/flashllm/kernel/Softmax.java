package com.flashllm.kernel;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;

/**
 * Softmax kernel - Softmax and Cross-Entropy operations.
 *
 * <p>Corresponds to llm.c's softmax_forward and crossentropy_softmax_backward.</p>
 *
 * <p>IMPORTANT: llm.c uses padded vocab size (Vp=50304) for memory layout but
 * only computes softmax over the real vocab size (V=50257). This prevents
 * the padded region from affecting probability distribution.</p>
 *
 * <h2>Softmax Forward:</h2>
 * <pre>probs = softmax(logits[:V])  // only over real vocab</pre>
 *
 * <h2>Cross-Entropy Forward:</h2>
 * <pre>losses[i] = -log(probs[i, targets[i]])</pre>
 *
 * <h2>Fused Backward:</h2>
 * <pre>dlogits = probs - one_hot(targets)</pre>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public final class Softmax {

    private Softmax() {}

    /**
     * Softmax forward pass with separate V (real vocab) and Vp (padded vocab).
     *
     * <p>Computes softmax only over the first V logits, ignoring padded region.</p>
     * <p>This matches llm.c's softmax_forward(probs, logits, B, T, V, Vp).</p>
     *
     * @param probs  output probabilities (B*T, Vp) - only first V are valid
     * @param logits input logits (B*T, Vp)
     * @param B      batch size
     * @param T      sequence length
     * @param V      real vocabulary size (50257)
     * @param Vp     padded vocabulary size (50304)
     */
    public static void forward(CudaTensor probs, CudaTensor logits, int B, int T, int V, int Vp) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaDevice device = backend.getDevice();
        
        float[] logitsData = logits.toFloatArray();
        float[] probsData = new float[B * T * Vp];
        
        for (int i = 0; i < B * T; i++) {
            int offset = i * Vp;  // stride is Vp
            
            // Find max for numerical stability (only in V range!)
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int v = 0; v < V; v++) {
                maxVal = Math.max(maxVal, logitsData[offset + v]);
            }
            
            // Compute exp(x - max) and sum (only in V range!)
            float sum = 0;
            for (int v = 0; v < V; v++) {
                probsData[offset + v] = (float) Math.exp(logitsData[offset + v] - maxVal);
                sum += probsData[offset + v];
            }
            
            // Normalize (only in V range!)
            float invSum = 1.0f / sum;
            for (int v = 0; v < V; v++) {
                probsData[offset + v] *= invSum;
            }
            
            // Padded region stays 0
            for (int v = V; v < Vp; v++) {
                probsData[offset + v] = 0.0f;
            }
        }
        
        // Copy to GPU
        TensorUtils.copyFromHost(device, probsData, probs);
    }

    /**
     * Softmax forward pass (convenience method when V == Vp).
     *
     * @param probs  output probabilities (B*T, V)
     * @param logits input logits (B*T, V)
     * @param B      batch size
     * @param T      sequence length
     * @param V      vocabulary size
     */
    public static void forward(CudaTensor probs, CudaTensor logits, int B, int T, int V) {
        forward(probs, logits, B, T, V, V);
    }

    /**
     * Cross-Entropy forward pass with separate V and Vp.
     *
     * <p>Computes per-token cross-entropy loss.</p>
     *
     * @param losses  output losses (B*T,)
     * @param probs   input probabilities (B*T, Vp)
     * @param targets target token IDs (B*T,)
     * @param B       batch size
     * @param T       sequence length
     * @param V       real vocabulary size
     * @param Vp      padded vocabulary size
     */
    public static void crossEntropyForward(
            CudaTensor losses,
            CudaTensor probs,
            int[] targets,
            int B, int T, int V, int Vp
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaDevice device = backend.getDevice();

        float[] probsData = probs.toFloatArray();
        float[] lossesData = new float[B * T];

        for (int i = 0; i < B * T; i++) {
            int target = targets[i];
            if (target < 0 || target >= V) {
                throw new IllegalArgumentException("Target " + target + " out of range [0, " + V + ")");
            }
            float prob = probsData[i * Vp + target];  // stride is Vp
            lossesData[i] = (float) (-Math.log(Math.max(prob, 1e-10)));
        }

        // Copy back to GPU
        try (CudaTensor temp = CudaTensor.fromFloat(device, lossesData, losses.getPrecision())) {
            CudaOps.fill(device, losses, 0.0);
            CudaOps.axpy(device, 1.0, temp, losses);
        }
    }

    /**
     * Cross-Entropy forward pass (convenience method when V == Vp).
     */
    public static void crossEntropyForward(
            CudaTensor losses,
            CudaTensor probs,
            int[] targets,
            int B, int T, int V
    ) {
        crossEntropyForward(losses, probs, targets, B, T, V, V);
    }

    /**
     * Computes mean loss from per-token losses.
     *
     * @param losses per-token losses (B*T,)
     * @param B      batch size
     * @param T      sequence length
     * @return mean loss
     */
    public static float meanLoss(CudaTensor losses, int B, int T) {
        float[] lossData = losses.toFloatArray();
        float sum = 0;
        for (int i = 0; i < B * T; i++) {
            sum += lossData[i];
        }
        return sum / (B * T);
    }

    /**
     * Fused Cross-Entropy + Softmax backward pass with separate V and Vp.
     *
     * <p>Computes: dlogits = (probs - one_hot(targets)) / (B * T)</p>
     *
     * @param dlogits gradient for logits (B*T, Vp) - output
     * @param probs   softmax probabilities (B*T, Vp)
     * @param targets target token IDs (B*T,)
     * @param B       batch size
     * @param T       sequence length
     * @param V       real vocabulary size
     * @param Vp      padded vocabulary size
     */
    public static void crossEntropySoftmaxBackward(
            CudaTensor dlogits,
            CudaTensor probs,
            int[] targets,
            int B, int T, int V, int Vp
    ) {
        crossEntropySoftmaxBackward(dlogits, probs, targets, 1.0f / (B * T), B, T, V, Vp);
    }

    /**
     * Fused Cross-Entropy + Softmax backward (convenience method when V == Vp).
     */
    public static void crossEntropySoftmaxBackward(
            CudaTensor dlogits,
            CudaTensor probs,
            int[] targets,
            int B, int T, int V
    ) {
        crossEntropySoftmaxBackward(dlogits, probs, targets, B, T, V, V);
    }

    /**
     * Fused Cross-Entropy + Softmax backward with custom scale and separate V/Vp.
     *
     * @param dlogits gradient for logits (B*T, Vp) - output
     * @param probs   softmax probabilities (B*T, Vp)
     * @param targets target token IDs (B*T,)
     * @param scale   gradient scale factor
     * @param B       batch size
     * @param T       sequence length
     * @param V       real vocabulary size
     * @param Vp      padded vocabulary size
     */
    public static void crossEntropySoftmaxBackward(
            CudaTensor dlogits,
            CudaTensor probs,
            int[] targets,
            float scale,
            int B, int T, int V, int Vp
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaDevice device = backend.getDevice();

        float[] probsData = probs.toFloatArray();
        float[] dlogitsData = new float[B * T * Vp];

        for (int i = 0; i < B * T; i++) {
            int target = targets[i];
            int offset = i * Vp;  // stride is Vp
            
            // Gradient for real vocab
            for (int v = 0; v < V; v++) {
                float grad = probsData[offset + v];
                if (v == target) {
                    grad -= 1.0f;
                }
                dlogitsData[offset + v] = grad * scale;
            }
            
            // Padded region gradient is 0
            for (int v = V; v < Vp; v++) {
                dlogitsData[offset + v] = 0.0f;
            }
        }

        // Copy back to GPU
        try (CudaTensor temp = CudaTensor.fromFloat(device, dlogitsData, dlogits.getPrecision())) {
            CudaOps.fill(device, dlogits, 0.0);
            CudaOps.axpy(device, 1.0, temp, dlogits);
        }
    }

    /**
     * Fused Cross-Entropy + Softmax backward with custom scale (convenience method when V == Vp).
     */
    public static void crossEntropySoftmaxBackward(
            CudaTensor dlogits,
            CudaTensor probs,
            int[] targets,
            float scale,
            int B, int T, int V
    ) {
        crossEntropySoftmaxBackward(dlogits, probs, targets, scale, B, T, V, V);
    }
}
