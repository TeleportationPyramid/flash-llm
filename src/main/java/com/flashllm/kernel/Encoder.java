package com.flashllm.kernel;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;

/**
 * Encoder kernel - Token and Position Embedding.
 *
 * <p>Corresponds to llm.c's encoder_forward/backward.</p>
 *
 * <h2>Forward:</h2>
 * <pre>out[b][t] = wte[tokens[b][t]] + wpe[t]</pre>
 *
 * <h2>Backward:</h2>
 * <pre>
 * dwte[tokens[b][t]] += dout[b][t]
 * dwpe[t] += sum over b of dout[b][t]
 * </pre>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public final class Encoder {

    private Encoder() {}

    /**
     * Encoder forward pass.
     *
     * <p>Looks up token embeddings and adds position embeddings.</p>
     *
     * @param out    output tensor (B*T, C) - will contain wte[token] + wpe[pos]
     * @param tokens input token IDs (B*T,)
     * @param wte    token embedding table (V, C)
     * @param wpe    position embedding table (T, C)
     * @param B      batch size
     * @param T      sequence length
     * @param C      embedding dimension
     */
    public static void forward(
            CudaTensor out,
            int[] tokens,
            CudaTensor wte,
            CudaTensor wpe,
            int B, int T, int C
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaLlmKernels llm = backend.getLlmKernels();

        // Step 1: Token embedding lookup -> out = wte[tokens]
        llm.embeddingForward(out, tokens, wte, B, T, C);

        // Step 2: Add position embeddings -> out += wpe[0:T]
        // Need to broadcast wpe across batch dimension
        // For now, use a simple approach: add wpe to each batch
        addPositionEmbeddings(backend, out, wpe, B, T, C);
    }

    /**
     * Encoder backward pass.
     *
     * <p>Accumulates gradients into dwte and dwpe.</p>
     *
     * @param dwte   gradient for token embeddings (V, C) - accumulated
     * @param dwpe   gradient for position embeddings (T, C) - accumulated
     * @param dout   gradient from upstream (B*T, C)
     * @param tokens input token IDs (B*T,)
     * @param B      batch size
     * @param T      sequence length
     * @param C      embedding dimension
     */
    public static void backward(
            CudaTensor dwte,
            CudaTensor dwpe,
            CudaTensor dout,
            int[] tokens,
            int B, int T, int C
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaLlmKernels llm = backend.getLlmKernels();

        // Step 1: Accumulate gradients into token embeddings
        // dwte[tokens[i]] += dout[i]
        llm.embeddingBackward(dwte, tokens, dout, B, T, C);

        // Step 2: Accumulate gradients into position embeddings
        // dwpe[t] += sum over b of dout[b, t]
        addPositionEmbeddingsBackward(backend, dwpe, dout, B, T, C);
    }

    /**
     * Adds position embeddings to the output.
     * out[b, t, :] += wpe[t, :]
     */
    private static void addPositionEmbeddings(
            FlashBackend backend,
            CudaTensor out,
            CudaTensor wpe,
            int B, int T, int C
    ) {
        CudaDevice device = backend.getDevice();

        // Get the raw data and perform addition
        // out has shape (B*T, C), wpe has shape (T, C)
        // We need to add wpe[t % T] to each position

        // Use CudaOps for element-wise operations
        // For each batch, add the corresponding position embedding
        // This is a broadcast add: out[b*T + t, :] += wpe[t, :]

        // Since Flash doesn't have a direct broadcast add, we'll use a loop
        // or implement it efficiently using the available ops

        // Simple approach: create a temporary tensor with broadcasted wpe
        // and add it to out

        // For efficiency, we can use the fact that wpe repeats every T positions
        // out[i] += wpe[i % T] for i in 0..B*T*C

        // Using axpy with stride would be ideal, but let's use a simpler approach
        // that works with available Flash APIs

        // Actually, we can reshape and use broadcasting if available
        // For now, let's do it in a straightforward way using element offsets

        // Get float arrays (temporary, for correctness first)
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

        // Copy back to GPU
        TensorUtils.copyFromHost(device, outData, out);
    }

    /**
     * Backward pass for position embeddings.
     * dwpe[t] += sum over b of dout[b, t]
     */
    private static void addPositionEmbeddingsBackward(
            FlashBackend backend,
            CudaTensor dwpe,
            CudaTensor dout,
            int B, int T, int C
    ) {
        CudaDevice device = backend.getDevice();

        // Get float arrays
        float[] dwpeData = dwpe.toFloatArray();
        float[] doutData = dout.toFloatArray();

        // Sum across batch dimension for each position
        for (int t = 0; t < T; t++) {
            int dwpeOffset = t * C;
            for (int b = 0; b < B; b++) {
                int doutOffset = (b * T + t) * C;
                for (int c = 0; c < C; c++) {
                    dwpeData[dwpeOffset + c] += doutData[doutOffset + c];
                }
            }
        }

        // Copy back to GPU
        TensorUtils.copyFromHost(device, dwpeData, dwpe);
    }
}
