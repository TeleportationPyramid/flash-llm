package com.flashllm.model;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.config.GPT2Config;
import com.flashllm.kernel.*;
import com.flashllm.memory.*;

/**
 * GPT2 - Complete GPT-2 model implementation.
 *
 * <p>Corresponds to llm.c's GPT2 struct with forward and backward passes.</p>
 *
 * <h2>Architecture:</h2>
 * <pre>
 * 1. Token Embedding + Position Embedding
 * 2. L x TransformerBlock
 * 3. Final LayerNorm
 * 4. Output projection (tied with token embedding)
 * 5. Softmax + Cross-Entropy Loss
 * </pre>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public class GPT2 implements AutoCloseable {

    private final GPT2Config config;
    private final int B;  // batch size
    private final int T;  // sequence length

    // Memory management
    private final ParameterTensors params;
    private final GradTensors grads;
    private final ActivationTensors acts;

    // Transformer blocks
    private final TransformerBlock[] blocks;

    // Backend
    private final FlashBackend backend;

    // State
    private boolean closed = false;
    private float meanLoss = 0.0f;

    /**
     * Creates a GPT2 model.
     *
     * @param config model configuration
     * @param B batch size
     * @param T sequence length
     */
    public GPT2(GPT2Config config, int B, int T) {
        this.config = config;
        this.B = B;
        this.T = T;
        this.backend = FlashBackend.getInstance();

        // Allocate memory
        this.params = new ParameterTensors(config);
        this.grads = new GradTensors(config);
        this.acts = new ActivationTensors(config, B, T);

        // Create transformer blocks
        this.blocks = new TransformerBlock[config.numLayers];
        for (int l = 0; l < config.numLayers; l++) {
            blocks[l] = new TransformerBlock(l, B, T, config.channels, config.numHeads);
        }

        System.out.println("GPT2 model created:");
        System.out.println("  Config: " + config.numLayers + " layers, " + config.channels + " channels, " + config.numHeads + " heads");
        System.out.println("  Batch: " + B + ", SeqLen: " + T);
        System.out.println("  Parameters: " + params.numParameters());
    }

    /**
     * Forward pass through the model.
     *
     * @param tokens input token IDs (B * T)
     * @param targets target token IDs for loss computation (B * T), can be null for inference
     * @return loss if targets provided, 0 otherwise
     */
    public float forward(int[] tokens, int[] targets) {
        ensureNotClosed();

        int C = config.channels;
        int V = config.vocabSize;
        int L = config.numLayers;
        int NH = config.numHeads;
        int BT = B * T;

        CudaDevice device = backend.getDevice();

        // ============================================================
        // 1. Encoder: Token Embedding + Position Embedding
        // ============================================================
        Encoder.forward(acts.encoded, tokens, params.wte, params.wpe, B, T, C);

        // ============================================================
        // 2. Transformer Blocks
        // ============================================================
        CudaTensor residual = acts.encoded;

        for (int l = 0; l < L; l++) {
            CudaTensor blockOut = (l == L - 1) ? acts.residual3 : acts.getLayerOutput(l);

            blocks[l].forward(
                    blockOut,
                    residual,
                    // Attention
                    params.getLn1w(l), params.getLn1b(l),
                    params.getQkvw(l), params.getQkvb(l),
                    params.getAttprojw(l), params.getAttprojb(l),
                    // MLP
                    params.getLn2w(l), params.getLn2b(l),
                    params.getFcw(l), params.getFcb(l),
                    params.getFcprojw(l), params.getFcprojb(l),
                    // Saved activations
                    acts.getLn1(l), acts.getLn1Mean(l), acts.getLn1Rstd(l),
                    acts.getQkv(l), acts.getAtty(l), acts.getAttLse(l), acts.getAttnOut(l),
                    acts.getLn2(l), acts.getLn2Mean(l), acts.getLn2Rstd(l),
                    acts.getFch(l), acts.getFchGelu(l)
            );

            residual = blockOut;
        }

        // ============================================================
        // 3. Final LayerNorm
        // ============================================================
        LayerNorm.forward(acts.lnf, acts.lnfMean, acts.lnfRstd, 
                acts.residual3, params.lnfw, params.lnfb, BT, C);

        // ============================================================
        // 4. Output Projection (tied weights with wte)
        // ============================================================
        // logits = lnf @ wte.T  (tied weights)
        Matmul.forward(acts.logits, acts.lnf, params.wte, BT, V, C);

        // ============================================================
        // 5. Loss Computation (if targets provided)
        // ============================================================
        if (targets != null) {
            // Softmax
            Softmax.forward(acts.probs, acts.logits, B, T, V);

            // Cross-Entropy
            Softmax.crossEntropyForward(acts.losses, acts.probs, targets, B, T, V);

            // Mean loss
            meanLoss = Softmax.meanLoss(acts.losses, B, T);
            return meanLoss;
        }

        return 0.0f;
    }

    /**
     * Backward pass through the model.
     *
     * @param tokens input token IDs (B * T)
     * @param targets target token IDs (B * T)
     */
    public void backward(int[] tokens, int[] targets) {
        ensureNotClosed();

        int C = config.channels;
        int V = config.vocabSize;
        int L = config.numLayers;
        int NH = config.numHeads;
        int BT = B * T;

        CudaDevice device = backend.getDevice();

        // Zero gradients
        grads.zero();

        // ============================================================
        // 5. Loss Backward: dlogits = (probs - one_hot(targets)) / (B * T)
        // ============================================================
        Softmax.crossEntropySoftmaxBackward(acts.dlogits, acts.probs, targets, B, T, V);

        // ============================================================
        // 4. Output Projection Backward (tied weights)
        // ============================================================
        // dlnf = dlogits @ wte
        // dwte += lnf.T @ dlogits
        Matmul.backward(acts.dlnf, grads.wte, null, acts.dlogits, acts.lnf, params.wte, BT, V, C);

        // ============================================================
        // 3. Final LayerNorm Backward
        // ============================================================
        LayerNorm.backward(acts.dresidual3, grads.lnfw, grads.lnfb,
                acts.dlnf, acts.residual3, params.lnfw, acts.lnfMean, acts.lnfRstd, BT, C);

        // ============================================================
        // 2. Transformer Blocks Backward (reverse order)
        // ============================================================
        CudaTensor dout = acts.dresidual3;

        for (int l = L - 1; l >= 0; l--) {
            CudaTensor inp = (l == 0) ? acts.encoded : acts.getLayerOutput(l - 1);
            CudaTensor dinp = (l == 0) ? acts.dencoded : acts.getLayerDOutput(l - 1);

            blocks[l].backward(
                    dinp,
                    dout,
                    inp,
                    // Weights
                    params.getLn1w(l), params.getQkvw(l), params.getAttprojw(l),
                    params.getLn2w(l), params.getFcw(l), params.getFcprojw(l),
                    // Saved activations
                    acts.getLn1(l), acts.getLn1Mean(l), acts.getLn1Rstd(l),
                    acts.getQkv(l), acts.getAtty(l), acts.getAttLse(l), acts.getAttnOut(l),
                    acts.getLn2(l), acts.getLn2Mean(l), acts.getLn2Rstd(l),
                    acts.getFch(l), acts.getFchGelu(l),
                    // Weight gradients
                    grads.getLn1w(l), grads.getLn1b(l),
                    grads.getQkvw(l), grads.getQkvb(l),
                    grads.getAttprojw(l), grads.getAttprojb(l),
                    grads.getLn2w(l), grads.getLn2b(l),
                    grads.getFcw(l), grads.getFcb(l),
                    grads.getFcprojw(l), grads.getFcprojb(l)
            );

            dout = dinp;
        }

        // ============================================================
        // 1. Encoder Backward
        // ============================================================
        Encoder.backward(grads.wte, grads.wpe, acts.dencoded, tokens, B, T, C);
    }

    /**
     * Performs a single training step (forward + backward).
     *
     * @param tokens input tokens
     * @param targets target tokens
     * @return loss value
     */
    public float step(int[] tokens, int[] targets) {
        float loss = forward(tokens, targets);
        backward(tokens, targets);
        return loss;
    }

    /**
     * Gets the model parameters.
     */
    public ParameterTensors getParams() {
        return params;
    }

    /**
     * Gets the gradients.
     */
    public GradTensors getGrads() {
        return grads;
    }

    /**
     * Gets the activations.
     */
    public ActivationTensors getActs() {
        return acts;
    }

    /**
     * Gets the configuration.
     */
    public GPT2Config getConfig() {
        return config;
    }

    /**
     * Gets the last computed mean loss.
     */
    public float getMeanLoss() {
        return meanLoss;
    }

    /**
     * Gets batch size.
     */
    public int getBatchSize() {
        return B;
    }

    /**
     * Gets sequence length.
     */
    public int getSeqLen() {
        return T;
    }

    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("GPT2 model is closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            params.close();
            grads.close();
            acts.close();
            closed = true;
        }
    }

    @Override
    public String toString() {
        return String.format("GPT2[%s, B=%d, T=%d, params=%d]",
                config.name, B, T, params.numParameters());
    }
}
