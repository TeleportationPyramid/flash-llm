package com.flashllm.config;

/**
 * GPT-2 model configuration.
 *
 * <p>Mirrors the configuration structure from llm.c's GPT2Config.</p>
 *
 * <h2>Model Sizes:</h2>
 * <ul>
 *   <li>GPT-2 124M: 12 layers, 12 heads, 768 channels</li>
 *   <li>GPT-2 350M: 24 layers, 16 heads, 1024 channels</li>
 *   <li>GPT-2 774M: 36 layers, 20 heads, 1280 channels</li>
 *   <li>GPT-2 1558M: 48 layers, 25 heads, 1600 channels</li>
 * </ul>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public class GPT2Config {

    /** Configuration name */
    public String name = "gpt2";

    /** Maximum sequence length (context window) */
    public int maxSeqLen = 1024;

    /** Vocabulary size (GPT-2 BPE tokenizer) */
    public int vocabSize = 50257;

    /** Padded vocabulary size (aligned to 128 for CUDA efficiency) */
    public int paddedVocabSize = 50304;

    /** Number of transformer layers */
    public int numLayers = 12;

    /** Number of attention heads */
    public int numHeads = 12;

    /** Hidden dimension (embedding size) */
    public int channels = 768;

    // ========================================================================
    // Constructors
    // ========================================================================

    public GPT2Config() {
    }

    public GPT2Config(int maxSeqLen, int vocabSize, int numLayers, int numHeads, int channels) {
        this.maxSeqLen = maxSeqLen;
        this.vocabSize = vocabSize;
        this.paddedVocabSize = padVocabSize(vocabSize);
        this.numLayers = numLayers;
        this.numHeads = numHeads;
        this.channels = channels;
    }

    // ========================================================================
    // Preset Configurations
    // ========================================================================

    /**
     * Tiny configuration for testing (2 layers, 64 channels).
     */
    public static GPT2Config tiny() {
        GPT2Config config = new GPT2Config();
        config.name = "tiny";
        config.maxSeqLen = 64;
        config.vocabSize = 256;
        config.paddedVocabSize = 256;
        config.numLayers = 2;
        config.numHeads = 2;
        config.channels = 64;
        return config;
    }

    /**
     * Small configuration for testing (4 layers, 128 channels).
     */
    public static GPT2Config small() {
        GPT2Config config = new GPT2Config();
        config.name = "small";
        config.maxSeqLen = 256;
        config.vocabSize = 1024;
        config.paddedVocabSize = 1024;
        config.numLayers = 4;
        config.numHeads = 4;
        config.channels = 128;
        return config;
    }

    /**
     * GPT-2 124M configuration.
     */
    public static GPT2Config gpt2_124M() {
        GPT2Config config = new GPT2Config();
        config.maxSeqLen = 1024;
        config.vocabSize = 50257;
        config.paddedVocabSize = 50304;
        config.numLayers = 12;
        config.numHeads = 12;
        config.channels = 768;
        return config;
    }

    /**
     * GPT-2 350M configuration.
     */
    public static GPT2Config gpt2_350M() {
        GPT2Config config = new GPT2Config();
        config.maxSeqLen = 1024;
        config.vocabSize = 50257;
        config.paddedVocabSize = 50304;
        config.numLayers = 24;
        config.numHeads = 16;
        config.channels = 1024;
        return config;
    }

    /**
     * GPT-2 774M configuration.
     */
    public static GPT2Config gpt2_774M() {
        GPT2Config config = new GPT2Config();
        config.maxSeqLen = 1024;
        config.vocabSize = 50257;
        config.paddedVocabSize = 50304;
        config.numLayers = 36;
        config.numHeads = 20;
        config.channels = 1280;
        return config;
    }

    /**
     * GPT-2 1558M (1.5B) configuration.
     */
    public static GPT2Config gpt2_1558M() {
        GPT2Config config = new GPT2Config();
        config.maxSeqLen = 1024;
        config.vocabSize = 50257;
        config.paddedVocabSize = 50304;
        config.numLayers = 48;
        config.numHeads = 25;
        config.channels = 1600;
        return config;
    }

    // ========================================================================
    // Derived Properties
    // ========================================================================

    /**
     * Head dimension (channels / numHeads).
     */
    public int headDim() {
        return channels / numHeads;
    }

    /**
     * MLP hidden dimension (4 * channels for GPT-2).
     */
    public int mlpDim() {
        return 4 * channels;
    }

    /**
     * Calculate total number of parameters.
     *
     * <p>Parameter breakdown for GPT-2:</p>
     * <ul>
     *   <li>wte: vocabSize * channels (token embeddings)</li>
     *   <li>wpe: maxSeqLen * channels (position embeddings)</li>
     *   <li>Per layer:
     *     <ul>
     *       <li>ln1: 2 * channels (weight + bias)</li>
     *       <li>qkv: channels * 3*channels + 3*channels</li>
     *       <li>attn_proj: channels * channels + channels</li>
     *       <li>ln2: 2 * channels</li>
     *       <li>fc: channels * 4*channels + 4*channels</li>
     *       <li>fc_proj: 4*channels * channels + channels</li>
     *     </ul>
     *   </li>
     *   <li>lnf: 2 * channels (final layer norm)</li>
     * </ul>
     *
     * @return total parameter count
     */
    public long totalParameters() {
        long C = channels;
        long V = vocabSize;
        long T = maxSeqLen;
        long L = numLayers;

        // Embeddings
        long wte = V * C;
        long wpe = T * C;

        // Per-layer parameters
        long ln1 = 2 * C;                           // weight + bias
        long qkv = C * 3 * C + 3 * C;               // qkvw + qkvb
        long attnProj = C * C + C;                   // attprojw + attprojb
        long ln2 = 2 * C;
        long fc = C * 4 * C + 4 * C;                 // fcw + fcb
        long fcProj = 4 * C * C + C;                 // fcprojw + fcprojb

        long perLayer = ln1 + qkv + attnProj + ln2 + fc + fcProj;

        // Final layer norm
        long lnf = 2 * C;

        return wte + wpe + L * perLayer + lnf;
    }

    /**
     * Calculate total activations for a given batch size.
     *
     * @param B batch size
     * @param T sequence length (must be <= maxSeqLen)
     * @return total activation count (number of floats)
     */
    public long totalActivations(int B, int T) {
        long C = channels;
        long L = numLayers;
        long V = paddedVocabSize;

        // Encoder output
        long encoded = (long) B * T * C;

        // Per-layer activations
        long ln1 = (long) B * T * C;
        long ln1Stats = 2L * B * T;  // mean + rstd
        long atty = (long) B * T * C;
        long lse = (long) B * numHeads * T;  // log-sum-exp for flash attention
        long residual2 = (long) B * T * C;
        long ln2 = (long) B * T * C;
        long ln2Stats = 2L * B * T;
        long fch = (long) B * T * 4 * C;
        long fchGelu = (long) B * T * 4 * C;
        long fcproj = (long) B * T * C;
        long residual3 = (long) B * T * C;

        long perLayer = ln1 + ln1Stats + atty + lse + residual2 + ln2 + ln2Stats + fch + fchGelu + fcproj + residual3;

        // Output
        long lnf = (long) B * T * C;
        long lnfStats = 2L * B * T;
        long logits = (long) B * T * V;
        long probs = (long) B * T * V;
        long losses = (long) B * T;

        return encoded + L * perLayer + lnf + lnfStats + logits + probs + losses;
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /**
     * Pad vocabulary size to multiple of 128 for CUDA efficiency.
     */
    public static int padVocabSize(int vocabSize) {
        return ((vocabSize + 127) / 128) * 128;
    }

    /**
     * Validate configuration.
     *
     * @throws IllegalStateException if configuration is invalid
     */
    public void validate() {
        if (maxSeqLen <= 0) {
            throw new IllegalStateException("maxSeqLen must be positive");
        }
        if (vocabSize <= 0) {
            throw new IllegalStateException("vocabSize must be positive");
        }
        if (numLayers <= 0) {
            throw new IllegalStateException("numLayers must be positive");
        }
        if (numHeads <= 0) {
            throw new IllegalStateException("numHeads must be positive");
        }
        if (channels <= 0) {
            throw new IllegalStateException("channels must be positive");
        }
        if (channels % numHeads != 0) {
            throw new IllegalStateException("channels must be divisible by numHeads");
        }
    }

    @Override
    public String toString() {
        return String.format(
            "GPT2Config[maxSeqLen=%d, vocabSize=%d, numLayers=%d, numHeads=%d, channels=%d, params=%,d]",
            maxSeqLen, vocabSize, numLayers, numHeads, channels, totalParameters()
        );
    }
}
