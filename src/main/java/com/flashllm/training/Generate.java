package com.flashllm.training;

import java.util.Arrays;
import java.util.Random;

/**
 * Text generation utilities for GPT-2.
 * 
 * <p>Supports various sampling strategies:</p>
 * <ul>
 *   <li>Greedy (temperature=0)</li>
 *   <li>Temperature sampling</li>
 *   <li>Top-k sampling</li>
 *   <li>Top-p (nucleus) sampling</li>
 *   <li>Combined top-k + top-p</li>
 * </ul>
 * 
 * <p>Corresponds to generate functions in llm.c and nanoGPT.</p>
 */
public final class Generate {

    private Generate() {}

    /**
     * Sampling configuration.
     */
    public static class SamplingConfig {
        public float temperature = 1.0f;
        public int topK = 0;           // 0 = disabled
        public float topP = 1.0f;      // 1.0 = disabled
        public long seed = -1;         // -1 = random seed
        
        public SamplingConfig() {}
        
        public SamplingConfig temperature(float t) {
            this.temperature = t;
            return this;
        }
        
        public SamplingConfig topK(int k) {
            this.topK = k;
            return this;
        }
        
        public SamplingConfig topP(float p) {
            this.topP = p;
            return this;
        }
        
        public SamplingConfig seed(long s) {
            this.seed = s;
            return this;
        }
        
        /**
         * Greedy decoding (always pick highest probability).
         */
        public static SamplingConfig greedy() {
            return new SamplingConfig().temperature(0);
        }
        
        /**
         * Default sampling with temperature.
         */
        public static SamplingConfig withTemperature(float t) {
            return new SamplingConfig().temperature(t);
        }
        
        /**
         * Top-k sampling.
         */
        public static SamplingConfig withTopK(int k) {
            return new SamplingConfig().topK(k);
        }
        
        /**
         * Top-p (nucleus) sampling.
         */
        public static SamplingConfig withTopP(float p) {
            return new SamplingConfig().topP(p);
        }
        
        /**
         * Combined top-k and top-p (recommended).
         * 
         * @param k top-k value (e.g., 50)
         * @param p top-p value (e.g., 0.95)
         * @param t temperature (e.g., 0.8)
         */
        public static SamplingConfig topKTopP(int k, float p, float t) {
            return new SamplingConfig().topK(k).topP(p).temperature(t);
        }
    }

    /**
     * Sample next token from logits.
     * 
     * @param logits raw logits array (size V or larger)
     * @param offset starting offset in logits array
     * @param vocabSize vocabulary size (V)
     * @param config sampling configuration
     * @param rng random number generator
     * @return sampled token index
     */
    public static int sample(float[] logits, int offset, int vocabSize, 
                             SamplingConfig config, Random rng) {
        // Greedy decoding
        if (config.temperature <= 0) {
            return argmax(logits, offset, vocabSize);
        }
        
        // Extract logits for this position
        float[] logitsSlice = new float[vocabSize];
        System.arraycopy(logits, offset, logitsSlice, 0, vocabSize);
        
        // Apply temperature
        if (config.temperature != 1.0f) {
            for (int i = 0; i < vocabSize; i++) {
                logitsSlice[i] /= config.temperature;
            }
        }
        
        // Convert to probabilities (softmax)
        float[] probs = softmax(logitsSlice);
        
        // Apply top-k filtering
        if (config.topK > 0 && config.topK < vocabSize) {
            probs = applyTopK(probs, config.topK);
        }
        
        // Apply top-p (nucleus) filtering
        if (config.topP < 1.0f) {
            probs = applyTopP(probs, config.topP);
        }
        
        // Sample from filtered distribution
        return sampleFromProbs(probs, rng);
    }
    
    /**
     * Sample from probability distribution.
     */
    public static int sampleFromProbs(float[] probs, Random rng) {
        double r = rng.nextDouble();
        double cumsum = 0;
        for (int i = 0; i < probs.length; i++) {
            cumsum += probs[i];
            if (r < cumsum) {
                return i;
            }
        }
        return probs.length - 1;
    }
    
    /**
     * Greedy selection (argmax).
     */
    public static int argmax(float[] logits, int offset, int size) {
        int maxIdx = 0;
        float maxVal = logits[offset];
        for (int i = 1; i < size; i++) {
            if (logits[offset + i] > maxVal) {
                maxVal = logits[offset + i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    /**
     * Softmax over logits.
     */
    public static float[] softmax(float[] logits) {
        float[] probs = new float[logits.length];
        
        // Find max for numerical stability
        float maxLogit = Float.NEGATIVE_INFINITY;
        for (float l : logits) {
            maxLogit = Math.max(maxLogit, l);
        }
        
        // Compute exp and sum
        float sum = 0;
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - maxLogit);
            sum += probs[i];
        }
        
        // Normalize
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        
        return probs;
    }
    
    /**
     * Apply top-k filtering.
     * 
     * <p>Keep only the top-k highest probability tokens, set others to 0.</p>
     * 
     * @param probs probability distribution
     * @param k number of top tokens to keep
     * @return filtered and renormalized probabilities
     */
    public static float[] applyTopK(float[] probs, int k) {
        int V = probs.length;
        k = Math.min(k, V);
        
        // Find k-th largest value using partial sort
        float[] sorted = probs.clone();
        Arrays.sort(sorted);
        float threshold = sorted[V - k];
        
        // Zero out tokens below threshold
        float[] filtered = new float[V];
        float sum = 0;
        for (int i = 0; i < V; i++) {
            if (probs[i] >= threshold) {
                filtered[i] = probs[i];
                sum += probs[i];
            }
        }
        
        // Renormalize
        if (sum > 0) {
            for (int i = 0; i < V; i++) {
                filtered[i] /= sum;
            }
        }
        
        return filtered;
    }
    
    /**
     * Apply top-p (nucleus) filtering.
     * 
     * <p>Keep the smallest set of tokens whose cumulative probability >= p.</p>
     * 
     * @param probs probability distribution
     * @param p cumulative probability threshold (e.g., 0.9 or 0.95)
     * @return filtered and renormalized probabilities
     */
    public static float[] applyTopP(float[] probs, float p) {
        int V = probs.length;
        
        // Create index-probability pairs and sort by probability descending
        IndexedProb[] indexed = new IndexedProb[V];
        for (int i = 0; i < V; i++) {
            indexed[i] = new IndexedProb(i, probs[i]);
        }
        Arrays.sort(indexed, (a, b) -> Float.compare(b.prob, a.prob));
        
        // Find cutoff index where cumulative probability >= p
        float cumsum = 0;
        int cutoff = V;
        for (int i = 0; i < V; i++) {
            cumsum += indexed[i].prob;
            if (cumsum >= p) {
                cutoff = i + 1;
                break;
            }
        }
        
        // Build filtered distribution
        float[] filtered = new float[V];
        float sum = 0;
        for (int i = 0; i < cutoff; i++) {
            int idx = indexed[i].index;
            filtered[idx] = probs[idx];
            sum += probs[idx];
        }
        
        // Renormalize
        if (sum > 0) {
            for (int i = 0; i < V; i++) {
                filtered[i] /= sum;
            }
        }
        
        return filtered;
    }
    
    /**
     * Helper class for sorting probabilities with indices.
     */
    private static class IndexedProb {
        final int index;
        final float prob;
        
        IndexedProb(int index, float prob) {
            this.index = index;
            this.prob = prob;
        }
    }
    
    // ========================================================================
    // Convenience methods for common sampling strategies
    // ========================================================================
    
    /**
     * Greedy sampling (always pick highest probability).
     */
    public static int sampleGreedy(float[] logits, int offset, int vocabSize) {
        return argmax(logits, offset, vocabSize);
    }
    
    /**
     * Temperature sampling.
     */
    public static int sampleWithTemperature(float[] logits, int offset, int vocabSize,
                                            float temperature, Random rng) {
        return sample(logits, offset, vocabSize, 
                     SamplingConfig.withTemperature(temperature), rng);
    }
    
    /**
     * Top-k sampling with default temperature.
     */
    public static int sampleTopK(float[] logits, int offset, int vocabSize,
                                 int k, Random rng) {
        return sample(logits, offset, vocabSize, 
                     SamplingConfig.withTopK(k), rng);
    }
    
    /**
     * Top-p sampling with default temperature.
     */
    public static int sampleTopP(float[] logits, int offset, int vocabSize,
                                 float p, Random rng) {
        return sample(logits, offset, vocabSize, 
                     SamplingConfig.withTopP(p), rng);
    }
    
    /**
     * Combined top-k + top-p + temperature (recommended for production).
     * 
     * <p>Example: sampleTopKTopP(logits, 0, V, 50, 0.95f, 0.8f, rng)</p>
     */
    public static int sampleTopKTopP(float[] logits, int offset, int vocabSize,
                                     int k, float p, float temperature, Random rng) {
        return sample(logits, offset, vocabSize,
                     SamplingConfig.topKTopP(k, p, temperature), rng);
    }
}
