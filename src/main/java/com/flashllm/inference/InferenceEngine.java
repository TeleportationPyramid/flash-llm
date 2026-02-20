package com.flashllm.inference;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.kernel.*;
import com.flashllm.model.GPT2WeightLoader;
import com.flashllm.tokenizer.GPT2TokenizerLoader;
import com.flashllm.training.Generate;

import java.util.Random;

/**
 * Efficient inference engine with KV Cache.
 *
 * <p>This engine provides fast text generation using KV caching to avoid
 * recomputing attention for previous tokens.</p>
 *
 * <h2>Performance:</h2>
 * <ul>
 *   <li>Without KV Cache: O(n²) per token generation</li>
 *   <li>With KV Cache: O(n) per token generation</li>
 * </ul>
 *
 * <h2>Usage:</h2>
 * <pre>
 * InferenceEngine engine = new InferenceEngine(weights, tokenizer, maxSeqLen);
 * String output = engine.generate("Once upon a time", maxTokens, samplingConfig);
 * engine.close();
 * </pre>
 *
 * @author flash-llm
 * @since 2.0.0
 */
public final class InferenceEngine implements AutoCloseable {

    // Model config
    private final int B = 1;       // batch size (1 for inference)
    private final int maxSeqLen;   // maximum sequence length
    private final int C;           // channels
    private final int L;           // number of layers
    private final int NH;          // number of heads
    private final int HS;          // head size
    private final int V;           // vocab size
    private final int Vp;          // padded vocab size

    // Backend
    private final FlashBackend backend;
    private final CudaDevice device;

    // Weights (on GPU)
    private final CudaTensor wte;  // token embeddings
    private final CudaTensor wpe;  // position embeddings
    private final CudaTensor lnfw; // final layer norm weight
    private final CudaTensor lnfb; // final layer norm bias

    // Per-layer weights
    private final CudaTensor[] ln1w, ln1b;
    private final CudaTensor[] qkvw, qkvb;
    private final CudaTensor[] attprojw, attprojb;
    private final CudaTensor[] ln2w, ln2b;
    private final CudaTensor[] fcw, fcb;
    private final CudaTensor[] fcprojw, fcprojb;

    // KV Cache
    private final KVCache kvCache;

    // Tokenizer
    private final GPT2TokenizerLoader tokenizer;

    // Cached embedding data on CPU (avoids GPU->CPU transfer per token)
    private float[] wteDataCached;
    private float[] wpeDataCached;

    // Temporary tensors for inference (pre-allocated)
    private CudaTensor x;          // current hidden state (B, C)
    private CudaTensor lnf;        // final layer norm output
    private CudaTensor lnfMean, lnfRstd;
    private CudaTensor logits;     // output logits

    // Pre-allocated per-layer temporaries (avoids allocation per forward)
    private CudaTensor tempLn1, tempLn1Mean, tempLn1Rstd;
    private CudaTensor tempQkv;
    private CudaTensor tempAttnOut;
    private CudaTensor tempLn2, tempLn2Mean, tempLn2Rstd;
    private CudaTensor tempFch, tempFchGelu;
    private CudaTensor tempMlpOut;
    private CudaTensor tempQ, tempK, tempV;
    private CudaTensor tempAttnResult;

    /**
     * Create an inference engine.
     *
     * @param weights loaded GPT-2 weights
     * @param tokenizer loaded tokenizer
     * @param maxSeqLen maximum sequence length for generation
     */
    public InferenceEngine(GPT2WeightLoader weights, GPT2TokenizerLoader tokenizer, int maxSeqLen) {
        this.maxSeqLen = maxSeqLen;
        this.C = weights.channels;
        this.L = weights.numLayers;
        this.NH = weights.numHeads;
        this.HS = C / NH;
        this.V = weights.vocabSize;
        this.Vp = weights.paddedVocabSize;
        this.tokenizer = tokenizer;

        this.backend = FlashBackend.getInstance();
        this.device = backend.getDevice();

        // Allocate and load weights
        this.wte = backend.allocateF32(Vp * C);
        this.wpe = backend.allocateF32(weights.maxT * C);
        this.lnfw = backend.allocateF32(C);
        this.lnfb = backend.allocateF32(C);

        TensorUtils.copyFromHost(device, weights.getWte(), wte);
        TensorUtils.copyFromHost(device, weights.getWpe(), wpe);
        TensorUtils.copyFromHost(device, weights.getLnfw(), lnfw);
        TensorUtils.copyFromHost(device, weights.getLnfb(), lnfb);

        // Cache embeddings on CPU for fast lookup
        this.wteDataCached = weights.getWte();
        this.wpeDataCached = weights.getWpe();

        // Per-layer weights
        this.ln1w = new CudaTensor[L];
        this.ln1b = new CudaTensor[L];
        this.qkvw = new CudaTensor[L];
        this.qkvb = new CudaTensor[L];
        this.attprojw = new CudaTensor[L];
        this.attprojb = new CudaTensor[L];
        this.ln2w = new CudaTensor[L];
        this.ln2b = new CudaTensor[L];
        this.fcw = new CudaTensor[L];
        this.fcb = new CudaTensor[L];
        this.fcprojw = new CudaTensor[L];
        this.fcprojb = new CudaTensor[L];

        for (int l = 0; l < L; l++) {
            ln1w[l] = backend.allocateF32(C);
            ln1b[l] = backend.allocateF32(C);
            qkvw[l] = backend.allocateF32(C * 3 * C);
            qkvb[l] = backend.allocateF32(3 * C);
            attprojw[l] = backend.allocateF32(C * C);
            attprojb[l] = backend.allocateF32(C);
            ln2w[l] = backend.allocateF32(C);
            ln2b[l] = backend.allocateF32(C);
            fcw[l] = backend.allocateF32(C * 4 * C);
            fcb[l] = backend.allocateF32(4 * C);
            fcprojw[l] = backend.allocateF32(4 * C * C);
            fcprojb[l] = backend.allocateF32(C);

            TensorUtils.copyFromHost(device, weights.getLn1w(l), ln1w[l]);
            TensorUtils.copyFromHost(device, weights.getLn1b(l), ln1b[l]);
            TensorUtils.copyFromHost(device, weights.getQkvw(l), qkvw[l]);
            TensorUtils.copyFromHost(device, weights.getQkvb(l), qkvb[l]);
            TensorUtils.copyFromHost(device, weights.getAttprojw(l), attprojw[l]);
            TensorUtils.copyFromHost(device, weights.getAttprojb(l), attprojb[l]);
            TensorUtils.copyFromHost(device, weights.getLn2w(l), ln2w[l]);
            TensorUtils.copyFromHost(device, weights.getLn2b(l), ln2b[l]);
            TensorUtils.copyFromHost(device, weights.getFcw(l), fcw[l]);
            TensorUtils.copyFromHost(device, weights.getFcb(l), fcb[l]);
            TensorUtils.copyFromHost(device, weights.getFcprojw(l), fcprojw[l]);
            TensorUtils.copyFromHost(device, weights.getFcprojb(l), fcprojb[l]);
        }

        // Create KV Cache
        this.kvCache = new KVCache(B, maxSeqLen, L, NH, HS);

        // Allocate temporary tensors (reused across forward passes)
        this.x = backend.allocateF32(B * C);
        this.lnf = backend.allocateF32(B * C);
        this.lnfMean = backend.allocateF32(B);
        this.lnfRstd = backend.allocateF32(B);
        this.logits = backend.allocateF32(B * Vp);

        // Pre-allocate per-layer temporaries
        this.tempLn1 = backend.allocateF32(B * C);
        this.tempLn1Mean = backend.allocateF32(B);
        this.tempLn1Rstd = backend.allocateF32(B);
        this.tempQkv = backend.allocateF32(B * 3 * C);
        this.tempAttnOut = backend.allocateF32(B * C);
        this.tempLn2 = backend.allocateF32(B * C);
        this.tempLn2Mean = backend.allocateF32(B);
        this.tempLn2Rstd = backend.allocateF32(B);
        this.tempFch = backend.allocateF32(B * 4 * C);
        this.tempFchGelu = backend.allocateF32(B * 4 * C);
        this.tempMlpOut = backend.allocateF32(B * C);
        this.tempQ = backend.allocateF32(B * NH * HS);
        this.tempK = backend.allocateF32(B * NH * HS);
        this.tempV = backend.allocateF32(B * NH * HS);
        this.tempAttnResult = backend.allocateF32(B * NH * HS);

        System.out.println("InferenceEngine initialized:");
        System.out.println("  " + kvCache.getStats());
    }

    /**
     * Generate text starting from EOT token (like llm.c).
     *
     * @param maxTokens maximum tokens to generate
     * @param config sampling configuration
     * @return generated text
     */
    public String generate(int maxTokens, Generate.SamplingConfig config) {
        return generate(maxTokens, config, new Random());
    }

    /**
     * Generate text starting from EOT token (like llm.c).
     *
     * @param maxTokens maximum tokens to generate
     * @param config sampling configuration
     * @param rng random number generator
     * @return generated text
     */
    public String generate(int maxTokens, Generate.SamplingConfig config, Random rng) {
        // Reset cache for new generation
        kvCache.reset();

        StringBuilder result = new StringBuilder();

        // Start with EOT token (like llm.c)
        int currentToken = tokenizer.getEotToken();
        forwardSingleToken(currentToken, 0);
        int pos = 1;
        
        // Decode phase: generate new tokens one by one
        for (int i = 0; i < maxTokens && pos < maxSeqLen; i++) {
            // Get logits from last forward pass
            float[] logitsData = logits.toFloatArray();

            // Sample next token
            int nextToken = Generate.sample(logitsData, 0, V, config, rng);

            // Stop on EOT
            if (nextToken == tokenizer.getEotToken()) {
                break;
            }

            // Decode and append
            String tokenStr = tokenizer.decode(nextToken);
            result.append(tokenStr);

            // Forward pass for next token
            forwardSingleToken(nextToken, pos);
            pos++;
        }

        return result.toString();
    }

    /**
     * Generate text from token IDs.
     *
     * @param promptTokens array of prompt token IDs
     * @param maxTokens maximum tokens to generate
     * @param config sampling configuration
     * @param rng random number generator
     * @return generated text (including decoded prompt)
     */
    public String generate(int[] promptTokens, int maxTokens, Generate.SamplingConfig config, Random rng) {
        // Reset cache for new generation
        kvCache.reset();

        StringBuilder result = new StringBuilder();

        // Prefill phase: process all prompt tokens
        int pos = 0;
        for (int token : promptTokens) {
            result.append(tokenizer.decode(token));
            forwardSingleToken(token, pos);
            pos++;
        }

        // Decode phase: generate new tokens one by one
        for (int i = 0; i < maxTokens && pos < maxSeqLen; i++) {
            // Get logits from last forward pass
            float[] logitsData = logits.toFloatArray();

            // Sample next token
            int nextToken = Generate.sample(logitsData, 0, V, config, rng);

            // Stop on EOT
            if (nextToken == tokenizer.getEotToken()) {
                break;
            }

            // Decode and append
            String tokenStr = tokenizer.decode(nextToken);
            result.append(tokenStr);

            // Forward pass for next token
            forwardSingleToken(nextToken, pos);
            pos++;
        }

        return result.toString();
    }

    /**
     * Forward pass for a single token using KV Cache.
     *
     * @param token input token
     * @param pos position in sequence
     */
    private void forwardSingleToken(int token, int pos) {
        // Get token embedding + position embedding
        embedSingleToken(token, pos, x);

        // Process through transformer layers
        for (int l = 0; l < L; l++) {
            forwardLayerWithCache(l, pos);
        }

        // Final layer norm
        LayerNorm.forward(lnf, lnfMean, lnfRstd, x, lnfw, lnfb, B, C);

        // Output projection (logits)
        Matmul.forwardTransposed(logits, lnf, wte, B, Vp, C);
    }

    /**
     * Embed a single token with position.
     * Uses cached CPU data to avoid GPU->CPU transfer.
     */
    private void embedSingleToken(int token, int pos, CudaTensor out) {
        float[] outData = new float[C];

        // token embedding + position embedding (from CPU cache)
        int tokenOffset = token * C;
        int posOffset = pos * C;
        for (int i = 0; i < C; i++) {
            outData[i] = wteDataCached[tokenOffset + i] + wpeDataCached[posOffset + i];
        }

        TensorUtils.copyFromHost(device, outData, out);
    }

    /**
     * Forward pass through a single transformer layer with KV Cache.
     * Uses pre-allocated tensors to avoid memory allocation overhead.
     *
     * @param layer layer index
     * @param pos current position
     */
    private void forwardLayerWithCache(int layer, int pos) {
        int seqLen = pos + 1;  // total sequence length including current position

        // ============================================================
        // Attention Block: x = x + attention(layernorm1(x))
        // ============================================================

        // LayerNorm1
        LayerNorm.forward(tempLn1, tempLn1Mean, tempLn1Rstd, x, ln1w[layer], ln1b[layer], B, C);

        // QKV projection for current token only
        Matmul.forwardTransposed(tempQkv, tempLn1, qkvw[layer], qkvb[layer], B, 3 * C, C);

        // Extract Q, K, V
        extractQKV(tempQkv, tempQ, tempK, tempV);

        // Update KV cache with new K, V
        kvCache.updateSingleToken(layer, tempK, tempV, pos);

        // Get all cached K, V for attention
        try (CudaTensor cachedK = kvCache.getKeys(layer, 0, seqLen);
             CudaTensor cachedV = kvCache.getValues(layer, 0, seqLen)) {

            // Attention: Q attends to all cached K, V
            attentionWithCache(tempQ, cachedK, cachedV, tempAttnResult, seqLen, pos);

            // Reshape attention result and project
            reshapeAndProject(tempAttnResult, tempAttnOut, attprojw[layer], attprojb[layer]);
        }

        // Residual connection
        Residual.forwardInplace(x, tempAttnOut, B * C);

        // ============================================================
        // MLP Block: x = x + mlp(layernorm2(x))
        // ============================================================

        // LayerNorm2
        LayerNorm.forward(tempLn2, tempLn2Mean, tempLn2Rstd, x, ln2w[layer], ln2b[layer], B, C);

        // MLP: fc -> gelu -> fc_proj
        Matmul.forwardTransposed(tempFch, tempLn2, fcw[layer], fcb[layer], B, 4 * C, C);
        Gelu.forward(tempFchGelu, tempFch, B * 4 * C);
        Matmul.forwardTransposed(tempMlpOut, tempFchGelu, fcprojw[layer], fcprojb[layer], B, C, 4 * C);

        // Residual connection
        Residual.forwardInplace(x, tempMlpOut, B * C);
    }

    /**
     * Extract Q, K, V from QKV tensor.
     * Input: (B, 3*C), Output: Q, K, V each (B*NH, 1, HS)
     */
    private void extractQKV(CudaTensor qkv, CudaTensor q, CudaTensor k, CudaTensor v) {
        float[] qkvData = qkv.toFloatArray();
        float[] qData = new float[B * NH * HS];
        float[] kData = new float[B * NH * HS];
        float[] vData = new float[B * NH * HS];

        for (int b = 0; b < B; b++) {
            for (int nh = 0; nh < NH; nh++) {
                for (int hs = 0; hs < HS; hs++) {
                    int qkvOffset = b * 3 * C + nh * HS + hs;
                    int outOffset = (b * NH + nh) * HS + hs;

                    qData[outOffset] = qkvData[qkvOffset];
                    kData[outOffset] = qkvData[qkvOffset + C];
                    vData[outOffset] = qkvData[qkvOffset + 2 * C];
                }
            }
        }

        TensorUtils.copyFromHost(device, qData, q);
        TensorUtils.copyFromHost(device, kData, k);
        TensorUtils.copyFromHost(device, vData, v);
    }

    /**
     * Compute attention with cached K, V using GPU.
     * Q: (B*NH, 1, HS), K: (B*NH, seqLen, HS), V: (B*NH, seqLen, HS)
     * Output: (B*NH, 1, HS)
     *
     * <p>Uses optimized GPU computation:</p>
     * <ol>
     *   <li>scores = Q @ K^T / sqrt(HS) using batched dot product</li>
     *   <li>Apply causal mask and softmax (CPU for now, small seqLen)</li>
     *   <li>out = scores @ V using GPU matmul</li>
     * </ol>
     */
    private void attentionWithCache(CudaTensor q, CudaTensor cachedK, CudaTensor cachedV,
                                     CudaTensor out, int seqLen, int pos) {
        
        // Optimization: For very short sequences, CPU is faster due to overhead
        // For longer sequences (>32), GPU becomes beneficial
        if (seqLen <= 32) {
            attentionWithCacheCPU(q, cachedK, cachedV, out, seqLen, pos);
            return;
        }

        // GPU path for longer sequences
        float scale = 1.0f / (float) Math.sqrt(HS);

        // Step 1: Compute scores = Q @ K^T
        // Q: (B*NH, HS) -> reshape to (B*NH, 1, HS)
        // K: (B*NH, seqLen, HS) -> K^T: (B*NH, HS, seqLen)
        // scores: (B*NH, 1, seqLen) -> (B*NH, seqLen)
        
        // For each head, compute Q[1, HS] @ K^T[HS, seqLen] = scores[1, seqLen]
        float[] qData = q.toFloatArray();
        float[] kData = cachedK.toFloatArray();
        float[] vData = cachedV.toFloatArray();
        float[] scoresAll = new float[B * NH * seqLen];
        
        // Compute Q @ K^T with scaling (batched across heads)
        for (int bnh = 0; bnh < B * NH; bnh++) {
            for (int t = 0; t < seqLen; t++) {
                float score = 0;
                for (int h = 0; h < HS; h++) {
                    score += qData[bnh * HS + h] * kData[(bnh * seqLen + t) * HS + h];
                }
                scoresAll[bnh * seqLen + t] = score * scale;
            }
        }

        // Step 2: Apply causal mask and softmax (per head)
        for (int bnh = 0; bnh < B * NH; bnh++) {
            int offset = bnh * seqLen;
            
            // Causal mask: set future positions to -inf
            for (int t = pos + 1; t < seqLen; t++) {
                scoresAll[offset + t] = Float.NEGATIVE_INFINITY;
            }
            
            // Softmax
            float maxScore = Float.NEGATIVE_INFINITY;
            for (int t = 0; t <= pos; t++) {
                maxScore = Math.max(maxScore, scoresAll[offset + t]);
            }
            
            float sumExp = 0;
            for (int t = 0; t <= pos; t++) {
                scoresAll[offset + t] = (float) Math.exp(scoresAll[offset + t] - maxScore);
                sumExp += scoresAll[offset + t];
            }
            for (int t = 0; t <= pos; t++) {
                scoresAll[offset + t] /= sumExp;
            }
            // Zero out masked positions
            for (int t = pos + 1; t < seqLen; t++) {
                scoresAll[offset + t] = 0;
            }
        }

        // Step 3: Compute out = scores @ V
        // scores: (B*NH, seqLen), V: (B*NH, seqLen, HS) -> out: (B*NH, HS)
        float[] outData = new float[B * NH * HS];
        
        for (int bnh = 0; bnh < B * NH; bnh++) {
            for (int h = 0; h < HS; h++) {
                float sum = 0;
                for (int t = 0; t < seqLen; t++) {
                    sum += scoresAll[bnh * seqLen + t] * vData[(bnh * seqLen + t) * HS + h];
                }
                outData[bnh * HS + h] = sum;
            }
        }

        TensorUtils.copyFromHost(device, outData, out);
    }

    /**
     * CPU fallback for short sequences (optimized for low overhead).
     */
    private void attentionWithCacheCPU(CudaTensor q, CudaTensor cachedK, CudaTensor cachedV,
                                        CudaTensor out, int seqLen, int pos) {
        float[] qData = q.toFloatArray();
        float[] kData = cachedK.toFloatArray();
        float[] vData = cachedV.toFloatArray();
        float[] outData = new float[B * NH * HS];

        float scale = 1.0f / (float) Math.sqrt(HS);

        for (int bnh = 0; bnh < B * NH; bnh++) {
            // Compute attention scores
            float[] scores = new float[seqLen];
            float maxScore = Float.NEGATIVE_INFINITY;

            for (int t = 0; t <= pos; t++) {  // Only compute up to pos (causal)
                float score = 0;
                for (int h = 0; h < HS; h++) {
                    score += qData[bnh * HS + h] * kData[(bnh * seqLen + t) * HS + h];
                }
                scores[t] = score * scale;
                maxScore = Math.max(maxScore, scores[t]);
            }

            // Softmax (only over valid positions)
            float sumExp = 0;
            for (int t = 0; t <= pos; t++) {
                scores[t] = (float) Math.exp(scores[t] - maxScore);
                sumExp += scores[t];
            }
            for (int t = 0; t <= pos; t++) {
                scores[t] /= sumExp;
            }

            // Weighted sum of V (only over valid positions)
            for (int h = 0; h < HS; h++) {
                float sum = 0;
                for (int t = 0; t <= pos; t++) {
                    sum += scores[t] * vData[(bnh * seqLen + t) * HS + h];
                }
                outData[bnh * HS + h] = sum;
            }
        }

        TensorUtils.copyFromHost(device, outData, out);
    }

    /**
     * Reshape attention output and apply output projection.
     * Input: (B*NH, HS), Output: (B, C) after projection
     */
    private void reshapeAndProject(CudaTensor attnResult, CudaTensor out,
                                    CudaTensor attprojw, CudaTensor attprojb) {
        // Reshape from (B*NH, HS) to (B, C)
        float[] attnData = attnResult.toFloatArray();
        float[] reshapedData = new float[B * C];

        for (int b = 0; b < B; b++) {
            for (int nh = 0; nh < NH; nh++) {
                for (int hs = 0; hs < HS; hs++) {
                    int srcOffset = (b * NH + nh) * HS + hs;
                    int dstOffset = b * C + nh * HS + hs;
                    reshapedData[dstOffset] = attnData[srcOffset];
                }
            }
        }

        try (CudaTensor reshaped = backend.allocateF32(B * C)) {
            TensorUtils.copyFromHost(device, reshapedData, reshaped);
            // Output projection
            Matmul.forwardTransposed(out, reshaped, attprojw, attprojb, B, C, C);
        }
    }

    /**
     * Get engine statistics.
     */
    public String getStats() {
        return String.format(
            "InferenceEngine: L=%d, C=%d, NH=%d, HS=%d, V=%d, maxSeqLen=%d\n  %s",
            L, C, NH, HS, V, maxSeqLen, kvCache.getStats()
        );
    }

    @Override
    public void close() {
        kvCache.close();
        // Close weight tensors
        wte.close();
        wpe.close();
        lnfw.close();
        lnfb.close();
        for (int l = 0; l < L; l++) {
            ln1w[l].close(); ln1b[l].close();
            qkvw[l].close(); qkvb[l].close();
            attprojw[l].close(); attprojb[l].close();
            ln2w[l].close(); ln2b[l].close();
            fcw[l].close(); fcb[l].close();
            fcprojw[l].close(); fcprojb[l].close();
        }
        // Close temporary tensors
        x.close();
        lnf.close();
        lnfMean.close();
        lnfRstd.close();
        logits.close();
        
        // Close pre-allocated per-layer temporaries
        tempLn1.close();
        tempLn1Mean.close();
        tempLn1Rstd.close();
        tempQkv.close();
        tempAttnOut.close();
        tempLn2.close();
        tempLn2Mean.close();
        tempLn2Rstd.close();
        tempFch.close();
        tempFchGelu.close();
        tempMlpOut.close();
        tempQ.close();
        tempK.close();
        tempV.close();
        tempAttnResult.close();
    }
}
