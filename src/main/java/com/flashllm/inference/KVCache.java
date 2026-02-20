package com.flashllm.inference;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.kernel.TensorUtils;

/**
 * KV Cache for efficient autoregressive inference.
 *
 * <p>During text generation, we don't need to recompute K and V for previous tokens.
 * This class caches K and V values for each layer, enabling O(n) generation instead of O(n²).</p>
 *
 * <h2>Memory Layout:</h2>
 * <pre>
 * For each layer l:
 *   keyCache[l]: (B, NH, maxSeqLen, HS)
 *   valueCache[l]: (B, NH, maxSeqLen, HS)
 * </pre>
 *
 * <h2>Usage:</h2>
 * <pre>
 * KVCache cache = new KVCache(B, maxSeqLen, numLayers, numHeads, headSize);
 * 
 * // First token (prefill)
 * cache.update(0, keys, values, 0, seqLen);
 * 
 * // Subsequent tokens (decode)
 * cache.update(0, newKey, newValue, currentPos, 1);
 * cachedK = cache.getKeys(0, 0, currentPos + 1);
 * cachedV = cache.getValues(0, 0, currentPos + 1);
 * </pre>
 *
 * @author flash-llm
 * @since 2.0.0
 */
public final class KVCache implements AutoCloseable {

    private final int B;          // batch size
    private final int maxSeqLen;  // maximum sequence length
    private final int numLayers;  // number of transformer layers
    private final int NH;         // number of heads
    private final int HS;         // head size
    private final int C;          // channels = NH * HS

    // Cache tensors: [layer][batch * numHeads * maxSeqLen * headSize]
    private final CudaTensor[] keyCache;
    private final CudaTensor[] valueCache;

    // Current sequence length for each batch
    private final int[] seqLens;

    // Backend reference
    private final FlashBackend backend;
    private final CudaDevice device;

    /**
     * Create a new KV Cache.
     *
     * @param B batch size
     * @param maxSeqLen maximum sequence length to cache
     * @param numLayers number of transformer layers
     * @param NH number of attention heads
     * @param HS head size
     */
    public KVCache(int B, int maxSeqLen, int numLayers, int NH, int HS) {
        this.B = B;
        this.maxSeqLen = maxSeqLen;
        this.numLayers = numLayers;
        this.NH = NH;
        this.HS = HS;
        this.C = NH * HS;

        this.backend = FlashBackend.getInstance();
        this.device = backend.getDevice();

        // Allocate cache tensors for each layer
        // Layout: (B, NH, maxSeqLen, HS) stored as flat array
        long cacheSize = (long) B * NH * maxSeqLen * HS;
        
        this.keyCache = new CudaTensor[numLayers];
        this.valueCache = new CudaTensor[numLayers];

        for (int l = 0; l < numLayers; l++) {
            keyCache[l] = backend.allocateF32((int) cacheSize);
            valueCache[l] = backend.allocateF32((int) cacheSize);
            backend.zeroFill(keyCache[l]);
            backend.zeroFill(valueCache[l]);
        }

        this.seqLens = new int[B];
    }

    /**
     * Update the cache with new K/V values for a layer.
     *
     * @param layer layer index
     * @param newKeys new key tensor (B, NH, newLen, HS) or (B*NH, newLen, HS)
     * @param newValues new value tensor (B, NH, newLen, HS) or (B*NH, newLen, HS)
     * @param startPos starting position in the sequence
     * @param newLen number of new tokens
     */
    public void update(int layer, CudaTensor newKeys, CudaTensor newValues, int startPos, int newLen) {
        // Copy new K/V to cache at the correct position
        // Source layout: (B*NH, newLen, HS) contiguous
        // Target layout: (B, NH, maxSeqLen, HS) with stride for maxSeqLen
        
        float[] newK = newKeys.toFloatArray();
        float[] newV = newValues.toFloatArray();
        float[] cachedK = keyCache[layer].toFloatArray();
        float[] cachedV = valueCache[layer].toFloatArray();

        // Copy each head's data to the cache
        for (int b = 0; b < B; b++) {
            for (int nh = 0; nh < NH; nh++) {
                for (int t = 0; t < newLen; t++) {
                    int srcOffset = ((b * NH + nh) * newLen + t) * HS;
                    int dstOffset = ((b * NH + nh) * maxSeqLen + startPos + t) * HS;
                    
                    System.arraycopy(newK, srcOffset, cachedK, dstOffset, HS);
                    System.arraycopy(newV, srcOffset, cachedV, dstOffset, HS);
                }
            }
        }

        // Update cache on GPU
        TensorUtils.copyFromHost(device, cachedK, keyCache[layer]);
        TensorUtils.copyFromHost(device, cachedV, valueCache[layer]);

        // Update sequence lengths
        for (int b = 0; b < B; b++) {
            seqLens[b] = Math.max(seqLens[b], startPos + newLen);
        }
    }

    /**
     * Update cache with a single new token (decode phase).
     * Optimized for single-token updates - minimizes GPU transfers.
     *
     * @param layer layer index
     * @param newKey new key tensor (B*NH, 1, HS) - will be kept on GPU
     * @param newValue new value tensor (B*NH, 1, HS) - will be kept on GPU
     * @param pos position in the sequence
     */
    public void updateSingleToken(int layer, CudaTensor newKey, CudaTensor newValue, int pos) {
        // For single token, we can use GPU-to-GPU copy if available
        // For now, fall back to CPU path but optimized for single token
        
        float[] newK = newKey.toFloatArray();
        float[] newV = newValue.toFloatArray();
        
        // Get current cache data
        float[] cachedK = keyCache[layer].toFloatArray();
        float[] cachedV = valueCache[layer].toFloatArray();

        // Update single position for each head
        for (int b = 0; b < B; b++) {
            for (int nh = 0; nh < NH; nh++) {
                int srcOffset = (b * NH + nh) * HS;
                int dstOffset = ((b * NH + nh) * maxSeqLen + pos) * HS;
                
                System.arraycopy(newK, srcOffset, cachedK, dstOffset, HS);
                System.arraycopy(newV, srcOffset, cachedV, dstOffset, HS);
            }
        }

        // Update cache on GPU
        TensorUtils.copyFromHost(device, cachedK, keyCache[layer]);
        TensorUtils.copyFromHost(device, cachedV, valueCache[layer]);

        // Update sequence lengths
        for (int b = 0; b < B; b++) {
            seqLens[b] = Math.max(seqLens[b], pos + 1);
        }
    }

    /**
     * Update cache with a single new token using direct memory copy.
     * This version keeps data on GPU as much as possible.
     *
     * @param layer layer index
     * @param newKeyData key data array (B*NH*HS floats)
     * @param newValueData value data array (B*NH*HS floats)
     * @param pos position in the sequence
     */
    public void updateSingleTokenDirect(int layer, float[] newKeyData, float[] newValueData, int pos) {
        // Get current cache data
        float[] cachedK = keyCache[layer].toFloatArray();
        float[] cachedV = valueCache[layer].toFloatArray();

        // Update single position for each head
        for (int b = 0; b < B; b++) {
            for (int nh = 0; nh < NH; nh++) {
                int srcOffset = (b * NH + nh) * HS;
                int dstOffset = ((b * NH + nh) * maxSeqLen + pos) * HS;
                
                System.arraycopy(newKeyData, srcOffset, cachedK, dstOffset, HS);
                System.arraycopy(newValueData, srcOffset, cachedV, dstOffset, HS);
            }
        }

        // Update cache on GPU
        TensorUtils.copyFromHost(device, cachedK, keyCache[layer]);
        TensorUtils.copyFromHost(device, cachedV, valueCache[layer]);

        // Update sequence lengths
        for (int b = 0; b < B; b++) {
            seqLens[b] = Math.max(seqLens[b], pos + 1);
        }
    }

    /**
     * Get cached keys up to a given length.
     *
     * @param layer layer index
     * @param startPos starting position
     * @param length number of positions to retrieve
     * @return key tensor (B*NH, length, HS)
     */
    public CudaTensor getKeys(int layer, int startPos, int length) {
        CudaTensor result = backend.allocateF32(B * NH * length * HS);
        
        float[] cachedK = keyCache[layer].toFloatArray();
        float[] outK = new float[B * NH * length * HS];

        for (int b = 0; b < B; b++) {
            for (int nh = 0; nh < NH; nh++) {
                for (int t = 0; t < length; t++) {
                    int srcOffset = ((b * NH + nh) * maxSeqLen + startPos + t) * HS;
                    int dstOffset = ((b * NH + nh) * length + t) * HS;
                    System.arraycopy(cachedK, srcOffset, outK, dstOffset, HS);
                }
            }
        }

        TensorUtils.copyFromHost(device, outK, result);
        return result;
    }

    /**
     * Get cached values up to a given length.
     *
     * @param layer layer index
     * @param startPos starting position
     * @param length number of positions to retrieve
     * @return value tensor (B*NH, length, HS)
     */
    public CudaTensor getValues(int layer, int startPos, int length) {
        CudaTensor result = backend.allocateF32(B * NH * length * HS);
        
        float[] cachedV = valueCache[layer].toFloatArray();
        float[] outV = new float[B * NH * length * HS];

        for (int b = 0; b < B; b++) {
            for (int nh = 0; nh < NH; nh++) {
                for (int t = 0; t < length; t++) {
                    int srcOffset = ((b * NH + nh) * maxSeqLen + startPos + t) * HS;
                    int dstOffset = ((b * NH + nh) * length + t) * HS;
                    System.arraycopy(cachedV, srcOffset, outV, dstOffset, HS);
                }
            }
        }

        TensorUtils.copyFromHost(device, outV, result);
        return result;
    }

    /**
     * Get the current sequence length for a batch.
     *
     * @param batchIdx batch index
     * @return current sequence length
     */
    public int getSeqLen(int batchIdx) {
        return seqLens[batchIdx];
    }

    /**
     * Get the current sequence length (assumes all batches have same length).
     *
     * @return current sequence length
     */
    public int getSeqLen() {
        return seqLens[0];
    }

    /**
     * Reset the cache (clear all cached values).
     */
    public void reset() {
        for (int l = 0; l < numLayers; l++) {
            backend.zeroFill(keyCache[l]);
            backend.zeroFill(valueCache[l]);
        }
        for (int b = 0; b < B; b++) {
            seqLens[b] = 0;
        }
    }

    /**
     * Get cache statistics for debugging.
     */
    public String getStats() {
        long totalBytes = 2L * numLayers * B * NH * maxSeqLen * HS * 4; // float32
        double totalMB = totalBytes / (1024.0 * 1024.0);
        return String.format(
            "KVCache: B=%d, maxSeqLen=%d, layers=%d, NH=%d, HS=%d, memory=%.2f MB",
            B, maxSeqLen, numLayers, NH, HS, totalMB
        );
    }

    // Getters
    public int getBatchSize() { return B; }
    public int getMaxSeqLen() { return maxSeqLen; }
    public int getNumLayers() { return numLayers; }
    public int getNumHeads() { return NH; }
    public int getHeadSize() { return HS; }
    public int getChannels() { return C; }

    @Override
    public void close() {
        for (int l = 0; l < numLayers; l++) {
            if (keyCache[l] != null) keyCache[l].close();
            if (valueCache[l] != null) valueCache[l].close();
        }
    }
}
