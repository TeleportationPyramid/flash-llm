package com.flashllm.memory;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.config.GPT2Config;

/**
 * Container for GPT-2 forward pass activations.
 *
 * <p>All activations are stored in a single contiguous GPU memory block.
 * This follows the llm.c approach of pre-allocating all memory at once.</p>
 *
 * <h2>Activation Layout:</h2>
 * <pre>
 * [encoded]    Encoder output        (B, T, C)
 * -- Per layer (L times) --
 * [ln1]        LayerNorm 1 output    (B, T, C)
 * [ln1Mean]    LN1 mean              (B, T)
 * [ln1Rstd]    LN1 reciprocal std    (B, T)
 * [qkv]        QKV projection        (B, T, 3C)
 * [atty]       Attention output      (B, T, C)
 * [lse]        Log-sum-exp (FlashAttn) (B, NH, T)
 * [residual2]  After attention add   (B, T, C)
 * [ln2]        LayerNorm 2 output    (B, T, C)
 * [ln2Mean]    LN2 mean              (B, T)
 * [ln2Rstd]    LN2 reciprocal std    (B, T)
 * [fch]        MLP hidden            (B, T, 4C)
 * [fchGelu]    After GELU            (B, T, 4C)
 * [fcproj]     MLP projection        (B, T, C)
 * [residual3]  After MLP add         (B, T, C)
 * -- End per layer --
 * [lnf]        Final LayerNorm       (B, T, C)
 * [lnfMean]    Final LN mean         (B, T)
 * [lnfRstd]    Final LN rstd         (B, T)
 * [logits]     Output logits         (B, T, V)
 * [probs]      Softmax output        (B, T, V)
 * [losses]     Per-token losses      (B, T)
 * </pre>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public class ActivationTensors implements AutoCloseable {

    private final GPT2Config config;
    private final Precision precision;
    private final int batchSize;
    private final int seqLen;

    // Contiguous memory block
    private CudaTensor memory;
    private final long totalElements;

    // === Encoder output ===
    public long encodedOffset;
    public long encodedSize;

    // === Per-layer activations ===
    public long[] ln1Offsets;
    public long[] ln1MeanOffsets;
    public long[] ln1RstdOffsets;
    public long[] qkvOffsets;
    public long[] attyOffsets;
    public long[] lseOffsets;       // log-sum-exp for Flash Attention
    public long[] residual2Offsets;
    public long[] ln2Offsets;
    public long[] ln2MeanOffsets;
    public long[] ln2RstdOffsets;
    public long[] fchOffsets;
    public long[] fchGeluOffsets;
    public long[] fcprojOffsets;
    public long[] residual3Offsets;

    // Per-layer sizes
    public long lnSize;      // B * T * C
    public long lnStatsSize; // B * T
    public long qkvSize;     // B * T * 3C
    public long lseSize;     // B * NH * T
    public long fchSize;     // B * T * 4C

    // === Output activations ===
    public long lnfOffset;
    public long lnfMeanOffset;
    public long lnfRstdOffset;
    public long logitsOffset;
    public long probsOffset;
    public long lossesOffset;

    public long logitsSize;  // B * T * V
    public long lossesSize;  // B * T

    private boolean closed = false;

    // ========================================================================
    // Constructor
    // ========================================================================

    /**
     * Creates activation tensors for the given configuration and batch size.
     *
     * @param config model configuration
     * @param precision data precision
     * @param batchSize batch size (B)
     * @param seqLen sequence length (T), must be <= config.maxSeqLen
     */
    public ActivationTensors(GPT2Config config, Precision precision, int batchSize, int seqLen) {
        if (seqLen > config.maxSeqLen) {
            throw new IllegalArgumentException(
                "seqLen " + seqLen + " exceeds maxSeqLen " + config.maxSeqLen
            );
        }

        this.config = config;
        this.precision = precision;
        this.batchSize = batchSize;
        this.seqLen = seqLen;

        int L = config.numLayers;

        // Initialize offset arrays
        ln1Offsets = new long[L];
        ln1MeanOffsets = new long[L];
        ln1RstdOffsets = new long[L];
        qkvOffsets = new long[L];
        attyOffsets = new long[L];
        lseOffsets = new long[L];
        residual2Offsets = new long[L];
        ln2Offsets = new long[L];
        ln2MeanOffsets = new long[L];
        ln2RstdOffsets = new long[L];
        fchOffsets = new long[L];
        fchGeluOffsets = new long[L];
        fcprojOffsets = new long[L];
        residual3Offsets = new long[L];

        // Calculate sizes and offsets
        calculateSizes();
        this.totalElements = calculateOffsets();
    }

    /**
     * Creates activation tensors with FP32 precision and auto-allocates.
     *
     * @param config model configuration
     * @param batchSize batch size (B)
     * @param seqLen sequence length (T)
     */
    public ActivationTensors(GPT2Config config, int batchSize, int seqLen) {
        this(config, Precision.FP32, batchSize, seqLen);
        FlashBackend backend = FlashBackend.getInstance();
        allocate(backend);
        createViews(backend);
    }

    // ========================================================================
    // Memory Allocation
    // ========================================================================

    /**
     * Allocates GPU memory for all activations.
     *
     * @param backend the Flash backend
     */
    public void allocate(FlashBackend backend) {
        if (memory != null) {
            throw new IllegalStateException("Memory already allocated");
        }
        memory = backend.allocate(totalElements, precision);
        System.out.printf("Allocated %.2f MiB for activations (B=%d, T=%d)%n",
            memory.sizeInBytes() / (1024.0 * 1024.0), batchSize, seqLen);
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    public CudaTensor getMemory() {
        ensureAllocated();
        return memory;
    }

    public long getMemoryHandle() {
        ensureAllocated();
        return memory.getHandle();
    }

    public long getTotalElements() {
        return totalElements;
    }

    public long getSizeInBytes() {
        return totalElements * precision.getBytesPerElement();
    }

    public Precision getPrecision() {
        return precision;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getSeqLen() {
        return seqLen;
    }

    // ========================================================================
    // Handle Access (GPU pointers)
    // ========================================================================

    private long offsetToHandle(long offset) {
        return memory.getHandle() + offset * precision.getBytesPerElement();
    }

    public long getEncodedHandle() { return offsetToHandle(encodedOffset); }

    public long getLn1Handle(int layer) { return offsetToHandle(ln1Offsets[layer]); }
    public long getLn1MeanHandle(int layer) { return offsetToHandle(ln1MeanOffsets[layer]); }
    public long getLn1RstdHandle(int layer) { return offsetToHandle(ln1RstdOffsets[layer]); }
    public long getQkvHandle(int layer) { return offsetToHandle(qkvOffsets[layer]); }
    public long getAttyHandle(int layer) { return offsetToHandle(attyOffsets[layer]); }
    public long getLseHandle(int layer) { return offsetToHandle(lseOffsets[layer]); }
    public long getResidual2Handle(int layer) { return offsetToHandle(residual2Offsets[layer]); }
    public long getLn2Handle(int layer) { return offsetToHandle(ln2Offsets[layer]); }
    public long getLn2MeanHandle(int layer) { return offsetToHandle(ln2MeanOffsets[layer]); }
    public long getLn2RstdHandle(int layer) { return offsetToHandle(ln2RstdOffsets[layer]); }
    public long getFchHandle(int layer) { return offsetToHandle(fchOffsets[layer]); }
    public long getFchGeluHandle(int layer) { return offsetToHandle(fchGeluOffsets[layer]); }
    public long getFcprojHandle(int layer) { return offsetToHandle(fcprojOffsets[layer]); }
    public long getResidual3Handle(int layer) { return offsetToHandle(residual3Offsets[layer]); }

    public long getLnfHandle() { return offsetToHandle(lnfOffset); }
    public long getLnfMeanHandle() { return offsetToHandle(lnfMeanOffset); }
    public long getLnfRstdHandle() { return offsetToHandle(lnfRstdOffset); }
    public long getLogitsHandle() { return offsetToHandle(logitsOffset); }
    public long getProbsHandle() { return offsetToHandle(probsOffset); }
    public long getLossesHandle() { return offsetToHandle(lossesOffset); }

    /**
     * Gets the residual input for a given layer.
     * Layer 0 uses encoded, others use residual3 of previous layer.
     */
    public long getResidualInputHandle(int layer) {
        if (layer == 0) {
            return getEncodedHandle();
        } else {
            return getResidual3Handle(layer - 1);
        }
    }

    // ========================================================================
    // Size Calculations
    // ========================================================================

    private void calculateSizes() {
        int B = batchSize;
        int T = seqLen;
        int C = config.channels;
        int NH = config.numHeads;
        int V = config.paddedVocabSize;

        // Per-layer sizes
        lnSize = (long) B * T * C;
        lnStatsSize = (long) B * T;
        qkvSize = (long) B * T * 3 * C;
        lseSize = (long) B * NH * T;
        fchSize = (long) B * T * 4 * C;

        // Output sizes
        encodedSize = (long) B * T * C;
        logitsSize = (long) B * T * V;
        lossesSize = (long) B * T;
    }

    private long calculateOffsets() {
        long offset = 0;
        int L = config.numLayers;

        // Encoder output
        encodedOffset = offset;
        offset += encodedSize;

        // Per-layer activations
        for (int l = 0; l < L; l++) {
            ln1Offsets[l] = offset;
            offset += lnSize;

            ln1MeanOffsets[l] = offset;
            offset += lnStatsSize;

            ln1RstdOffsets[l] = offset;
            offset += lnStatsSize;

            qkvOffsets[l] = offset;
            offset += qkvSize;

            attyOffsets[l] = offset;
            offset += lnSize;

            lseOffsets[l] = offset;
            offset += lseSize;

            residual2Offsets[l] = offset;
            offset += lnSize;

            ln2Offsets[l] = offset;
            offset += lnSize;

            ln2MeanOffsets[l] = offset;
            offset += lnStatsSize;

            ln2RstdOffsets[l] = offset;
            offset += lnStatsSize;

            fchOffsets[l] = offset;
            offset += fchSize;

            fchGeluOffsets[l] = offset;
            offset += fchSize;

            fcprojOffsets[l] = offset;
            offset += lnSize;

            residual3Offsets[l] = offset;
            offset += lnSize;
        }

        // Output activations
        lnfOffset = offset;
        offset += lnSize;

        lnfMeanOffset = offset;
        offset += lnStatsSize;

        lnfRstdOffset = offset;
        offset += lnStatsSize;

        logitsOffset = offset;
        offset += logitsSize;

        probsOffset = offset;
        offset += logitsSize;

        lossesOffset = offset;
        offset += lossesSize;

        return offset;
    }

    /**
     * Calculates total activation count for given parameters.
     */
    public static long calculateTotalElements(GPT2Config config, int B, int T) {
        int C = config.channels;
        int L = config.numLayers;
        int NH = config.numHeads;
        int V = config.paddedVocabSize;

        // Encoder
        long encoded = (long) B * T * C;

        // Per-layer
        long ln = (long) B * T * C;
        long lnStats = 2L * B * T;  // mean + rstd
        long qkv = (long) B * T * 3 * C;
        long atty = (long) B * T * C;
        long lse = (long) B * NH * T;
        long residual = (long) B * T * C;
        long fch = (long) B * T * 4 * C;
        long fchGelu = (long) B * T * 4 * C;
        long fcproj = (long) B * T * C;

        long perLayer = ln + lnStats + qkv + atty + lse + residual + ln + lnStats + fch + fchGelu + fcproj + residual;

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
     * Prints activation layout information.
     */
    public void printLayout() {
        System.out.println("=== Activation Layout ===");
        System.out.printf("Batch size: %d, Sequence length: %d%n", batchSize, seqLen);
        System.out.printf("Total elements: %,d%n", totalElements);
        System.out.printf("Memory size: %.2f MiB (%s)%n",
            getSizeInBytes() / (1024.0 * 1024.0), precision);
        System.out.println();
        System.out.printf("encoded:  offset=%,d  size=%,d  (B*T*C)%n", encodedOffset, encodedSize);
        System.out.println();
        System.out.printf("Per layer (x%d):%n", config.numLayers);
        System.out.printf("  ln1:       size=%,d  (B*T*C)%n", lnSize);
        System.out.printf("  ln1 stats: size=%,d  (B*T * 2)%n", lnStatsSize * 2);
        System.out.printf("  qkv:       size=%,d  (B*T*3C)%n", qkvSize);
        System.out.printf("  atty:      size=%,d  (B*T*C)%n", lnSize);
        System.out.printf("  lse:       size=%,d  (B*NH*T)%n", lseSize);
        System.out.printf("  residual2: size=%,d  (B*T*C)%n", lnSize);
        System.out.printf("  ln2:       size=%,d  (B*T*C)%n", lnSize);
        System.out.printf("  ln2 stats: size=%,d  (B*T * 2)%n", lnStatsSize * 2);
        System.out.printf("  fch:       size=%,d  (B*T*4C)%n", fchSize);
        System.out.printf("  fchGelu:   size=%,d  (B*T*4C)%n", fchSize);
        System.out.printf("  fcproj:    size=%,d  (B*T*C)%n", lnSize);
        System.out.printf("  residual3: size=%,d  (B*T*C)%n", lnSize);
        System.out.println();
        System.out.printf("lnf:      offset=%,d  size=%,d%n", lnfOffset, lnSize);
        System.out.printf("logits:   offset=%,d  size=%,d  (B*T*V)%n", logitsOffset, logitsSize);
        System.out.printf("probs:    offset=%,d  size=%,d  (B*T*V)%n", probsOffset, logitsSize);
        System.out.printf("losses:   offset=%,d  size=%,d  (B*T)%n", lossesOffset, lossesSize);
    }

    private void ensureAllocated() {
        if (memory == null) {
            throw new IllegalStateException("Memory not allocated. Call allocate() first.");
        }
    }

    // ========================================================================
    // Resource Management
    // ========================================================================

    // ========================================================================
    // CudaTensor Views for Phase 3
    // ========================================================================

    // Cached tensor views
    public CudaTensor encoded;
    public CudaTensor residual3;  // Last layer output
    public CudaTensor lnf, lnfMean, lnfRstd;
    public CudaTensor logits, probs, losses;
    public CudaTensor dlogits, dlnf, dresidual3, dencoded;

    // Per-layer views
    private CudaTensor[] ln1Views, ln1MeanViews, ln1RstdViews;
    private CudaTensor[] qkvViews, attyViews, attLseViews, attnOutViews;
    private CudaTensor[] ln2Views, ln2MeanViews, ln2RstdViews;
    private CudaTensor[] fchViews, fchGeluViews;
    private CudaTensor[] layerOutputViews;
    private CudaTensor[] layerDOutputViews;

    /**
     * Creates CudaTensor views for all activations.
     * Must be called after allocate().
     */
    public void createViews(FlashBackend backend) {
        ensureAllocated();
        CudaDevice device = backend.getDevice();
        int L = config.numLayers;
        int BT = batchSize * seqLen;
        int C = config.channels;
        int V = config.paddedVocabSize;
        int NH = config.numHeads;

        // Main views
        encoded = createView(encodedOffset, BT * C);
        residual3 = createView(residual3Offsets[L - 1], BT * C);
        lnf = createView(lnfOffset, BT * C);
        lnfMean = createView(lnfMeanOffset, BT);
        lnfRstd = createView(lnfRstdOffset, BT);
        logits = createView(logitsOffset, BT * V);
        probs = createView(probsOffset, BT * V);
        losses = createView(lossesOffset, BT);

        // Gradient views (reuse probs/losses space or allocate separate)
        // For now, allocate separate
        dlogits = backend.allocateF32(BT * V);
        dlnf = backend.allocateF32(BT * C);
        dresidual3 = backend.allocateF32(BT * C);
        dencoded = backend.allocateF32(BT * C);

        // Per-layer views
        ln1Views = new CudaTensor[L];
        ln1MeanViews = new CudaTensor[L];
        ln1RstdViews = new CudaTensor[L];
        qkvViews = new CudaTensor[L];
        attyViews = new CudaTensor[L];
        attLseViews = new CudaTensor[L];
        attnOutViews = new CudaTensor[L];
        ln2Views = new CudaTensor[L];
        ln2MeanViews = new CudaTensor[L];
        ln2RstdViews = new CudaTensor[L];
        fchViews = new CudaTensor[L];
        fchGeluViews = new CudaTensor[L];
        layerOutputViews = new CudaTensor[L];
        layerDOutputViews = new CudaTensor[L];

        for (int l = 0; l < L; l++) {
            ln1Views[l] = createView(ln1Offsets[l], BT * C);
            ln1MeanViews[l] = createView(ln1MeanOffsets[l], BT);
            ln1RstdViews[l] = createView(ln1RstdOffsets[l], BT);
            qkvViews[l] = createView(qkvOffsets[l], BT * 3 * C);
            attyViews[l] = createView(attyOffsets[l], BT * C);
            attLseViews[l] = createView(lseOffsets[l], batchSize * NH * seqLen);
            attnOutViews[l] = createView(residual2Offsets[l], BT * C);
            ln2Views[l] = createView(ln2Offsets[l], BT * C);
            ln2MeanViews[l] = createView(ln2MeanOffsets[l], BT);
            ln2RstdViews[l] = createView(ln2RstdOffsets[l], BT);
            fchViews[l] = createView(fchOffsets[l], BT * 4 * C);
            fchGeluViews[l] = createView(fchGeluOffsets[l], BT * 4 * C);
            layerOutputViews[l] = createView(residual3Offsets[l], BT * C);
            layerDOutputViews[l] = backend.allocateF32(BT * C);  // Separate allocation for gradients
        }
    }

    private CudaTensor createView(long offset, long size) {
        // For now, we return the whole memory with offset information
        // In a real implementation, we'd create proper views
        // This is a simplified version that works with our kernel APIs
        return memory;  // Kernels will use offset-based access
    }

    // Accessors for per-layer tensors
    public CudaTensor getLn1(int layer) { return ln1Views[layer]; }
    public CudaTensor getLn1Mean(int layer) { return ln1MeanViews[layer]; }
    public CudaTensor getLn1Rstd(int layer) { return ln1RstdViews[layer]; }
    public CudaTensor getQkv(int layer) { return qkvViews[layer]; }
    public CudaTensor getAtty(int layer) { return attyViews[layer]; }
    public CudaTensor getAttLse(int layer) { return attLseViews[layer]; }
    public CudaTensor getAttnOut(int layer) { return attnOutViews[layer]; }
    public CudaTensor getLn2(int layer) { return ln2Views[layer]; }
    public CudaTensor getLn2Mean(int layer) { return ln2MeanViews[layer]; }
    public CudaTensor getLn2Rstd(int layer) { return ln2RstdViews[layer]; }
    public CudaTensor getFch(int layer) { return fchViews[layer]; }
    public CudaTensor getFchGelu(int layer) { return fchGeluViews[layer]; }
    public CudaTensor getLayerOutput(int layer) { return layerOutputViews[layer]; }
    public CudaTensor getLayerDOutput(int layer) { return layerDOutputViews[layer]; }

    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        // Close gradient tensors
        if (dlogits != null) { dlogits.close(); dlogits = null; }
        if (dlnf != null) { dlnf.close(); dlnf = null; }
        if (dresidual3 != null) { dresidual3.close(); dresidual3 = null; }
        if (dencoded != null) { dencoded.close(); dencoded = null; }
        if (layerDOutputViews != null) {
            for (CudaTensor t : layerDOutputViews) {
                if (t != null) t.close();
            }
            layerDOutputViews = null;
        }
        if (!closed) {
            closed = true;
            if (memory != null) {
                memory.close();
                memory = null;
            }
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "ActivationTensors[CLOSED]";
        }
        return String.format("ActivationTensors[B=%d, T=%d, elements=%,d, allocated=%b]",
            batchSize, seqLen, totalElements, memory != null);
    }
}
