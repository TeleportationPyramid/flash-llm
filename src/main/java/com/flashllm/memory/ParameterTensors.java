package com.flashllm.memory;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.config.GPT2Config;

/**
 * Container for GPT-2 model parameters.
 *
 * <p>All parameters are stored in a single contiguous GPU memory block.
 * Individual parameter tensors are views into this memory at specific offsets.</p>
 *
 * <h2>Parameter Layout (following llm.c):</h2>
 * <pre>
 * [wte]      Token embeddings    (V, C)
 * [wpe]      Position embeddings (T, C)
 * -- Per layer (L times) --
 * [ln1w]     LayerNorm 1 weight  (C,)
 * [ln1b]     LayerNorm 1 bias    (C,)
 * [qkvw]     QKV projection      (C, 3C)
 * [qkvb]     QKV bias            (3C,)
 * [attprojw] Attn output proj    (C, C)
 * [attprojb] Attn output bias    (C,)
 * [ln2w]     LayerNorm 2 weight  (C,)
 * [ln2b]     LayerNorm 2 bias    (C,)
 * [fcw]      MLP fc weight       (C, 4C)
 * [fcb]      MLP fc bias         (4C,)
 * [fcprojw]  MLP proj weight     (4C, C)
 * [fcprojb]  MLP proj bias       (C,)
 * -- End per layer --
 * [lnfw]     Final LN weight     (C,)
 * [lnfb]     Final LN bias       (C,)
 * </pre>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public class ParameterTensors implements AutoCloseable {

    private final GPT2Config config;
    private final Precision precision;

    // Contiguous memory block for all parameters
    private CudaTensor memory;
    private final long totalElements;

    // === Embedding parameters (CudaTensor views) ===
    public CudaTensor wte;   // (V, C)
    public CudaTensor wpe;   // (T, C)

    // === Per-layer parameters ===
    private CudaTensor[] ln1w;     // (C,)
    private CudaTensor[] ln1b;     // (C,)
    private CudaTensor[] qkvw;     // (C, 3C)
    private CudaTensor[] qkvb;     // (3C,)
    private CudaTensor[] attprojw; // (C, C)
    private CudaTensor[] attprojb; // (C,)
    private CudaTensor[] ln2w;     // (C,)
    private CudaTensor[] ln2b;     // (C,)
    private CudaTensor[] fcw;      // (C, 4C)
    private CudaTensor[] fcb;      // (4C,)
    private CudaTensor[] fcprojw;  // (4C, C)
    private CudaTensor[] fcprojb;  // (C,)

    // === Final LayerNorm ===
    public CudaTensor lnfw;  // (C,)
    public CudaTensor lnfb;  // (C,)

    // === Offsets (for compatibility) ===
    public long wteOffset;
    public long wteSize;
    public long wpeOffset;
    public long wpeSize;
    public long[] ln1wOffsets;
    public long[] ln1bOffsets;
    public long[] qkvwOffsets;
    public long[] qkvbOffsets;
    public long[] attprojwOffsets;
    public long[] attprojbOffsets;
    public long[] ln2wOffsets;
    public long[] ln2bOffsets;
    public long[] fcwOffsets;
    public long[] fcbOffsets;
    public long[] fcprojwOffsets;
    public long[] fcprojbOffsets;
    public long ln1Size;
    public long qkvwSize;
    public long qkvbSize;
    public long attprojwSize;
    public long attprojbSize;
    public long fcwSize;
    public long fcbSize;
    public long fcprojwSize;
    public long fcprojbSize;
    public long lnfwOffset;
    public long lnfbOffset;
    public long lnfSize;

    private boolean closed = false;

    // ========================================================================
    // Constructor
    // ========================================================================

    /**
     * Creates and allocates parameter tensors for the given configuration.
     *
     * @param config model configuration
     */
    public ParameterTensors(GPT2Config config) {
        this(config, Precision.FP32);
        allocateAndCreateViews();
    }

    /**
     * Creates parameter tensors for the given configuration.
     *
     * @param config model configuration
     * @param precision data precision
     */
    public ParameterTensors(GPT2Config config, Precision precision) {
        this.config = config;
        this.precision = precision;
        this.totalElements = calculateTotalParameters(config);

        int L = config.numLayers;

        // Initialize offset arrays
        ln1wOffsets = new long[L];
        ln1bOffsets = new long[L];
        qkvwOffsets = new long[L];
        qkvbOffsets = new long[L];
        attprojwOffsets = new long[L];
        attprojbOffsets = new long[L];
        ln2wOffsets = new long[L];
        ln2bOffsets = new long[L];
        fcwOffsets = new long[L];
        fcbOffsets = new long[L];
        fcprojwOffsets = new long[L];
        fcprojbOffsets = new long[L];

        // Initialize tensor arrays
        ln1w = new CudaTensor[L];
        ln1b = new CudaTensor[L];
        qkvw = new CudaTensor[L];
        qkvb = new CudaTensor[L];
        attprojw = new CudaTensor[L];
        attprojb = new CudaTensor[L];
        ln2w = new CudaTensor[L];
        ln2b = new CudaTensor[L];
        fcw = new CudaTensor[L];
        fcb = new CudaTensor[L];
        fcprojw = new CudaTensor[L];
        fcprojb = new CudaTensor[L];

        // Calculate sizes
        calculateSizes();

        // Calculate offsets
        calculateOffsets();
    }

    // ========================================================================
    // Memory Allocation
    // ========================================================================

    /**
     * Allocates GPU memory and creates tensor views.
     */
    private void allocateAndCreateViews() {
        FlashBackend backend = FlashBackend.getInstance();
        
        int V = config.vocabSize;
        int T = config.maxSeqLen;
        int C = config.channels;
        int L = config.numLayers;

        // Allocate contiguous memory for all parameters
        memory = backend.allocate(totalElements, precision);
        backend.zeroFill(memory);

        // For Phase 3, we use separate allocations for simplicity
        // These will be properly initialized
        wte = backend.allocateF32((long) V * C);
        wpe = backend.allocateF32((long) T * C);
        
        backend.zeroFill(wte);
        backend.zeroFill(wpe);

        for (int l = 0; l < L; l++) {
            ln1w[l] = backend.allocateF32(C);
            ln1b[l] = backend.allocateF32(C);
            qkvw[l] = backend.allocateF32((long) C * 3 * C);
            qkvb[l] = backend.allocateF32(3 * C);
            attprojw[l] = backend.allocateF32((long) C * C);
            attprojb[l] = backend.allocateF32(C);
            ln2w[l] = backend.allocateF32(C);
            ln2b[l] = backend.allocateF32(C);
            fcw[l] = backend.allocateF32((long) C * 4 * C);
            fcb[l] = backend.allocateF32(4 * C);
            fcprojw[l] = backend.allocateF32((long) 4 * C * C);
            fcprojb[l] = backend.allocateF32(C);
            
            // Zero fill all
            backend.zeroFill(ln1w[l]);
            backend.zeroFill(ln1b[l]);
            backend.zeroFill(qkvw[l]);
            backend.zeroFill(qkvb[l]);
            backend.zeroFill(attprojw[l]);
            backend.zeroFill(attprojb[l]);
            backend.zeroFill(ln2w[l]);
            backend.zeroFill(ln2b[l]);
            backend.zeroFill(fcw[l]);
            backend.zeroFill(fcb[l]);
            backend.zeroFill(fcprojw[l]);
            backend.zeroFill(fcprojb[l]);
        }

        lnfw = backend.allocateF32(C);
        lnfb = backend.allocateF32(C);
        backend.zeroFill(lnfw);
        backend.zeroFill(lnfb);

        System.out.printf("Allocated %.2f MiB for %,d parameters%n",
            totalElements * precision.getBytesPerElement() / (1024.0 * 1024.0), totalElements);
    }

    /**
     * Allocates GPU memory for all parameters (legacy method).
     *
     * @param backend the Flash backend
     */
    public void allocate(FlashBackend backend) {
        if (memory != null) {
            throw new IllegalStateException("Memory already allocated");
        }
        memory = backend.allocate(totalElements, precision);
        System.out.printf("Allocated %.2f MiB for %,d parameters%n",
            memory.sizeInBytes() / (1024.0 * 1024.0), totalElements);
    }

    // ========================================================================
    // Per-layer Accessors
    // ========================================================================

    public CudaTensor getLn1w(int layer) { return ln1w[layer]; }
    public CudaTensor getLn1b(int layer) { return ln1b[layer]; }
    public CudaTensor getQkvw(int layer) { return qkvw[layer]; }
    public CudaTensor getQkvb(int layer) { return qkvb[layer]; }
    public CudaTensor getAttprojw(int layer) { return attprojw[layer]; }
    public CudaTensor getAttprojb(int layer) { return attprojb[layer]; }
    public CudaTensor getLn2w(int layer) { return ln2w[layer]; }
    public CudaTensor getLn2b(int layer) { return ln2b[layer]; }
    public CudaTensor getFcw(int layer) { return fcw[layer]; }
    public CudaTensor getFcb(int layer) { return fcb[layer]; }
    public CudaTensor getFcprojw(int layer) { return fcprojw[layer]; }
    public CudaTensor getFcprojb(int layer) { return fcprojb[layer]; }

    /**
     * Sets the memory from an external tensor (e.g., loaded from checkpoint).
     *
     * @param tensor pre-allocated tensor with parameter data
     */
    public void setMemory(CudaTensor tensor) {
        if (tensor.getElementCount() != totalElements) {
            throw new IllegalArgumentException(
                "Tensor size mismatch: expected " + totalElements +
                ", got " + tensor.getElementCount()
            );
        }
        if (tensor.getPrecision() != precision) {
            throw new IllegalArgumentException(
                "Precision mismatch: expected " + precision +
                ", got " + tensor.getPrecision()
            );
        }
        this.memory = tensor;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /**
     * Gets the underlying contiguous memory block.
     */
    public CudaTensor getMemory() {
        ensureAllocated();
        return memory;
    }

    /**
     * Gets the memory handle (GPU pointer).
     */
    public long getMemoryHandle() {
        ensureAllocated();
        return memory.getHandle();
    }

    /**
     * Gets the total number of parameters.
     */
    public long getTotalElements() {
        return totalElements;
    }

    /**
     * Gets the total size in bytes.
     */
    public long getSizeInBytes() {
        return totalElements * precision.getBytesPerElement();
    }

    /**
     * Gets the data precision.
     */
    public Precision getPrecision() {
        return precision;
    }

    // ========================================================================
    // Offset-based Access
    // ========================================================================

    // These methods return GPU memory offsets (in elements, not bytes)
    // Callers can use: memory.getHandle() + offset * bytesPerElement

    public long getWteHandle() { return memory.getHandle(); }
    public long getWpeHandle() { return memory.getHandle() + wpeOffset * precision.getBytesPerElement(); }

    public long getLn1wHandle(int layer) { return memory.getHandle() + ln1wOffsets[layer] * precision.getBytesPerElement(); }
    public long getLn1bHandle(int layer) { return memory.getHandle() + ln1bOffsets[layer] * precision.getBytesPerElement(); }
    public long getQkvwHandle(int layer) { return memory.getHandle() + qkvwOffsets[layer] * precision.getBytesPerElement(); }
    public long getQkvbHandle(int layer) { return memory.getHandle() + qkvbOffsets[layer] * precision.getBytesPerElement(); }
    public long getAttprojwHandle(int layer) { return memory.getHandle() + attprojwOffsets[layer] * precision.getBytesPerElement(); }
    public long getAttprojbHandle(int layer) { return memory.getHandle() + attprojbOffsets[layer] * precision.getBytesPerElement(); }
    public long getLn2wHandle(int layer) { return memory.getHandle() + ln2wOffsets[layer] * precision.getBytesPerElement(); }
    public long getLn2bHandle(int layer) { return memory.getHandle() + ln2bOffsets[layer] * precision.getBytesPerElement(); }
    public long getFcwHandle(int layer) { return memory.getHandle() + fcwOffsets[layer] * precision.getBytesPerElement(); }
    public long getFcbHandle(int layer) { return memory.getHandle() + fcbOffsets[layer] * precision.getBytesPerElement(); }
    public long getFcprojwHandle(int layer) { return memory.getHandle() + fcprojwOffsets[layer] * precision.getBytesPerElement(); }
    public long getFcprojbHandle(int layer) { return memory.getHandle() + fcprojbOffsets[layer] * precision.getBytesPerElement(); }

    public long getLnfwHandle() { return memory.getHandle() + lnfwOffset * precision.getBytesPerElement(); }
    public long getLnfbHandle() { return memory.getHandle() + lnfbOffset * precision.getBytesPerElement(); }

    // ========================================================================
    // Size Calculations
    // ========================================================================

    private void calculateSizes() {
        int C = config.channels;

        // Embeddings
        wteSize = (long) config.vocabSize * C;
        wpeSize = (long) config.maxSeqLen * C;

        // Per-layer sizes
        ln1Size = C;
        qkvwSize = (long) C * 3 * C;
        qkvbSize = 3L * C;
        attprojwSize = (long) C * C;
        attprojbSize = C;
        fcwSize = (long) C * 4 * C;
        fcbSize = 4L * C;
        fcprojwSize = (long) 4 * C * C;
        fcprojbSize = C;

        // Final LayerNorm
        lnfSize = C;
    }

    private void calculateOffsets() {
        long offset = 0;
        int L = config.numLayers;

        // Embeddings
        wteOffset = offset;
        offset += wteSize;

        wpeOffset = offset;
        offset += wpeSize;

        // Per-layer parameters
        for (int l = 0; l < L; l++) {
            ln1wOffsets[l] = offset;
            offset += ln1Size;

            ln1bOffsets[l] = offset;
            offset += ln1Size;

            qkvwOffsets[l] = offset;
            offset += qkvwSize;

            qkvbOffsets[l] = offset;
            offset += qkvbSize;

            attprojwOffsets[l] = offset;
            offset += attprojwSize;

            attprojbOffsets[l] = offset;
            offset += attprojbSize;

            ln2wOffsets[l] = offset;
            offset += ln1Size;  // same size as ln1

            ln2bOffsets[l] = offset;
            offset += ln1Size;

            fcwOffsets[l] = offset;
            offset += fcwSize;

            fcbOffsets[l] = offset;
            offset += fcbSize;

            fcprojwOffsets[l] = offset;
            offset += fcprojwSize;

            fcprojbOffsets[l] = offset;
            offset += fcprojbSize;
        }

        // Final LayerNorm
        lnfwOffset = offset;
        offset += lnfSize;

        lnfbOffset = offset;
        offset += lnfSize;

        // Verify total
        if (offset != totalElements) {
            throw new IllegalStateException(
                "Offset calculation error: expected " + totalElements + ", got " + offset
            );
        }
    }

    /**
     * Calculates total parameter count for a configuration.
     */
    public static long calculateTotalElements(GPT2Config config) {
        return calculateTotalParameters(config);
    }

    /**
     * Calculates total parameter count for a configuration.
     */
    public static long calculateTotalParameters(GPT2Config config) {
        int C = config.channels;
        int V = config.vocabSize;
        int T = config.maxSeqLen;
        int L = config.numLayers;

        // Embeddings
        long wte = (long) V * C;
        long wpe = (long) T * C;

        // Per-layer
        long ln1 = 2L * C;  // weight + bias
        long qkv = (long) C * 3 * C + 3L * C;
        long attnProj = (long) C * C + C;
        long ln2 = 2L * C;
        long fc = (long) C * 4 * C + 4L * C;
        long fcProj = (long) 4 * C * C + C;

        long perLayer = ln1 + qkv + attnProj + ln2 + fc + fcProj;

        // Final LayerNorm
        long lnf = 2L * C;

        return wte + wpe + L * perLayer + lnf;
    }

    /**
     * Gets the total number of parameters.
     */
    public long numParameters() {
        return totalElements;
    }

    /**
     * Gets all parameters as a single contiguous tensor.
     */
    public CudaTensor getAll() {
        ensureAllocated();
        return memory;
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /**
     * Prints parameter layout information.
     */
    public void printLayout() {
        System.out.println("=== Parameter Layout ===");
        System.out.printf("Total parameters: %,d%n", totalElements);
        System.out.printf("Memory size: %.2f MiB (%s)%n",
            getSizeInBytes() / (1024.0 * 1024.0), precision);
        System.out.println();
        System.out.printf("wte:      offset=%,d  size=%,d  (V=%d, C=%d)%n",
            wteOffset, wteSize, config.vocabSize, config.channels);
        System.out.printf("wpe:      offset=%,d  size=%,d  (T=%d, C=%d)%n",
            wpeOffset, wpeSize, config.maxSeqLen, config.channels);
        System.out.println();
        System.out.printf("Per layer (x%d):%n", config.numLayers);
        System.out.printf("  ln1w:     size=%,d%n", ln1Size);
        System.out.printf("  ln1b:     size=%,d%n", ln1Size);
        System.out.printf("  qkvw:     size=%,d  (C=%d, 3C=%d)%n", qkvwSize, config.channels, 3 * config.channels);
        System.out.printf("  qkvb:     size=%,d%n", qkvbSize);
        System.out.printf("  attprojw: size=%,d%n", attprojwSize);
        System.out.printf("  attprojb: size=%,d%n", attprojbSize);
        System.out.printf("  ln2w:     size=%,d%n", ln1Size);
        System.out.printf("  ln2b:     size=%,d%n", ln1Size);
        System.out.printf("  fcw:      size=%,d  (C=%d, 4C=%d)%n", fcwSize, config.channels, 4 * config.channels);
        System.out.printf("  fcb:      size=%,d%n", fcbSize);
        System.out.printf("  fcprojw:  size=%,d%n", fcprojwSize);
        System.out.printf("  fcprojb:  size=%,d%n", fcprojbSize);
        System.out.println();
        System.out.printf("lnfw:     offset=%,d  size=%,d%n", lnfwOffset, lnfSize);
        System.out.printf("lnfb:     offset=%,d  size=%,d%n", lnfbOffset, lnfSize);
    }

    private void ensureAllocated() {
        if (memory == null) {
            throw new IllegalStateException("Memory not allocated. Call allocate() first.");
        }
    }

    // ========================================================================
    // Resource Management
    // ========================================================================

    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        if (closed) {
            return;  // Already closed, avoid double-free
        }
        closed = true;

        // Close embedding tensors
        closeTensor(wte); wte = null;
        closeTensor(wpe); wpe = null;

        // Close per-layer tensors
        if (ln1w != null) {
            for (int l = 0; l < config.numLayers; l++) {
                closeTensor(ln1w[l]); ln1w[l] = null;
                closeTensor(ln1b[l]); ln1b[l] = null;
                closeTensor(qkvw[l]); qkvw[l] = null;
                closeTensor(qkvb[l]); qkvb[l] = null;
                closeTensor(attprojw[l]); attprojw[l] = null;
                closeTensor(attprojb[l]); attprojb[l] = null;
                closeTensor(ln2w[l]); ln2w[l] = null;
                closeTensor(ln2b[l]); ln2b[l] = null;
                closeTensor(fcw[l]); fcw[l] = null;
                closeTensor(fcb[l]); fcb[l] = null;
                closeTensor(fcprojw[l]); fcprojw[l] = null;
                closeTensor(fcprojb[l]); fcprojb[l] = null;
            }
        }

        // Close final layer norm
        closeTensor(lnfw); lnfw = null;
        closeTensor(lnfb); lnfb = null;

        // Close contiguous memory last
        closeTensor(memory); memory = null;
    }

    /**
     * Safely closes a tensor if not null.
     */
    private void closeTensor(CudaTensor tensor) {
        if (tensor != null) {
            try {
                tensor.close();
            } catch (Exception e) {
                // Ignore close errors
                System.err.println("Warning: Failed to close tensor: " + e.getMessage());
            }
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "ParameterTensors[CLOSED]";
        }
        return String.format("ParameterTensors[params=%,d, precision=%s, allocated=%b]",
            totalElements, precision, memory != null);
    }
}
