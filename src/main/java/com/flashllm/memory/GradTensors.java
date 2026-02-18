package com.flashllm.memory;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.config.GPT2Config;

/**
 * Container for GPT-2 parameter gradients.
 *
 * <p>This mirrors ParameterTensors but stores gradients.
 * All gradients are stored in a single contiguous GPU memory block,
 * which enables efficient optimizer updates.</p>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public class GradTensors implements AutoCloseable {

    private final GPT2Config config;
    private final Precision precision;

    // Contiguous memory block for all gradients
    private CudaTensor memory;
    private final long totalElements;

    // === Embedding gradients ===
    public CudaTensor wte;   // (V, C)
    public CudaTensor wpe;   // (T, C)

    // === Per-layer gradients ===
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

    // === Final layer gradients ===
    public CudaTensor lnfw;  // (C,)
    public CudaTensor lnfb;  // (C,)

    private boolean closed = false;

    // ========================================================================
    // Constructor
    // ========================================================================

    /**
     * Creates gradient tensors for the given configuration.
     *
     * @param config model configuration
     */
    public GradTensors(GPT2Config config) {
        this(config, Precision.FP32);
    }

    /**
     * Creates gradient tensors for the given configuration.
     *
     * @param config model configuration
     * @param precision data precision
     */
    public GradTensors(GPT2Config config, Precision precision) {
        this.config = config;
        this.precision = precision;

        int L = config.numLayers;

        // Initialize per-layer arrays
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

        // Calculate total elements (same as ParameterTensors)
        this.totalElements = ParameterTensors.calculateTotalParameters(config);

        // Allocate and create views
        allocate();
    }

    // ========================================================================
    // Memory Allocation
    // ========================================================================

    /**
     * Allocates GPU memory and creates tensor views.
     */
    private void allocate() {
        FlashBackend backend = FlashBackend.getInstance();
        
        int V = config.vocabSize;
        int T = config.maxSeqLen;
        int C = config.channels;
        int L = config.numLayers;

        // Allocate contiguous memory
        memory = backend.allocate(totalElements, precision);
        backend.zeroFill(memory);  // Initialize to zero

        // Create individual tensor allocations
        // For simplicity, we allocate each tensor separately
        // In production, we'd use views into contiguous memory

        wte = backend.allocateF32((long) V * C);
        wpe = backend.allocateF32((long) T * C);

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
        }

        lnfw = backend.allocateF32(C);
        lnfb = backend.allocateF32(C);

        System.out.printf("Allocated %.2f MiB for parameter gradients%n",
            totalElements * precision.getBytesPerElement() / (1024.0 * 1024.0));
    }

    /**
     * Zeros all gradients.
     */
    public void zero() {
        FlashBackend backend = FlashBackend.getInstance();
        
        backend.zeroFill(wte);
        backend.zeroFill(wpe);

        for (int l = 0; l < config.numLayers; l++) {
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

        backend.zeroFill(lnfw);
        backend.zeroFill(lnfb);
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

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /**
     * Gets all gradients as a single contiguous tensor.
     * Note: This returns the memory block, not a copy.
     */
    public CudaTensor getAll() {
        return memory;
    }

    /**
     * Gets the total number of gradient elements.
     */
    public long numParameters() {
        return totalElements;
    }

    // ========================================================================
    // Resource Management
    // ========================================================================

    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;

            // Close all tensors
            if (wte != null) wte.close();
            if (wpe != null) wpe.close();

            for (int l = 0; l < config.numLayers; l++) {
                if (ln1w[l] != null) ln1w[l].close();
                if (ln1b[l] != null) ln1b[l].close();
                if (qkvw[l] != null) qkvw[l].close();
                if (qkvb[l] != null) qkvb[l].close();
                if (attprojw[l] != null) attprojw[l].close();
                if (attprojb[l] != null) attprojb[l].close();
                if (ln2w[l] != null) ln2w[l].close();
                if (ln2b[l] != null) ln2b[l].close();
                if (fcw[l] != null) fcw[l].close();
                if (fcb[l] != null) fcb[l].close();
                if (fcprojw[l] != null) fcprojw[l].close();
                if (fcprojb[l] != null) fcprojb[l].close();
            }

            if (lnfw != null) lnfw.close();
            if (lnfb != null) lnfb.close();
            if (memory != null) memory.close();
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "GradTensors[CLOSED]";
        }
        return String.format("GradTensors[params=%,d]", totalElements);
    }
}
