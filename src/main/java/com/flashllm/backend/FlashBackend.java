package com.flashllm.backend;

import io.github.teleportationpyramid.flash.*;

/**
 * Unified Flash backend manager.
 *
 * <p>This singleton class manages the lifecycle of all Flash CUDA resources:</p>
 * <ul>
 *   <li>{@link CudaDevice} - GPU device context</li>
 *   <li>{@link CudaBlas} - Basic BLAS operations</li>
 *   <li>{@link CudaBlasExt} - Extended BLAS (transpose, batched GEMM)</li>
 *   <li>{@link CudaDnn} - DNN operations (softmax, activation)</li>
 *   <li>{@link CudaLlmKernels} - LLM-specific kernels</li>
 *   <li>{@link CudaRand} - Random number generation</li>
 * </ul>
 *
 * <h2>Usage:</h2>
 * <pre>{@code
 * // Get singleton instance (uses GPU 0)
 * FlashBackend backend = FlashBackend.getInstance();
 *
 * // Or specify GPU device
 * FlashBackend backend = FlashBackend.getInstance(1);
 *
 * // Use Flash APIs
 * CudaBlas blas = backend.getBlas();
 * CudaLlmKernels llm = backend.getLlmKernels();
 *
 * // Clean up (call once at program end)
 * backend.close();
 * }</pre>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public class FlashBackend implements AutoCloseable {

    private static volatile FlashBackend instance;
    private static final Object lock = new Object();

    private final int deviceId;
    private final CudaDevice device;
    private final CudaBlas blas;
    private final CudaBlasExt blasExt;
    private final CudaDnn dnn;
    private final CudaLlmKernels llmKernels;
    private final CudaRand rand;

    private boolean closed = false;

    // ========================================================================
    // Singleton Access
    // ========================================================================

    /**
     * Gets the singleton instance using default GPU (device 0).
     *
     * @return the FlashBackend instance
     * @throws RuntimeException if initialization fails
     */
    public static FlashBackend getInstance() {
        return getInstance(0);
    }

    /**
     * Gets the singleton instance for a specific GPU device.
     *
     * <p>Note: Once initialized with a device ID, subsequent calls with
     * different device IDs will return the existing instance.</p>
     *
     * @param deviceId GPU device index
     * @return the FlashBackend instance
     * @throws RuntimeException if initialization fails
     */
    public static FlashBackend getInstance(int deviceId) {
        if (instance == null) {
            synchronized (lock) {
                if (instance == null) {
                    instance = new FlashBackend(deviceId);
                }
            }
        }
        return instance;
    }

    /**
     * Checks if the backend has been initialized.
     *
     * @return true if getInstance() has been called
     */
    public static boolean isInitialized() {
        return instance != null && !instance.closed;
    }

    /**
     * Resets the singleton instance, allowing reinitialization.
     *
     * <p>This closes the current instance if it exists.</p>
     */
    public static void reset() {
        synchronized (lock) {
            if (instance != null) {
                instance.close();
                instance = null;
            }
        }
    }

    // ========================================================================
    // Constructor
    // ========================================================================

    private FlashBackend(int deviceId) {
        this.deviceId = deviceId;

        try {
            // Load native library using Flash's NativeLibraryLoader
            NativeLibraryLoader.getLibrary();

            // Validate device ID
            if (deviceId < 0) {
                throw new IllegalArgumentException("Invalid device ID: " + deviceId);
            }

            // Initialize device
            this.device = new CudaDevice(deviceId);

            // Initialize CUDA libraries
            this.blas = new CudaBlas(device);
            this.blasExt = new CudaBlasExt(blas);
            this.dnn = new CudaDnn(device);
            this.llmKernels = new CudaLlmKernels(device);
            this.rand = new CudaRand(device, System.nanoTime());

            System.out.printf("FlashBackend initialized on GPU %d: %s%n",
                    deviceId, device.getName());

        } catch (Exception e) {
            throw new RuntimeException("Failed to initialize FlashBackend", e);
        }
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /**
     * Gets the GPU device ID.
     */
    public int getDeviceId() {
        ensureNotClosed();
        return deviceId;
    }

    /**
     * Gets the CUDA device context.
     */
    public CudaDevice getDevice() {
        ensureNotClosed();
        return device;
    }

    /**
     * Gets the cuBLAS handle for basic BLAS operations.
     */
    public CudaBlas getBlas() {
        ensureNotClosed();
        return blas;
    }

    /**
     * Gets extended BLAS operations (transpose, batched GEMM).
     */
    public CudaBlasExt getBlasExt() {
        ensureNotClosed();
        return blasExt;
    }

    /**
     * Gets the cuDNN handle for DNN operations.
     */
    public CudaDnn getDnn() {
        ensureNotClosed();
        return dnn;
    }

    /**
     * Gets LLM-specific CUDA kernels.
     */
    public CudaLlmKernels getLlmKernels() {
        ensureNotClosed();
        return llmKernels;
    }

    /**
     * Gets the cuRAND handle for random number generation.
     */
    public CudaRand getRand() {
        ensureNotClosed();
        return rand;
    }

    // ========================================================================
    // Tensor Factory Methods
    // ========================================================================

    /**
     * Allocates a GPU tensor.
     *
     * @param elementCount number of elements
     * @param precision data type
     * @return allocated tensor
     */
    public CudaTensor allocate(long elementCount, Precision precision) {
        ensureNotClosed();
        return CudaTensor.allocate(device, elementCount, precision);
    }

    /**
     * Allocates a GPU tensor with FP32 precision.
     */
    public CudaTensor allocateF32(long elementCount) {
        return allocate(elementCount, Precision.FP32);
    }

    /**
     * Allocates a GPU tensor with FP16 precision.
     */
    public CudaTensor allocateF16(long elementCount) {
        return allocate(elementCount, Precision.FP16);
    }

    /**
     * Creates a GPU tensor from float array.
     *
     * @param data source data
     * @param precision target precision
     * @return GPU tensor
     */
    public CudaTensor fromFloat(float[] data, Precision precision) {
        ensureNotClosed();
        return CudaTensor.fromFloat(device, data, precision);
    }

    /**
     * Creates an FP32 GPU tensor from float array.
     */
    public CudaTensor fromFloat(float[] data) {
        return fromFloat(data, Precision.FP32);
    }

    /**
     * Creates a GPU tensor from double array.
     *
     * @param data source data
     * @param precision target precision
     * @return GPU tensor
     */
    public CudaTensor fromDouble(double[] data, Precision precision) {
        ensureNotClosed();
        return CudaTensor.fromDouble(device, data, precision);
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    /**
     * Fills a tensor with zeros.
     */
    public void zeroFill(CudaTensor tensor) {
        ensureNotClosed();
        CudaOps.fill(device, tensor, 0.0);
    }

    /**
     * Fills a tensor with a constant value.
     */
    public void fill(CudaTensor tensor, double value) {
        ensureNotClosed();
        CudaOps.fill(device, tensor, value);
    }

    /**
     * Synchronizes the GPU device.
     */
    public void synchronize() {
        ensureNotClosed();
        device.synchronize();
    }

    // ========================================================================
    // Resource Management
    // ========================================================================

    /**
     * Checks if the backend has been closed.
     */
    public boolean isClosed() {
        return closed;
    }

    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("FlashBackend has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;

            // Close in reverse order of creation
            try {
                if (rand != null) rand.close();
            } catch (Exception e) {
                System.err.println("Error closing CudaRand: " + e.getMessage());
            }

            try {
                if (llmKernels != null) llmKernels.close();
            } catch (Exception e) {
                System.err.println("Error closing CudaLlmKernels: " + e.getMessage());
            }

            try {
                if (dnn != null) dnn.close();
            } catch (Exception e) {
                System.err.println("Error closing CudaDnn: " + e.getMessage());
            }

            try {
                if (blas != null) blas.close();
            } catch (Exception e) {
                System.err.println("Error closing CudaBlas: " + e.getMessage());
            }

            try {
                if (device != null) device.close();
            } catch (Exception e) {
                System.err.println("Error closing CudaDevice: " + e.getMessage());
            }

            System.out.println("FlashBackend closed");
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "FlashBackend[CLOSED]";
        }
        return String.format("FlashBackend[device=%d, gpu=%s]", deviceId, device.getName());
    }
}