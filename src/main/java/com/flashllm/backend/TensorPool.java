package com.flashllm.backend;

import io.github.teleportationpyramid.flash.*;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * GPU tensor memory pool for efficient allocation.
 *
 * <p>This pool reduces allocation overhead by reusing tensors of the same size.
 * Tensors are cached when released and reused on subsequent acquisitions.</p>
 *
 * <h2>Usage:</h2>
 * <pre>{@code
 * FlashBackend backend = FlashBackend.getInstance();
 * TensorPool pool = new TensorPool(backend);
 *
 * // Acquire a tensor
 * CudaTensor t1 = pool.acquire(1024, Precision.FP32);
 *
 * // Use tensor...
 *
 * // Release back to pool (instead of close)
 * pool.release(t1);
 *
 * // Next acquire of same size will reuse t1
 * CudaTensor t2 = pool.acquire(1024, Precision.FP32); // reuses t1
 *
 * // Clean up all cached tensors
 * pool.clear();
 * }</pre>
 *
 * <h2>Thread Safety:</h2>
 * <p>This class is thread-safe.</p>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public class TensorPool implements AutoCloseable {

    private final FlashBackend backend;

    // Pool storage: (size, precision) -> queue of available tensors
    private final Map<PoolKey, Queue<CudaTensor>> pool;

    // Statistics
    private final AtomicLong hits = new AtomicLong(0);
    private final AtomicLong misses = new AtomicLong(0);
    private final AtomicLong allocatedBytes = new AtomicLong(0);

    // Configuration
    private final int maxCachedPerSize;
    private boolean closed = false;

    /**
     * Pool key combining size and precision.
     */
    private record PoolKey(long size, Precision precision) {}

    // ========================================================================
    // Constructors
    // ========================================================================

    /**
     * Creates a tensor pool with default settings.
     *
     * @param backend the Flash backend
     */
    public TensorPool(FlashBackend backend) {
        this(backend, 8);
    }

    /**
     * Creates a tensor pool with custom cache size.
     *
     * @param backend the Flash backend
     * @param maxCachedPerSize maximum tensors to cache per (size, precision) combination
     */
    public TensorPool(FlashBackend backend, int maxCachedPerSize) {
        if (backend == null) {
            throw new IllegalArgumentException("Backend cannot be null");
        }
        if (maxCachedPerSize < 1) {
            throw new IllegalArgumentException("maxCachedPerSize must be at least 1");
        }

        this.backend = backend;
        this.maxCachedPerSize = maxCachedPerSize;
        this.pool = new ConcurrentHashMap<>();
    }

    // ========================================================================
    // Core Operations
    // ========================================================================

    /**
     * Acquires a tensor of the specified size and precision.
     *
     * <p>If a matching tensor is available in the pool, it is reused.
     * Otherwise, a new tensor is allocated.</p>
     *
     * @param elementCount number of elements
     * @param precision data type
     * @return a tensor (may be reused or newly allocated)
     */
    public CudaTensor acquire(long elementCount, Precision precision) {
        ensureNotClosed();

        PoolKey key = new PoolKey(elementCount, precision);
        Queue<CudaTensor> queue = pool.get(key);

        CudaTensor tensor = null;
        if (queue != null) {
            tensor = queue.poll();
        }

        if (tensor != null) {
            hits.incrementAndGet();
            return tensor;
        } else {
            misses.incrementAndGet();
            tensor = backend.allocate(elementCount, precision);
            allocatedBytes.addAndGet(tensor.sizeInBytes());
            return tensor;
        }
    }

    /**
     * Acquires an FP32 tensor.
     */
    public CudaTensor acquireF32(long elementCount) {
        return acquire(elementCount, Precision.FP32);
    }

    /**
     * Acquires an FP16 tensor.
     */
    public CudaTensor acquireF16(long elementCount) {
        return acquire(elementCount, Precision.FP16);
    }

    /**
     * Releases a tensor back to the pool for reuse.
     *
     * <p>If the pool for this size is full, the tensor is closed instead.</p>
     *
     * @param tensor the tensor to release
     */
    public void release(CudaTensor tensor) {
        if (tensor == null || tensor.isClosed()) {
            return;
        }

        if (closed) {
            // Pool is closed, just close the tensor
            tensor.close();
            return;
        }

        PoolKey key = new PoolKey(tensor.getElementCount(), tensor.getPrecision());
        Queue<CudaTensor> queue = pool.computeIfAbsent(key, k -> new ConcurrentLinkedQueue<>());

        if (queue.size() < maxCachedPerSize) {
            queue.offer(tensor);
        } else {
            // Pool is full for this size, close the tensor
            allocatedBytes.addAndGet(-tensor.sizeInBytes());
            tensor.close();
        }
    }

    /**
     * Releases multiple tensors back to the pool.
     */
    public void release(CudaTensor... tensors) {
        for (CudaTensor tensor : tensors) {
            release(tensor);
        }
    }

    // ========================================================================
    // Batch Allocation
    // ========================================================================

    /**
     * Acquires multiple tensors of the same size.
     *
     * @param count number of tensors
     * @param elementCount elements per tensor
     * @param precision data type
     * @return array of tensors
     */
    public CudaTensor[] acquireMultiple(int count, long elementCount, Precision precision) {
        CudaTensor[] tensors = new CudaTensor[count];
        for (int i = 0; i < count; i++) {
            tensors[i] = acquire(elementCount, precision);
        }
        return tensors;
    }

    // ========================================================================
    // Pool Management
    // ========================================================================

    /**
     * Clears all cached tensors, freeing GPU memory.
     */
    public void clear() {
        for (Queue<CudaTensor> queue : pool.values()) {
            CudaTensor tensor;
            while ((tensor = queue.poll()) != null) {
                allocatedBytes.addAndGet(-tensor.sizeInBytes());
                tensor.close();
            }
        }
        pool.clear();
    }

    /**
     * Clears tensors of a specific size.
     */
    public void clear(long elementCount, Precision precision) {
        PoolKey key = new PoolKey(elementCount, precision);
        Queue<CudaTensor> queue = pool.remove(key);
        if (queue != null) {
            CudaTensor tensor;
            while ((tensor = queue.poll()) != null) {
                allocatedBytes.addAndGet(-tensor.sizeInBytes());
                tensor.close();
            }
        }
    }

    /**
     * Shrinks the pool by removing excess cached tensors.
     *
     * @param maxPerSize new maximum cache size per (size, precision)
     */
    public void shrink(int maxPerSize) {
        for (Queue<CudaTensor> queue : pool.values()) {
            while (queue.size() > maxPerSize) {
                CudaTensor tensor = queue.poll();
                if (tensor != null) {
                    allocatedBytes.addAndGet(-tensor.sizeInBytes());
                    tensor.close();
                }
            }
        }
    }

    // ========================================================================
    // Statistics
    // ========================================================================

    /**
     * Gets the cache hit count.
     */
    public long getHits() {
        return hits.get();
    }

    /**
     * Gets the cache miss count.
     */
    public long getMisses() {
        return misses.get();
    }

    /**
     * Gets the cache hit ratio.
     */
    public double getHitRatio() {
        long h = hits.get();
        long m = misses.get();
        long total = h + m;
        return total > 0 ? (double) h / total : 0.0;
    }

    /**
     * Gets the number of currently cached tensors.
     */
    public int getCachedCount() {
        int count = 0;
        for (Queue<CudaTensor> queue : pool.values()) {
            count += queue.size();
        }
        return count;
    }

    /**
     * Gets the total bytes allocated through this pool.
     */
    public long getAllocatedBytes() {
        return allocatedBytes.get();
    }

    /**
     * Gets the estimated bytes currently cached.
     */
    public long getCachedBytes() {
        long bytes = 0;
        for (Map.Entry<PoolKey, Queue<CudaTensor>> entry : pool.entrySet()) {
            PoolKey key = entry.getKey();
            int count = entry.getValue().size();
            bytes += count * key.size() * key.precision().getBytesPerElement();
        }
        return bytes;
    }

    /**
     * Resets statistics.
     */
    public void resetStats() {
        hits.set(0);
        misses.set(0);
    }

    /**
     * Prints pool statistics.
     */
    public void printStats() {
        System.out.printf("TensorPool Stats:%n");
        System.out.printf("  Hits: %,d%n", hits.get());
        System.out.printf("  Misses: %,d%n", misses.get());
        System.out.printf("  Hit Ratio: %.2f%%%n", getHitRatio() * 100);
        System.out.printf("  Cached Tensors: %d%n", getCachedCount());
        System.out.printf("  Cached Bytes: %,d MiB%n", getCachedBytes() / (1024 * 1024));
        System.out.printf("  Total Allocated: %,d MiB%n", allocatedBytes.get() / (1024 * 1024));
    }

    // ========================================================================
    // Resource Management
    // ========================================================================

    public boolean isClosed() {
        return closed;
    }

    private void ensureNotClosed() {
        if (closed) {
            throw new IllegalStateException("TensorPool has been closed");
        }
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            clear();
        }
    }

    @Override
    public String toString() {
        if (closed) {
            return "TensorPool[CLOSED]";
        }
        return String.format("TensorPool[cached=%d, hitRatio=%.2f%%]",
            getCachedCount(), getHitRatio() * 100);
    }
}
