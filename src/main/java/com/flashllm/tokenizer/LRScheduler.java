package com.flashllm.tokenizer;

/**
 * Learning rate scheduler for GPT-2 training.
 * 
 * <p>Supports common scheduling strategies:</p>
 * <ul>
 *   <li><b>Constant</b> - No decay</li>
 *   <li><b>Linear warmup</b> - Gradual increase at start</li>
 *   <li><b>Cosine decay</b> - Smooth decay to minimum</li>
 *   <li><b>Warmup + Cosine</b> - Standard GPT training schedule</li>
 * </ul>
 * 
 * <h2>Usage Example</h2>
 * <pre>{@code
 * // Standard GPT-2 training schedule
 * LRScheduler scheduler = LRScheduler.warmupCosine(
 *     maxLr: 6e-4f,
 *     minLr: 6e-5f,
 *     warmupSteps: 100,
 *     totalSteps: 10000
 * );
 * 
 * for (int step = 0; step < totalSteps; step++) {
 *     float lr = scheduler.getLr(step);
 *     optimizer.step(lr);
 * }
 * }</pre>
 * 
 * <p>Corresponds to get_lr() in llm.c and nanoGPT.</p>
 */
public final class LRScheduler {

    private final ScheduleType type;
    private final float maxLr;
    private final float minLr;
    private final int warmupSteps;
    private final int totalSteps;

    public enum ScheduleType {
        CONSTANT,
        LINEAR_WARMUP,
        COSINE_DECAY,
        WARMUP_COSINE
    }

    private LRScheduler(ScheduleType type, float maxLr, float minLr, 
                        int warmupSteps, int totalSteps) {
        this.type = type;
        this.maxLr = maxLr;
        this.minLr = minLr;
        this.warmupSteps = warmupSteps;
        this.totalSteps = totalSteps;
    }

    /**
     * Get learning rate for given step.
     * 
     * @param step current training step (0-indexed)
     * @return learning rate for this step
     */
    public float getLr(int step) {
        switch (type) {
            case CONSTANT:
                return maxLr;
                
            case LINEAR_WARMUP:
                return linearWarmup(step);
                
            case COSINE_DECAY:
                return cosineDecay(step);
                
            case WARMUP_COSINE:
                return warmupCosine(step);
                
            default:
                return maxLr;
        }
    }

    // ========================================================================
    // Schedule implementations
    // ========================================================================

    private float linearWarmup(int step) {
        if (step < warmupSteps) {
            return maxLr * (step + 1) / warmupSteps;
        }
        return maxLr;
    }

    private float cosineDecay(int step) {
        if (step >= totalSteps) {
            return minLr;
        }
        float progress = (float) step / totalSteps;
        float coeff = 0.5f * (1.0f + (float) Math.cos(Math.PI * progress));
        return minLr + (maxLr - minLr) * coeff;
    }

    private float warmupCosine(int step) {
        // Phase 1: Linear warmup
        if (step < warmupSteps) {
            return maxLr * (step + 1) / warmupSteps;
        }
        
        // Phase 2: Cosine decay
        if (step >= totalSteps) {
            return minLr;
        }
        
        int decaySteps = totalSteps - warmupSteps;
        float progress = (float) (step - warmupSteps) / decaySteps;
        float coeff = 0.5f * (1.0f + (float) Math.cos(Math.PI * progress));
        return minLr + (maxLr - minLr) * coeff;
    }

    // ========================================================================
    // Factory methods
    // ========================================================================

    /**
     * Constant learning rate (no decay).
     */
    public static LRScheduler constant(float lr) {
        return new LRScheduler(ScheduleType.CONSTANT, lr, lr, 0, 0);
    }

    /**
     * Linear warmup only.
     * 
     * @param maxLr target learning rate after warmup
     * @param warmupSteps number of warmup steps
     */
    public static LRScheduler linearWarmup(float maxLr, int warmupSteps) {
        return new LRScheduler(ScheduleType.LINEAR_WARMUP, maxLr, maxLr, warmupSteps, 0);
    }

    /**
     * Cosine decay only (no warmup).
     * 
     * @param maxLr starting learning rate
     * @param minLr ending learning rate
     * @param totalSteps total training steps
     */
    public static LRScheduler cosineDecay(float maxLr, float minLr, int totalSteps) {
        return new LRScheduler(ScheduleType.COSINE_DECAY, maxLr, minLr, 0, totalSteps);
    }

    /**
     * Linear warmup followed by cosine decay.
     * 
     * <p>This is the standard schedule for GPT-2/GPT-3 training:</p>
     * <ol>
     *   <li>Learning rate linearly increases from 0 to maxLr over warmupSteps</li>
     *   <li>Learning rate decays following cosine curve to minLr</li>
     * </ol>
     * 
     * @param maxLr peak learning rate (e.g., 6e-4)
     * @param minLr minimum learning rate (e.g., 6e-5, typically maxLr/10)
     * @param warmupSteps number of warmup steps (e.g., 100-1000)
     * @param totalSteps total training steps
     */
    public static LRScheduler warmupCosine(float maxLr, float minLr, 
                                           int warmupSteps, int totalSteps) {
        return new LRScheduler(ScheduleType.WARMUP_COSINE, maxLr, minLr, warmupSteps, totalSteps);
    }

    /**
     * Create warmup cosine schedule with default minLr = maxLr / 10.
     */
    public static LRScheduler warmupCosine(float maxLr, int warmupSteps, int totalSteps) {
        return warmupCosine(maxLr, maxLr / 10, warmupSteps, totalSteps);
    }

    // ========================================================================
    // Utility methods
    // ========================================================================

    /**
     * Print learning rate schedule for debugging.
     */
    public void printSchedule(int steps) {
        System.out.println("Learning Rate Schedule:");
        System.out.println("========================");
        System.out.printf("Type: %s%n", type);
        System.out.printf("Max LR: %.2e%n", maxLr);
        System.out.printf("Min LR: %.2e%n", minLr);
        System.out.printf("Warmup steps: %d%n", warmupSteps);
        System.out.printf("Total steps: %d%n", totalSteps);
        System.out.println();
        
        // Print sample learning rates
        int[] checkpoints = {0, warmupSteps / 2, warmupSteps, 
                            totalSteps / 4, totalSteps / 2, 
                            3 * totalSteps / 4, totalSteps - 1};
        for (int step : checkpoints) {
            if (step >= 0 && step < steps) {
                System.out.printf("  Step %6d: LR = %.6e%n", step, getLr(step));
            }
        }
    }

    @Override
    public String toString() {
        return String.format("LRScheduler(%s, maxLr=%.2e, minLr=%.2e, warmup=%d, total=%d)",
                            type, maxLr, minLr, warmupSteps, totalSteps);
    }
}
