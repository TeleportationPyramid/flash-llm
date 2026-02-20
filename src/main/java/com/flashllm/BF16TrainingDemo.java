package com.flashllm;

import com.flashllm.config.GPT2Config;
import com.flashllm.model.BF16GPT2;

/**
 * BF16 GPT-2 Training Demo.
 *
 * Compared to FP16 mixed-precision:
 *   - No LossScaler
 *   - No overflow detection / skip step
 *   - No gradient unscaling
 *   - Much simpler training loop
 *
 * @author flash-llm
 * @since 3.0.0
 */
public class BF16TrainingDemo {

    public static void main(String[] args) throws Exception {
        // GPT-2 124M config
        GPT2Config config = new GPT2Config();
        config.vocabSize = 50257;
        config.channels = 768;
        config.numLayers = 12;
        config.numHeads = 12;
        config.maxSeqLen = 1024;

        int B = 4;
        int T = 64;
        int numSteps = 20;

        // Hyperparameters
        float lr = 3e-4f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float weightDecay = 0.01f;

        String weightsPath = "src/main/resources/gpt2/gpt2_124M.bin";

        System.out.println("========================================");
        System.out.println("BF16 GPT-2 Training Demo");
        System.out.println("========================================");
        System.out.println("  No LossScaler needed (BF16 range = FP32)");
        System.out.println("  No overflow detection needed");
        System.out.println();

        try (BF16GPT2 model = new BF16GPT2(config, B, T)) {
            model.loadWeights(weightsPath);

            // Create dummy data (same as FP16 demo for comparison)
            int[] tokens = new int[B * T];
            int[] targets = new int[B * T];
            for (int i = 0; i < B * T; i++) {
                tokens[i] = i % config.vocabSize;
                targets[i] = (i + 1) % config.vocabSize;
            }

            System.out.println("========================================");
            System.out.println("BF16 Training:");
            System.out.println("========================================\n");

            long totalTime = 0;

            for (int step = 0; step < numSteps; step++) {
                long stepStart = System.currentTimeMillis();

                // Forward
                float loss = model.forward(tokens, targets);

                // Zero gradients
                model.zeroGradients();

                // Backward (no loss scaling!)
                model.backward(tokens, targets);

                // Update weights (no overflow check, no unscale!)
                model.updateWeights(lr, beta1, beta2, eps, weightDecay, step + 1);

                long stepTime = System.currentTimeMillis() - stepStart;
                totalTime += stepTime;

                if (step % 5 == 0 || step == numSteps - 1) {
                    System.out.printf("  Step %3d: loss=%.4f, time=%dms%n",
                            step, loss, stepTime);
                }
            }

            // Summary
            System.out.println("\n========================================");
            System.out.println("Training Summary:");
            System.out.println("========================================");
            System.out.printf("  Total steps: %d%n", numSteps);
            System.out.printf("  Total time: %d ms%n", totalTime);
            System.out.printf("  Avg time/step: %d ms%n", totalTime / numSteps);
            System.out.printf("  Final loss: %f%n", model.getMeanLoss());
            System.out.println("  Overflow skips: 0 (BF16 never overflows)");
        }

        System.out.println("✓ BF16 Training Demo Complete!");
    }
}
