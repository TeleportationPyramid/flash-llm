package com.flashllm;

import com.flashllm.config.GPT2Config;
import com.flashllm.model.MixedPrecisionFP32GPT2;
import io.github.teleportationpyramid.flash.LossScaler;

import java.io.File;
import java.util.Random;

/**
 * Demo for FP16 Mixed-Precision Training.
 *
 * <p>Demonstrates training GPT-2 124M with FP16 forward/backward
 * and FP32 master weights for optimizer updates.</p>
 *
 * <h2>Expected Benefits:</h2>
 * <ul>
 *   <li>~2x memory reduction for activations</li>
 *   <li>~2x speedup on Tensor Core GPUs</li>
 *   <li>Same convergence as FP32</li>
 * </ul>
 *
 * @author flash-llm
 * @since 2.3.0
 */
public class MixedPrecisionTrainingFP32Demo {

    public static void main(String[] args) {
        System.out.println("================================================================");
        System.out.println("     GPT-2 124M FP16 Mixed-Precision Training Demo");
        System.out.println("     Phase 2.3: Quantized Training");
        System.out.println("================================================================\n");

        // Configuration
        int B = 4;      // Batch size
        int T = 64;     // Sequence length
        int numSteps = 20;
        
        // Hyperparameters
        float lr = 3e-4f;
        float beta1 = 0.9f;
        float beta2 = 0.999f;
        float eps = 1e-8f;
        float weightDecay = 0.01f;

        try {
            // Find weights file
            String weightsPath = findFile("gpt2_124M.bin", new String[]{
                "src/main/resources/gpt2/gpt2_124M.bin",
                "gpt2_124M.bin",
                "models/gpt2_124M.bin"
            });

            // Load config
            GPT2Config config = GPT2Config.gpt2_124M();

            System.out.println("\n========================================");
            System.out.println("Model Configuration:");
            System.out.println("========================================");
            System.out.println("  Layers: " + config.numLayers);
            System.out.println("  Channels: " + config.channels);
            System.out.println("  Heads: " + config.numHeads);
            System.out.println("  Vocab: " + config.vocabSize);
            System.out.println("  Batch: " + B);
            System.out.println("  SeqLen: " + T);
            System.out.println("  Precision: FP16 (mixed)");

            // Create model
            System.out.println("\n========================================");
            System.out.println("Creating Mixed-Precision Model:");
            System.out.println("========================================");
            
            long startCreate = System.currentTimeMillis();
            MixedPrecisionFP32GPT2 model = new MixedPrecisionFP32GPT2(config, B, T);
            long createTime = System.currentTimeMillis() - startCreate;
            System.out.println("  Model created in " + createTime + " ms");

            // Load weights
            System.out.println("\nLoading weights...");
            long startLoad = System.currentTimeMillis();
            model.loadWeights(weightsPath);
            long loadTime = System.currentTimeMillis() - startLoad;
            System.out.println("  Weights loaded in " + loadTime + " ms");

            // Create loss scaler
            LossScaler scaler = LossScaler.forFp16Training();
            System.out.println("\nLoss Scaler: " + scaler);

            // Generate dummy data
            Random rng = new Random(42);
            int[] tokens = new int[B * T];
            int[] targets = new int[B * T];
            for (int i = 0; i < B * T; i++) {
                tokens[i] = rng.nextInt(config.vocabSize);
                targets[i] = rng.nextInt(config.vocabSize);
            }

            // Training loop
            System.out.println("\n========================================");
            System.out.println("FP16 Mixed-Precision Training:");
            System.out.println("========================================\n");

            long totalTime = 0;
            int skippedSteps = 0;

            for (int step = 0; step < numSteps; step++) {
                long stepStart = System.currentTimeMillis();

                // Forward pass (FP16)
                float loss = model.forward(tokens, targets);
                if (Float.isNaN(loss)) {
                    System.out.println("Step " + step + ": forward loss is NaN! scale=" + scaler.getScale());
                }

                // Zero gradients before backward
                model.zeroGradients();

                // Backward pass with scaled loss (FP16)
                model.backward(tokens, targets, scaler.getScale());
                System.out.println("Step " + step + ": hasOverflow=" + model.hasGradientOverflow()
                        + ", backwardOverflow=" + model.isBackwardOverflow());
                // Check for gradient overflow
                if (!model.hasGradientOverflow()) {
                    model.unscaleGradients(1.0f / scaler.getScale());
                    model.updateMasterWeights(lr, beta1, beta2, eps, weightDecay, step + 1);
                    scaler.update();
                    model.syncWeightsToFp16();
                } else {
                    scaler.decreaseScale();
                    model.zeroGradients();
                    System.out.println("  Step " + step + ": OVERFLOW - skipping (scale: " + scaler.getScale() + ")");
                    continue;
                }

                long stepTime = System.currentTimeMillis() - stepStart;
                totalTime += stepTime;

                if (step % 5 == 0 || step == numSteps - 1) {
                    System.out.printf("  Step %3d: loss=%.4f, scale=%.0f, time=%dms%n",
                            step, loss, scaler.getScale(), stepTime);
                }
            }

            // Summary
            System.out.println("\n========================================");
            System.out.println("Training Summary:");
            System.out.println("========================================");
            System.out.println("  Total steps: " + numSteps);
            System.out.println("  Skipped steps (overflow): " + skippedSteps);
            System.out.println("  Total time: " + totalTime + " ms");
            System.out.println("  Avg time/step: " + (totalTime / numSteps) + " ms");
            System.out.println("  Final loss: " + model.getMeanLoss());
            System.out.println("  Final scale: " + scaler.getScale());
            System.out.println("  Total overflows: " + scaler.getTotalOverflows());

            // Cleanup
            model.close();

            System.out.println("\n✓ FP16 Mixed-Precision Training Demo Complete!");

        } catch (Exception e) {
            System.err.println("Training failed:");
            e.printStackTrace();
            System.exit(1);
        }
    }

    static String findFile(String name, String[] paths) {
        for (String path : paths) {
            File f = new File(path);
            if (f.exists()) {
                System.out.println("Found " + name + " at: " + path);
                return path;
            }
        }
        throw new RuntimeException("Could not find " + name + "!");
    }

}
