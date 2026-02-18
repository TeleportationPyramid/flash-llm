package com.flashllm;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.kernel.*;

/**
 * End-to-End Training Test.
 *
 * Tests complete training loop:
 * 1. Forward pass
 * 2. Backward pass
 * 3. AdamW optimizer update
 * 4. Verify loss decreases
 * 5. Measure training speed (ms/step)
 *
 * This is the FIRST VALIDATION POINT for comparing with nanoGPT CPU!
 */
public class TrainingTest {

    public static void main(String[] args) {
        System.out.println("=== End-to-End Training Test ===\n");

        try {
            FlashBackend backend = FlashBackend.getInstance();
            CudaDevice device = backend.getDevice();

            // Small config for testing
            int B = 4;     // batch size
            int T = 64;    // sequence length
            int C = 128;   // channels
            int NH = 4;    // num heads
            int V = 256;   // vocab size
            int numSteps = 100;

            System.out.printf("Config: B=%d, T=%d, C=%d, NH=%d, V=%d%n", B, T, C, NH, V);
            System.out.printf("Training steps: %d%n%n", numSteps);

            // Test 1: Simple model training (Embedding -> LN -> Proj -> Loss)
            testSimpleModelTraining(backend, device, B, T, C, V, numSteps);

            // Test 2: Speed benchmark
            testTrainingSpeed(backend, device, B, T, C, V);

            System.out.println("\n==================================================");
            System.out.println("✓ All Training tests PASSED");

            backend.close();

        } catch (Exception e) {
            System.err.println("✗ Test failed:");
            e.printStackTrace();
            System.exit(1);
        }
    }

    // ========================================================================
    // Test 1: Simple Model Training
    // ========================================================================
    static void testSimpleModelTraining(FlashBackend backend, CudaDevice device,
                                         int B, int T, int C, int V, int numSteps) {
        System.out.println("Test 1: Simple Model Training (verify loss decreases)");

        int BT = B * T;
        long numParams = (long) V * C + (long) T * C + C + C;  // wte + wpe + lnw + lnb

        // Model parameters
        CudaTensor wte = backend.allocateF32(V * C);
        CudaTensor wpe = backend.allocateF32(T * C);
        CudaTensor lnw = backend.allocateF32(C);
        CudaTensor lnb = backend.allocateF32(C);

        // Activations
        CudaTensor encoded = backend.allocateF32(BT * C);
        CudaTensor lnOut = backend.allocateF32(BT * C);
        CudaTensor lnMean = backend.allocateF32(BT);
        CudaTensor lnRstd = backend.allocateF32(BT);
        CudaTensor logits = backend.allocateF32(BT * V);
        CudaTensor probs = backend.allocateF32(BT * V);
        CudaTensor losses = backend.allocateF32(BT);

        // Gradients
        CudaTensor dwte = backend.allocateF32(V * C);
        CudaTensor dwpe = backend.allocateF32(T * C);
        CudaTensor dlnw = backend.allocateF32(C);
        CudaTensor dlnb = backend.allocateF32(C);
        CudaTensor dlogits = backend.allocateF32(BT * V);
        CudaTensor dlnOut = backend.allocateF32(BT * C);
        CudaTensor dencoded = backend.allocateF32(BT * C);

        // AdamW optimizer state
        CudaTensor mWte = backend.allocateF32(V * C);
        CudaTensor vWte = backend.allocateF32(V * C);
        CudaTensor mWpe = backend.allocateF32(T * C);
        CudaTensor vWpe = backend.allocateF32(T * C);
        CudaTensor mLnw = backend.allocateF32(C);
        CudaTensor vLnw = backend.allocateF32(C);
        CudaTensor mLnb = backend.allocateF32(C);
        CudaTensor vLnb = backend.allocateF32(C);

        // Initialize parameters
        initRandom(device, wte, 0.02f);
        initRandom(device, wpe, 0.02f);
        initOnes(device, lnw);
        backend.zeroFill(lnb);

        // Initialize optimizer state
        backend.zeroFill(mWte); backend.zeroFill(vWte);
        backend.zeroFill(mWpe); backend.zeroFill(vWpe);
        backend.zeroFill(mLnw); backend.zeroFill(vLnw);
        backend.zeroFill(mLnb); backend.zeroFill(vLnb);

        // Training data (random tokens)
        int[] tokens = new int[BT];
        int[] targets = new int[BT];
        java.util.Random rng = new java.util.Random(42);
        for (int i = 0; i < BT; i++) {
            tokens[i] = rng.nextInt(V);
            targets[i] = rng.nextInt(V);
        }

        // Hyperparameters
        float lr = 1e-3f;
        float weightDecay = 0.01f;

        float initialLoss = 0;
        float finalLoss = 0;

        System.out.printf("  Parameters: %,d%n", numParams);
        System.out.println("  Training...");

        for (int step = 0; step < numSteps; step++) {
            // Zero gradients
            backend.zeroFill(dwte);
            backend.zeroFill(dwpe);
            backend.zeroFill(dlnw);
            backend.zeroFill(dlnb);

            // ==================== FORWARD ====================
            Encoder.forward(encoded, tokens, wte, wpe, B, T, C);
            LayerNorm.forward(lnOut, lnMean, lnRstd, encoded, lnw, lnb, BT, C);
            Matmul.forward(logits, lnOut, wte, BT, V, C);
            Softmax.forward(probs, logits, B, T, V);
            Softmax.crossEntropyForward(losses, probs, targets, B, T, V);
            float loss = Softmax.meanLoss(losses, B, T);

            if (step == 0) initialLoss = loss;
            if (step == numSteps - 1) finalLoss = loss;

            // ==================== BACKWARD ====================
            Softmax.crossEntropySoftmaxBackward(dlogits, probs, targets, B, T, V);
            Matmul.backward(dlnOut, dwte, null, dlogits, lnOut, wte, BT, V, C);
            LayerNorm.backward(dencoded, dlnw, dlnb, dlnOut, encoded, lnw, lnMean, lnRstd, BT, C);
            Encoder.backward(dwte, dwpe, dencoded, tokens, B, T, C);

            // ==================== OPTIMIZER ====================
            int t = step + 1;
            AdamW.update(wte, dwte, mWte, vWte, lr, weightDecay, t, (long) V * C);
            AdamW.update(wpe, dwpe, mWpe, vWpe, lr, weightDecay, t, (long) T * C);
            AdamW.update(lnw, dlnw, mLnw, vLnw, lr, weightDecay, t, C);
            AdamW.update(lnb, dlnb, mLnb, vLnb, lr, weightDecay, t, C);

            // Log progress
            if (step % 20 == 0 || step == numSteps - 1) {
                System.out.printf("    Step %3d: loss = %.4f%n", step, loss);
            }
        }

        // Verify loss decreased
        float lossReduction = (initialLoss - finalLoss) / initialLoss * 100;
        System.out.printf("\n  Initial loss: %.4f%n", initialLoss);
        System.out.printf("  Final loss:   %.4f%n", finalLoss);
        System.out.printf("  Reduction:    %.1f%%%n", lossReduction);

        boolean lossDecreased = finalLoss < initialLoss;
        System.out.println("  Loss decreased: " + (lossDecreased ? "Yes ✓" : "No ✗"));

        assert lossDecreased : "Loss should decrease during training";

        // Cleanup
        wte.close(); wpe.close(); lnw.close(); lnb.close();
        encoded.close(); lnOut.close(); lnMean.close(); lnRstd.close();
        logits.close(); probs.close(); losses.close();
        dwte.close(); dwpe.close(); dlnw.close(); dlnb.close();
        dlogits.close(); dlnOut.close(); dencoded.close();
        mWte.close(); vWte.close(); mWpe.close(); vWpe.close();
        mLnw.close(); vLnw.close(); mLnb.close(); vLnb.close();

        System.out.println("  ✓ Simple model training works\n");
    }

    // ========================================================================
    // Test 2: Training Speed Benchmark
    // ========================================================================
    static void testTrainingSpeed(FlashBackend backend, CudaDevice device,
                                   int B, int T, int C, int V) {
        System.out.println("Test 2: Training Speed Benchmark");
        System.out.println("  ★ FIRST VALIDATION POINT: Comparing with nanoGPT CPU ★\n");

        int BT = B * T;
        int warmupSteps = 10;
        int benchmarkSteps = 50;

        // Model parameters
        CudaTensor wte = backend.allocateF32(V * C);
        CudaTensor wpe = backend.allocateF32(T * C);
        CudaTensor lnw = backend.allocateF32(C);
        CudaTensor lnb = backend.allocateF32(C);

        // Activations
        CudaTensor encoded = backend.allocateF32(BT * C);
        CudaTensor lnOut = backend.allocateF32(BT * C);
        CudaTensor lnMean = backend.allocateF32(BT);
        CudaTensor lnRstd = backend.allocateF32(BT);
        CudaTensor logits = backend.allocateF32(BT * V);
        CudaTensor probs = backend.allocateF32(BT * V);
        CudaTensor losses = backend.allocateF32(BT);

        // Gradients
        CudaTensor dwte = backend.allocateF32(V * C);
        CudaTensor dwpe = backend.allocateF32(T * C);
        CudaTensor dlnw = backend.allocateF32(C);
        CudaTensor dlnb = backend.allocateF32(C);
        CudaTensor dlogits = backend.allocateF32(BT * V);
        CudaTensor dlnOut = backend.allocateF32(BT * C);
        CudaTensor dencoded = backend.allocateF32(BT * C);

        // Optimizer state
        CudaTensor mWte = backend.allocateF32(V * C);
        CudaTensor vWte = backend.allocateF32(V * C);
        CudaTensor mWpe = backend.allocateF32(T * C);
        CudaTensor vWpe = backend.allocateF32(T * C);
        CudaTensor mLnw = backend.allocateF32(C);
        CudaTensor vLnw = backend.allocateF32(C);
        CudaTensor mLnb = backend.allocateF32(C);
        CudaTensor vLnb = backend.allocateF32(C);

        // Initialize
        initRandom(device, wte, 0.02f);
        initRandom(device, wpe, 0.02f);
        initOnes(device, lnw);
        backend.zeroFill(lnb);
        backend.zeroFill(mWte); backend.zeroFill(vWte);
        backend.zeroFill(mWpe); backend.zeroFill(vWpe);
        backend.zeroFill(mLnw); backend.zeroFill(vLnw);
        backend.zeroFill(mLnb); backend.zeroFill(vLnb);

        int[] tokens = new int[BT];
        int[] targets = new int[BT];
        java.util.Random rng = new java.util.Random(42);
        for (int i = 0; i < BT; i++) {
            tokens[i] = rng.nextInt(V);
            targets[i] = rng.nextInt(V);
        }

        float lr = 1e-3f;
        float weightDecay = 0.01f;

        // Warmup
        System.out.printf("  Warmup: %d steps...%n", warmupSteps);
        for (int step = 0; step < warmupSteps; step++) {
            trainStep(backend, device, B, T, C, V, BT,
                    tokens, targets, wte, wpe, lnw, lnb,
                    encoded, lnOut, lnMean, lnRstd, logits, probs, losses,
                    dwte, dwpe, dlnw, dlnb, dlogits, dlnOut, dencoded,
                    mWte, vWte, mWpe, vWpe, mLnw, vLnw, mLnb, vLnb,
                    lr, weightDecay, step + 1);
        }

        // Benchmark
        System.out.printf("  Benchmark: %d steps...%n", benchmarkSteps);

        // GPU sync before timing
        device.synchronize();

        long startTime = System.nanoTime();

        for (int step = 0; step < benchmarkSteps; step++) {
            trainStep(backend, device, B, T, C, V, BT,
                    tokens, targets, wte, wpe, lnw, lnb,
                    encoded, lnOut, lnMean, lnRstd, logits, probs, losses,
                    dwte, dwpe, dlnw, dlnb, dlogits, dlnOut, dencoded,
                    mWte, vWte, mWpe, vWpe, mLnw, vLnw, mLnb, vLnb,
                    lr, weightDecay, warmupSteps + step + 1);
        }

        // GPU sync after timing
        device.synchronize();

        long endTime = System.nanoTime();
        double totalMs = (endTime - startTime) / 1_000_000.0;
        double msPerStep = totalMs / benchmarkSteps;
        double stepsPerSec = 1000.0 / msPerStep;
        double tokensPerSec = stepsPerSec * BT;

        System.out.println("\n  ========================================");
        System.out.println("  SPEED RESULTS:");
        System.out.println("  ========================================");
        System.out.printf("  Total time:       %.2f ms%n", totalMs);
        System.out.printf("  Time per step:    %.2f ms%n", msPerStep);
        System.out.printf("  Steps per second: %.1f%n", stepsPerSec);
        System.out.printf("  Tokens per second: %.0f (B=%d, T=%d)%n", tokensPerSec, B, T);
        System.out.println("  ========================================");

        // Estimate 10K steps time
        double estimatedTimeFor10K = msPerStep * 10000 / 1000 / 60;  // in minutes
        System.out.printf("\n  Estimated time for 10K steps: %.1f minutes%n", estimatedTimeFor10K);

        // Cleanup
        wte.close(); wpe.close(); lnw.close(); lnb.close();
        encoded.close(); lnOut.close(); lnMean.close(); lnRstd.close();
        logits.close(); probs.close(); losses.close();
        dwte.close(); dwpe.close(); dlnw.close(); dlnb.close();
        dlogits.close(); dlnOut.close(); dencoded.close();
        mWte.close(); vWte.close(); mWpe.close(); vWpe.close();
        mLnw.close(); vLnw.close(); mLnb.close(); vLnb.close();

        System.out.println("\n  ✓ Speed benchmark complete");
    }

    static void trainStep(FlashBackend backend, CudaDevice device,
                          int B, int T, int C, int V, int BT,
                          int[] tokens, int[] targets,
                          CudaTensor wte, CudaTensor wpe, CudaTensor lnw, CudaTensor lnb,
                          CudaTensor encoded, CudaTensor lnOut, CudaTensor lnMean, CudaTensor lnRstd,
                          CudaTensor logits, CudaTensor probs, CudaTensor losses,
                          CudaTensor dwte, CudaTensor dwpe, CudaTensor dlnw, CudaTensor dlnb,
                          CudaTensor dlogits, CudaTensor dlnOut, CudaTensor dencoded,
                          CudaTensor mWte, CudaTensor vWte,
                          CudaTensor mWpe, CudaTensor vWpe,
                          CudaTensor mLnw, CudaTensor vLnw,
                          CudaTensor mLnb, CudaTensor vLnb,
                          float lr, float weightDecay, int t) {

        // Zero gradients
        backend.zeroFill(dwte);
        backend.zeroFill(dwpe);
        backend.zeroFill(dlnw);
        backend.zeroFill(dlnb);

        // Forward
        Encoder.forward(encoded, tokens, wte, wpe, B, T, C);
        LayerNorm.forward(lnOut, lnMean, lnRstd, encoded, lnw, lnb, BT, C);
        Matmul.forward(logits, lnOut, wte, BT, V, C);
        Softmax.forward(probs, logits, B, T, V);
        Softmax.crossEntropyForward(losses, probs, targets, B, T, V);

        // Backward
        Softmax.crossEntropySoftmaxBackward(dlogits, probs, targets, B, T, V);
        Matmul.backward(dlnOut, dwte, null, dlogits, lnOut, wte, BT, V, C);
        LayerNorm.backward(dencoded, dlnw, dlnb, dlnOut, encoded, lnw, lnMean, lnRstd, BT, C);
        Encoder.backward(dwte, dwpe, dencoded, tokens, B, T, C);

        // Optimizer
        AdamW.update(wte, dwte, mWte, vWte, lr, weightDecay, t, (long) V * C);
        AdamW.update(wpe, dwpe, mWpe, vWpe, lr, weightDecay, t, (long) T * C);
        AdamW.update(lnw, dlnw, mLnw, vLnw, lr, weightDecay, t, C);
        AdamW.update(lnb, dlnb, mLnb, vLnb, lr, weightDecay, t, C);
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    static void initRandom(CudaDevice device, CudaTensor tensor, float scale) {
        int size = (int) tensor.getElementCount();
        float[] data = new float[size];
        java.util.Random rng = new java.util.Random(42);
        for (int i = 0; i < size; i++) {
            data[i] = (float) (rng.nextGaussian() * scale);
        }
        TensorUtils.copyFromHost(device, data, tensor);
    }

    static void initOnes(CudaDevice device, CudaTensor tensor) {
        int size = (int) tensor.getElementCount();
        float[] data = new float[size];
        java.util.Arrays.fill(data, 1.0f);
        TensorUtils.copyFromHost(device, data, tensor);
    }
}
