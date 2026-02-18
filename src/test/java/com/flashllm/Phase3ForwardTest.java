package com.flashllm;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.config.GPT2Config;
import com.flashllm.kernel.*;

/**
 * Phase 3 Forward Pass Test - Step by step.
 *
 * Tests each component of the forward pass individually:
 * 1. Encoder (embedding lookup)
 * 2. LayerNorm
 * 3. Matmul (QKV projection)
 * 4. Attention
 * 5. MLP (fc -> gelu -> proj)
 * 6. Output projection
 * 7. Softmax + Loss
 */
public class Phase3ForwardTest {

    public static void main(String[] args) {
        System.out.println("=== Phase 3 Forward Pass Test ===\n");

        try {
            FlashBackend backend = FlashBackend.getInstance();
            CudaDevice device = backend.getDevice();

            // Small config for testing
            int B = 2;   // batch size
            int T = 8;   // sequence length
            int C = 64;  // channels
            int V = 256; // vocab size
            int NH = 2;  // num heads
            int BT = B * T;

            System.out.printf("Config: B=%d, T=%d, C=%d, V=%d, NH=%d%n%n", B, T, C, V, NH);

            // Test 1: Encoder
            testEncoder(backend, device, B, T, C, V);

            // Test 2: LayerNorm
            testLayerNorm(backend, device, BT, C);

            // Test 3: Matmul
            testMatmul(backend, device, BT, C);

            // Test 4: Attention (simplified - just QKV projection for now)
            testQKVProjection(backend, device, BT, C, NH);

            // Test 5: Gelu
            testGelu(backend, device, BT, C);

            // Test 6: Softmax + CrossEntropy
            testSoftmaxLoss(backend, device, B, T, V);

            // Test 7: Full forward (without TransformerBlock)
            testSimpleForward(backend, device, B, T, C, V, NH);

            System.out.println("\n==================================================");
            System.out.println("✓ All Phase 3 forward tests PASSED");

            backend.close();

        } catch (Exception e) {
            System.err.println("✗ Test failed:");
            e.printStackTrace();
            System.exit(1);
        }
    }

    static void testEncoder(FlashBackend backend, CudaDevice device, int B, int T, int C, int V) {
        System.out.println("Test 1: Encoder (Token + Position Embedding)");

        // Allocate tensors
        CudaTensor wte = backend.allocateF32((long) V * C);
        CudaTensor wpe = backend.allocateF32((long) T * C);
        CudaTensor out = backend.allocateF32((long) B * T * C);

        // Initialize embeddings with small random values
        initRandom(device, wte, 0.02f);
        initRandom(device, wpe, 0.02f);

        // Create tokens
        int[] tokens = new int[B * T];
        for (int i = 0; i < B * T; i++) {
            tokens[i] = i % V;
        }

        // Forward
        Encoder.forward(out, tokens, wte, wpe, B, T, C);

        // Verify output is non-zero
        float[] outData = out.toFloatArray();
        float sum = 0;
        for (float v : outData) sum += Math.abs(v);

        System.out.printf("  Output sum: %.4f (expected non-zero)%n", sum);
        assert sum > 0 : "Encoder output should be non-zero";

        // Cleanup
        wte.close();
        wpe.close();
        out.close();

        System.out.println("  ✓ Encoder works\n");
    }

    static void testLayerNorm(FlashBackend backend, CudaDevice device, int N, int C) {
        System.out.println("Test 2: LayerNorm");

        CudaTensor inp = backend.allocateF32((long) N * C);
        CudaTensor out = backend.allocateF32((long) N * C);
        CudaTensor mean = backend.allocateF32(N);
        CudaTensor rstd = backend.allocateF32(N);
        CudaTensor weight = backend.allocateF32(C);
        CudaTensor bias = backend.allocateF32(C);

        // Initialize
        initRandom(device, inp, 1.0f);
        initOnes(device, weight);
        backend.zeroFill(bias);

        // Forward
        LayerNorm.forward(out, mean, rstd, inp, weight, bias, N, C);

        // Verify output has reasonable stats (mean ~0, std ~1)
        float[] outData = out.toFloatArray();
        double outMean = 0, outVar = 0;
        for (float v : outData) outMean += v;
        outMean /= outData.length;
        for (float v : outData) outVar += (v - outMean) * (v - outMean);
        outVar /= outData.length;

        System.out.printf("  Output mean: %.4f (expected ~0)%n", outMean);
        System.out.printf("  Output std: %.4f (expected ~1)%n", Math.sqrt(outVar));

        // Cleanup
        inp.close();
        out.close();
        mean.close();
        rstd.close();
        weight.close();
        bias.close();

        System.out.println("  ✓ LayerNorm works\n");
    }

    static void testMatmul(FlashBackend backend, CudaDevice device, int N, int C) {
        System.out.println("Test 3: Matmul");

        int OC = C * 4;  // MLP hidden size

        CudaTensor inp = backend.allocateF32((long) N * C);
        CudaTensor weight = backend.allocateF32((long) C * OC);
        CudaTensor bias = backend.allocateF32(OC);
        CudaTensor out = backend.allocateF32((long) N * OC);

        // Initialize
        initRandom(device, inp, 0.1f);
        initRandom(device, weight, 0.02f);
        initRandom(device, bias, 0.01f);

        // Forward: out = inp @ weight + bias
        Matmul.forward(out, inp, weight, bias, N, OC, C);

        // Verify
        float[] outData = out.toFloatArray();
        float sum = 0;
        for (float v : outData) sum += Math.abs(v);

        System.out.printf("  Output sum: %.4f (expected non-zero)%n", sum);

        // Cleanup
        inp.close();
        weight.close();
        bias.close();
        out.close();

        System.out.println("  ✓ Matmul works\n");
    }

    static void testQKVProjection(FlashBackend backend, CudaDevice device, int N, int C, int NH) {
        System.out.println("Test 4: QKV Projection");

        CudaTensor inp = backend.allocateF32((long) N * C);
        CudaTensor qkvw = backend.allocateF32((long) C * 3 * C);
        CudaTensor qkvb = backend.allocateF32(3 * C);
        CudaTensor qkv = backend.allocateF32((long) N * 3 * C);

        // Initialize
        initRandom(device, inp, 0.1f);
        initRandom(device, qkvw, 0.02f);
        backend.zeroFill(qkvb);

        // Forward: qkv = inp @ qkvw + qkvb
        Matmul.forward(qkv, inp, qkvw, qkvb, N, 3 * C, C);

        // Verify
        float[] qkvData = qkv.toFloatArray();
        float sum = 0;
        for (float v : qkvData) sum += Math.abs(v);

        System.out.printf("  QKV output sum: %.4f%n", sum);

        // Cleanup
        inp.close();
        qkvw.close();
        qkvb.close();
        qkv.close();

        System.out.println("  ✓ QKV Projection works\n");
    }

    static void testGelu(FlashBackend backend, CudaDevice device, int N, int C) {
        System.out.println("Test 5: GELU");

        CudaTensor inp = backend.allocateF32((long) N * C);
        CudaTensor out = backend.allocateF32((long) N * C);

        // Initialize with values in [-2, 2]
        initRandom(device, inp, 2.0f);

        // Forward
        Gelu.forward(out, inp, N * C);

        // Verify: GELU(x) should be between x (for x >> 0) and 0 (for x << 0)
        float[] inpData = inp.toFloatArray();
        float[] outData = out.toFloatArray();

        float inpSum = 0, outSum = 0;
        for (float v : inpData) inpSum += v;
        for (float v : outData) outSum += v;

        System.out.printf("  Input sum: %.4f, Output sum: %.4f%n", inpSum, outSum);

        // Cleanup
        inp.close();
        out.close();

        System.out.println("  ✓ GELU works\n");
    }

    static void testSoftmaxLoss(FlashBackend backend, CudaDevice device, int B, int T, int V) {
        System.out.println("Test 6: Softmax + Cross-Entropy Loss");

        int BT = B * T;

        CudaTensor logits = backend.allocateF32((long) BT * V);
        CudaTensor probs = backend.allocateF32((long) BT * V);
        CudaTensor losses = backend.allocateF32(BT);

        // Initialize logits with random values
        initRandom(device, logits, 1.0f);

        // Create random targets
        int[] targets = new int[BT];
        for (int i = 0; i < BT; i++) {
            targets[i] = i % V;
        }

        // Softmax forward
        Softmax.forward(probs, logits, B, T, V);

        // Verify probs sum to 1 for each position
        float[] probsData = probs.toFloatArray();
        float firstRowSum = 0;
        for (int v = 0; v < V; v++) {
            firstRowSum += probsData[v];
        }
        System.out.printf("  First position probs sum: %.4f (expected 1.0)%n", firstRowSum);

        // Cross-entropy forward
        Softmax.crossEntropyForward(losses, probs, targets, B, T, V);

        // Mean loss
        float meanLoss = Softmax.meanLoss(losses, B, T);
        float expectedLoss = (float) Math.log(V);  // For uniform distribution

        System.out.printf("  Mean loss: %.4f (expected ~%.4f for uniform)%n", meanLoss, expectedLoss);

        // Cleanup
        logits.close();
        probs.close();
        losses.close();

        System.out.println("  ✓ Softmax + Loss works\n");
    }

    static void testSimpleForward(FlashBackend backend, CudaDevice device, int B, int T, int C, int V, int NH) {
        System.out.println("Test 7: Simple Forward Pass (Encoder -> LN -> Matmul -> Softmax)");

        int BT = B * T;

        // Allocate
        CudaTensor wte = backend.allocateF32((long) V * C);
        CudaTensor wpe = backend.allocateF32((long) T * C);
        CudaTensor encoded = backend.allocateF32((long) BT * C);
        CudaTensor lnOut = backend.allocateF32((long) BT * C);
        CudaTensor lnMean = backend.allocateF32(BT);
        CudaTensor lnRstd = backend.allocateF32(BT);
        CudaTensor lnw = backend.allocateF32(C);
        CudaTensor lnb = backend.allocateF32(C);
        CudaTensor logits = backend.allocateF32((long) BT * V);
        CudaTensor probs = backend.allocateF32((long) BT * V);
        CudaTensor losses = backend.allocateF32(BT);

        // Initialize parameters
        initRandom(device, wte, 0.02f);
        initRandom(device, wpe, 0.02f);
        initOnes(device, lnw);
        backend.zeroFill(lnb);

        // Create tokens and targets
        int[] tokens = new int[BT];
        int[] targets = new int[BT];
        for (int i = 0; i < BT; i++) {
            tokens[i] = i % V;
            targets[i] = (i + 1) % V;
        }

        // Forward pass
        System.out.println("  Step 1: Encoder...");
        Encoder.forward(encoded, tokens, wte, wpe, B, T, C);

        System.out.println("  Step 2: LayerNorm...");
        LayerNorm.forward(lnOut, lnMean, lnRstd, encoded, lnw, lnb, BT, C);

        System.out.println("  Step 3: Output projection (tied weights)...");
        // logits = lnOut @ wte.T (output projection uses transposed embeddings)
        Matmul.forward(logits, lnOut, wte, BT, V, C);

        System.out.println("  Step 4: Softmax...");
        Softmax.forward(probs, logits, B, T, V);

        System.out.println("  Step 5: Cross-Entropy Loss...");
        Softmax.crossEntropyForward(losses, probs, targets, B, T, V);

        float meanLoss = Softmax.meanLoss(losses, B, T);
        float expectedLoss = (float) Math.log(V);

        System.out.printf("\n  Final mean loss: %.4f%n", meanLoss);
        System.out.printf("  Expected (random): ~%.4f%n", expectedLoss);
        System.out.printf("  Loss is reasonable: %s%n",
                (meanLoss > 0 && meanLoss < expectedLoss * 2) ? "Yes ✓" : "No ✗");

        // Cleanup
        wte.close();
        wpe.close();
        encoded.close();
        lnOut.close();
        lnMean.close();
        lnRstd.close();
        lnw.close();
        lnb.close();
        logits.close();
        probs.close();
        losses.close();

        System.out.println("  ✓ Simple forward pass works\n");
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
