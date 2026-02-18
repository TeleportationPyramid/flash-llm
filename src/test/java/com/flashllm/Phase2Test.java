package com.flashllm;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.kernel.*;

import java.util.Arrays;

/**
 * Phase 2 verification tests - Kernel unit tests.
 *
 * <p>Tests all kernel wrappers with GPU execution.</p>
 *
 * <p>Run with: mvn exec:java -Dexec.mainClass="com.flashllm.Phase2Test"</p>
 */
public class Phase2Test {

    private static FlashBackend backend;

    public static void main(String[] args) {
        System.out.println("=== Flash-LLM Phase 2 Tests ===\n");

        try {
            // Initialize backend
            backend = FlashBackend.getInstance();
            System.out.println("Backend initialized: " + backend);
            System.out.println();

            boolean allPassed = true;

            allPassed &= testLayerNorm();
            allPassed &= testGelu();
            allPassed &= testMatmul();
            allPassed &= testSoftmax();
            allPassed &= testResidual();
            allPassed &= testAdamW();
            allPassed &= testEncoder();

            System.out.println("\n" + "=".repeat(50));
            if (allPassed) {
                System.out.println("✓ All Phase 2 tests PASSED");
            } else {
                System.out.println("✗ Some Phase 2 tests FAILED");
            }

        } catch (Exception e) {
            System.err.println("Error during tests: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (backend != null) {
                backend.close();
            }
        }
    }

    // ========================================================================
    // Test: LayerNorm
    // ========================================================================

    private static boolean testLayerNorm() {
        System.out.println("Test: LayerNorm");

        try {
            int N = 2, C = 4;
            float[] inp = {1, 2, 3, 4, 5, 6, 7, 8};
            float[] weight = {1, 1, 1, 1};
            float[] bias = {0, 0, 0, 0};

            try (CudaTensor inpT = backend.fromFloat(inp);
                 CudaTensor weightT = backend.fromFloat(weight);
                 CudaTensor biasT = backend.fromFloat(bias);
                 CudaTensor outT = backend.allocateF32(N * C);
                 CudaTensor meanT = backend.allocateF32(N);
                 CudaTensor rstdT = backend.allocateF32(N)) {

                LayerNorm.forward(outT, meanT, rstdT, inpT, weightT, biasT, N, C);

                float[] out = outT.toFloatArray();
                float[] mean = meanT.toFloatArray();

                // Check mean values
                boolean meanOk = Math.abs(mean[0] - 2.5f) < 0.01f &&
                        Math.abs(mean[1] - 6.5f) < 0.01f;

                // Output should be normalized (mean≈0, std≈1)
                float outMean0 = (out[0] + out[1] + out[2] + out[3]) / 4;
                boolean normalizedOk = Math.abs(outMean0) < 0.01f;

                if (meanOk && normalizedOk) {
                    System.out.println("  ✓ LayerNorm forward correct");
                    System.out.println("    mean=" + Arrays.toString(mean));
                    System.out.println("    out=" + Arrays.toString(out));
                    return true;
                } else {
                    System.out.println("  ✗ LayerNorm forward incorrect");
                    return false;
                }
            }
        } catch (Exception e) {
            System.out.println("  ✗ LayerNorm FAILED: " + e.getMessage());
            return false;
        }
    }

    // ========================================================================
    // Test: GELU
    // ========================================================================

    private static boolean testGelu() {
        System.out.println("\nTest: GELU");

        try {
            int N = 8;
            float[] inp = {-2, -1, -0.5f, 0, 0.5f, 1, 2, 3};

            try (CudaTensor inpT = backend.fromFloat(inp);
                 CudaTensor outT = backend.allocateF32(N)) {

                Gelu.forward(outT, inpT, N);

                float[] out = outT.toFloatArray();

                // GELU(0) should be 0
                boolean zeroOk = Math.abs(out[3]) < 0.01f;

                // GELU(x) > 0 for x > 0
                boolean positiveOk = out[4] > 0 && out[5] > 0 && out[6] > 0;

                // GELU(-2) < 0 (small negative)
                boolean negativeOk = out[0] < 0;

                if (zeroOk && positiveOk && negativeOk) {
                    System.out.println("  ✓ GELU forward correct");
                    System.out.println("    out=" + Arrays.toString(out));
                    return true;
                } else {
                    System.out.println("  ✗ GELU forward incorrect");
                    return false;
                }
            }
        } catch (Exception e) {
            System.out.println("  ✗ GELU FAILED: " + e.getMessage());
            return false;
        }
    }

    // ========================================================================
    // Test: Matmul
    // ========================================================================

    private static boolean testMatmul() {
        System.out.println("\nTest: Matmul");

        try {
            // C[2,2] = A[2,3] @ B[3,2]
            int N = 2, OC = 2, IC = 3;

            float[] a = {1, 2, 3, 4, 5, 6};  // [2, 3]
            float[] b = {1, 2, 3, 4, 5, 6};  // [3, 2]
            float[] c = new float[4];

            try (CudaTensor aT = backend.fromFloat(a);
                 CudaTensor bT = backend.fromFloat(b);
                 CudaTensor cT = backend.fromFloat(c)) {

                Matmul.forward(cT, aT, bT, N, OC, IC);

                float[] out = cT.toFloatArray();

                // Expected:
                // C[0,0] = 1*1 + 2*3 + 3*5 = 22
                // C[0,1] = 1*2 + 2*4 + 3*6 = 28
                // C[1,0] = 4*1 + 5*3 + 6*5 = 49
                // C[1,1] = 4*2 + 5*4 + 6*6 = 64

                boolean correct = Math.abs(out[0] - 22) < 0.01f &&
                        Math.abs(out[1] - 28) < 0.01f &&
                        Math.abs(out[2] - 49) < 0.01f &&
                        Math.abs(out[3] - 64) < 0.01f;

                if (correct) {
                    System.out.println("  ✓ Matmul forward correct");
                    System.out.println("    out=" + Arrays.toString(out));
                    return true;
                } else {
                    System.out.println("  ✗ Matmul forward incorrect");
                    System.out.println("    expected=[22, 28, 49, 64]");
                    System.out.println("    got=" + Arrays.toString(out));
                    return false;
                }
            }
        } catch (Exception e) {
            System.out.println("  ✗ Matmul FAILED: " + e.getMessage());
            return false;
        }
    }

    // ========================================================================
    // Test: Softmax
    // ========================================================================

    private static boolean testSoftmax() {
        System.out.println("\nTest: Softmax");

        try {
            int B = 1, T = 1, V = 4;
            float[] logits = {1, 2, 3, 4};

            try (CudaTensor logitsT = backend.fromFloat(logits);
                 CudaTensor probsT = backend.allocateF32(B * T * V)) {

                Softmax.forward(probsT, logitsT, B, T, V);

                float[] probs = probsT.toFloatArray();

                // Sum should be 1
                float sum = 0;
                for (float p : probs) sum += p;

                // Probs should be increasing (since logits are increasing)
                boolean increasing = probs[0] < probs[1] && probs[1] < probs[2] && probs[2] < probs[3];

                if (Math.abs(sum - 1.0f) < 0.01f && increasing) {
                    System.out.println("  ✓ Softmax forward correct");
                    System.out.println("    probs=" + Arrays.toString(probs) + " (sum=" + sum + ")");
                    return true;
                } else {
                    System.out.println("  ✗ Softmax forward incorrect");
                    return false;
                }
            }
        } catch (Exception e) {
            System.out.println("  ✗ Softmax FAILED: " + e.getMessage());
            return false;
        }
    }

    // ========================================================================
    // Test: Residual
    // ========================================================================

    private static boolean testResidual() {
        System.out.println("\nTest: Residual");

        try {
            int N = 4;
            float[] a = {1, 2, 3, 4};
            float[] b = {10, 20, 30, 40};

            try (CudaTensor aT = backend.fromFloat(a);
                 CudaTensor bT = backend.fromFloat(b);
                 CudaTensor outT = backend.allocateF32(N)) {

                Residual.forward(outT, aT, bT, N);

                float[] out = outT.toFloatArray();

                // out = a + b = [11, 22, 33, 44]
                boolean correct = Math.abs(out[0] - 11) < 0.01f &&
                        Math.abs(out[1] - 22) < 0.01f &&
                        Math.abs(out[2] - 33) < 0.01f &&
                        Math.abs(out[3] - 44) < 0.01f;

                if (correct) {
                    System.out.println("  ✓ Residual forward correct");
                    System.out.println("    out=" + Arrays.toString(out));
                    return true;
                } else {
                    System.out.println("  ✗ Residual forward incorrect");
                    return false;
                }
            }
        } catch (Exception e) {
            System.out.println("  ✗ Residual FAILED: " + e.getMessage());
            return false;
        }
    }

    // ========================================================================
    // Test: AdamW
    // ========================================================================

    private static boolean testAdamW() {
        System.out.println("\nTest: AdamW");

        try {
            int N = 4;
            float[] params = {1.0f, 2.0f, 3.0f, 4.0f};
            float[] grads = {0.1f, 0.2f, 0.3f, 0.4f};
            float[] m = new float[N];
            float[] v = new float[N];

            float[] paramsBefore = params.clone();

            try (CudaTensor paramsT = backend.fromFloat(params);
                 CudaTensor gradsT = backend.fromFloat(grads);
                 CudaTensor mT = backend.fromFloat(m);
                 CudaTensor vT = backend.fromFloat(v)) {

                // One update step
                AdamW.update(paramsT, gradsT, mT, vT, 0.001f, 0.0f, 1, N);

                float[] paramsAfter = paramsT.toFloatArray();
                float[] mAfter = mT.toFloatArray();

                // Check that params changed (decreased for positive grads)
                boolean paramsChanged = true;
                for (int i = 0; i < N; i++) {
                    if (Math.abs(paramsAfter[i] - paramsBefore[i]) < 1e-6f) {
                        paramsChanged = false;
                        break;
                    }
                }

                // Check that m is updated (should be non-zero)
                boolean mUpdated = mAfter[0] != 0;

                if (paramsChanged && mUpdated) {
                    System.out.println("  ✓ AdamW update correct");
                    System.out.println("    params: " + Arrays.toString(paramsBefore) + " -> " + Arrays.toString(paramsAfter));
                    return true;
                } else {
                    System.out.println("  ✗ AdamW update incorrect");
                    return false;
                }
            }
        } catch (Exception e) {
            System.out.println("  ✗ AdamW FAILED: " + e.getMessage());
            return false;
        }
    }

    // ========================================================================
    // Test: Encoder
    // ========================================================================

    private static boolean testEncoder() {
        System.out.println("\nTest: Encoder");

        try {
            int B = 2, T = 3, C = 4, V = 10;

            // Token embeddings [V, C]
            float[] wte = new float[V * C];
            for (int i = 0; i < V * C; i++) {
                wte[i] = i * 0.01f;
            }

            // Position embeddings [T, C]
            float[] wpe = new float[T * C];
            for (int i = 0; i < T * C; i++) {
                wpe[i] = i * 0.001f;
            }

            // Input tokens
            int[] tokens = {0, 1, 2, 3, 4, 5};

            try (CudaTensor wteT = backend.fromFloat(wte);
                 CudaTensor wpeT = backend.fromFloat(wpe);
                 CudaTensor outT = backend.allocateF32(B * T * C)) {

                Encoder.forward(outT, tokens, wteT, wpeT, B, T, C);

                float[] out = outT.toFloatArray();

                // out[0,0,:] should be wte[0,:] + wpe[0,:]
                // wte[0,0] = 0, wpe[0,0] = 0 -> out[0] = 0
                // wte[0,1] = 0.01, wpe[0,1] = 0.001 -> out[1] = 0.011

                boolean firstTokenOk = Math.abs(out[0] - 0.0f) < 0.01f;
                boolean secondElementOk = Math.abs(out[1] - 0.011f) < 0.01f;

                // out[1,0,:] should be wte[3,:] + wpe[0,:]
                // (second batch, first position, token 3)
                // wte[3,0] = 12 * 0.01 = 0.12, wpe[0,0] = 0
                int idx = T * C; // start of second batch
                boolean secondBatchOk = Math.abs(out[idx] - 0.12f) < 0.01f;

                if (firstTokenOk && secondElementOk) {
                    System.out.println("  ✓ Encoder forward correct");
                    System.out.println("    out[0:8]=" + Arrays.toString(Arrays.copyOf(out, 8)));
                    return true;
                } else {
                    System.out.println("  ✗ Encoder forward incorrect");
                    System.out.println("    out[0]=" + out[0] + " (expected 0)");
                    System.out.println("    out[1]=" + out[1] + " (expected 0.011)");
                    return false;
                }
            }
        } catch (Exception e) {
            System.out.println("  ✗ Encoder FAILED: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
}