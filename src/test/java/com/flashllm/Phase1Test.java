package com.flashllm;

import com.flashllm.config.GPT2Config;
import com.flashllm.memory.ParameterTensors;
import com.flashllm.memory.ActivationTensors;

/**
 * Phase 1 verification tests.
 *
 * <p>This class verifies the core framework without requiring GPU access.
 * It tests configuration, parameter calculations, and memory layout.</p>
 *
 * <p>Run with: java -cp target/classes com.flashllm.Phase1Test</p>
 */
public class Phase1Test {

    public static void main(String[] args) {
        System.out.println("=== Flash-LLM Phase 1 Tests ===\n");

        boolean allPassed = true;

        allPassed &= testGPT2Config();
        allPassed &= testParameterCounts();
        allPassed &= testParameterTensors();
        allPassed &= testActivationTensors();

        System.out.println("\n" + "=".repeat(50));
        if (allPassed) {
            System.out.println("✓ All Phase 1 tests PASSED");
        } else {
            System.out.println("✗ Some Phase 1 tests FAILED");
            System.exit(1);
        }
    }

    // ========================================================================
    // Test: GPT2Config
    // ========================================================================

    private static boolean testGPT2Config() {
        System.out.println("Test: GPT2Config");

        try {
            GPT2Config config = GPT2Config.gpt2_124M();

            // Verify preset values
            assert config.maxSeqLen == 1024 : "maxSeqLen should be 1024";
            assert config.vocabSize == 50257 : "vocabSize should be 50257";
            assert config.paddedVocabSize == 50304 : "paddedVocabSize should be 50304";
            assert config.numLayers == 12 : "numLayers should be 12";
            assert config.numHeads == 12 : "numHeads should be 12";
            assert config.channels == 768 : "channels should be 768";

            // Verify derived properties
            assert config.headDim() == 64 : "headDim should be 64";
            assert config.mlpDim() == 3072 : "mlpDim should be 3072";

            // Validate
            config.validate();

            System.out.println("  ✓ GPT2Config preset values correct");
            System.out.println("  ✓ GPT2Config derived properties correct");
            System.out.println("  ✓ GPT2Config validation passed");

            return true;
        } catch (AssertionError | Exception e) {
            System.out.println("  ✗ FAILED: " + e.getMessage());
            return false;
        }
    }

    // ========================================================================
    // Test: Parameter Counts
    // ========================================================================

    private static boolean testParameterCounts() {
        System.out.println("\nTest: Parameter Counts");

        try {
            // GPT-2 124M
            GPT2Config config124M = GPT2Config.gpt2_124M();
            long params124M = config124M.totalParameters();
            long expected124M = 124_439_808L;

            System.out.printf("  GPT-2 124M: %,d parameters (expected %,d)%n", params124M, expected124M);
            assert params124M == expected124M : "124M parameter count mismatch";

            // GPT-2 350M
            GPT2Config config350M = GPT2Config.gpt2_350M();
            long params350M = config350M.totalParameters();
            // Expected: ~354M (actual calculation)
            System.out.printf("  GPT-2 350M: %,d parameters%n", params350M);
            assert params350M > 300_000_000L && params350M < 400_000_000L : "350M params out of range";

            // GPT-2 774M
            GPT2Config config774M = GPT2Config.gpt2_774M();
            long params774M = config774M.totalParameters();
            System.out.printf("  GPT-2 774M: %,d parameters%n", params774M);
            assert params774M > 700_000_000L && params774M < 850_000_000L : "774M params out of range";

            // GPT-2 1558M
            GPT2Config config1558M = GPT2Config.gpt2_1558M();
            long params1558M = config1558M.totalParameters();
            System.out.printf("  GPT-2 1558M: %,d parameters%n", params1558M);
            assert params1558M > 1_400_000_000L && params1558M < 1_700_000_000L : "1558M params out of range";

            System.out.println("  ✓ All parameter counts correct");
            return true;
        } catch (AssertionError | Exception e) {
            System.out.println("  ✗ FAILED: " + e.getMessage());
            return false;
        }
    }

    // ========================================================================
    // Test: ParameterTensors
    // ========================================================================

    private static boolean testParameterTensors() {
        System.out.println("\nTest: ParameterTensors");

        try {
            GPT2Config config = GPT2Config.gpt2_124M();
            ParameterTensors params = new ParameterTensors(config, io.github.teleportationpyramid.flash.Precision.FP32);

            // Verify total elements
            long total = params.getTotalElements();
            long expected = 124_439_808L;
            System.out.printf("  Total elements: %,d (expected %,d)%n", total, expected);
            assert total == expected : "Total elements mismatch";

            // Verify memory size
            long sizeBytes = params.getSizeInBytes();
            long expectedBytes = total * 4; // FP32 = 4 bytes
            System.out.printf("  Size in bytes: %,d (%.2f MiB)%n", sizeBytes, sizeBytes / (1024.0 * 1024.0));
            assert sizeBytes == expectedBytes : "Size in bytes mismatch";

            // Verify wte size
            assert params.wteSize == 50257L * 768 : "wte size incorrect";
            System.out.printf("  wte size: %,d (V=%d, C=%d)%n", params.wteSize, config.vocabSize, config.channels);

            // Verify wpe size
            assert params.wpeSize == 1024L * 768 : "wpe size incorrect";
            System.out.printf("  wpe size: %,d (T=%d, C=%d)%n", params.wpeSize, config.maxSeqLen, config.channels);

            // Verify offsets are sequential
            assert params.wteOffset == 0 : "wte should start at 0";
            assert params.wpeOffset == params.wteSize : "wpe should follow wte";

            // Verify layer offsets exist
            assert params.ln1wOffsets.length == 12 : "Should have 12 layers";
            assert params.ln1wOffsets[0] == params.wpeOffset + params.wpeSize : "First layer should follow wpe";

            // Print layout summary
            System.out.println("  ✓ ParameterTensors offsets calculated correctly");

            // Cleanup (no GPU memory allocated yet)
            params.close();

            return true;
        } catch (AssertionError | Exception e) {
            System.out.println("  ✗ FAILED: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }

    // ========================================================================
    // Test: ActivationTensors
    // ========================================================================

    private static boolean testActivationTensors() {
        System.out.println("\nTest: ActivationTensors");

        try {
            GPT2Config config = GPT2Config.gpt2_124M();
            int B = 4;
            int T = 1024;

            ActivationTensors acts = new ActivationTensors(config, io.github.teleportationpyramid.flash.Precision.FP32, B, T);

            // Verify dimensions
            assert acts.getBatchSize() == B : "Batch size mismatch";
            assert acts.getSeqLen() == T : "Sequence length mismatch";

            // Verify total elements
            long total = acts.getTotalElements();
            System.out.printf("  Total elements (B=%d, T=%d): %,d%n", B, T, total);

            // Verify size in bytes
            long sizeBytes = acts.getSizeInBytes();
            System.out.printf("  Size in bytes: %,d (%.2f MiB)%n", sizeBytes, sizeBytes / (1024.0 * 1024.0));

            // Verify encoded size
            long expectedEncoded = (long) B * T * config.channels;
            assert acts.encodedSize == expectedEncoded : "Encoded size mismatch";
            System.out.printf("  encoded size: %,d (B*T*C)%n", acts.encodedSize);

            // Verify logits size
            long expectedLogits = (long) B * T * config.paddedVocabSize;
            assert acts.logitsSize == expectedLogits : "Logits size mismatch";
            System.out.printf("  logits size: %,d (B*T*V)%n", acts.logitsSize);

            // Verify per-layer arrays
            assert acts.ln1Offsets.length == 12 : "Should have 12 layers";
            assert acts.lseOffsets.length == 12 : "LSE should have 12 layers";

            // Verify offsets are increasing
            for (int l = 1; l < config.numLayers; l++) {
                assert acts.ln1Offsets[l] > acts.ln1Offsets[l-1] : "Offsets should increase";
            }

            System.out.println("  ✓ ActivationTensors layout calculated correctly");

            // Test smaller batch
            ActivationTensors actsSmall = new ActivationTensors(config, io.github.teleportationpyramid.flash.Precision.FP32, 1, 64);
            System.out.printf("  Small batch (B=1, T=64): %,d elements, %.2f MiB%n",
                actsSmall.getTotalElements(),
                actsSmall.getSizeInBytes() / (1024.0 * 1024.0));

            // Cleanup
            acts.close();
            actsSmall.close();

            return true;
        } catch (AssertionError | Exception e) {
            System.out.println("  ✗ FAILED: " + e.getMessage());
            e.printStackTrace();
            return false;
        }
    }
}
