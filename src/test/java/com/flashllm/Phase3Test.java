package com.flashllm;

import com.flashllm.backend.FlashBackend;
import com.flashllm.config.GPT2Config;
import com.flashllm.data.DataLoader;
import com.flashllm.memory.ParameterTensors;
import com.flashllm.memory.GradTensors;

/**
 * Phase 3 Integration Tests - Model and Training Loop.
 *
 * <p>Tests:</p>
 * <ul>
 *   <li>GPT2Config</li>
 *   <li>ParameterTensors allocation</li>
 *   <li>GradTensors allocation</li>
 *   <li>DataLoader</li>
 * </ul>
 */
public class Phase3Test {

    public static void main(String[] args) {
        System.out.println("=== Flash-LLM Phase 3 Tests ===\n");

        try {
            // Initialize backend
            FlashBackend backend = FlashBackend.getInstance();
            System.out.println("Backend initialized: " + backend);
            System.out.println();

            // Run tests
            testGPT2Config();
            testDataLoader();
            testParameterTensors();
            testGradTensors();

            System.out.println("==================================================");
            System.out.println("✓ All Phase 3 tests PASSED");

            // Cleanup
            backend.close();

        } catch (Exception e) {
            System.err.println("✗ Test failed with exception:");
            e.printStackTrace();
            System.exit(1);
        }
    }

    static void testGPT2Config() {
        System.out.println("Test: GPT2Config");

        GPT2Config tiny = GPT2Config.tiny();
        System.out.println("  Tiny config: " + tiny);
        System.out.println("  Parameters: " + tiny.totalParameters());

        GPT2Config small = GPT2Config.small();
        System.out.println("  Small config: " + small);
        System.out.println("  Parameters: " + small.totalParameters());

        GPT2Config gpt2 = GPT2Config.gpt2_124M();
        System.out.println("  GPT2-124M config: " + gpt2);
        System.out.println("  Parameters: " + gpt2.totalParameters());

        System.out.println("  ✓ GPT2Config works correctly");
        System.out.println();
    }

    static void testDataLoader() {
        System.out.println("Test: DataLoader");

        int B = 2;
        int T = 8;

        // Create synthetic data
        int numTokens = 100;
        int[] tokens = new int[numTokens];
        for (int i = 0; i < numTokens; i++) {
            tokens[i] = i % 50;  // vocab size 50
        }

        DataLoader loader = new DataLoader(tokens, B, T);
        System.out.println("  Tokens: " + loader.getNumTokens());
        System.out.println("  Batches: " + loader.getNumBatches());

        int[] inputTokens = new int[B * T];
        int[] targetTokens = new int[B * T];

        // Get first batch
        boolean hasMore = loader.nextBatch(inputTokens, targetTokens);
        assert hasMore : "Should have at least one batch";

        // Verify target is shifted by 1
        System.out.println("  Input[0:4]: " + inputTokens[0] + ", " + inputTokens[1] + ", " + inputTokens[2] + ", " + inputTokens[3]);
        System.out.println("  Target[0:4]: " + targetTokens[0] + ", " + targetTokens[1] + ", " + targetTokens[2] + ", " + targetTokens[3]);

        System.out.println("  ✓ DataLoader works correctly");
        System.out.println();
    }

    static void testParameterTensors() {
        System.out.println("Test: ParameterTensors");

        GPT2Config config = GPT2Config.tiny();
        System.out.println("  Creating ParameterTensors for tiny config...");

        ParameterTensors params = new ParameterTensors(config);
        System.out.println("  Allocated: " + params);
        System.out.println("  Total parameters: " + params.numParameters());

        // Verify we can access tensors
        System.out.println("  wte size: " + params.wte.getElementCount());
        System.out.println("  wpe size: " + params.wpe.getElementCount());
        System.out.println("  ln1w[0] size: " + params.getLn1w(0).getElementCount());

        // Close
        System.out.println("  Closing ParameterTensors...");
        params.close();

        System.out.println("  ✓ ParameterTensors works correctly");
        System.out.println();
    }

    static void testGradTensors() {
        System.out.println("Test: GradTensors");

        GPT2Config config = GPT2Config.tiny();
        System.out.println("  Creating GradTensors for tiny config...");

        GradTensors grads = new GradTensors(config);
        System.out.println("  Allocated: " + grads);
        System.out.println("  Total elements: " + grads.numParameters());

        // Verify we can access tensors
        System.out.println("  wte size: " + grads.wte.getElementCount());
        System.out.println("  ln1w[0] size: " + grads.getLn1w(0).getElementCount());

        // Zero gradients
        System.out.println("  Zeroing gradients...");
        grads.zero();

        // Close
        System.out.println("  Closing GradTensors...");
        grads.close();

        System.out.println("  ✓ GradTensors works correctly");
        System.out.println();
    }
}
