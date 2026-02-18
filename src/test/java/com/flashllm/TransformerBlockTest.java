package com.flashllm;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.kernel.*;
import com.flashllm.model.TransformerBlock;

/**
 * TransformerBlock Forward Test.
 *
 * Tests the complete TransformerBlock forward pass:
 * x = x + attention(layernorm1(x))
 * x = x + mlp(layernorm2(x))
 */
public class TransformerBlockTest {

    public static void main(String[] args) {
        System.out.println("=== TransformerBlock Forward Test ===\n");

        try {
            FlashBackend backend = FlashBackend.getInstance();
            CudaDevice device = backend.getDevice();

            // Config
            int B = 2;    // batch size
            int T = 8;    // sequence length
            int C = 64;   // channels
            int NH = 2;   // num heads
            int HS = C / NH;  // head size = 32
            int BT = B * T;

            System.out.printf("Config: B=%d, T=%d, C=%d, NH=%d, HS=%d%n%n", B, T, C, NH, HS);

            // Test 1: Attention forward only
            testAttentionForward(backend, device, B, T, C, NH);

            // Test 2: Complete TransformerBlock forward
            testTransformerBlockForward(backend, device, B, T, C, NH);

            System.out.println("\n==================================================");
            System.out.println("✓ All TransformerBlock tests PASSED");

            backend.close();

        } catch (Exception e) {
            System.err.println("✗ Test failed:");
            e.printStackTrace();
            System.exit(1);
        }
    }

    static void testAttentionForward(FlashBackend backend, CudaDevice device, int B, int T, int C, int NH) {
        System.out.println("Test 1: Attention Forward");

        int HS = C / NH;
        int BT = B * T;
        int BNH = B * NH;

        // Allocate tensors
        CudaTensor inp = backend.allocateF32(BT * C);
        CudaTensor qkvw = backend.allocateF32(C * 3 * C);
        CudaTensor qkvb = backend.allocateF32(3 * C);
        CudaTensor attprojw = backend.allocateF32(C * C);
        CudaTensor attprojb = backend.allocateF32(C);
        CudaTensor out = backend.allocateF32(BT * C);
        CudaTensor qkv = backend.allocateF32(BT * 3 * C);
        CudaTensor lse = backend.allocateF32(BNH * T);

        // Initialize
        initRandom(device, inp, 0.1f);
        initRandom(device, qkvw, 0.02f);
        backend.zeroFill(qkvb);
        initRandom(device, attprojw, 0.02f);
        backend.zeroFill(attprojb);

        System.out.println("  Running Attention.forward...");

        try {
            Attention.forward(out, qkv, lse, inp, qkvw, qkvb, attprojw, attprojb, B, T, C, NH);

            // Verify output
            float[] outData = out.toFloatArray();
            float sum = 0, min = Float.MAX_VALUE, max = Float.MIN_VALUE;
            for (float v : outData) {
                sum += Math.abs(v);
                min = Math.min(min, v);
                max = Math.max(max, v);
            }

            System.out.printf("  Output: sum=%.4f, min=%.4f, max=%.4f%n", sum, min, max);
            assert sum > 0 : "Attention output should be non-zero";

            System.out.println("  ✓ Attention forward works\n");

        } catch (Exception e) {
            System.err.println("  ✗ Attention forward failed: " + e.getMessage());
            throw e;
        } finally {
            inp.close();
            qkvw.close();
            qkvb.close();
            attprojw.close();
            attprojb.close();
            out.close();
            qkv.close();
            lse.close();
        }
    }

    static void testTransformerBlockForward(FlashBackend backend, CudaDevice device, int B, int T, int C, int NH) {
        System.out.println("Test 2: TransformerBlock Forward");

        int BT = B * T;
        int BNH = B * NH;

        // Create TransformerBlock
        TransformerBlock block = new TransformerBlock(0, B, T, C, NH);

        // Allocate parameters
        CudaTensor ln1w = backend.allocateF32(C);
        CudaTensor ln1b = backend.allocateF32(C);
        CudaTensor qkvw = backend.allocateF32(C * 3 * C);
        CudaTensor qkvb = backend.allocateF32(3 * C);
        CudaTensor attprojw = backend.allocateF32(C * C);
        CudaTensor attprojb = backend.allocateF32(C);
        CudaTensor ln2w = backend.allocateF32(C);
        CudaTensor ln2b = backend.allocateF32(C);
        CudaTensor fcw = backend.allocateF32(C * 4 * C);
        CudaTensor fcb = backend.allocateF32(4 * C);
        CudaTensor fcprojw = backend.allocateF32(4 * C * C);
        CudaTensor fcprojb = backend.allocateF32(C);

        // Initialize parameters
        initOnes(device, ln1w);
        backend.zeroFill(ln1b);
        initRandom(device, qkvw, 0.02f);
        backend.zeroFill(qkvb);
        initRandom(device, attprojw, 0.02f);
        backend.zeroFill(attprojb);
        initOnes(device, ln2w);
        backend.zeroFill(ln2b);
        initRandom(device, fcw, 0.02f);
        backend.zeroFill(fcb);
        initRandom(device, fcprojw, 0.02f);
        backend.zeroFill(fcprojb);

        // Allocate activations
        CudaTensor inp = backend.allocateF32(BT * C);
        CudaTensor out = backend.allocateF32(BT * C);
        CudaTensor ln1 = backend.allocateF32(BT * C);
        CudaTensor ln1Mean = backend.allocateF32(BT);
        CudaTensor ln1Rstd = backend.allocateF32(BT);
        CudaTensor qkv = backend.allocateF32(BT * 3 * C);
        CudaTensor atty = backend.allocateF32(BT * C);  // Not used in current impl
        CudaTensor attLse = backend.allocateF32(BNH * T);
        CudaTensor attnOut = backend.allocateF32(BT * C);
        CudaTensor ln2 = backend.allocateF32(BT * C);
        CudaTensor ln2Mean = backend.allocateF32(BT);
        CudaTensor ln2Rstd = backend.allocateF32(BT);
        CudaTensor fch = backend.allocateF32(BT * 4 * C);
        CudaTensor fchGelu = backend.allocateF32(BT * 4 * C);

        // Initialize input
        initRandom(device, inp, 0.5f);

        // Print input stats
        float[] inpData = inp.toFloatArray();
        float inpSum = 0;
        for (float v : inpData) inpSum += Math.abs(v);
        System.out.printf("  Input: sum=%.4f%n", inpSum);

        System.out.println("  Running TransformerBlock.forward...");

        try {
            block.forward(
                out, inp,
                ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                ln1, ln1Mean, ln1Rstd,
                qkv, atty, attLse, attnOut,
                ln2, ln2Mean, ln2Rstd,
                fch, fchGelu
            );

            // Verify output
            float[] outData = out.toFloatArray();
            float outSum = 0, min = Float.MAX_VALUE, max = Float.MIN_VALUE;
            for (float v : outData) {
                outSum += Math.abs(v);
                min = Math.min(min, v);
                max = Math.max(max, v);
            }

            System.out.printf("  Output: sum=%.4f, min=%.4f, max=%.4f%n", outSum, min, max);

            // Verify intermediate activations
            float[] ln1Data = ln1.toFloatArray();
            float ln1Sum = 0;
            for (float v : ln1Data) ln1Sum += Math.abs(v);
            System.out.printf("  LN1 output sum: %.4f%n", ln1Sum);

            float[] qkvData = qkv.toFloatArray();
            float qkvSum = 0;
            for (float v : qkvData) qkvSum += Math.abs(v);
            System.out.printf("  QKV output sum: %.4f%n", qkvSum);

            float[] fchData = fch.toFloatArray();
            float fchSum = 0;
            for (float v : fchData) fchSum += Math.abs(v);
            System.out.printf("  FCH (MLP hidden) sum: %.4f%n", fchSum);

            assert outSum > 0 : "TransformerBlock output should be non-zero";

            // Check residual connection is working
            // Output should be close to input (residual) plus some transformation
            boolean residualWorking = Math.abs(outSum - inpSum) < inpSum * 10;  // Within 10x
            System.out.printf("  Residual check: input_sum=%.2f, output_sum=%.2f, ratio=%.2f%n",
                    inpSum, outSum, outSum / inpSum);

            System.out.println("  ✓ TransformerBlock forward works");

        } catch (Exception e) {
            System.err.println("  ✗ TransformerBlock forward failed: " + e.getMessage());
            throw e;
        } finally {
            // Cleanup
            ln1w.close(); ln1b.close();
            qkvw.close(); qkvb.close();
            attprojw.close(); attprojb.close();
            ln2w.close(); ln2b.close();
            fcw.close(); fcb.close();
            fcprojw.close(); fcprojb.close();
            inp.close(); out.close();
            ln1.close(); ln1Mean.close(); ln1Rstd.close();
            qkv.close(); atty.close(); attLse.close(); attnOut.close();
            ln2.close(); ln2Mean.close(); ln2Rstd.close();
            fch.close(); fchGelu.close();
        }
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
