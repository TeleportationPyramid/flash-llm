package com.flashllm;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.kernel.*;
import com.flashllm.model.TransformerBlock;

/**
 * Backward Pass Test.
 *
 * Tests backward pass for each kernel:
 * 1. LayerNorm backward
 * 2. Matmul backward
 * 3. GELU backward
 * 4. Residual backward
 * 5. Attention backward
 * 6. TransformerBlock backward
 * 7. End-to-end forward + backward
 */
public class BackwardPassTest {

    public static void main(String[] args) {
        System.out.println("=== Backward Pass Test ===\n");

        try {
            FlashBackend backend = FlashBackend.getInstance();
            CudaDevice device = backend.getDevice();

            // Config
            int B = 2;    // batch size
            int T = 8;    // sequence length
            int C = 64;   // channels
            int NH = 2;   // num heads
            int V = 256;  // vocab size
            int BT = B * T;

            System.out.printf("Config: B=%d, T=%d, C=%d, NH=%d, V=%d%n%n", B, T, C, NH, V);

            // Test each kernel's backward
            testLayerNormBackward(backend, device, BT, C);
            testMatmulBackward(backend, device, BT, C);
            testGeluBackward(backend, device, BT, C);
            testResidualBackward(backend, device, BT, C);
            testAttentionBackward(backend, device, B, T, C, NH);
            testTransformerBlockBackward(backend, device, B, T, C, NH);
            testEndToEndForwardBackward(backend, device, B, T, C, NH, V);

            System.out.println("\n==================================================");
            System.out.println("✓ All Backward Pass tests PASSED");

            backend.close();

        } catch (Exception e) {
            System.err.println("✗ Test failed:");
            e.printStackTrace();
            System.exit(1);
        }
    }

    // ========================================================================
    // Test 1: LayerNorm Backward
    // ========================================================================
    static void testLayerNormBackward(FlashBackend backend, CudaDevice device, int N, int C) {
        System.out.println("Test 1: LayerNorm Backward");

        CudaTensor inp = backend.allocateF32(N * C);
        CudaTensor out = backend.allocateF32(N * C);
        CudaTensor mean = backend.allocateF32(N);
        CudaTensor rstd = backend.allocateF32(N);
        CudaTensor weight = backend.allocateF32(C);
        CudaTensor bias = backend.allocateF32(C);

        CudaTensor dout = backend.allocateF32(N * C);
        CudaTensor dinp = backend.allocateF32(N * C);
        CudaTensor dweight = backend.allocateF32(C);
        CudaTensor dbias = backend.allocateF32(C);

        // Initialize
        initRandom(device, inp, 1.0f);
        initOnes(device, weight);
        backend.zeroFill(bias);
        initRandom(device, dout, 0.1f);  // Upstream gradient
        backend.zeroFill(dweight);
        backend.zeroFill(dbias);

        // Forward
        LayerNorm.forward(out, mean, rstd, inp, weight, bias, N, C);

        // Backward
        LayerNorm.backward(dinp, dweight, dbias, dout, inp, weight, mean, rstd, N, C);

        // Verify gradients are non-zero
        float dinpSum = sumAbs(dinp);
        float dweightSum = sumAbs(dweight);
        float dbiasSum = sumAbs(dbias);

        System.out.printf("  dinp sum: %.4f, dweight sum: %.4f, dbias sum: %.4f%n",
                dinpSum, dweightSum, dbiasSum);

        assert dinpSum > 0 : "dinp should be non-zero";
        assert dweightSum > 0 : "dweight should be non-zero";
        assert dbiasSum > 0 : "dbias should be non-zero";

        // Cleanup
        inp.close(); out.close(); mean.close(); rstd.close();
        weight.close(); bias.close();
        dout.close(); dinp.close(); dweight.close(); dbias.close();

        System.out.println("  ✓ LayerNorm backward works\n");
    }

    // ========================================================================
    // Test 2: Matmul Backward
    // ========================================================================
    static void testMatmulBackward(FlashBackend backend, CudaDevice device, int N, int C) {
        System.out.println("Test 2: Matmul Backward");

        int OC = C * 4;  // Output channels

        CudaTensor inp = backend.allocateF32(N * C);
        CudaTensor weight = backend.allocateF32(C * OC);
        CudaTensor bias = backend.allocateF32(OC);
        CudaTensor out = backend.allocateF32(N * OC);

        CudaTensor dout = backend.allocateF32(N * OC);
        CudaTensor dinp = backend.allocateF32(N * C);
        CudaTensor dweight = backend.allocateF32(C * OC);
        CudaTensor dbias = backend.allocateF32(OC);

        // Initialize
        initRandom(device, inp, 0.1f);
        initRandom(device, weight, 0.02f);
        initRandom(device, bias, 0.01f);
        initRandom(device, dout, 0.1f);
        backend.zeroFill(dweight);
        backend.zeroFill(dbias);

        // Forward
        Matmul.forward(out, inp, weight, bias, N, OC, C);

        // Backward
        Matmul.backward(dinp, dweight, dbias, dout, inp, weight, N, OC, C);

        // Verify
        float dinpSum = sumAbs(dinp);
        float dweightSum = sumAbs(dweight);
        float dbiasSum = sumAbs(dbias);

        System.out.printf("  dinp sum: %.4f, dweight sum: %.4f, dbias sum: %.4f%n",
                dinpSum, dweightSum, dbiasSum);

        assert dinpSum > 0 : "dinp should be non-zero";
        assert dweightSum > 0 : "dweight should be non-zero";
        assert dbiasSum > 0 : "dbias should be non-zero";

        // Cleanup
        inp.close(); weight.close(); bias.close(); out.close();
        dout.close(); dinp.close(); dweight.close(); dbias.close();

        System.out.println("  ✓ Matmul backward works\n");
    }

    // ========================================================================
    // Test 3: GELU Backward
    // ========================================================================
    static void testGeluBackward(FlashBackend backend, CudaDevice device, int N, int C) {
        System.out.println("Test 3: GELU Backward");

        int size = N * C;

        CudaTensor inp = backend.allocateF32(size);
        CudaTensor out = backend.allocateF32(size);
        CudaTensor dout = backend.allocateF32(size);
        CudaTensor dinp = backend.allocateF32(size);

        // Initialize
        initRandom(device, inp, 1.0f);
        initRandom(device, dout, 0.1f);

        // Forward
        Gelu.forward(out, inp, size);

        // Backward
        Gelu.backward(dinp, inp, dout, size);

        // Verify
        float dinpSum = sumAbs(dinp);
        System.out.printf("  dinp sum: %.4f%n", dinpSum);

        assert dinpSum > 0 : "dinp should be non-zero";

        // Cleanup
        inp.close(); out.close(); dout.close(); dinp.close();

        System.out.println("  ✓ GELU backward works\n");
    }

    // ========================================================================
    // Test 4: Residual Backward
    // ========================================================================
    static void testResidualBackward(FlashBackend backend, CudaDevice device, int N, int C) {
        System.out.println("Test 4: Residual Backward");

        int size = N * C;

        CudaTensor a = backend.allocateF32(size);
        CudaTensor b = backend.allocateF32(size);
        CudaTensor out = backend.allocateF32(size);
        CudaTensor dout = backend.allocateF32(size);
        CudaTensor da = backend.allocateF32(size);
        CudaTensor db = backend.allocateF32(size);

        // Initialize
        initRandom(device, a, 0.5f);
        initRandom(device, b, 0.5f);
        initRandom(device, dout, 0.1f);
        backend.zeroFill(da);
        backend.zeroFill(db);

        // Forward
        Residual.forward(out, a, b, size);

        // Backward
        Residual.backward(da, db, dout, size);

        // Verify: da and db should equal dout (gradient flows to both branches)
        float doutSum = sumAbs(dout);
        float daSum = sumAbs(da);
        float dbSum = sumAbs(db);

        System.out.printf("  dout sum: %.4f, da sum: %.4f, db sum: %.4f%n", doutSum, daSum, dbSum);

        // da and db should be close to dout
        assert Math.abs(daSum - doutSum) < 0.01 : "da should equal dout";
        assert Math.abs(dbSum - doutSum) < 0.01 : "db should equal dout";

        // Cleanup
        a.close(); b.close(); out.close();
        dout.close(); da.close(); db.close();

        System.out.println("  ✓ Residual backward works\n");
    }

    // ========================================================================
    // Test 5: Attention Backward
    // ========================================================================
    static void testAttentionBackward(FlashBackend backend, CudaDevice device, int B, int T, int C, int NH) {
        System.out.println("Test 5: Attention Backward");

        int BT = B * T;
        int BNH = B * NH;

        // Forward tensors
        CudaTensor inp = backend.allocateF32(BT * C);
        CudaTensor qkvw = backend.allocateF32(C * 3 * C);
        CudaTensor qkvb = backend.allocateF32(3 * C);
        CudaTensor attprojw = backend.allocateF32(C * C);
        CudaTensor attprojb = backend.allocateF32(C);
        CudaTensor out = backend.allocateF32(BT * C);
        CudaTensor qkv = backend.allocateF32(BT * 3 * C);
        CudaTensor lse = backend.allocateF32(BNH * T);

        // Backward tensors
        CudaTensor dout = backend.allocateF32(BT * C);
        CudaTensor dinp = backend.allocateF32(BT * C);
        CudaTensor dqkvw = backend.allocateF32(C * 3 * C);
        CudaTensor dqkvb = backend.allocateF32(3 * C);
        CudaTensor dattprojw = backend.allocateF32(C * C);
        CudaTensor dattprojb = backend.allocateF32(C);

        // Initialize
        initRandom(device, inp, 0.1f);
        initRandom(device, qkvw, 0.02f);
        backend.zeroFill(qkvb);
        initRandom(device, attprojw, 0.02f);
        backend.zeroFill(attprojb);
        initRandom(device, dout, 0.1f);
        backend.zeroFill(dqkvw);
        backend.zeroFill(dqkvb);
        backend.zeroFill(dattprojw);
        backend.zeroFill(dattprojb);

        // Forward
        Attention.forward(out, qkv, lse, inp, qkvw, qkvb, attprojw, attprojb, B, T, C, NH);

        // Backward
        Attention.backward(dinp, dqkvw, dqkvb, dattprojw, dattprojb,
                dout, inp, qkv, lse, qkvw, attprojw, B, T, C, NH);

        // Verify
        float dinpSum = sumAbs(dinp);
        float dqkvwSum = sumAbs(dqkvw);
        float dattprojwSum = sumAbs(dattprojw);

        System.out.printf("  dinp sum: %.4f, dqkvw sum: %.4f, dattprojw sum: %.4f%n",
                dinpSum, dqkvwSum, dattprojwSum);

        assert dinpSum > 0 : "dinp should be non-zero";
        assert dqkvwSum > 0 : "dqkvw should be non-zero";
        assert dattprojwSum > 0 : "dattprojw should be non-zero";

        // Cleanup
        inp.close(); qkvw.close(); qkvb.close(); attprojw.close(); attprojb.close();
        out.close(); qkv.close(); lse.close();
        dout.close(); dinp.close(); dqkvw.close(); dqkvb.close();
        dattprojw.close(); dattprojb.close();

        System.out.println("  ✓ Attention backward works\n");
    }

    // ========================================================================
    // Test 6: TransformerBlock Backward
    // ========================================================================
    static void testTransformerBlockBackward(FlashBackend backend, CudaDevice device,
                                              int B, int T, int C, int NH) {
        System.out.println("Test 6: TransformerBlock Backward");

        int BT = B * T;
        int BNH = B * NH;

        TransformerBlock block = new TransformerBlock(0, B, T, C, NH);

        // Parameters
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
        initOnes(device, ln1w); backend.zeroFill(ln1b);
        initRandom(device, qkvw, 0.02f); backend.zeroFill(qkvb);
        initRandom(device, attprojw, 0.02f); backend.zeroFill(attprojb);
        initOnes(device, ln2w); backend.zeroFill(ln2b);
        initRandom(device, fcw, 0.02f); backend.zeroFill(fcb);
        initRandom(device, fcprojw, 0.02f); backend.zeroFill(fcprojb);

        // Activations
        CudaTensor inp = backend.allocateF32(BT * C);
        CudaTensor out = backend.allocateF32(BT * C);
        CudaTensor ln1 = backend.allocateF32(BT * C);
        CudaTensor ln1Mean = backend.allocateF32(BT);
        CudaTensor ln1Rstd = backend.allocateF32(BT);
        CudaTensor qkv = backend.allocateF32(BT * 3 * C);
        CudaTensor atty = backend.allocateF32(BT * C);
        CudaTensor attLse = backend.allocateF32(BNH * T);
        CudaTensor attnOut = backend.allocateF32(BT * C);
        CudaTensor ln2 = backend.allocateF32(BT * C);
        CudaTensor ln2Mean = backend.allocateF32(BT);
        CudaTensor ln2Rstd = backend.allocateF32(BT);
        CudaTensor fch = backend.allocateF32(BT * 4 * C);
        CudaTensor fchGelu = backend.allocateF32(BT * 4 * C);

        // Gradients
        CudaTensor dout = backend.allocateF32(BT * C);
        CudaTensor dinp = backend.allocateF32(BT * C);
        CudaTensor dln1w = backend.allocateF32(C);
        CudaTensor dln1b = backend.allocateF32(C);
        CudaTensor dqkvw = backend.allocateF32(C * 3 * C);
        CudaTensor dqkvb = backend.allocateF32(3 * C);
        CudaTensor dattprojw = backend.allocateF32(C * C);
        CudaTensor dattprojb = backend.allocateF32(C);
        CudaTensor dln2w = backend.allocateF32(C);
        CudaTensor dln2b = backend.allocateF32(C);
        CudaTensor dfcw = backend.allocateF32(C * 4 * C);
        CudaTensor dfcb = backend.allocateF32(4 * C);
        CudaTensor dfcprojw = backend.allocateF32(4 * C * C);
        CudaTensor dfcprojb = backend.allocateF32(C);

        // Initialize
        initRandom(device, inp, 0.5f);
        initRandom(device, dout, 0.1f);
        backend.zeroFill(dln1w); backend.zeroFill(dln1b);
        backend.zeroFill(dqkvw); backend.zeroFill(dqkvb);
        backend.zeroFill(dattprojw); backend.zeroFill(dattprojb);
        backend.zeroFill(dln2w); backend.zeroFill(dln2b);
        backend.zeroFill(dfcw); backend.zeroFill(dfcb);
        backend.zeroFill(dfcprojw); backend.zeroFill(dfcprojb);

        // Forward
        block.forward(out, inp,
                ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                ln1, ln1Mean, ln1Rstd,
                qkv, atty, attLse, attnOut,
                ln2, ln2Mean, ln2Rstd,
                fch, fchGelu);

        System.out.println("  Forward done, running backward...");

        // Backward
        block.backward(dinp, dout, inp,
                ln1w, qkvw, attprojw, ln2w, fcw, fcprojw,
                ln1, ln1Mean, ln1Rstd,
                qkv, atty, attLse, attnOut,
                ln2, ln2Mean, ln2Rstd,
                fch, fchGelu,
                dln1w, dln1b, dqkvw, dqkvb, dattprojw, dattprojb,
                dln2w, dln2b, dfcw, dfcb, dfcprojw, dfcprojb);

        // Verify gradients
        float dinpSum = sumAbs(dinp);
        float dln1wSum = sumAbs(dln1w);
        float dqkvwSum = sumAbs(dqkvw);
        float dfcwSum = sumAbs(dfcw);

        System.out.printf("  dinp sum: %.4f%n", dinpSum);
        System.out.printf("  dln1w sum: %.4f, dqkvw sum: %.4f, dfcw sum: %.4f%n",
                dln1wSum, dqkvwSum, dfcwSum);

        assert dinpSum > 0 : "dinp should be non-zero";
        assert dln1wSum > 0 : "dln1w should be non-zero";
        assert dqkvwSum > 0 : "dqkvw should be non-zero";
        assert dfcwSum > 0 : "dfcw should be non-zero";

        // Cleanup (abbreviated for brevity)
        ln1w.close(); ln1b.close(); qkvw.close(); qkvb.close();
        attprojw.close(); attprojb.close(); ln2w.close(); ln2b.close();
        fcw.close(); fcb.close(); fcprojw.close(); fcprojb.close();
        inp.close(); out.close(); ln1.close(); ln1Mean.close(); ln1Rstd.close();
        qkv.close(); atty.close(); attLse.close(); attnOut.close();
        ln2.close(); ln2Mean.close(); ln2Rstd.close(); fch.close(); fchGelu.close();
        dout.close(); dinp.close();
        dln1w.close(); dln1b.close(); dqkvw.close(); dqkvb.close();
        dattprojw.close(); dattprojb.close(); dln2w.close(); dln2b.close();
        dfcw.close(); dfcb.close(); dfcprojw.close(); dfcprojb.close();

        System.out.println("  ✓ TransformerBlock backward works\n");
    }

    // ========================================================================
    // Test 7: End-to-End Forward + Backward
    // ========================================================================
    static void testEndToEndForwardBackward(FlashBackend backend, CudaDevice device,
                                             int B, int T, int C, int NH, int V) {
        System.out.println("Test 7: End-to-End Forward + Backward");

        int BT = B * T;

        // Simplified model: Embedding -> LayerNorm -> Output Projection -> Loss
        CudaTensor wte = backend.allocateF32(V * C);
        CudaTensor wpe = backend.allocateF32(T * C);
        CudaTensor lnw = backend.allocateF32(C);
        CudaTensor lnb = backend.allocateF32(C);

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

        // Initialize
        initRandom(device, wte, 0.02f);
        initRandom(device, wpe, 0.02f);
        initOnes(device, lnw);
        backend.zeroFill(lnb);
        backend.zeroFill(dwte);
        backend.zeroFill(dwpe);
        backend.zeroFill(dlnw);
        backend.zeroFill(dlnb);

        // Input tokens and targets
        int[] tokens = new int[BT];
        int[] targets = new int[BT];
        for (int i = 0; i < BT; i++) {
            tokens[i] = i % V;
            targets[i] = (i + 1) % V;
        }

        // ==================== FORWARD ====================
        System.out.println("  Forward pass...");

        // 1. Encoder
        Encoder.forward(encoded, tokens, wte, wpe, B, T, C);

        // 2. LayerNorm
        LayerNorm.forward(lnOut, lnMean, lnRstd, encoded, lnw, lnb, BT, C);

        // 3. Output projection (tied weights)
        Matmul.forward(logits, lnOut, wte, BT, V, C);

        // 4. Softmax
        Softmax.forward(probs, logits, B, T, V);

        // 5. Loss
        Softmax.crossEntropyForward(losses, probs, targets, B, T, V);
        float loss = Softmax.meanLoss(losses, B, T);

        System.out.printf("  Loss: %.4f (expected ~%.4f)%n", loss, Math.log(V));

        // ==================== BACKWARD ====================
        System.out.println("  Backward pass...");

        // 5. Loss backward
        Softmax.crossEntropySoftmaxBackward(dlogits, probs, targets, B, T, V);

        // 4. Output projection backward (accumulate into dwte)
        Matmul.backward(dlnOut, dwte, null, dlogits, lnOut, wte, BT, V, C);

        // 3. LayerNorm backward
        LayerNorm.backward(dencoded, dlnw, dlnb, dlnOut, encoded, lnw, lnMean, lnRstd, BT, C);

        // 2. Encoder backward
        Encoder.backward(dwte, dwpe, dencoded, tokens, B, T, C);

        // Verify gradients
        float dwteSum = sumAbs(dwte);
        float dwpeSum = sumAbs(dwpe);
        float dlnwSum = sumAbs(dlnw);

        System.out.printf("  dwte sum: %.4f, dwpe sum: %.4f, dlnw sum: %.4f%n",
                dwteSum, dwpeSum, dlnwSum);

        assert dwteSum > 0 : "dwte should be non-zero";
        assert dwpeSum > 0 : "dwpe should be non-zero";
        assert dlnwSum > 0 : "dlnw should be non-zero";

        // Cleanup
        wte.close(); wpe.close(); lnw.close(); lnb.close();
        encoded.close(); lnOut.close(); lnMean.close(); lnRstd.close();
        logits.close(); probs.close(); losses.close();
        dwte.close(); dwpe.close(); dlnw.close(); dlnb.close();
        dlogits.close(); dlnOut.close(); dencoded.close();

        System.out.println("  ✓ End-to-end forward + backward works\n");
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

    static float sumAbs(CudaTensor tensor) {
        float[] data = tensor.toFloatArray();
        float sum = 0;
        for (float v : data) sum += Math.abs(v);
        return sum;
    }
}
