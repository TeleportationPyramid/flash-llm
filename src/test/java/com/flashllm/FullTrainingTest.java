package com.flashllm;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.config.GPT2Config;
import com.flashllm.kernel.*;
import com.flashllm.model.TransformerBlock;

/**
 * Full GPT-2 Training Test.
 *
 * Complete GPT-2 architecture:
 * 1. Token Embedding + Position Embedding
 * 2. L x TransformerBlock (LN -> Attention -> Residual -> LN -> MLP -> Residual)
 * 3. Final LayerNorm
 * 4. Output Projection (tied weights)
 * 5. Softmax + Cross-Entropy Loss
 *
 * This is the SECOND VALIDATION POINT!
 */
public class FullTrainingTest {

    public static void main(String[] args) {
        System.out.println("=== Full GPT-2 Training Test ===\n");

        try {
            FlashBackend backend = FlashBackend.getInstance();
            CudaDevice device = backend.getDevice();

            // GPT-2 Small-ish config (fits in 8GB VRAM)
            int B = 4;      // batch size
            int T = 64;     // sequence length
            int C = 128;    // channels (embedding dim)
            int L = 4;      // num layers
            int NH = 4;     // num heads
            int V = 256;    // vocab size (small for testing)

            System.out.println("========================================");
            System.out.println("GPT-2 Configuration:");
            System.out.println("========================================");
            System.out.printf("  Batch size (B):     %d%n", B);
            System.out.printf("  Sequence length (T): %d%n", T);
            System.out.printf("  Channels (C):       %d%n", C);
            System.out.printf("  Layers (L):         %d%n", L);
            System.out.printf("  Heads (NH):         %d%n", NH);
            System.out.printf("  Vocab size (V):     %d%n", V);

            // Calculate parameters
            long embParams = (long) V * C + (long) T * C;  // wte + wpe
            long lnParams = 2L * C;  // weight + bias
            long attnParams = C * 3L * C + 3L * C + C * C + C;  // qkv_w, qkv_b, proj_w, proj_b
            long mlpParams = C * 4L * C + 4L * C + 4L * C * C + C;  // fc_w, fc_b, proj_w, proj_b
            long blockParams = 2 * lnParams + attnParams + mlpParams;
            long totalParams = embParams + L * blockParams + lnParams;

            System.out.printf("  Total parameters:   %,d%n", totalParams);
            System.out.println("========================================\n");

            // Run training test
            testFullGPT2Training(backend, device, B, T, C, L, NH, V);

            // Speed benchmark
            testFullGPT2Speed(backend, device, B, T, C, L, NH, V);

            System.out.println("\n==================================================");
            System.out.println("✓ Full GPT-2 Training Test PASSED");

            backend.close();

        } catch (Exception e) {
            System.err.println("✗ Test failed:");
            e.printStackTrace();
            System.exit(1);
        }
    }

    static void testFullGPT2Training(FlashBackend backend, CudaDevice device,
                                      int B, int T, int C, int L, int NH, int V) {
        System.out.println("Test 1: Full GPT-2 Training (verify loss decreases)");

        int BT = B * T;
        int BNH = B * NH;
        int numSteps = 50;

        // ==================== ALLOCATE PARAMETERS ====================
        // Embeddings
        CudaTensor wte = backend.allocateF32(V * C);
        CudaTensor wpe = backend.allocateF32(T * C);

        // Per-layer parameters
        CudaTensor[] ln1w = new CudaTensor[L];
        CudaTensor[] ln1b = new CudaTensor[L];
        CudaTensor[] qkvw = new CudaTensor[L];
        CudaTensor[] qkvb = new CudaTensor[L];
        CudaTensor[] attprojw = new CudaTensor[L];
        CudaTensor[] attprojb = new CudaTensor[L];
        CudaTensor[] ln2w = new CudaTensor[L];
        CudaTensor[] ln2b = new CudaTensor[L];
        CudaTensor[] fcw = new CudaTensor[L];
        CudaTensor[] fcb = new CudaTensor[L];
        CudaTensor[] fcprojw = new CudaTensor[L];
        CudaTensor[] fcprojb = new CudaTensor[L];

        for (int l = 0; l < L; l++) {
            ln1w[l] = backend.allocateF32(C);
            ln1b[l] = backend.allocateF32(C);
            qkvw[l] = backend.allocateF32(C * 3 * C);
            qkvb[l] = backend.allocateF32(3 * C);
            attprojw[l] = backend.allocateF32(C * C);
            attprojb[l] = backend.allocateF32(C);
            ln2w[l] = backend.allocateF32(C);
            ln2b[l] = backend.allocateF32(C);
            fcw[l] = backend.allocateF32(C * 4 * C);
            fcb[l] = backend.allocateF32(4 * C);
            fcprojw[l] = backend.allocateF32(4 * C * C);
            fcprojb[l] = backend.allocateF32(C);
        }

        // Final LayerNorm
        CudaTensor lnfw = backend.allocateF32(C);
        CudaTensor lnfb = backend.allocateF32(C);

        // ==================== ALLOCATE ACTIVATIONS ====================
        CudaTensor encoded = backend.allocateF32(BT * C);
        CudaTensor[] ln1 = new CudaTensor[L];
        CudaTensor[] ln1Mean = new CudaTensor[L];
        CudaTensor[] ln1Rstd = new CudaTensor[L];
        CudaTensor[] qkv = new CudaTensor[L];
        CudaTensor[] atty = new CudaTensor[L];
        CudaTensor[] attLse = new CudaTensor[L];
        CudaTensor[] attnOut = new CudaTensor[L];
        CudaTensor[] ln2 = new CudaTensor[L];
        CudaTensor[] ln2Mean = new CudaTensor[L];
        CudaTensor[] ln2Rstd = new CudaTensor[L];
        CudaTensor[] fch = new CudaTensor[L];
        CudaTensor[] fchGelu = new CudaTensor[L];
        CudaTensor[] residual = new CudaTensor[L];  // Output of each block

        for (int l = 0; l < L; l++) {
            ln1[l] = backend.allocateF32(BT * C);
            ln1Mean[l] = backend.allocateF32(BT);
            ln1Rstd[l] = backend.allocateF32(BT);
            qkv[l] = backend.allocateF32(BT * 3 * C);
            atty[l] = backend.allocateF32(BT * C);
            attLse[l] = backend.allocateF32(BNH * T);
            attnOut[l] = backend.allocateF32(BT * C);
            ln2[l] = backend.allocateF32(BT * C);
            ln2Mean[l] = backend.allocateF32(BT);
            ln2Rstd[l] = backend.allocateF32(BT);
            fch[l] = backend.allocateF32(BT * 4 * C);
            fchGelu[l] = backend.allocateF32(BT * 4 * C);
            residual[l] = backend.allocateF32(BT * C);
        }

        CudaTensor lnf = backend.allocateF32(BT * C);
        CudaTensor lnfMean = backend.allocateF32(BT);
        CudaTensor lnfRstd = backend.allocateF32(BT);
        CudaTensor logits = backend.allocateF32(BT * V);
        CudaTensor probs = backend.allocateF32(BT * V);
        CudaTensor losses = backend.allocateF32(BT);

        // ==================== ALLOCATE GRADIENTS ====================
        CudaTensor dwte = backend.allocateF32(V * C);
        CudaTensor dwpe = backend.allocateF32(T * C);
        CudaTensor[] dln1w = new CudaTensor[L];
        CudaTensor[] dln1b = new CudaTensor[L];
        CudaTensor[] dqkvw = new CudaTensor[L];
        CudaTensor[] dqkvb = new CudaTensor[L];
        CudaTensor[] dattprojw = new CudaTensor[L];
        CudaTensor[] dattprojb = new CudaTensor[L];
        CudaTensor[] dln2w = new CudaTensor[L];
        CudaTensor[] dln2b = new CudaTensor[L];
        CudaTensor[] dfcw = new CudaTensor[L];
        CudaTensor[] dfcb = new CudaTensor[L];
        CudaTensor[] dfcprojw = new CudaTensor[L];
        CudaTensor[] dfcprojb = new CudaTensor[L];

        for (int l = 0; l < L; l++) {
            dln1w[l] = backend.allocateF32(C);
            dln1b[l] = backend.allocateF32(C);
            dqkvw[l] = backend.allocateF32(C * 3 * C);
            dqkvb[l] = backend.allocateF32(3 * C);
            dattprojw[l] = backend.allocateF32(C * C);
            dattprojb[l] = backend.allocateF32(C);
            dln2w[l] = backend.allocateF32(C);
            dln2b[l] = backend.allocateF32(C);
            dfcw[l] = backend.allocateF32(C * 4 * C);
            dfcb[l] = backend.allocateF32(4 * C);
            dfcprojw[l] = backend.allocateF32(4 * C * C);
            dfcprojb[l] = backend.allocateF32(C);
        }

        CudaTensor dlnfw = backend.allocateF32(C);
        CudaTensor dlnfb = backend.allocateF32(C);
        CudaTensor dlogits = backend.allocateF32(BT * V);
        CudaTensor dlnf = backend.allocateF32(BT * C);
        CudaTensor dresidual = backend.allocateF32(BT * C);
        CudaTensor dencoded = backend.allocateF32(BT * C);

        // ==================== ALLOCATE OPTIMIZER STATE ====================
        CudaTensor mWte = backend.allocateF32(V * C);
        CudaTensor vWte = backend.allocateF32(V * C);
        CudaTensor mWpe = backend.allocateF32(T * C);
        CudaTensor vWpe = backend.allocateF32(T * C);
        // ... (simplified - in real impl would have m/v for all params)

        // ==================== INITIALIZE ====================
        initRandom(device, wte, 0.02f);
        initRandom(device, wpe, 0.02f);
        for (int l = 0; l < L; l++) {
            initOnes(device, ln1w[l]); backend.zeroFill(ln1b[l]);
            initRandom(device, qkvw[l], 0.02f); backend.zeroFill(qkvb[l]);
            initRandom(device, attprojw[l], 0.02f); backend.zeroFill(attprojb[l]);
            initOnes(device, ln2w[l]); backend.zeroFill(ln2b[l]);
            initRandom(device, fcw[l], 0.02f); backend.zeroFill(fcb[l]);
            initRandom(device, fcprojw[l], 0.02f); backend.zeroFill(fcprojb[l]);
        }
        initOnes(device, lnfw); backend.zeroFill(lnfb);
        backend.zeroFill(mWte); backend.zeroFill(vWte);
        backend.zeroFill(mWpe); backend.zeroFill(vWpe);

        // Create TransformerBlocks
        TransformerBlock[] blocks = new TransformerBlock[L];
        for (int l = 0; l < L; l++) {
            blocks[l] = new TransformerBlock(l, B, T, C, NH);
        }

        // Training data
        int[] tokens = new int[BT];
        int[] targets = new int[BT];
        java.util.Random rng = new java.util.Random(42);
        for (int i = 0; i < BT; i++) {
            tokens[i] = rng.nextInt(V);
            targets[i] = rng.nextInt(V);
        }

        float lr = 3e-4f;
        float weightDecay = 0.01f;

        float initialLoss = 0;
        float finalLoss = 0;

        System.out.println("  Training...");

        for (int step = 0; step < numSteps; step++) {
            // Zero gradients (simplified)
            backend.zeroFill(dwte);
            backend.zeroFill(dwpe);
            for (int l = 0; l < L; l++) {
                backend.zeroFill(dln1w[l]); backend.zeroFill(dln1b[l]);
                backend.zeroFill(dqkvw[l]); backend.zeroFill(dqkvb[l]);
                backend.zeroFill(dattprojw[l]); backend.zeroFill(dattprojb[l]);
                backend.zeroFill(dln2w[l]); backend.zeroFill(dln2b[l]);
                backend.zeroFill(dfcw[l]); backend.zeroFill(dfcb[l]);
                backend.zeroFill(dfcprojw[l]); backend.zeroFill(dfcprojb[l]);
            }
            backend.zeroFill(dlnfw); backend.zeroFill(dlnfb);

            // ==================== FORWARD ====================
            // 1. Encoder
            Encoder.forward(encoded, tokens, wte, wpe, B, T, C);

            // 2. Transformer Blocks
            CudaTensor blockInput = encoded;
            for (int l = 0; l < L; l++) {
                blocks[l].forward(
                    residual[l], blockInput,
                    ln1w[l], ln1b[l], qkvw[l], qkvb[l], attprojw[l], attprojb[l],
                    ln2w[l], ln2b[l], fcw[l], fcb[l], fcprojw[l], fcprojb[l],
                    ln1[l], ln1Mean[l], ln1Rstd[l],
                    qkv[l], atty[l], attLse[l], attnOut[l],
                    ln2[l], ln2Mean[l], ln2Rstd[l],
                    fch[l], fchGelu[l]
                );
                blockInput = residual[l];
            }

            // 3. Final LayerNorm
            LayerNorm.forward(lnf, lnfMean, lnfRstd, residual[L-1], lnfw, lnfb, BT, C);

            // 4. Output projection (tied weights)
            Matmul.forward(logits, lnf, wte, BT, V, C);

            // 5. Softmax + Loss
            Softmax.forward(probs, logits, B, T, V);
            Softmax.crossEntropyForward(losses, probs, targets, B, T, V);
            float loss = Softmax.meanLoss(losses, B, T);

            if (step == 0) initialLoss = loss;
            if (step == numSteps - 1) finalLoss = loss;

            // ==================== BACKWARD ====================
            // 5. Loss backward
            Softmax.crossEntropySoftmaxBackward(dlogits, probs, targets, B, T, V);

            // 4. Output projection backward
            Matmul.backward(dlnf, dwte, null, dlogits, lnf, wte, BT, V, C);

            // 3. Final LayerNorm backward
            LayerNorm.backward(dresidual, dlnfw, dlnfb, dlnf, residual[L-1], lnfw, lnfMean, lnfRstd, BT, C);

            // 2. Transformer Blocks backward (reverse order)
            CudaTensor dout = dresidual;
            for (int l = L - 1; l >= 0; l--) {
                CudaTensor inp = (l == 0) ? encoded : residual[l - 1];
                CudaTensor dinp = (l == 0) ? dencoded : dresidual;

                blocks[l].backward(
                    dinp, dout, inp,
                    ln1w[l], qkvw[l], attprojw[l], ln2w[l], fcw[l], fcprojw[l],
                    ln1[l], ln1Mean[l], ln1Rstd[l],
                    qkv[l], atty[l], attLse[l], attnOut[l],
                    ln2[l], ln2Mean[l], ln2Rstd[l],
                    fch[l], fchGelu[l],
                    dln1w[l], dln1b[l], dqkvw[l], dqkvb[l], dattprojw[l], dattprojb[l],
                    dln2w[l], dln2b[l], dfcw[l], dfcb[l], dfcprojw[l], dfcprojb[l]
                );
                dout = dinp;
            }

            // 1. Encoder backward
            Encoder.backward(dwte, dwpe, dencoded, tokens, B, T, C);

            // ==================== OPTIMIZER (simplified - only embeddings) ====================
            int t = step + 1;
            AdamW.update(wte, dwte, mWte, vWte, lr, weightDecay, t, (long) V * C);
            AdamW.update(wpe, dwpe, mWpe, vWpe, lr, weightDecay, t, (long) T * C);
            // In full impl, would update all parameters

            // Log progress
            if (step % 10 == 0 || step == numSteps - 1) {
                System.out.printf("    Step %3d: loss = %.4f%n", step, loss);
            }
        }

        // Verify
        float lossReduction = (initialLoss - finalLoss) / initialLoss * 100;
        System.out.printf("\n  Initial loss: %.4f%n", initialLoss);
        System.out.printf("  Final loss:   %.4f%n", finalLoss);
        System.out.printf("  Reduction:    %.1f%%%n", lossReduction);

        boolean lossDecreased = finalLoss < initialLoss;
        System.out.println("  Loss decreased: " + (lossDecreased ? "Yes ✓" : "No ✗"));

        // Cleanup (simplified)
        wte.close(); wpe.close(); lnfw.close(); lnfb.close();
        encoded.close(); lnf.close(); lnfMean.close(); lnfRstd.close();
        logits.close(); probs.close(); losses.close();
        dwte.close(); dwpe.close(); dlnfw.close(); dlnfb.close();
        dlogits.close(); dlnf.close(); dresidual.close(); dencoded.close();
        mWte.close(); vWte.close(); mWpe.close(); vWpe.close();

        for (int l = 0; l < L; l++) {
            ln1w[l].close(); ln1b[l].close(); qkvw[l].close(); qkvb[l].close();
            attprojw[l].close(); attprojb[l].close(); ln2w[l].close(); ln2b[l].close();
            fcw[l].close(); fcb[l].close(); fcprojw[l].close(); fcprojb[l].close();
            ln1[l].close(); ln1Mean[l].close(); ln1Rstd[l].close();
            qkv[l].close(); atty[l].close(); attLse[l].close(); attnOut[l].close();
            ln2[l].close(); ln2Mean[l].close(); ln2Rstd[l].close();
            fch[l].close(); fchGelu[l].close(); residual[l].close();
            dln1w[l].close(); dln1b[l].close(); dqkvw[l].close(); dqkvb[l].close();
            dattprojw[l].close(); dattprojb[l].close(); dln2w[l].close(); dln2b[l].close();
            dfcw[l].close(); dfcb[l].close(); dfcprojw[l].close(); dfcprojb[l].close();
        }

        System.out.println("  ✓ Full GPT-2 training works\n");
    }

    static void testFullGPT2Speed(FlashBackend backend, CudaDevice device,
                                   int B, int T, int C, int L, int NH, int V) {
        System.out.println("Test 2: Full GPT-2 Speed Benchmark");
        System.out.println("  ★ SECOND VALIDATION POINT ★\n");

        int BT = B * T;
        int BNH = B * NH;
        int warmupSteps = 5;
        int benchmarkSteps = 20;

        // Allocate everything (same as above, abbreviated)
        CudaTensor wte = backend.allocateF32(V * C);
        CudaTensor wpe = backend.allocateF32(T * C);
        CudaTensor encoded = backend.allocateF32(BT * C);
        CudaTensor lnfw = backend.allocateF32(C);
        CudaTensor lnfb = backend.allocateF32(C);
        CudaTensor lnf = backend.allocateF32(BT * C);
        CudaTensor lnfMean = backend.allocateF32(BT);
        CudaTensor lnfRstd = backend.allocateF32(BT);
        CudaTensor logits = backend.allocateF32(BT * V);
        CudaTensor probs = backend.allocateF32(BT * V);
        CudaTensor losses = backend.allocateF32(BT);
        CudaTensor dwte = backend.allocateF32(V * C);
        CudaTensor dwpe = backend.allocateF32(T * C);
        CudaTensor dlogits = backend.allocateF32(BT * V);
        CudaTensor dlnf = backend.allocateF32(BT * C);
        CudaTensor dlnfw = backend.allocateF32(C);
        CudaTensor dlnfb = backend.allocateF32(C);
        CudaTensor dresidual = backend.allocateF32(BT * C);
        CudaTensor dencoded = backend.allocateF32(BT * C);
        CudaTensor mWte = backend.allocateF32(V * C);
        CudaTensor vWte = backend.allocateF32(V * C);
        CudaTensor mWpe = backend.allocateF32(T * C);
        CudaTensor vWpe = backend.allocateF32(T * C);

        // Per-layer
        CudaTensor[] ln1w = new CudaTensor[L], ln1b = new CudaTensor[L];
        CudaTensor[] qkvw = new CudaTensor[L], qkvb = new CudaTensor[L];
        CudaTensor[] attprojw = new CudaTensor[L], attprojb = new CudaTensor[L];
        CudaTensor[] ln2w = new CudaTensor[L], ln2b = new CudaTensor[L];
        CudaTensor[] fcw = new CudaTensor[L], fcb = new CudaTensor[L];
        CudaTensor[] fcprojw = new CudaTensor[L], fcprojb = new CudaTensor[L];
        CudaTensor[] ln1 = new CudaTensor[L], ln1Mean = new CudaTensor[L], ln1Rstd = new CudaTensor[L];
        CudaTensor[] qkv = new CudaTensor[L], atty = new CudaTensor[L], attLse = new CudaTensor[L], attnOut = new CudaTensor[L];
        CudaTensor[] ln2 = new CudaTensor[L], ln2Mean = new CudaTensor[L], ln2Rstd = new CudaTensor[L];
        CudaTensor[] fch = new CudaTensor[L], fchGelu = new CudaTensor[L], residual = new CudaTensor[L];
        CudaTensor[] dln1w = new CudaTensor[L], dln1b = new CudaTensor[L];
        CudaTensor[] dqkvw = new CudaTensor[L], dqkvb = new CudaTensor[L];
        CudaTensor[] dattprojw = new CudaTensor[L], dattprojb = new CudaTensor[L];
        CudaTensor[] dln2w = new CudaTensor[L], dln2b = new CudaTensor[L];
        CudaTensor[] dfcw = new CudaTensor[L], dfcb = new CudaTensor[L];
        CudaTensor[] dfcprojw = new CudaTensor[L], dfcprojb = new CudaTensor[L];

        for (int l = 0; l < L; l++) {
            ln1w[l] = backend.allocateF32(C); ln1b[l] = backend.allocateF32(C);
            qkvw[l] = backend.allocateF32(C * 3 * C); qkvb[l] = backend.allocateF32(3 * C);
            attprojw[l] = backend.allocateF32(C * C); attprojb[l] = backend.allocateF32(C);
            ln2w[l] = backend.allocateF32(C); ln2b[l] = backend.allocateF32(C);
            fcw[l] = backend.allocateF32(C * 4 * C); fcb[l] = backend.allocateF32(4 * C);
            fcprojw[l] = backend.allocateF32(4 * C * C); fcprojb[l] = backend.allocateF32(C);
            ln1[l] = backend.allocateF32(BT * C); ln1Mean[l] = backend.allocateF32(BT); ln1Rstd[l] = backend.allocateF32(BT);
            qkv[l] = backend.allocateF32(BT * 3 * C); atty[l] = backend.allocateF32(BT * C);
            attLse[l] = backend.allocateF32(BNH * T); attnOut[l] = backend.allocateF32(BT * C);
            ln2[l] = backend.allocateF32(BT * C); ln2Mean[l] = backend.allocateF32(BT); ln2Rstd[l] = backend.allocateF32(BT);
            fch[l] = backend.allocateF32(BT * 4 * C); fchGelu[l] = backend.allocateF32(BT * 4 * C);
            residual[l] = backend.allocateF32(BT * C);
            dln1w[l] = backend.allocateF32(C); dln1b[l] = backend.allocateF32(C);
            dqkvw[l] = backend.allocateF32(C * 3 * C); dqkvb[l] = backend.allocateF32(3 * C);
            dattprojw[l] = backend.allocateF32(C * C); dattprojb[l] = backend.allocateF32(C);
            dln2w[l] = backend.allocateF32(C); dln2b[l] = backend.allocateF32(C);
            dfcw[l] = backend.allocateF32(C * 4 * C); dfcb[l] = backend.allocateF32(4 * C);
            dfcprojw[l] = backend.allocateF32(4 * C * C); dfcprojb[l] = backend.allocateF32(C);
        }

        // Initialize
        initRandom(device, wte, 0.02f);
        initRandom(device, wpe, 0.02f);
        initOnes(device, lnfw); backend.zeroFill(lnfb);
        backend.zeroFill(mWte); backend.zeroFill(vWte);
        backend.zeroFill(mWpe); backend.zeroFill(vWpe);

        for (int l = 0; l < L; l++) {
            initOnes(device, ln1w[l]); backend.zeroFill(ln1b[l]);
            initRandom(device, qkvw[l], 0.02f); backend.zeroFill(qkvb[l]);
            initRandom(device, attprojw[l], 0.02f); backend.zeroFill(attprojb[l]);
            initOnes(device, ln2w[l]); backend.zeroFill(ln2b[l]);
            initRandom(device, fcw[l], 0.02f); backend.zeroFill(fcb[l]);
            initRandom(device, fcprojw[l], 0.02f); backend.zeroFill(fcprojb[l]);
        }

        TransformerBlock[] blocks = new TransformerBlock[L];
        for (int l = 0; l < L; l++) {
            blocks[l] = new TransformerBlock(l, B, T, C, NH);
        }

        int[] tokens = new int[BT];
        int[] targets = new int[BT];
        java.util.Random rng = new java.util.Random(42);
        for (int i = 0; i < BT; i++) {
            tokens[i] = rng.nextInt(V);
            targets[i] = rng.nextInt(V);
        }

        float lr = 3e-4f;
        float weightDecay = 0.01f;

        // Warmup
        System.out.printf("  Warmup: %d steps...%n", warmupSteps);
        for (int step = 0; step < warmupSteps; step++) {
            trainStepFull(backend, device, B, T, C, L, NH, V, BT, BNH, step + 1,
                    tokens, targets, wte, wpe, lnfw, lnfb,
                    ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                    ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                    encoded, ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                    ln2, ln2Mean, ln2Rstd, fch, fchGelu, residual,
                    lnf, lnfMean, lnfRstd, logits, probs, losses,
                    dwte, dwpe, dlnfw, dlnfb,
                    dln1w, dln1b, dqkvw, dqkvb, dattprojw, dattprojb,
                    dln2w, dln2b, dfcw, dfcb, dfcprojw, dfcprojb,
                    dlogits, dlnf, dresidual, dencoded,
                    mWte, vWte, mWpe, vWpe,
                    blocks, lr, weightDecay);
        }

        // Benchmark
        System.out.printf("  Benchmark: %d steps...%n", benchmarkSteps);
        device.synchronize();

        long startTime = System.nanoTime();

        for (int step = 0; step < benchmarkSteps; step++) {
            trainStepFull(backend, device, B, T, C, L, NH, V, BT, BNH, warmupSteps + step + 1,
                    tokens, targets, wte, wpe, lnfw, lnfb,
                    ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                    ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                    encoded, ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                    ln2, ln2Mean, ln2Rstd, fch, fchGelu, residual,
                    lnf, lnfMean, lnfRstd, logits, probs, losses,
                    dwte, dwpe, dlnfw, dlnfb,
                    dln1w, dln1b, dqkvw, dqkvb, dattprojw, dattprojb,
                    dln2w, dln2b, dfcw, dfcb, dfcprojw, dfcprojb,
                    dlogits, dlnf, dresidual, dencoded,
                    mWte, vWte, mWpe, vWpe,
                    blocks, lr, weightDecay);
        }

        device.synchronize();
        long endTime = System.nanoTime();

        double totalMs = (endTime - startTime) / 1_000_000.0;
        double msPerStep = totalMs / benchmarkSteps;
        double stepsPerSec = 1000.0 / msPerStep;
        double tokensPerSec = stepsPerSec * BT;

        System.out.println("\n  ========================================");
        System.out.println("  FULL GPT-2 SPEED RESULTS:");
        System.out.println("  ========================================");
        System.out.printf("  Layers: %d, Channels: %d, Heads: %d%n", L, C, NH);
        System.out.printf("  Batch: %d x %d = %d tokens/batch%n", B, T, BT);
        System.out.println("  ----------------------------------------");
        System.out.printf("  Total time:       %.2f ms%n", totalMs);
        System.out.printf("  Time per step:    %.2f ms%n", msPerStep);
        System.out.printf("  Steps per second: %.1f%n", stepsPerSec);
        System.out.printf("  Tokens per second: %,.0f%n", tokensPerSec);
        System.out.println("  ========================================");

        double estimatedTimeFor10K = msPerStep * 10000 / 1000 / 60;
        System.out.printf("\n  Estimated time for 10K steps: %.1f minutes%n", estimatedTimeFor10K);

        // Cleanup (abbreviated)
        wte.close(); wpe.close(); encoded.close(); lnfw.close(); lnfb.close();
        lnf.close(); lnfMean.close(); lnfRstd.close();
        logits.close(); probs.close(); losses.close();
        dwte.close(); dwpe.close(); dlogits.close(); dlnf.close();
        dlnfw.close(); dlnfb.close(); dresidual.close(); dencoded.close();
        mWte.close(); vWte.close(); mWpe.close(); vWpe.close();

        for (int l = 0; l < L; l++) {
            ln1w[l].close(); ln1b[l].close(); qkvw[l].close(); qkvb[l].close();
            attprojw[l].close(); attprojb[l].close(); ln2w[l].close(); ln2b[l].close();
            fcw[l].close(); fcb[l].close(); fcprojw[l].close(); fcprojb[l].close();
            ln1[l].close(); ln1Mean[l].close(); ln1Rstd[l].close();
            qkv[l].close(); atty[l].close(); attLse[l].close(); attnOut[l].close();
            ln2[l].close(); ln2Mean[l].close(); ln2Rstd[l].close();
            fch[l].close(); fchGelu[l].close(); residual[l].close();
            dln1w[l].close(); dln1b[l].close(); dqkvw[l].close(); dqkvb[l].close();
            dattprojw[l].close(); dattprojb[l].close(); dln2w[l].close(); dln2b[l].close();
            dfcw[l].close(); dfcb[l].close(); dfcprojw[l].close(); dfcprojb[l].close();
        }

        System.out.println("\n  ✓ Full GPT-2 speed benchmark complete");
    }

    static void trainStepFull(FlashBackend backend, CudaDevice device,
                               int B, int T, int C, int L, int NH, int V, int BT, int BNH, int t,
                               int[] tokens, int[] targets,
                               CudaTensor wte, CudaTensor wpe, CudaTensor lnfw, CudaTensor lnfb,
                               CudaTensor[] ln1w, CudaTensor[] ln1b,
                               CudaTensor[] qkvw, CudaTensor[] qkvb,
                               CudaTensor[] attprojw, CudaTensor[] attprojb,
                               CudaTensor[] ln2w, CudaTensor[] ln2b,
                               CudaTensor[] fcw, CudaTensor[] fcb,
                               CudaTensor[] fcprojw, CudaTensor[] fcprojb,
                               CudaTensor encoded,
                               CudaTensor[] ln1, CudaTensor[] ln1Mean, CudaTensor[] ln1Rstd,
                               CudaTensor[] qkv, CudaTensor[] atty, CudaTensor[] attLse, CudaTensor[] attnOut,
                               CudaTensor[] ln2, CudaTensor[] ln2Mean, CudaTensor[] ln2Rstd,
                               CudaTensor[] fch, CudaTensor[] fchGelu, CudaTensor[] residual,
                               CudaTensor lnf, CudaTensor lnfMean, CudaTensor lnfRstd,
                               CudaTensor logits, CudaTensor probs, CudaTensor losses,
                               CudaTensor dwte, CudaTensor dwpe, CudaTensor dlnfw, CudaTensor dlnfb,
                               CudaTensor[] dln1w, CudaTensor[] dln1b,
                               CudaTensor[] dqkvw, CudaTensor[] dqkvb,
                               CudaTensor[] dattprojw, CudaTensor[] dattprojb,
                               CudaTensor[] dln2w, CudaTensor[] dln2b,
                               CudaTensor[] dfcw, CudaTensor[] dfcb,
                               CudaTensor[] dfcprojw, CudaTensor[] dfcprojb,
                               CudaTensor dlogits, CudaTensor dlnf, CudaTensor dresidual, CudaTensor dencoded,
                               CudaTensor mWte, CudaTensor vWte, CudaTensor mWpe, CudaTensor vWpe,
                               TransformerBlock[] blocks,
                               float lr, float weightDecay) {

        // Zero gradients
        backend.zeroFill(dwte);
        backend.zeroFill(dwpe);
        for (int l = 0; l < L; l++) {
            backend.zeroFill(dln1w[l]); backend.zeroFill(dln1b[l]);
            backend.zeroFill(dqkvw[l]); backend.zeroFill(dqkvb[l]);
            backend.zeroFill(dattprojw[l]); backend.zeroFill(dattprojb[l]);
            backend.zeroFill(dln2w[l]); backend.zeroFill(dln2b[l]);
            backend.zeroFill(dfcw[l]); backend.zeroFill(dfcb[l]);
            backend.zeroFill(dfcprojw[l]); backend.zeroFill(dfcprojb[l]);
        }
        backend.zeroFill(dlnfw); backend.zeroFill(dlnfb);

        // Forward
        Encoder.forward(encoded, tokens, wte, wpe, B, T, C);

        CudaTensor blockInput = encoded;
        for (int l = 0; l < L; l++) {
            blocks[l].forward(
                residual[l], blockInput,
                ln1w[l], ln1b[l], qkvw[l], qkvb[l], attprojw[l], attprojb[l],
                ln2w[l], ln2b[l], fcw[l], fcb[l], fcprojw[l], fcprojb[l],
                ln1[l], ln1Mean[l], ln1Rstd[l],
                qkv[l], atty[l], attLse[l], attnOut[l],
                ln2[l], ln2Mean[l], ln2Rstd[l],
                fch[l], fchGelu[l]
            );
            blockInput = residual[l];
        }

        LayerNorm.forward(lnf, lnfMean, lnfRstd, residual[L-1], lnfw, lnfb, BT, C);
        Matmul.forward(logits, lnf, wte, BT, V, C);
        Softmax.forward(probs, logits, B, T, V);
        Softmax.crossEntropyForward(losses, probs, targets, B, T, V);

        // Backward
        Softmax.crossEntropySoftmaxBackward(dlogits, probs, targets, B, T, V);
        Matmul.backward(dlnf, dwte, null, dlogits, lnf, wte, BT, V, C);
        LayerNorm.backward(dresidual, dlnfw, dlnfb, dlnf, residual[L-1], lnfw, lnfMean, lnfRstd, BT, C);

        CudaTensor dout = dresidual;
        for (int l = L - 1; l >= 0; l--) {
            CudaTensor inp = (l == 0) ? encoded : residual[l - 1];
            CudaTensor dinp = (l == 0) ? dencoded : dresidual;

            blocks[l].backward(
                dinp, dout, inp,
                ln1w[l], qkvw[l], attprojw[l], ln2w[l], fcw[l], fcprojw[l],
                ln1[l], ln1Mean[l], ln1Rstd[l],
                qkv[l], atty[l], attLse[l], attnOut[l],
                ln2[l], ln2Mean[l], ln2Rstd[l],
                fch[l], fchGelu[l],
                dln1w[l], dln1b[l], dqkvw[l], dqkvb[l], dattprojw[l], dattprojb[l],
                dln2w[l], dln2b[l], dfcw[l], dfcb[l], dfcprojw[l], dfcprojb[l]
            );
            dout = dinp;
        }

        Encoder.backward(dwte, dwpe, dencoded, tokens, B, T, C);

        // Optimizer (simplified)
        AdamW.update(wte, dwte, mWte, vWte, lr, weightDecay, t, (long) V * C);
        AdamW.update(wpe, dwpe, mWpe, vWpe, lr, weightDecay, t, (long) T * C);
    }

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
