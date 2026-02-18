package com.flashllm;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.data.*;
import com.flashllm.kernel.*;
import com.flashllm.model.TransformerBlock;

/**
 * Shakespeare Training - Train on real text data!
 *
 * This matches the llm.c example:
 * - Train on tinyshakespeare (input.txt)
 * - Character-level tokenization
 * - Full GPT-2 architecture
 * - Generate sample text after training
 *
 * Comparison target (llm.c on M3 Max CPU):
 * - ~1300 ms/step
 * - Loss: 5.35 → 3.97 after 40 steps
 */
public class ShakespeareTraining {

    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║         Shakespeare Training with flash-llm              ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝\n");

        try {
            // Find input.txt
            String dataPath = findDataPath();
            System.out.println("Data path: " + dataPath + "\n");

            FlashBackend backend = FlashBackend.getInstance();
            CudaDevice device = backend.getDevice();

            // Configuration (similar to llm.c example)
            int B = 4;      // batch size
            int T = 64;     // sequence length
            int C = 384;    // channels (smaller than 768 for testing)
            int L = 8;      // layers (smaller than 12 for testing)
            int NH = 8;     // heads
            int numSteps = 10000;

            // Load data
            TextDataLoader dataLoader = new TextDataLoader(dataPath, B, T, 0.9);
            int V = dataLoader.getVocabSize();

            // Print configuration
            System.out.println("========================================");
            System.out.println("[GPT-2]");
            System.out.printf("max_seq_len: %d%n", T);
            System.out.printf("vocab_size: %d%n", V);
            System.out.printf("num_layers: %d%n", L);
            System.out.printf("num_heads: %d%n", NH);
            System.out.printf("channels: %d%n", C);

            long numParams = calculateParams(V, T, C, L);
            System.out.printf("num_parameters: %,d%n", numParams);

            System.out.printf("train dataset num_batches: %d%n", dataLoader.getNumTrainBatches());
            System.out.printf("val dataset num_batches: %d%n", dataLoader.getNumValBatches());
            System.out.println("========================================\n");

            // Train
            train(backend, device, dataLoader, B, T, C, L, NH, V, numSteps);

            backend.close();

        } catch (Exception e) {
            System.err.println("Training failed:");
            e.printStackTrace();
            System.exit(1);
        }
    }

    static String findDataPath() {
        String[] paths = {
            "src/main/resources/input.txt",
            "src/main/java/com/flashllm/data/input.txt",
            "data/input.txt",
            "input.txt"
        };

        for (String path : paths) {
            java.io.File f = new java.io.File(path);
            if (f.exists()) {
                return path;
            }
        }

        throw new RuntimeException("Could not find input.txt! Please place it in src/main/resources/ or data/");
    }

    static long calculateParams(int V, int T, int C, int L) {
        long embParams = (long) V * C + (long) T * C;
        long lnParams = 2L * C;
        long attnParams = C * 3L * C + 3L * C + C * C + C;
        long mlpParams = C * 4L * C + 4L * C + 4L * C * C + C;
        long blockParams = 2 * lnParams + attnParams + mlpParams;
        return embParams + L * blockParams + lnParams;
    }

    static void train(FlashBackend backend, CudaDevice device,
                      TextDataLoader dataLoader,
                      int B, int T, int C, int L, int NH, int V,
                      int numSteps) {

        int BT = B * T;
        int BNH = B * NH;

        // ==================== ALLOCATE ====================
        // Parameters
        CudaTensor wte = backend.allocateF32(V * C);
        CudaTensor wpe = backend.allocateF32(T * C);
        CudaTensor[] ln1w = new CudaTensor[L], ln1b = new CudaTensor[L];
        CudaTensor[] qkvw = new CudaTensor[L], qkvb = new CudaTensor[L];
        CudaTensor[] attprojw = new CudaTensor[L], attprojb = new CudaTensor[L];
        CudaTensor[] ln2w = new CudaTensor[L], ln2b = new CudaTensor[L];
        CudaTensor[] fcw = new CudaTensor[L], fcb = new CudaTensor[L];
        CudaTensor[] fcprojw = new CudaTensor[L], fcprojb = new CudaTensor[L];
        CudaTensor lnfw = backend.allocateF32(C);
        CudaTensor lnfb = backend.allocateF32(C);

        // Activations
        CudaTensor encoded = backend.allocateF32(BT * C);
        CudaTensor[] ln1 = new CudaTensor[L], ln1Mean = new CudaTensor[L], ln1Rstd = new CudaTensor[L];
        CudaTensor[] qkv = new CudaTensor[L], atty = new CudaTensor[L], attLse = new CudaTensor[L], attnOut = new CudaTensor[L];
        CudaTensor[] ln2 = new CudaTensor[L], ln2Mean = new CudaTensor[L], ln2Rstd = new CudaTensor[L];
        CudaTensor[] fch = new CudaTensor[L], fchGelu = new CudaTensor[L], residual = new CudaTensor[L];
        CudaTensor lnf = backend.allocateF32(BT * C);
        CudaTensor lnfMean = backend.allocateF32(BT);
        CudaTensor lnfRstd = backend.allocateF32(BT);
        CudaTensor logits = backend.allocateF32(BT * V);
        CudaTensor probs = backend.allocateF32(BT * V);
        CudaTensor losses = backend.allocateF32(BT);

        // Gradients
        CudaTensor dwte = backend.allocateF32(V * C);
        CudaTensor dwpe = backend.allocateF32(T * C);
        CudaTensor[] dln1w = new CudaTensor[L], dln1b = new CudaTensor[L];
        CudaTensor[] dqkvw = new CudaTensor[L], dqkvb = new CudaTensor[L];
        CudaTensor[] dattprojw = new CudaTensor[L], dattprojb = new CudaTensor[L];
        CudaTensor[] dln2w = new CudaTensor[L], dln2b = new CudaTensor[L];
        CudaTensor[] dfcw = new CudaTensor[L], dfcb = new CudaTensor[L];
        CudaTensor[] dfcprojw = new CudaTensor[L], dfcprojb = new CudaTensor[L];
        CudaTensor dlnfw = backend.allocateF32(C);
        CudaTensor dlnfb = backend.allocateF32(C);
        CudaTensor dlogits = backend.allocateF32(BT * V);
        CudaTensor dlnf = backend.allocateF32(BT * C);
        CudaTensor dresidual = backend.allocateF32(BT * C);
        CudaTensor dencoded = backend.allocateF32(BT * C);

        // Optimizer state
        CudaTensor mWte = backend.allocateF32(V * C);
        CudaTensor vWte = backend.allocateF32(V * C);
        CudaTensor mWpe = backend.allocateF32(T * C);
        CudaTensor vWpe = backend.allocateF32(T * C);

        // Per-layer allocation
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

        // Initialize parameters
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

        // Create TransformerBlocks
        TransformerBlock[] blocks = new TransformerBlock[L];
        for (int l = 0; l < L; l++) {
            blocks[l] = new TransformerBlock(l, B, T, C, NH);
        }

        // Hyperparameters
        float lr = 3e-4f;
        float weightDecay = 0.01f;

        // ==================== VALIDATION LOSS (before training) ====================
        float valLoss = evaluateVal(backend, device, dataLoader, B, T, C, L, NH, V, BT, BNH,
                wte, wpe, lnfw, lnfb,
                ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                encoded, ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                ln2, ln2Mean, ln2Rstd, fch, fchGelu, residual,
                lnf, lnfMean, lnfRstd, logits, probs, losses, blocks, 10);

        System.out.printf("val loss %.6f%n", valLoss);

        // ==================== TRAINING LOOP ====================
        for (int step = 0; step < numSteps; step++) {
            long startTime = System.nanoTime();

            // Get batch
            int[][] batch = dataLoader.nextTrainBatch();
            int[] tokens = batch[0];
            int[] targets = batch[1];

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
            float loss = forward(backend, device, B, T, C, L, NH, V, BT, BNH,
                    tokens, targets, wte, wpe, lnfw, lnfb,
                    ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                    ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                    encoded, ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                    ln2, ln2Mean, ln2Rstd, fch, fchGelu, residual,
                    lnf, lnfMean, lnfRstd, logits, probs, losses, blocks);

            // Backward
            backward(backend, device, B, T, C, L, NH, V, BT, BNH,
                    tokens, targets, wte, wpe, lnfw, lnfb,
                    ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                    ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                    encoded, ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                    ln2, ln2Mean, ln2Rstd, fch, fchGelu, residual,
                    lnf, lnfMean, lnfRstd, logits, probs, losses,
                    dwte, dwpe, dlnfw, dlnfb,
                    dln1w, dln1b, dqkvw, dqkvb, dattprojw, dattprojb,
                    dln2w, dln2b, dfcw, dfcb, dfcprojw, dfcprojb,
                    dlogits, dlnf, dresidual, dencoded, blocks);

            // Optimizer
            int t = step + 1;
            AdamW.update(wte, dwte, mWte, vWte, lr, weightDecay, t, (long) V * C);
            AdamW.update(wpe, dwpe, mWpe, vWpe, lr, weightDecay, t, (long) T * C);

            device.synchronize();
            long endTime = System.nanoTime();
            double ms = (endTime - startTime) / 1_000_000.0;

            // Log progress
            if (step % 200 == 0 || step == numSteps - 1) {
                System.out.printf("step %d: train loss %.6f (took %.3f ms)%n", step, loss, ms);
            }
        }

        // ==================== VALIDATION LOSS (after training) ====================
        valLoss = evaluateVal(backend, device, dataLoader, B, T, C, L, NH, V, BT, BNH,
                wte, wpe, lnfw, lnfb,
                ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                encoded, ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                ln2, ln2Mean, ln2Rstd, fch, fchGelu, residual,
                lnf, lnfMean, lnfRstd, logits, probs, losses, blocks, 10);

        System.out.printf("val loss %.6f%n", valLoss);

        // ==================== GENERATE TEXT ====================
        System.out.println("generating:");
        System.out.println("---");
        String generated = generate(backend, device, dataLoader.getTokenizer(),
                B, T, C, L, NH, V, BT, BNH,
                wte, wpe, lnfw, lnfb,
                ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                encoded, ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                ln2, ln2Mean, ln2Rstd, fch, fchGelu, residual,
                lnf, lnfMean, lnfRstd, logits, probs, blocks,
                500);  // Generate 500 characters
        System.out.println(generated);
        System.out.println("---");

        // Cleanup (abbreviated)
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
    }

    static float forward(FlashBackend backend, CudaDevice device,
                          int B, int T, int C, int L, int NH, int V, int BT, int BNH,
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
                          TransformerBlock[] blocks) {

        // Encoder
        Encoder.forward(encoded, tokens, wte, wpe, B, T, C);

        // Transformer blocks
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

        // Final LN
        LayerNorm.forward(lnf, lnfMean, lnfRstd, residual[L-1], lnfw, lnfb, BT, C);

        // Output projection
        Matmul.forward(logits, lnf, wte, BT, V, C);

        // Softmax + Loss
        Softmax.forward(probs, logits, B, T, V);
        Softmax.crossEntropyForward(losses, probs, targets, B, T, V);

        return Softmax.meanLoss(losses, B, T);
    }

    static void backward(FlashBackend backend, CudaDevice device,
                          int B, int T, int C, int L, int NH, int V, int BT, int BNH,
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
                          TransformerBlock[] blocks) {

        // Loss backward
        Softmax.crossEntropySoftmaxBackward(dlogits, probs, targets, B, T, V);

        // Output projection backward
        Matmul.backward(dlnf, dwte, null, dlogits, lnf, wte, BT, V, C);

        // Final LN backward
        LayerNorm.backward(dresidual, dlnfw, dlnfb, dlnf, residual[L-1], lnfw, lnfMean, lnfRstd, BT, C);

        // Transformer blocks backward
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

        // Encoder backward
        Encoder.backward(dwte, dwpe, dencoded, tokens, B, T, C);
    }

    static float evaluateVal(FlashBackend backend, CudaDevice device,
                              TextDataLoader dataLoader,
                              int B, int T, int C, int L, int NH, int V, int BT, int BNH,
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
                              TransformerBlock[] blocks, int numBatches) {

        float totalLoss = 0;
        for (int i = 0; i < numBatches; i++) {
            int[][] batch = dataLoader.nextValBatch();
            float loss = forward(backend, device, B, T, C, L, NH, V, BT, BNH,
                    batch[0], batch[1], wte, wpe, lnfw, lnfb,
                    ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                    ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                    encoded, ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                    ln2, ln2Mean, ln2Rstd, fch, fchGelu, residual,
                    lnf, lnfMean, lnfRstd, logits, probs, losses, blocks);
            totalLoss += loss;
        }
        return totalLoss / numBatches;
    }

    static String generate(FlashBackend backend, CudaDevice device, CharTokenizer tokenizer,
                           int B, int T, int C, int L, int NH, int V, int BT, int BNH,
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
                           CudaTensor logits, CudaTensor probs,
                           TransformerBlock[] blocks,
                           int maxTokens) {

        StringBuilder sb = new StringBuilder();
        java.util.Random rng = new java.util.Random();

        // Start with newline
        int[] context = new int[T];
        context[0] = tokenizer.getId('\n');
        int contextLen = 1;

        for (int i = 0; i < maxTokens; i++) {
            // Pad context to T
            int[] tokens = new int[BT];
            int[] targets = new int[BT];  // Dummy
            int start = Math.max(0, contextLen - T);
            for (int j = 0; j < Math.min(contextLen, T); j++) {
                tokens[j] = context[start + j];
            }

            // Forward (only need logits)
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

            // Sample from last position
            int pos = Math.min(contextLen, T) - 1;
            float[] probsData = probs.toFloatArray();
            int offset = pos * V;

            // Sample with temperature
            int nextToken = sampleFromProbs(probsData, offset, V, rng, 0.8f);

            // Append to context
            if (contextLen < T) {
                context[contextLen++] = nextToken;
            } else {
                // Shift context
                System.arraycopy(context, 1, context, 0, T - 1);
                context[T - 1] = nextToken;
            }

            sb.append(tokenizer.getChar(nextToken));
        }

        return sb.toString();
    }

    static int sampleFromProbs(float[] probs, int offset, int V, java.util.Random rng, float temperature) {
        // Apply temperature
        double[] adjusted = new double[V];
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < V; i++) {
            adjusted[i] = Math.log(Math.max(probs[offset + i], 1e-10)) / temperature;
            maxLogit = Math.max(maxLogit, adjusted[i]);
        }

        // Softmax
        double sum = 0;
        for (int i = 0; i < V; i++) {
            adjusted[i] = Math.exp(adjusted[i] - maxLogit);
            sum += adjusted[i];
        }
        for (int i = 0; i < V; i++) {
            adjusted[i] /= sum;
        }

        // Sample
        double r = rng.nextDouble();
        double cumsum = 0;
        for (int i = 0; i < V; i++) {
            cumsum += adjusted[i];
            if (r < cumsum) {
                return i;
            }
        }
        return V - 1;
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
