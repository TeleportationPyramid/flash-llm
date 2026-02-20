package com.flashllm;

import com.flashllm.training.Generate;
import com.flashllm.tokenizer.LRScheduler;
import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.kernel.*;
import com.flashllm.model.*;
import com.flashllm.tokenizer.GPT2TokenizerLoader;
import com.flashllm.training.*;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.util.*;

/**
 * GPT-2 Pretrained Fine-tuning on Shakespeare.
 *
 * This loads the GPT-2 124M pretrained weights from llm.c format
 * and fine-tunes on tinyshakespeare, matching the llm.c demo.
 *
 * Phase 1.5 Features:
 * - Full parameter training with AdamW (124M parameters)
 * - Optional LR Schedule (warmup + cosine decay)
 * - Optional Gradient clipping
 * - Text generation with top-k/top-p sampling
 *
 * Expected output (similar to llm.c):
 * - val loss ~5.25 (before training)
 * - After 40 steps: train loss ~4.0
 * - val loss ~4.13 (after training)
 * - Generated text should be Shakespeare-like
 */
public class GPT2PretrainedTraining {

    // ==================== TRAINING CONFIGURATION ====================

    // Basic training config
    private static final int BATCH_SIZE = 4;
    private static final int SEQ_LEN = 64;
    private static final int NUM_STEPS = 40;

    // Optimizer config (matching llm.c defaults)
    private static final float LEARNING_RATE = 1e-4f;      // Base learning rate
    private static final float WEIGHT_DECAY = 0.0f;        // L2 regularization
    private static final float BETA1 = 0.9f;               // Adam beta1
    private static final float BETA2 = 0.999f;             // Adam beta2
    private static final float EPS = 1e-8f;                // Adam epsilon

    // LR Schedule config (set USE_LR_SCHEDULE = true to enable)
    private static final boolean USE_LR_SCHEDULE = false;  // Enable warmup + cosine decay
    private static final float MAX_LR = 6e-4f;             // Peak learning rate
    private static final float MIN_LR = 6e-5f;             // Minimum learning rate
    private static final int WARMUP_STEPS = 10;            // Linear warmup steps

    // Gradient clipping config (set GRAD_CLIP_NORM > 0 to enable)
    private static final float GRAD_CLIP_NORM = 0.0f;      // 0 = disabled, e.g., 1.0 to enable

    // Generation config
    private static final int GEN_MAX_TOKENS = 64;
    private static final int GEN_TOP_K = 50;               // Top-k sampling (0 = disabled)
    private static final float GEN_TOP_P = 0.95f;          // Nucleus sampling threshold
    private static final float GEN_TEMPERATURE = 0.8f;     // Temperature for sampling

    // ==================== MAIN ====================

    public static void main(String[] args) {
        System.out.println("================================================================");
        System.out.println("     GPT-2 124M Pretrained Fine-tuning on Shakespeare");
        System.out.println("     Phase 1.5: Full Training with Advanced Features");
        System.out.println("================================================================\n");

        try {
            // Find files
            String weightsPath = findFile("gpt2_124M.bin", new String[]{
                    "src/main/resources/gpt2/gpt2_124M.bin",
                    "gpt2_124M.bin",
                    "models/gpt2_124M.bin"
            });

            String tokenizerPath = findFile("gpt2_tokenizer.bin", new String[]{
                    "src/main/resources/gpt2/gpt2_tokenizer.bin",
                    "gpt2_tokenizer.bin",
                    "models/gpt2_tokenizer.bin"
            });

            String trainDataPath = findFile("tiny_shakespeare_train.bin", new String[]{
                    "src/main/resources/gpt2/tiny_shakespeare_train.bin",
                    "tiny_shakespeare_train.bin",
                    "dev/data/tinyshakespeare/tiny_shakespeare_train.bin"
            });

            String valDataPath = findFile("tiny_shakespeare_val.bin", new String[]{
                    "src/main/resources/gpt2/tiny_shakespeare_val.bin",
                    "tiny_shakespeare_val.bin",
                    "dev/data/tinyshakespeare/tiny_shakespeare_val.bin"
            });

            // Load weights
            GPT2WeightLoader weightLoader = new GPT2WeightLoader();
            weightLoader.load(weightsPath);

            // Load tokenizer
            GPT2TokenizerLoader tokenizer = new GPT2TokenizerLoader();
            tokenizer.load(tokenizerPath);

            // Load data
            int[] trainTokens = loadTokens(trainDataPath);
            int[] valTokens = loadTokens(valDataPath);
            System.out.printf("Train tokens: %,d%n", trainTokens.length);
            System.out.printf("Val tokens: %,d%n", valTokens.length);

            // Initialize backend
            FlashBackend backend = FlashBackend.getInstance();
            CudaDevice device = backend.getDevice();

            // Config
            int B = BATCH_SIZE;
            int T = SEQ_LEN;
            int C = weightLoader.channels;
            int L = weightLoader.numLayers;
            int NH = weightLoader.numHeads;
            int V = weightLoader.vocabSize;
            int Vp = weightLoader.paddedVocabSize;
            int maxT = weightLoader.maxT;

            // Print configuration
            System.out.println("\n========================================");
            System.out.println("Training Configuration:");
            System.out.printf("  batch_size: %d%n", B);
            System.out.printf("  seq_len: %d%n", T);
            System.out.printf("  num_steps: %d%n", NUM_STEPS);
            System.out.printf("  learning_rate: %.2e%n", LEARNING_RATE);
            System.out.printf("  weight_decay: %.2f%n", WEIGHT_DECAY);
            if (USE_LR_SCHEDULE) {
                System.out.printf("  lr_schedule: warmup=%d steps, max_lr=%.2e, min_lr=%.2e%n",
                        WARMUP_STEPS, MAX_LR, MIN_LR);
            } else {
                System.out.println("  lr_schedule: disabled (constant LR)");
            }
            if (GRAD_CLIP_NORM > 0) {
                System.out.printf("  grad_clip_norm: %.2f%n", GRAD_CLIP_NORM);
            } else {
                System.out.println("  grad_clip_norm: disabled");
            }
            System.out.printf("  generation: top_k=%d, top_p=%.2f, temp=%.2f%n",
                    GEN_TOP_K, GEN_TOP_P, GEN_TEMPERATURE);
            System.out.println("========================================\n");

            // Train
            train(backend, device, weightLoader, tokenizer,
                    trainTokens, valTokens, B, T, C, L, NH, V, Vp, NUM_STEPS);

            backend.close();

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
        throw new RuntimeException("Could not find " + name + "!\n" +
                "Please download from llm.c: ./dev/download_starter_pack.sh\n" +
                "Or run: python train_gpt2.py");
    }

    static int[] loadTokens(String path) throws IOException {
        try (RandomAccessFile raf = new RandomAccessFile(path, "r");
             FileChannel channel = raf.getChannel()) {

            ByteBuffer headerBuf = ByteBuffer.allocate(256 * 4);
            headerBuf.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(headerBuf);
            headerBuf.flip();

            int magic = headerBuf.getInt();
            int version = headerBuf.getInt();
            System.out.printf("Data file: magic=%d, version=%d%n", magic, version);

            long fileSize = channel.size();
            long headerSize = 256 * 4;
            long dataSize = fileSize - headerSize;

            int numTokens = (int) (dataSize / 2);
            int[] tokens = new int[numTokens];

            ByteBuffer dataBuf = ByteBuffer.allocate((int) dataSize);
            dataBuf.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(dataBuf);
            dataBuf.flip();

            ShortBuffer shortBuf = dataBuf.asShortBuffer();
            for (int i = 0; i < numTokens; i++) {
                tokens[i] = shortBuf.get() & 0xFFFF;
            }

            return tokens;
        }
    }

    static float min(float[] arr) {
        float m = Float.MAX_VALUE;
        for (float v : arr) m = Math.min(m, v);
        return m;
    }

    static float max(float[] arr) {
        float m = -Float.MAX_VALUE;
        for (float v : arr) m = Math.max(m, v);
        return m;
    }

    // ==================== PART 2: train() 函數 ====================

    static void train(FlashBackend backend, CudaDevice device,
                      GPT2WeightLoader weightLoader, GPT2TokenizerLoader tokenizer,
                      int[] trainTokens, int[] valTokens,
                      int B, int T, int C, int L, int NH, int V, int Vp, int numSteps) {

        int BT = B * T;
        int BNH = B * NH;
        int maxT = weightLoader.maxT;

        // ==================== ALLOCATE PARAMETERS ====================
        CudaTensor wte = backend.allocateF32(Vp * C);
        CudaTensor wpe = backend.allocateF32(maxT * C);
        CudaTensor[] ln1w = new CudaTensor[L], ln1b = new CudaTensor[L];
        CudaTensor[] qkvw = new CudaTensor[L], qkvb = new CudaTensor[L];
        CudaTensor[] attprojw = new CudaTensor[L], attprojb = new CudaTensor[L];
        CudaTensor[] ln2w = new CudaTensor[L], ln2b = new CudaTensor[L];
        CudaTensor[] fcw = new CudaTensor[L], fcb = new CudaTensor[L];
        CudaTensor[] fcprojw = new CudaTensor[L], fcprojb = new CudaTensor[L];
        CudaTensor lnfw = backend.allocateF32(C);
        CudaTensor lnfb = backend.allocateF32(C);

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

        // ==================== LOAD PRETRAINED WEIGHTS ====================
        System.out.println("Loading pretrained weights to GPU...");

        float[] wteData = weightLoader.getWte();
        float[] wpeData = weightLoader.getWpe();
        System.out.printf("wte stats: len=%d, min=%.6f, max=%.6f, first5=[%.4f,%.4f,%.4f,%.4f,%.4f]%n",
                wteData.length, min(wteData), max(wteData), wteData[0], wteData[1], wteData[2], wteData[3], wteData[4]);
        System.out.printf("wpe stats: len=%d, min=%.6f, max=%.6f, first5=[%.4f,%.4f,%.4f,%.4f,%.4f]%n",
                wpeData.length, min(wpeData), max(wpeData), wpeData[0], wpeData[1], wpeData[2], wpeData[3], wpeData[4]);

        float[] qkvw0 = weightLoader.getQkvw(0);
        System.out.printf("qkvw[0] stats: len=%d, min=%.6f, max=%.6f, first5=[%.4f,%.4f,%.4f,%.4f,%.4f]%n",
                qkvw0.length, min(qkvw0), max(qkvw0), qkvw0[0], qkvw0[1], qkvw0[2], qkvw0[3], qkvw0[4]);

        float[] ln1w0 = weightLoader.getLn1w(0);
        System.out.printf("ln1w[0] stats: len=%d, min=%.6f, max=%.6f, first5=[%.4f,%.4f,%.4f,%.4f,%.4f]%n",
                ln1w0.length, min(ln1w0), max(ln1w0), ln1w0[0], ln1w0[1], ln1w0[2], ln1w0[3], ln1w0[4]);

        float[] lnfwData = weightLoader.getLnfw();
        System.out.printf("lnfw stats: len=%d, min=%.6f, max=%.6f, first5=[%.4f,%.4f,%.4f,%.4f,%.4f]%n",
                lnfwData.length, min(lnfwData), max(lnfwData), lnfwData[0], lnfwData[1], lnfwData[2], lnfwData[3], lnfwData[4]);

        int zeroCount = 0, nanCount = 0;
        for (float v : wteData) {
            if (v == 0) zeroCount++;
            if (Float.isNaN(v)) nanCount++;
        }
        System.out.printf("wte: %d zeros, %d NaNs out of %d%n", zeroCount, nanCount, wteData.length);

        TensorUtils.copyFromHost(device, wteData, wte);
        TensorUtils.copyFromHost(device, wpeData, wpe);

        for (int l = 0; l < L; l++) {
            TensorUtils.copyFromHost(device, weightLoader.getLn1w(l), ln1w[l]);
            TensorUtils.copyFromHost(device, weightLoader.getLn1b(l), ln1b[l]);
            TensorUtils.copyFromHost(device, weightLoader.getQkvw(l), qkvw[l]);
            TensorUtils.copyFromHost(device, weightLoader.getQkvb(l), qkvb[l]);
            TensorUtils.copyFromHost(device, weightLoader.getAttprojw(l), attprojw[l]);
            TensorUtils.copyFromHost(device, weightLoader.getAttprojb(l), attprojb[l]);
            TensorUtils.copyFromHost(device, weightLoader.getLn2w(l), ln2w[l]);
            TensorUtils.copyFromHost(device, weightLoader.getLn2b(l), ln2b[l]);
            TensorUtils.copyFromHost(device, weightLoader.getFcw(l), fcw[l]);
            TensorUtils.copyFromHost(device, weightLoader.getFcb(l), fcb[l]);
            TensorUtils.copyFromHost(device, weightLoader.getFcprojw(l), fcprojw[l]);
            TensorUtils.copyFromHost(device, weightLoader.getFcprojb(l), fcprojb[l]);
        }

        TensorUtils.copyFromHost(device, weightLoader.getLnfw(), lnfw);
        TensorUtils.copyFromHost(device, weightLoader.getLnfb(), lnfb);
        System.out.println("Pretrained weights loaded!\n");

        // ==================== ALLOCATE ACTIVATIONS ====================
        CudaTensor encoded = backend.allocateF32(BT * C);
        CudaTensor[] ln1 = new CudaTensor[L], ln1Mean = new CudaTensor[L], ln1Rstd = new CudaTensor[L];
        CudaTensor[] qkv = new CudaTensor[L], atty = new CudaTensor[L], attLse = new CudaTensor[L], attnOut = new CudaTensor[L];
        CudaTensor[] ln2 = new CudaTensor[L], ln2Mean = new CudaTensor[L], ln2Rstd = new CudaTensor[L];
        CudaTensor[] fch = new CudaTensor[L], fchGelu = new CudaTensor[L], residual = new CudaTensor[L];
        CudaTensor lnf = backend.allocateF32(BT * C);
        CudaTensor lnfMean = backend.allocateF32(BT);
        CudaTensor lnfRstd = backend.allocateF32(BT);
        CudaTensor logits = backend.allocateF32(BT * Vp);
        CudaTensor probs = backend.allocateF32(BT * Vp);
        CudaTensor losses = backend.allocateF32(BT);

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

        // ==================== ALLOCATE GRADIENTS ====================
        CudaTensor dwte = backend.allocateF32(Vp * C);
        CudaTensor dwpe = backend.allocateF32(maxT * C);
        CudaTensor[] dln1w = new CudaTensor[L], dln1b = new CudaTensor[L];
        CudaTensor[] dqkvw = new CudaTensor[L], dqkvb = new CudaTensor[L];
        CudaTensor[] dattprojw = new CudaTensor[L], dattprojb = new CudaTensor[L];
        CudaTensor[] dln2w = new CudaTensor[L], dln2b = new CudaTensor[L];
        CudaTensor[] dfcw = new CudaTensor[L], dfcb = new CudaTensor[L];
        CudaTensor[] dfcprojw = new CudaTensor[L], dfcprojb = new CudaTensor[L];
        CudaTensor dlnfw = backend.allocateF32(C);
        CudaTensor dlnfb = backend.allocateF32(C);
        CudaTensor dlogits = backend.allocateF32(BT * Vp);
        CudaTensor dlnf = backend.allocateF32(BT * C);
        CudaTensor dresidual = backend.allocateF32(BT * C);
        CudaTensor dencoded = backend.allocateF32(BT * C);

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

        // ==================== ALLOCATE OPTIMIZER STATE ====================
        CudaTensor mWte = backend.allocateF32(Vp * C);
        CudaTensor vWte = backend.allocateF32(Vp * C);
        CudaTensor mWpe = backend.allocateF32(maxT * C);
        CudaTensor vWpe = backend.allocateF32(maxT * C);
        backend.zeroFill(mWte); backend.zeroFill(vWte);
        backend.zeroFill(mWpe); backend.zeroFill(vWpe);

        CudaTensor mLnfw = backend.allocateF32(C);
        CudaTensor vLnfw = backend.allocateF32(C);
        CudaTensor mLnfb = backend.allocateF32(C);
        CudaTensor vLnfb = backend.allocateF32(C);
        backend.zeroFill(mLnfw); backend.zeroFill(vLnfw);
        backend.zeroFill(mLnfb); backend.zeroFill(vLnfb);

        CudaTensor[] mLn1w = new CudaTensor[L], vLn1w = new CudaTensor[L];
        CudaTensor[] mLn1b = new CudaTensor[L], vLn1b = new CudaTensor[L];
        CudaTensor[] mQkvw = new CudaTensor[L], vQkvw = new CudaTensor[L];
        CudaTensor[] mQkvb = new CudaTensor[L], vQkvb = new CudaTensor[L];
        CudaTensor[] mAttprojw = new CudaTensor[L], vAttprojw = new CudaTensor[L];
        CudaTensor[] mAttprojb = new CudaTensor[L], vAttprojb = new CudaTensor[L];
        CudaTensor[] mLn2w = new CudaTensor[L], vLn2w = new CudaTensor[L];
        CudaTensor[] mLn2b = new CudaTensor[L], vLn2b = new CudaTensor[L];
        CudaTensor[] mFcw = new CudaTensor[L], vFcw = new CudaTensor[L];
        CudaTensor[] mFcb = new CudaTensor[L], vFcb = new CudaTensor[L];
        CudaTensor[] mFcprojw = new CudaTensor[L], vFcprojw = new CudaTensor[L];
        CudaTensor[] mFcprojb = new CudaTensor[L], vFcprojb = new CudaTensor[L];

        for (int l = 0; l < L; l++) {
            mLn1w[l] = backend.allocateF32(C); vLn1w[l] = backend.allocateF32(C);
            mLn1b[l] = backend.allocateF32(C); vLn1b[l] = backend.allocateF32(C);
            mQkvw[l] = backend.allocateF32(C * 3 * C); vQkvw[l] = backend.allocateF32(C * 3 * C);
            mQkvb[l] = backend.allocateF32(3 * C); vQkvb[l] = backend.allocateF32(3 * C);
            mAttprojw[l] = backend.allocateF32(C * C); vAttprojw[l] = backend.allocateF32(C * C);
            mAttprojb[l] = backend.allocateF32(C); vAttprojb[l] = backend.allocateF32(C);
            mLn2w[l] = backend.allocateF32(C); vLn2w[l] = backend.allocateF32(C);
            mLn2b[l] = backend.allocateF32(C); vLn2b[l] = backend.allocateF32(C);
            mFcw[l] = backend.allocateF32(C * 4 * C); vFcw[l] = backend.allocateF32(C * 4 * C);
            mFcb[l] = backend.allocateF32(4 * C); vFcb[l] = backend.allocateF32(4 * C);
            mFcprojw[l] = backend.allocateF32(4 * C * C); vFcprojw[l] = backend.allocateF32(4 * C * C);
            mFcprojb[l] = backend.allocateF32(C); vFcprojb[l] = backend.allocateF32(C);

            backend.zeroFill(mLn1w[l]); backend.zeroFill(vLn1w[l]);
            backend.zeroFill(mLn1b[l]); backend.zeroFill(vLn1b[l]);
            backend.zeroFill(mQkvw[l]); backend.zeroFill(vQkvw[l]);
            backend.zeroFill(mQkvb[l]); backend.zeroFill(vQkvb[l]);
            backend.zeroFill(mAttprojw[l]); backend.zeroFill(vAttprojw[l]);
            backend.zeroFill(mAttprojb[l]); backend.zeroFill(vAttprojb[l]);
            backend.zeroFill(mLn2w[l]); backend.zeroFill(vLn2w[l]);
            backend.zeroFill(mLn2b[l]); backend.zeroFill(vLn2b[l]);
            backend.zeroFill(mFcw[l]); backend.zeroFill(vFcw[l]);
            backend.zeroFill(mFcb[l]); backend.zeroFill(vFcb[l]);
            backend.zeroFill(mFcprojw[l]); backend.zeroFill(vFcprojw[l]);
            backend.zeroFill(mFcprojb[l]); backend.zeroFill(vFcprojb[l]);
        }

        // Create TransformerBlocks
        TransformerBlock[] blocks = new TransformerBlock[L];
        for (int l = 0; l < L; l++) {
            blocks[l] = new TransformerBlock(l, B, T, C, NH);
        }

        // Data loader state
        int trainPos = 0;
        Random rng = new Random(42);

        // ==================== Phase 1.5: LR SCHEDULER ====================
        LRScheduler lrScheduler;
        if (USE_LR_SCHEDULE) {
            lrScheduler = LRScheduler.warmupCosine(MAX_LR, MIN_LR, WARMUP_STEPS, numSteps);
        } else {
            lrScheduler = LRScheduler.constant(LEARNING_RATE);
        }

        // ==================== Phase 1.5: COLLECT GRADIENTS FOR CLIPPING ====================
        List<CudaTensor> allGradients = new ArrayList<>();
        allGradients.add(dwte);
        allGradients.add(dwpe);
        allGradients.add(dlnfw);
        allGradients.add(dlnfb);
        for (int l = 0; l < L; l++) {
            allGradients.add(dln1w[l]); allGradients.add(dln1b[l]);
            allGradients.add(dqkvw[l]); allGradients.add(dqkvb[l]);
            allGradients.add(dattprojw[l]); allGradients.add(dattprojb[l]);
            allGradients.add(dln2w[l]); allGradients.add(dln2b[l]);
            allGradients.add(dfcw[l]); allGradients.add(dfcb[l]);
            allGradients.add(dfcprojw[l]); allGradients.add(dfcprojb[l]);
        }

        // ==================== VALIDATION LOSS (before training) ====================
        float valLoss = evaluateVal(backend, device, valTokens, B, T, C, L, NH, V, Vp, BT, BNH,
                wte, wpe, lnfw, lnfb, ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                encoded, ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                ln2, ln2Mean, ln2Rstd, fch, fchGelu, residual,
                lnf, lnfMean, lnfRstd, logits, probs, losses, blocks, 10);
        System.out.printf("val loss %.6f%n", valLoss);

        // ==================== TRAINING LOOP ====================
        for (int step = 0; step < numSteps; step++) {
            long startTime = System.nanoTime();

            // Phase 1.5: Get learning rate from scheduler
            float lr = lrScheduler.getLr(step);

            // Get batch
            int[] tokens = new int[BT];
            int[] targets = new int[BT];
            for (int i = 0; i < BT; i++) {
                int pos = (trainPos + i) % (trainTokens.length - 1);
                tokens[i] = trainTokens[pos];
                targets[i] = trainTokens[pos + 1];
            }
            trainPos = (trainPos + BT) % (trainTokens.length - BT - 1);

            // Zero gradients
            backend.zeroFill(dwte);
            backend.zeroFill(dwpe);
            backend.zeroFill(dlnfw);
            backend.zeroFill(dlnfb);
            for (int l = 0; l < L; l++) {
                backend.zeroFill(dln1w[l]); backend.zeroFill(dln1b[l]);
                backend.zeroFill(dqkvw[l]); backend.zeroFill(dqkvb[l]);
                backend.zeroFill(dattprojw[l]); backend.zeroFill(dattprojb[l]);
                backend.zeroFill(dln2w[l]); backend.zeroFill(dln2b[l]);
                backend.zeroFill(dfcw[l]); backend.zeroFill(dfcb[l]);
                backend.zeroFill(dfcprojw[l]); backend.zeroFill(dfcprojb[l]);
            }

            // Forward
            float loss = forward(backend, device, B, T, C, L, NH, V, Vp, BT, BNH,
                    tokens, targets, wte, wpe, lnfw, lnfb,
                    ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                    ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                    encoded, ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                    ln2, ln2Mean, ln2Rstd, fch, fchGelu, residual,
                    lnf, lnfMean, lnfRstd, logits, probs, losses, blocks);

            // Backward
            backward(backend, device, B, T, C, L, NH, V, Vp, BT, BNH,
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

            // Phase 1.5: Gradient clipping (optional)
            float gradNorm = 0;
            if (GRAD_CLIP_NORM > 0) {
                gradNorm = GradientClipper.clipByGlobalNorm(allGradients, GRAD_CLIP_NORM, device);
            }

            // Optimizer - update ALL parameters
            int t = step + 1;

            AdamW.update(wte, dwte, mWte, vWte, lr, WEIGHT_DECAY, t, (long) Vp * C);
            AdamW.update(wpe, dwpe, mWpe, vWpe, lr, WEIGHT_DECAY, t, (long) maxT * C);
            AdamW.update(lnfw, dlnfw, mLnfw, vLnfw, lr, WEIGHT_DECAY, t, (long) C);
            AdamW.update(lnfb, dlnfb, mLnfb, vLnfb, lr, WEIGHT_DECAY, t, (long) C);

            for (int l = 0; l < L; l++) {
                AdamW.update(ln1w[l], dln1w[l], mLn1w[l], vLn1w[l], lr, WEIGHT_DECAY, t, (long) C);
                AdamW.update(ln1b[l], dln1b[l], mLn1b[l], vLn1b[l], lr, WEIGHT_DECAY, t, (long) C);
                AdamW.update(qkvw[l], dqkvw[l], mQkvw[l], vQkvw[l], lr, WEIGHT_DECAY, t, (long) C * 3 * C);
                AdamW.update(qkvb[l], dqkvb[l], mQkvb[l], vQkvb[l], lr, WEIGHT_DECAY, t, (long) 3 * C);
                AdamW.update(attprojw[l], dattprojw[l], mAttprojw[l], vAttprojw[l], lr, WEIGHT_DECAY, t, (long) C * C);
                AdamW.update(attprojb[l], dattprojb[l], mAttprojb[l], vAttprojb[l], lr, WEIGHT_DECAY, t, (long) C);
                AdamW.update(ln2w[l], dln2w[l], mLn2w[l], vLn2w[l], lr, WEIGHT_DECAY, t, (long) C);
                AdamW.update(ln2b[l], dln2b[l], mLn2b[l], vLn2b[l], lr, WEIGHT_DECAY, t, (long) C);
                AdamW.update(fcw[l], dfcw[l], mFcw[l], vFcw[l], lr, WEIGHT_DECAY, t, (long) C * 4 * C);
                AdamW.update(fcb[l], dfcb[l], mFcb[l], vFcb[l], lr, WEIGHT_DECAY, t, (long) 4 * C);
                AdamW.update(fcprojw[l], dfcprojw[l], mFcprojw[l], vFcprojw[l], lr, WEIGHT_DECAY, t, (long) 4 * C * C);
                AdamW.update(fcprojb[l], dfcprojb[l], mFcprojb[l], vFcprojb[l], lr, WEIGHT_DECAY, t, (long) C);
            }

            device.synchronize();
            long endTime = System.nanoTime();
            double ms = (endTime - startTime) / 1_000_000.0;

            // Print with optional LR and grad_norm
            if (USE_LR_SCHEDULE || GRAD_CLIP_NORM > 0) {
                if (GRAD_CLIP_NORM > 0) {
                    System.out.printf("step %d: train loss %.6f, lr %.2e, grad_norm %.4f (took %.3f ms)%n",
                            step, loss, lr, gradNorm, ms);
                } else {
                    System.out.printf("step %d: train loss %.6f, lr %.2e (took %.3f ms)%n",
                            step, loss, lr, ms);
                }
            } else {
                System.out.printf("step %d: train loss %.6f (took %.3f ms)%n", step, loss, ms);
            }
        }

        // ==================== VALIDATION LOSS (after training) ====================
        valLoss = evaluateVal(backend, device, valTokens, B, T, C, L, NH, V, Vp, BT, BNH,
                wte, wpe, lnfw, lnfb, ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                encoded, ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                ln2, ln2Mean, ln2Rstd, fch, fchGelu, residual,
                lnf, lnfMean, lnfRstd, logits, probs, losses, blocks, 10);
        System.out.printf("val loss %.6f%n", valLoss);

        // ==================== GENERATE TEXT (Phase 1.5: top-k/top-p) ====================
        System.out.printf("%ngenerating (top-k=%d, top-p=%.2f, temp=%.2f):%n",
                GEN_TOP_K, GEN_TOP_P, GEN_TEMPERATURE);
        System.out.println("---");

        Generate.SamplingConfig samplingConfig = Generate.SamplingConfig.topKTopP(
                GEN_TOP_K, GEN_TOP_P, GEN_TEMPERATURE
        );

        String generated = generate(backend, device, tokenizer, B, T, C, L, NH, V, Vp, BT, BNH,
                wte, wpe, lnfw, lnfb, ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                encoded, ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                ln2, ln2Mean, ln2Rstd, fch, fchGelu, residual,
                lnf, lnfMean, lnfRstd, logits, probs, blocks,
                GEN_MAX_TOKENS, samplingConfig, rng);
        System.out.println(generated);
        System.out.println("---");

        System.out.println("\nTraining complete!");
    }

    // ==================== PART 3: forward, backward, evaluateVal, generate ====================

    static boolean debugPrinted = false;

    static float forward(FlashBackend backend, CudaDevice device,
                         int B, int T, int C, int L, int NH, int V, int Vp, int BT, int BNH,
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
        Matmul.forwardTransposed(logits, lnf, wte, BT, Vp, C);
        Softmax.forward(probs, logits, B, T, V, Vp);
        Softmax.crossEntropyForward(losses, probs, targets, B, T, V, Vp);

        return Softmax.meanLoss(losses, B, T);
    }

    static void backward(FlashBackend backend, CudaDevice device,
                         int B, int T, int C, int L, int NH, int V, int Vp, int BT, int BNH,
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
        Softmax.crossEntropySoftmaxBackward(dlogits, probs, targets, B, T, V, Vp);

        // Output projection backward
        Matmul.backwardTransposed(dlnf, dwte, null, dlogits, lnf, wte, BT, Vp, C);

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
        Encoder.backward(dwte, dwpe, dout, tokens, B, T, C);
    }

    static float evaluateVal(FlashBackend backend, CudaDevice device, int[] valTokens,
                             int B, int T, int C, int L, int NH, int V, int Vp, int BT, int BNH,
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
        int valPos = 0;

        for (int i = 0; i < numBatches; i++) {
            int[] tokens = new int[BT];
            int[] targets = new int[BT];

            for (int j = 0; j < BT; j++) {
                int pos = (valPos + j) % (valTokens.length - 1);
                tokens[j] = valTokens[pos];
                targets[j] = valTokens[pos + 1];
            }
            valPos = (valPos + BT) % (valTokens.length - BT - 1);

            float loss = forward(backend, device, B, T, C, L, NH, V, Vp, BT, BNH,
                    tokens, targets, wte, wpe, lnfw, lnfb,
                    ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                    ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                    encoded, ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                    ln2, ln2Mean, ln2Rstd, fch, fchGelu, residual,
                    lnf, lnfMean, lnfRstd, logits, probs, losses, blocks);
            totalLoss += loss;
        }
        return totalLoss / numBatches;
    }

    // Phase 1.5: generate() 使用 Generate.sample() with top-k/top-p
    static String generate(FlashBackend backend, CudaDevice device, GPT2TokenizerLoader tokenizer,
                           int B, int T, int C, int L, int NH, int V, int Vp, int BT, int BNH,
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
                           TransformerBlock[] blocks, int maxTokens,
                           Generate.SamplingConfig samplingConfig, java.util.Random rng) {

        StringBuilder sb = new StringBuilder();

        // Start with EOT token (like llm.c)
        int[] context = new int[T];
        context[0] = tokenizer.getEotToken();
        int contextLen = 1;

        for (int i = 0; i < maxTokens; i++) {
            // Prepare input
            int[] tokens = new int[BT];
            int start = Math.max(0, contextLen - T);
            for (int j = 0; j < Math.min(contextLen, T); j++) {
                tokens[j] = context[start + j];
            }

            // Forward only
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
            Matmul.forwardTransposed(logits, lnf, wte, BT, Vp, C);

            // Phase 1.5: Sample using logits with top-k/top-p
            int pos = Math.min(contextLen, T) - 1;
            float[] logitsData = logits.toFloatArray();
            int offset = pos * Vp;

            int nextToken = Generate.sample(logitsData, offset, V, samplingConfig, rng);

            // Append to context
            if (contextLen < T) {
                context[contextLen++] = nextToken;
            } else {
                System.arraycopy(context, 1, context, 0, T - 1);
                context[T - 1] = nextToken;
            }

            // Decode and append
            String tokenStr = tokenizer.decode(nextToken);
            sb.append(tokenStr);

            // Stop on EOT
            if (nextToken == tokenizer.getEotToken()) {
                break;
            }
        }

        return sb.toString();
    }
}