package com.flashllm.model;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;

/**
 * GPT-2 Weight Loader for llm.c format.
 *
 * Reads the gpt2_124M.bin file format from llm.c:
 * - Header: 256 ints (magic, version, config)
 * - Weights: flat float32 array in specific order
 *
 * Weight order in llm.c:
 * 1. wte (vocab_size * C) - token embeddings
 * 2. wpe (maxT * C) - position embeddings
 * 3. For each layer:
 *    - ln1w (C) - layer norm 1 weight
 *    - ln1b (C) - layer norm 1 bias
 *    - qkvw (C * 3*C) - QKV projection weight
 *    - qkvb (3*C) - QKV projection bias
 *    - attprojw (C * C) - attention output projection weight
 *    - attprojb (C) - attention output projection bias
 *    - ln2w (C) - layer norm 2 weight
 *    - ln2b (C) - layer norm 2 bias
 *    - fcw (C * 4*C) - MLP fc weight
 *    - fcb (4*C) - MLP fc bias
 *    - fcprojw (4*C * C) - MLP projection weight
 *    - fcprojb (C) - MLP projection bias
 * 4. lnfw (C) - final layer norm weight
 * 5. lnfb (C) - final layer norm bias
 */
public class GPT2WeightLoader {

    // llm.c magic numbers
    private static final int MODEL_MAGIC_V3 = 20240326;  // Version 3 (FP32)
    private static final int MODEL_VERSION_3 = 3;

    // Config from header
    public int maxT;      // max sequence length (1024 for GPT-2)
    public int vocabSize; // vocabulary size (50257 for GPT-2)
    public int numLayers; // number of layers (12 for GPT-2 124M)
    public int numHeads;  // number of attention heads (12 for GPT-2 124M)
    public int channels;  // embedding dimension (768 for GPT-2 124M)
    public int paddedVocabSize; // padded vocab size for efficiency

    // All weights as flat float array
    private float[] weights;

    // Weight offsets
    private int wteOffset;
    private int wpeOffset;
    private int[] ln1wOffset, ln1bOffset;
    private int[] qkvwOffset, qkvbOffset;
    private int[] attprojwOffset, attprojbOffset;
    private int[] ln2wOffset, ln2bOffset;
    private int[] fcwOffset, fcbOffset;
    private int[] fcprojwOffset, fcprojbOffset;
    private int lnfwOffset, lnfbOffset;

    /**
     * Load GPT-2 weights from llm.c format .bin file.
     *
     * @param filePath path to gpt2_124M.bin
     * @throws IOException if file reading fails
     */
    public void load(String filePath) throws IOException {
        System.out.println("Loading GPT-2 weights from: " + filePath);

        try (RandomAccessFile raf = new RandomAccessFile(filePath, "r");
             FileChannel channel = raf.getChannel()) {

            // Read header (256 ints = 1024 bytes)
            ByteBuffer headerBuf = ByteBuffer.allocate(256 * 4);
            headerBuf.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(headerBuf);
            headerBuf.flip();

            int[] header = new int[256];
            for (int i = 0; i < 256; i++) {
                header[i] = headerBuf.getInt();
            }

            // Validate magic and version
            if (header[0] != MODEL_MAGIC_V3) {
                throw new IOException("Bad magic number: " + header[0] + ", expected: " + MODEL_MAGIC_V3);
            }
            if (header[1] != MODEL_VERSION_3) {
                throw new IOException("Bad version: " + header[1] + ", expected: " + MODEL_VERSION_3 +
                        "\n---> HINT: try to re-run `python train_gpt2.py`");
            }

            // Parse config from header
            maxT = header[2];
            vocabSize = header[3];
            numLayers = header[4];
            numHeads = header[5];
            channels = header[6];
            paddedVocabSize = header[7];

            System.out.println("========================================");
            System.out.println("[GPT-2 Loaded]");
            System.out.printf("max_seq_len: %d%n", maxT);
            System.out.printf("vocab_size: %d%n", vocabSize);
            System.out.printf("padded_vocab_size: %d%n", paddedVocabSize);
            System.out.printf("num_layers: %d%n", numLayers);
            System.out.printf("num_heads: %d%n", numHeads);
            System.out.printf("channels: %d%n", channels);

            // Calculate number of parameters
            long numParams = calculateNumParams();
            System.out.printf("num_parameters: %,d%n", numParams);
            System.out.println("========================================");

            // Allocate and read weights
            weights = new float[(int) numParams];

            // Read weights as float32
            ByteBuffer weightBuf = ByteBuffer.allocate((int) (numParams * 4));
            weightBuf.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(weightBuf);
            weightBuf.flip();

            FloatBuffer floatBuf = weightBuf.asFloatBuffer();
            floatBuf.get(weights);

            // Calculate offsets
            calculateOffsets();

            System.out.println("Weights loaded successfully!");
        }
    }

    /**
     * Calculate total number of parameters.
     */
    private long calculateNumParams() {
        int V = paddedVocabSize;
        int C = channels;
        int L = numLayers;
        int T = maxT;

        long params = 0;
        params += (long) V * C;       // wte
        params += (long) T * C;       // wpe

        // Per layer
        long perLayer = 0;
        perLayer += C;                // ln1w
        perLayer += C;                // ln1b
        perLayer += (long) C * 3 * C; // qkvw
        perLayer += 3 * C;            // qkvb
        perLayer += (long) C * C;     // attprojw
        perLayer += C;                // attprojb
        perLayer += C;                // ln2w
        perLayer += C;                // ln2b
        perLayer += (long) C * 4 * C; // fcw
        perLayer += 4 * C;            // fcb
        perLayer += (long) 4 * C * C; // fcprojw
        perLayer += C;                // fcprojb

        params += L * perLayer;

        params += C;  // lnfw
        params += C;  // lnfb

        return params;
    }

    /**
     * Calculate weight offsets for easy access.
     *
     * llm.c stores weights grouped by TYPE, not by layer:
     * wte, wpe, ln1w[all layers], ln1b[all layers], qkvw[all layers], ...
     */
    private void calculateOffsets() {
        int V = paddedVocabSize;
        int C = channels;
        int L = numLayers;
        int T = maxT;

        ln1wOffset = new int[L];
        ln1bOffset = new int[L];
        qkvwOffset = new int[L];
        qkvbOffset = new int[L];
        attprojwOffset = new int[L];
        attprojbOffset = new int[L];
        ln2wOffset = new int[L];
        ln2bOffset = new int[L];
        fcwOffset = new int[L];
        fcbOffset = new int[L];
        fcprojwOffset = new int[L];
        fcprojbOffset = new int[L];

        int offset = 0;

        // wte: (V, C)
        wteOffset = offset;
        offset += V * C;

        // wpe: (T, C)
        wpeOffset = offset;
        offset += T * C;

        // ln1w: (L, C) - all layers concatenated
        int ln1wBase = offset;
        for (int l = 0; l < L; l++) {
            ln1wOffset[l] = ln1wBase + l * C;
        }
        offset += L * C;

        // ln1b: (L, C)
        int ln1bBase = offset;
        for (int l = 0; l < L; l++) {
            ln1bOffset[l] = ln1bBase + l * C;
        }
        offset += L * C;

        // qkvw: (L, 3*C, C)
        int qkvwBase = offset;
        for (int l = 0; l < L; l++) {
            qkvwOffset[l] = qkvwBase + l * 3 * C * C;
        }
        offset += L * 3 * C * C;

        // qkvb: (L, 3*C)
        int qkvbBase = offset;
        for (int l = 0; l < L; l++) {
            qkvbOffset[l] = qkvbBase + l * 3 * C;
        }
        offset += L * 3 * C;

        // attprojw: (L, C, C)
        int attprojwBase = offset;
        for (int l = 0; l < L; l++) {
            attprojwOffset[l] = attprojwBase + l * C * C;
        }
        offset += L * C * C;

        // attprojb: (L, C)
        int attprojbBase = offset;
        for (int l = 0; l < L; l++) {
            attprojbOffset[l] = attprojbBase + l * C;
        }
        offset += L * C;

        // ln2w: (L, C)
        int ln2wBase = offset;
        for (int l = 0; l < L; l++) {
            ln2wOffset[l] = ln2wBase + l * C;
        }
        offset += L * C;

        // ln2b: (L, C)
        int ln2bBase = offset;
        for (int l = 0; l < L; l++) {
            ln2bOffset[l] = ln2bBase + l * C;
        }
        offset += L * C;

        // fcw: (L, 4*C, C)
        int fcwBase = offset;
        for (int l = 0; l < L; l++) {
            fcwOffset[l] = fcwBase + l * 4 * C * C;
        }
        offset += L * 4 * C * C;

        // fcb: (L, 4*C)
        int fcbBase = offset;
        for (int l = 0; l < L; l++) {
            fcbOffset[l] = fcbBase + l * 4 * C;
        }
        offset += L * 4 * C;

        // fcprojw: (L, C, 4*C)
        int fcprojwBase = offset;
        for (int l = 0; l < L; l++) {
            fcprojwOffset[l] = fcprojwBase + l * C * 4 * C;
        }
        offset += L * C * 4 * C;

        // fcprojb: (L, C)
        int fcprojbBase = offset;
        for (int l = 0; l < L; l++) {
            fcprojbOffset[l] = fcprojbBase + l * C;
        }
        offset += L * C;

        // lnfw: (C)
        lnfwOffset = offset;
        offset += C;

        // lnfb: (C)
        lnfbOffset = offset;
        offset += C;
    }

    // ========================================================================
    // Weight accessors - return float[] slices
    // ========================================================================

    public float[] getWte() {
        return slice(wteOffset, paddedVocabSize * channels);
    }

    public float[] getWpe() {
        return slice(wpeOffset, maxT * channels);
    }

    public float[] getLn1w(int layer) {
        return slice(ln1wOffset[layer], channels);
    }

    public float[] getLn1b(int layer) {
        return slice(ln1bOffset[layer], channels);
    }

    public float[] getQkvw(int layer) {
        // Try without transpose - llm.c may already store in correct layout
        return slice(qkvwOffset[layer], 3 * channels * channels);
    }

    public float[] getQkvb(int layer) {
        return slice(qkvbOffset[layer], 3 * channels);
    }

    public float[] getAttprojw(int layer) {
        // Try without transpose
        return slice(attprojwOffset[layer], channels * channels);
    }

    public float[] getAttprojb(int layer) {
        return slice(attprojbOffset[layer], channels);
    }

    public float[] getLn2w(int layer) {
        return slice(ln2wOffset[layer], channels);
    }

    public float[] getLn2b(int layer) {
        return slice(ln2bOffset[layer], channels);
    }

    public float[] getFcw(int layer) {
        // Try without transpose - llm.c may store (C, 4*C) already
        return slice(fcwOffset[layer], channels * 4 * channels);
    }

    public float[] getFcb(int layer) {
        return slice(fcbOffset[layer], 4 * channels);
    }

    public float[] getFcprojw(int layer) {
        // llm.c stores (4*C, C) - no transpose needed
        return slice(fcprojwOffset[layer], 4 * channels * channels);
    }

    public float[] getFcprojb(int layer) {
        return slice(fcprojbOffset[layer], channels);
    }

    public float[] getLnfw() {
        return slice(lnfwOffset, channels);
    }

    public float[] getLnfb() {
        return slice(lnfbOffset, channels);
    }

    private float[] slice(int offset, int length) {
        float[] result = new float[length];
        System.arraycopy(weights, offset, result, 0, length);
        return result;
    }

    /**
     * Get a transposed slice of weights.
     * llm.c stores weights as (out_features, in_features)
     * flash-llm expects (in_features, out_features)
     */
    private float[] sliceTransposed(int offset, int rows, int cols) {
        float[] result = new float[rows * cols];
        // Transpose: (rows, cols) -> (cols, rows)
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result[j * rows + i] = weights[offset + i * cols + j];
            }
        }
        return result;
    }

    /**
     * Get all weights as a flat array (for direct GPU upload).
     */
    public float[] getAllWeights() {
        return weights;
    }
}
