package com.flashllm.data;

import java.io.*;
import java.nio.file.*;
import java.util.Random;

/**
 * Text DataLoader for training.
 *
 * Loads text file, tokenizes it, and provides batches for training.
 * Supports train/val split.
 */
public class TextDataLoader {

    private final CharTokenizer tokenizer;
    private final int[] trainData;
    private final int[] valData;
    private final int batchSize;
    private final int seqLen;

    private int trainPos = 0;
    private int valPos = 0;
    private final Random rng;

    /**
     * Creates a TextDataLoader.
     *
     * @param filePath   path to the text file
     * @param batchSize  batch size (B)
     * @param seqLen     sequence length (T)
     * @param trainSplit fraction of data for training (e.g., 0.9)
     */
    public TextDataLoader(String filePath, int batchSize, int seqLen, double trainSplit) throws IOException {
        this.batchSize = batchSize;
        this.seqLen = seqLen;
        this.rng = new Random(42);

        // Read file
        String text = Files.readString(Path.of(filePath));
        System.out.printf("Loaded %,d characters from %s%n", text.length(), filePath);

        // Create tokenizer
        this.tokenizer = new CharTokenizer(text);

        // Tokenize
        int[] allTokens = tokenizer.encode(text);
        System.out.printf("Tokenized to %,d tokens%n", allTokens.length);

        // Split into train/val
        int splitIdx = (int) (allTokens.length * trainSplit);
        trainData = new int[splitIdx];
        valData = new int[allTokens.length - splitIdx];

        System.arraycopy(allTokens, 0, trainData, 0, splitIdx);
        System.arraycopy(allTokens, splitIdx, valData, 0, valData.length);

        int trainBatches = (trainData.length - 1) / (batchSize * seqLen);
        int valBatches = (valData.length - 1) / (batchSize * seqLen);

        System.out.printf("Train: %,d tokens (%d batches)%n", trainData.length, trainBatches);
        System.out.printf("Val:   %,d tokens (%d batches)%n", valData.length, valBatches);
    }

    /**
     * Gets the next training batch.
     *
     * @return [inputs, targets] where each is int[batchSize * seqLen]
     */
    public int[][] nextTrainBatch() {
        return getBatch(trainData, true);
    }

    /**
     * Gets the next validation batch.
     *
     * @return [inputs, targets] where each is int[batchSize * seqLen]
     */
    public int[][] nextValBatch() {
        return getBatch(valData, false);
    }

    private int[][] getBatch(int[] data, boolean isTrain) {
        int[] inputs = new int[batchSize * seqLen];
        int[] targets = new int[batchSize * seqLen];

        int pos = isTrain ? trainPos : valPos;
        int maxStart = data.length - seqLen - 1;

        for (int b = 0; b < batchSize; b++) {
            // Random start position for each sequence in batch
            int start;
            if (isTrain) {
                start = rng.nextInt(maxStart);
            } else {
                start = pos;
                pos += seqLen;
                if (pos >= maxStart) pos = 0;
            }

            // Copy sequence
            for (int t = 0; t < seqLen; t++) {
                int idx = b * seqLen + t;
                inputs[idx] = data[start + t];
                targets[idx] = data[start + t + 1];  // Next token prediction
            }
        }

        if (isTrain) {
            trainPos = (trainPos + batchSize * seqLen) % maxStart;
        } else {
            valPos = pos;
        }

        return new int[][] { inputs, targets };
    }

    /**
     * Resets the data loader positions.
     */
    public void reset() {
        trainPos = 0;
        valPos = 0;
    }

    /**
     * Gets the tokenizer.
     */
    public CharTokenizer getTokenizer() {
        return tokenizer;
    }

    /**
     * Gets the vocabulary size.
     */
    public int getVocabSize() {
        return tokenizer.getVocabSize();
    }

    /**
     * Gets number of training batches.
     */
    public int getNumTrainBatches() {
        return (trainData.length - 1) / (batchSize * seqLen);
    }

    /**
     * Gets number of validation batches.
     */
    public int getNumValBatches() {
        return (valData.length - 1) / (batchSize * seqLen);
    }

    /**
     * Decodes tokens to text.
     */
    public String decode(int[] tokens) {
        return tokenizer.decode(tokens);
    }
}
