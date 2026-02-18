package com.flashllm.data;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Random;

/**
 * DataLoader - Loads tokenized training data.
 *
 * <p>Supports llm.c binary format (.bin files containing int32 tokens).</p>
 *
 * <h2>File Format:</h2>
 * <pre>
 * Header (256 bytes):
 *   - magic: 0x4C4C4D44 ("LLMD")
 *   - version: 1
 *   - num_tokens: int64
 *   - ... reserved ...
 * Data:
 *   - tokens: int32[] (little-endian)
 * </pre>
 *
 * <p>Or simple format (no header, just raw int32 tokens).</p>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public class DataLoader implements AutoCloseable {

    private static final int LLMD_MAGIC = 0x4C4C4D44;  // "LLMD"
    private static final int HEADER_SIZE = 256;

    private final int B;  // batch size
    private final int T;  // sequence length

    private final int[] tokens;
    private final long numTokens;
    private final int numBatches;

    private int currentBatch;
    private final Random random;
    private int[] shuffleIndices;
    private boolean shuffle;

    /**
     * Creates a DataLoader from a binary file.
     *
     * @param path path to the .bin file
     * @param B batch size
     * @param T sequence length
     * @throws IOException if file cannot be read
     */
    public DataLoader(String path, int B, int T) throws IOException {
        this(Path.of(path), B, T);
    }

    /**
     * Creates a DataLoader from a binary file.
     *
     * @param path path to the .bin file
     * @param B batch size
     * @param T sequence length
     * @throws IOException if file cannot be read
     */
    public DataLoader(Path path, int B, int T) throws IOException {
        this.B = B;
        this.T = T;
        this.random = new Random();
        this.shuffle = false;

        // Read file
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            long fileSize = channel.size();

            // Check for header
            ByteBuffer headerBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(headerBuf);
            headerBuf.flip();

            int magic = headerBuf.getInt();
            boolean hasHeader = (magic == LLMD_MAGIC);

            long dataOffset;
            long dataSize;

            if (hasHeader) {
                // Read header
                int version = headerBuf.getInt();
                if (version != 1) {
                    throw new IOException("Unsupported version: " + version);
                }

                ByteBuffer numTokensBuf = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
                channel.position(8);
                channel.read(numTokensBuf);
                numTokensBuf.flip();
                this.numTokens = numTokensBuf.getLong();

                dataOffset = HEADER_SIZE;
                dataSize = fileSize - HEADER_SIZE;
            } else {
                // No header, raw tokens
                dataOffset = 0;
                dataSize = fileSize;
                this.numTokens = dataSize / 4;  // int32 = 4 bytes
            }

            // Calculate batches
            // Each batch needs B * (T + 1) tokens (input + target shift)
            int tokensPerBatch = B * (T + 1);
            this.numBatches = (int) (numTokens / tokensPerBatch);

            if (numBatches == 0) {
                throw new IOException("Not enough tokens for even one batch. " +
                        "Have " + numTokens + " tokens, need " + tokensPerBatch);
            }

            // Read all tokens
            channel.position(dataOffset);
            ByteBuffer dataBuf = ByteBuffer.allocate((int) dataSize).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(dataBuf);
            dataBuf.flip();

            this.tokens = new int[(int) numTokens];
            dataBuf.asIntBuffer().get(tokens);
        }

        // Initialize shuffle indices
        this.shuffleIndices = new int[numBatches];
        for (int i = 0; i < numBatches; i++) {
            shuffleIndices[i] = i;
        }

        this.currentBatch = 0;

        System.out.println("DataLoader initialized:");
        System.out.println("  Tokens: " + numTokens);
        System.out.println("  Batches: " + numBatches);
        System.out.println("  Batch size: " + B + ", Seq len: " + T);
    }

    /**
     * Creates a DataLoader from an int array (for testing).
     *
     * @param tokens token array
     * @param B batch size
     * @param T sequence length
     */
    public DataLoader(int[] tokens, int B, int T) {
        this.B = B;
        this.T = T;
        this.tokens = tokens;
        this.numTokens = tokens.length;
        this.random = new Random();
        this.shuffle = false;

        int tokensPerBatch = B * (T + 1);
        this.numBatches = (int) (numTokens / tokensPerBatch);

        this.shuffleIndices = new int[numBatches];
        for (int i = 0; i < numBatches; i++) {
            shuffleIndices[i] = i;
        }

        this.currentBatch = 0;
    }

    /**
     * Gets the next batch of tokens.
     *
     * @param inputTokens output array for input tokens (B * T)
     * @param targetTokens output array for target tokens (B * T)
     * @return true if batch was loaded, false if epoch ended
     */
    public boolean nextBatch(int[] inputTokens, int[] targetTokens) {
        if (currentBatch >= numBatches) {
            return false;
        }

        int batchIdx = shuffleIndices[currentBatch];
        int tokensPerBatch = B * (T + 1);
        int startIdx = batchIdx * tokensPerBatch;

        for (int b = 0; b < B; b++) {
            int batchStart = startIdx + b * (T + 1);
            for (int t = 0; t < T; t++) {
                int idx = b * T + t;
                inputTokens[idx] = tokens[batchStart + t];
                targetTokens[idx] = tokens[batchStart + t + 1];
            }
        }

        currentBatch++;
        return true;
    }

    /**
     * Resets the data loader for a new epoch.
     */
    public void reset() {
        currentBatch = 0;
        if (shuffle) {
            shuffleIndices();
        }
    }

    /**
     * Enables or disables shuffling.
     */
    public void setShuffle(boolean shuffle) {
        this.shuffle = shuffle;
    }

    /**
     * Sets the random seed.
     */
    public void setSeed(long seed) {
        random.setSeed(seed);
    }

    /**
     * Shuffles the batch indices.
     */
    private void shuffleIndices() {
        for (int i = numBatches - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = shuffleIndices[i];
            shuffleIndices[i] = shuffleIndices[j];
            shuffleIndices[j] = temp;
        }
    }

    /**
     * Gets the number of tokens.
     */
    public long getNumTokens() {
        return numTokens;
    }

    /**
     * Gets the number of batches per epoch.
     */
    public int getNumBatches() {
        return numBatches;
    }

    /**
     * Gets the current batch index.
     */
    public int getCurrentBatch() {
        return currentBatch;
    }

    /**
     * Gets the batch size.
     */
    public int getBatchSize() {
        return B;
    }

    /**
     * Gets the sequence length.
     */
    public int getSeqLen() {
        return T;
    }

    @Override
    public void close() {
        // Nothing to close for in-memory data
    }

    /**
     * Writes tokens to a binary file.
     *
     * @param path output path
     * @param tokens tokens to write
     * @throws IOException if write fails
     */
    public static void writeTokens(Path path, int[] tokens) throws IOException {
        try (FileChannel channel = FileChannel.open(path,
                StandardOpenOption.CREATE, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {

            // Write header
            ByteBuffer header = ByteBuffer.allocate(HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN);
            header.putInt(LLMD_MAGIC);
            header.putInt(1);  // version
            header.putLong(tokens.length);  // num_tokens
            header.position(HEADER_SIZE);
            header.flip();
            channel.write(header);

            // Write tokens
            ByteBuffer data = ByteBuffer.allocate(tokens.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            data.asIntBuffer().put(tokens);
            channel.write(data);
        }
    }
}
