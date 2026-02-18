package com.flashllm.tokenizer;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 * GPT-2 BPE Tokenizer Loader for llm.c format.
 *
 * Reads the gpt2_tokenizer.bin file format from llm.c:
 * - Header: version (uint32), vocab_size (uint32), eot_token (uint32)
 * - For each token: length (uint8), then bytes
 *
 * This is a simpler format than the full BPE tokenizer,
 * it's just a vocabulary lookup (encode via external BPE logic).
 */
public class GPT2TokenizerLoader {

    private int vocabSize;
    private int eotToken;  // End of text token
    private String[] vocab;  // token_id -> string
    private Map<String, Integer> encoder;  // string -> token_id

    /**
     * Load GPT-2 tokenizer from llm.c format .bin file.
     *
     * llm.c format (from train_gpt2.py write_tokenizer):
     * - uint32 version (20240328)
     * - uint32 vocab_size (50257)
     * - uint32 eot_token (50256)
     * - For each token: uint8 length, then bytes
     *
     * @param filePath path to gpt2_tokenizer.bin
     * @throws IOException if file reading fails
     */
    public void load(String filePath) throws IOException {
        System.out.println("Loading GPT-2 tokenizer from: " + filePath);

        File f = new File(filePath);
        System.out.printf("File size: %d bytes%n", f.length());

        try (RandomAccessFile raf = new RandomAccessFile(filePath, "r");
             FileChannel channel = raf.getChannel()) {

            // Map the entire file
            long fileSize = channel.size();
            ByteBuffer buf = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);
            buf.order(ByteOrder.LITTLE_ENDIAN);

            // Debug: print raw header bytes
            byte[] rawHeader = new byte[16];
            buf.get(rawHeader);
            buf.position(0);  // Reset position
            System.out.printf("Raw header bytes: %02x %02x %02x %02x | %02x %02x %02x %02x | %02x %02x %02x %02x | %02x %02x %02x %02x%n",
                    rawHeader[0] & 0xFF, rawHeader[1] & 0xFF, rawHeader[2] & 0xFF, rawHeader[3] & 0xFF,
                    rawHeader[4] & 0xFF, rawHeader[5] & 0xFF, rawHeader[6] & 0xFF, rawHeader[7] & 0xFF,
                    rawHeader[8] & 0xFF, rawHeader[9] & 0xFF, rawHeader[10] & 0xFF, rawHeader[11] & 0xFF,
                    rawHeader[12] & 0xFF, rawHeader[13] & 0xFF, rawHeader[14] & 0xFF, rawHeader[15] & 0xFF);

            // Read header - first int is version (magic number like 20240328)
            int version = buf.getInt();
            vocabSize = buf.getInt();    // Should be 50257
            eotToken = buf.getInt();     // Should be 50256

            System.out.printf("Tokenizer: version=%d (0x%08x), vocab_size=%d, eot_token=%d%n",
                    version, version, vocabSize, eotToken);

            // Check for format issues
            if (vocabSize < 256 || vocabSize > 200000) {
                System.out.println("WARNING: vocab_size looks wrong! Expected ~50257.");
                System.out.println("This tokenizer file may be corrupted or in a different format.");
                System.out.println("Please regenerate using: python train_gpt2.py");
                System.out.println("Or download from: ./dev/download_starter_pack.sh");

                // Fall back to basic tokenizer
                vocabSize = 50257;
                eotToken = 50256;
                createBasicTokenizer();
                return;
            }

            if (version != 20240328) {
                System.out.printf("WARNING: Unknown version %d, expected 20240328%n", version);
            }

            // Read vocabulary
            vocab = new String[vocabSize];
            encoder = new HashMap<>();

            for (int i = 0; i < vocabSize; i++) {
                // Read length (1 byte, unsigned)
                int length = buf.get() & 0xFF;

                // Read token bytes
                byte[] tokenBytes = new byte[length];
                buf.get(tokenBytes);

                String token = new String(tokenBytes, StandardCharsets.UTF_8);
                vocab[i] = token;
                encoder.put(token, i);
            }

            System.out.println("Tokenizer loaded successfully!");

            // Print some sample tokens
            System.out.println("Sample tokens:");
            System.out.printf("  [0] = '%s'%n", escape(vocab[0]));
            System.out.printf("  [1] = '%s'%n", escape(vocab[1]));
            if (vocabSize > 256) {
                System.out.printf("  [256] = '%s'%n", escape(vocab[256]));
            }
            if (vocabSize > 50256) {
                System.out.printf("  [50256] = '%s' (EOT)%n", escape(vocab[50256]));
            }
        }
    }

    /**
     * Create a basic byte-level tokenizer as fallback.
     */
    private void createBasicTokenizer() {
        System.out.println("Creating basic fallback tokenizer...");
        vocab = new String[vocabSize];
        encoder = new HashMap<>();

        // First 256 tokens are byte values
        for (int i = 0; i < 256; i++) {
            if (i >= 32 && i < 127) {
                vocab[i] = String.valueOf((char) i);
            } else {
                vocab[i] = String.format("<%02x>", i);
            }
            encoder.put(vocab[i], i);
        }

        // BPE tokens (placeholders)
        for (int i = 256; i < vocabSize - 1; i++) {
            vocab[i] = "<" + i + ">";
            encoder.put(vocab[i], i);
        }

        // EOT token
        vocab[eotToken] = "<|endoftext|>";
        encoder.put(vocab[eotToken], eotToken);

        System.out.println("Basic tokenizer created with " + vocabSize + " tokens");
    }

    /**
     * Decode token IDs to string.
     *
     * @param ids array of token IDs
     * @return decoded string
     */
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            if (id >= 0 && id < vocabSize) {
                sb.append(vocab[id]);
            }
        }
        return sb.toString();
    }

    /**
     * Decode a single token ID to string.
     *
     * @param id token ID
     * @return decoded string
     */
    public String decode(int id) {
        if (id >= 0 && id < vocabSize) {
            return vocab[id];
        }
        return "";
    }

    /**
     * Get vocabulary size.
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * Get end-of-text token ID.
     */
    public int getEotToken() {
        return eotToken;
    }

    /**
     * Get token string by ID.
     */
    public String getToken(int id) {
        if (id >= 0 && id < vocabSize) {
            return vocab[id];
        }
        return null;
    }

    /**
     * Get token ID by string (exact match).
     */
    public Integer getTokenId(String token) {
        return encoder.get(token);
    }

    private String escape(String s) {
        if (s == null) return "<null>";
        return s.replace("\n", "\\n")
                .replace("\t", "\\t")
                .replace("\r", "\\r");
    }
}
