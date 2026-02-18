package com.flashllm.data;

import java.util.*;

/**
 * Character-level Tokenizer.
 *
 * Simple tokenizer that maps each unique character to an integer ID.
 * Used for training on Shakespeare text (like nanoGPT).
 */
public class CharTokenizer {

    private final Map<Character, Integer> charToId;
    private final Map<Integer, Character> idToChar;
    private final int vocabSize;

    /**
     * Creates a tokenizer from the given text.
     *
     * @param text the text to build vocabulary from
     */
    public CharTokenizer(String text) {
        // Find all unique characters
        Set<Character> chars = new TreeSet<>();
        for (char c : text.toCharArray()) {
            chars.add(c);
        }

        // Build mappings
        charToId = new HashMap<>();
        idToChar = new HashMap<>();

        int id = 0;
        for (char c : chars) {
            charToId.put(c, id);
            idToChar.put(id, c);
            id++;
        }

        this.vocabSize = chars.size();

        System.out.printf("CharTokenizer: vocab_size=%d, chars=%s%n",
                vocabSize, getVocabPreview());
    }

    /**
     * Encodes text to token IDs.
     *
     * @param text the text to encode
     * @return array of token IDs
     */
    public int[] encode(String text) {
        int[] tokens = new int[text.length()];
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            Integer id = charToId.get(c);
            if (id == null) {
                // Unknown character, use 0 (or could throw)
                tokens[i] = 0;
            } else {
                tokens[i] = id;
            }
        }
        return tokens;
    }

    /**
     * Decodes token IDs to text.
     *
     * @param tokens array of token IDs
     * @return decoded text
     */
    public String decode(int[] tokens) {
        StringBuilder sb = new StringBuilder();
        for (int token : tokens) {
            Character c = idToChar.get(token);
            if (c != null) {
                sb.append(c);
            } else {
                sb.append('?');  // Unknown token
            }
        }
        return sb.toString();
    }

    /**
     * Gets the vocabulary size.
     */
    public int getVocabSize() {
        return vocabSize;
    }

    /**
     * Gets a preview of the vocabulary.
     */
    private String getVocabPreview() {
        StringBuilder sb = new StringBuilder();
        int count = 0;
        for (char c : charToId.keySet()) {
            if (count > 0) sb.append(", ");
            if (c == '\n') {
                sb.append("\\n");
            } else if (c == '\t') {
                sb.append("\\t");
            } else if (c == ' ') {
                sb.append("' '");
            } else {
                sb.append(c);
            }
            count++;
            if (count >= 10) {
                sb.append("...");
                break;
            }
        }
        return sb.toString();
    }

    /**
     * Gets the character for a token ID.
     */
    public char getChar(int id) {
        Character c = idToChar.get(id);
        return c != null ? c : '?';
    }

    /**
     * Gets the token ID for a character.
     */
    public int getId(char c) {
        Integer id = charToId.get(c);
        return id != null ? id : 0;
    }
}
