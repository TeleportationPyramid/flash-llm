package com.flashllm.tokenizer;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.kernel.TensorUtils;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;
import java.util.*;

/**
 * Checkpoint management for GPT-2 training.
 * 
 * <p>Supports two formats:</p>
 * <ul>
 *   <li><b>llm.c format (.bin)</b> - Compatible with llm.c, weights only</li>
 *   <li><b>Training checkpoint (.ckpt)</b> - Full state including optimizer</li>
 * </ul>
 * 
 * <h2>Training Checkpoint Format (.ckpt)</h2>
 * <pre>
 * Header (256 ints):
 *   [0] magic = 0x464C4348 ("FLCH")
 *   [1] version = 1
 *   [2] step
 *   [3] B (batch size)
 *   [4] T (sequence length)
 *   [5] C (channels)
 *   [6] L (num layers)
 *   [7] NH (num heads)
 *   [8] V (vocab size)
 *   [9] Vp (padded vocab size)
 *   [10] maxT (max sequence length)
 *   [11] has_optimizer (0 or 1)
 *   [12-255] reserved
 * 
 * Body:
 *   - Parameters (same order as llm.c)
 *   - If has_optimizer:
 *     - Adam m states
 *     - Adam v states
 *   - Training loss history (optional)
 * </pre>
 */
public final class Checkpoint {

    // Magic numbers
    private static final int CHECKPOINT_MAGIC = 0x464C4348;  // "FLCH"
    private static final int CHECKPOINT_VERSION = 1;
    
    // llm.c magic (for compatibility)
    private static final int LLM_C_MAGIC_V3 = 20240326;
    private static final int LLM_C_VERSION_3 = 3;

    private Checkpoint() {}

    // ========================================================================
    // Training Checkpoint Format (.ckpt)
    // ========================================================================

    /**
     * Save full training checkpoint including optimizer state.
     * 
     * @param path output file path
     * @param step current training step
     * @param config model configuration
     * @param params parameter tensors (on GPU)
     * @param optimM Adam first moment states (on GPU), can be null
     * @param optimV Adam second moment states (on GPU), can be null
     */
    public static void saveCheckpoint(
            String path,
            int step,
            CheckpointConfig config,
            Map<String, CudaTensor> params,
            Map<String, CudaTensor> optimM,
            Map<String, CudaTensor> optimV
    ) throws IOException {
        System.out.println("Saving checkpoint to: " + path);
        
        boolean hasOptimizer = (optimM != null && optimV != null);
        
        try (FileOutputStream fos = new FileOutputStream(path)) {
            // Write header
            ByteBuffer header = ByteBuffer.allocate(256 * 4).order(ByteOrder.LITTLE_ENDIAN);
            header.putInt(CHECKPOINT_MAGIC);
            header.putInt(CHECKPOINT_VERSION);
            header.putInt(step);
            header.putInt(config.B);
            header.putInt(config.T);
            header.putInt(config.C);
            header.putInt(config.L);
            header.putInt(config.NH);
            header.putInt(config.V);
            header.putInt(config.Vp);
            header.putInt(config.maxT);
            header.putInt(hasOptimizer ? 1 : 0);
            // Fill remaining with zeros
            for (int i = 12; i < 256; i++) {
                header.putInt(0);
            }
            fos.write(header.array());
            
            // Write parameters in llm.c order
            writeParamsLlmcOrder(fos, params, config);
            
            // Write optimizer states if present
            if (hasOptimizer) {
                writeParamsLlmcOrder(fos, optimM, config);
                writeParamsLlmcOrder(fos, optimV, config);
            }
        }
        
        System.out.printf("Checkpoint saved: step=%d, size=%.2f MB%n", 
                         step, new File(path).length() / 1024.0 / 1024.0);
    }
    
    /**
     * Load training checkpoint.
     * 
     * @param path checkpoint file path
     * @param params parameter tensors to load into (on GPU)
     * @param optimM Adam first moment states to load into (on GPU), can be null
     * @param optimV Adam second moment states to load into (on GPU), can be null
     * @return checkpoint metadata
     */
    public static CheckpointMetadata loadCheckpoint(
            String path,
            Map<String, CudaTensor> params,
            Map<String, CudaTensor> optimM,
            Map<String, CudaTensor> optimV
    ) throws IOException {
        System.out.println("Loading checkpoint from: " + path);
        
        try (RandomAccessFile raf = new RandomAccessFile(path, "r");
             FileChannel channel = raf.getChannel()) {
            
            // Read header
            ByteBuffer header = ByteBuffer.allocate(256 * 4).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(header);
            header.flip();
            
            int magic = header.getInt();
            if (magic != CHECKPOINT_MAGIC) {
                throw new IOException("Invalid checkpoint magic: " + magic);
            }
            
            int version = header.getInt();
            if (version != CHECKPOINT_VERSION) {
                throw new IOException("Unsupported checkpoint version: " + version);
            }
            
            int step = header.getInt();
            CheckpointConfig config = new CheckpointConfig();
            config.B = header.getInt();
            config.T = header.getInt();
            config.C = header.getInt();
            config.L = header.getInt();
            config.NH = header.getInt();
            config.V = header.getInt();
            config.Vp = header.getInt();
            config.maxT = header.getInt();
            boolean hasOptimizer = header.getInt() == 1;
            
            // Read parameters
            readParamsLlmcOrder(channel, params, config);
            
            // Read optimizer states if present
            if (hasOptimizer && optimM != null && optimV != null) {
                readParamsLlmcOrder(channel, optimM, config);
                readParamsLlmcOrder(channel, optimV, config);
            }
            
            System.out.printf("Checkpoint loaded: step=%d, hasOptimizer=%b%n", step, hasOptimizer);
            
            return new CheckpointMetadata(step, config, hasOptimizer);
        }
    }

    // ========================================================================
    // llm.c Compatible Format (.bin)
    // ========================================================================

    /**
     * Save weights in llm.c format (weights only, no optimizer).
     * 
     * <p>Compatible with llm.c's gpt2_124M.bin format.</p>
     */
    public static void saveLlmcFormat(
            String path,
            CheckpointConfig config,
            Map<String, CudaTensor> params
    ) throws IOException {
        System.out.println("Saving weights in llm.c format to: " + path);
        
        try (FileOutputStream fos = new FileOutputStream(path)) {
            // Write llm.c header
            ByteBuffer header = ByteBuffer.allocate(256 * 4).order(ByteOrder.LITTLE_ENDIAN);
            header.putInt(LLM_C_MAGIC_V3);
            header.putInt(LLM_C_VERSION_3);
            header.putInt(config.maxT);
            header.putInt(config.V);
            header.putInt(config.L);
            header.putInt(config.NH);
            header.putInt(config.C);
            header.putInt(config.Vp);
            // Fill remaining with zeros
            for (int i = 8; i < 256; i++) {
                header.putInt(0);
            }
            fos.write(header.array());
            
            // Write parameters in llm.c order
            writeParamsLlmcOrder(fos, params, config);
        }
        
        System.out.printf("Weights saved in llm.c format: %.2f MB%n", 
                         new File(path).length() / 1024.0 / 1024.0);
    }

    // ========================================================================
    // Helper methods
    // ========================================================================

    /**
     * Write parameters in llm.c order.
     * 
     * Order:
     * 1. wte (Vp * C)
     * 2. wpe (maxT * C)
     * 3. For each layer: ln1w, ln1b, qkvw, qkvb, attprojw, attprojb, 
     *                    ln2w, ln2b, fcw, fcb, fcprojw, fcprojb
     * 4. lnfw (C)
     * 5. lnfb (C)
     */
    private static void writeParamsLlmcOrder(
            OutputStream os, 
            Map<String, CudaTensor> params, 
            CheckpointConfig config
    ) throws IOException {
        int L = config.L;
        
        // Write embeddings
        writeTensor(os, params.get("wte"));
        writeTensor(os, params.get("wpe"));
        
        // Write per-layer parameters (llm.c groups by parameter type, not by layer)
        for (int l = 0; l < L; l++) writeTensor(os, params.get("ln1w." + l));
        for (int l = 0; l < L; l++) writeTensor(os, params.get("ln1b." + l));
        for (int l = 0; l < L; l++) writeTensor(os, params.get("qkvw." + l));
        for (int l = 0; l < L; l++) writeTensor(os, params.get("qkvb." + l));
        for (int l = 0; l < L; l++) writeTensor(os, params.get("attprojw." + l));
        for (int l = 0; l < L; l++) writeTensor(os, params.get("attprojb." + l));
        for (int l = 0; l < L; l++) writeTensor(os, params.get("ln2w." + l));
        for (int l = 0; l < L; l++) writeTensor(os, params.get("ln2b." + l));
        for (int l = 0; l < L; l++) writeTensor(os, params.get("fcw." + l));
        for (int l = 0; l < L; l++) writeTensor(os, params.get("fcb." + l));
        for (int l = 0; l < L; l++) writeTensor(os, params.get("fcprojw." + l));
        for (int l = 0; l < L; l++) writeTensor(os, params.get("fcprojb." + l));
        
        // Write final layer norm
        writeTensor(os, params.get("lnfw"));
        writeTensor(os, params.get("lnfb"));
    }
    
    /**
     * Read parameters in llm.c order.
     */
    private static void readParamsLlmcOrder(
            FileChannel channel,
            Map<String, CudaTensor> params,
            CheckpointConfig config
    ) throws IOException {
        int L = config.L;
        
        // Read embeddings
        readTensor(channel, params.get("wte"));
        readTensor(channel, params.get("wpe"));
        
        // Read per-layer parameters
        for (int l = 0; l < L; l++) readTensor(channel, params.get("ln1w." + l));
        for (int l = 0; l < L; l++) readTensor(channel, params.get("ln1b." + l));
        for (int l = 0; l < L; l++) readTensor(channel, params.get("qkvw." + l));
        for (int l = 0; l < L; l++) readTensor(channel, params.get("qkvb." + l));
        for (int l = 0; l < L; l++) readTensor(channel, params.get("attprojw." + l));
        for (int l = 0; l < L; l++) readTensor(channel, params.get("attprojb." + l));
        for (int l = 0; l < L; l++) readTensor(channel, params.get("ln2w." + l));
        for (int l = 0; l < L; l++) readTensor(channel, params.get("ln2b." + l));
        for (int l = 0; l < L; l++) readTensor(channel, params.get("fcw." + l));
        for (int l = 0; l < L; l++) readTensor(channel, params.get("fcb." + l));
        for (int l = 0; l < L; l++) readTensor(channel, params.get("fcprojw." + l));
        for (int l = 0; l < L; l++) readTensor(channel, params.get("fcprojb." + l));
        
        // Read final layer norm
        readTensor(channel, params.get("lnfw"));
        readTensor(channel, params.get("lnfb"));
    }
    
    private static void writeTensor(OutputStream os, CudaTensor tensor) throws IOException {
        float[] data = tensor.toFloatArray();
        ByteBuffer buf = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : data) {
            buf.putFloat(v);
        }
        os.write(buf.array());
    }
    
    private static void readTensor(FileChannel channel, CudaTensor tensor) throws IOException {
        // Get element count from tensor's byte size (assuming FP32)
        int size = (int) (tensor.sizeInBytes() / 4);
        ByteBuffer buf = ByteBuffer.allocate(size * 4).order(ByteOrder.LITTLE_ENDIAN);
        channel.read(buf);
        buf.flip();
        
        float[] data = new float[size];
        for (int i = 0; i < size; i++) {
            data[i] = buf.getFloat();
        }
        
        // Copy to GPU
        TensorUtils.copyFromHost(null, data, tensor);  // TODO: pass device
    }

    // ========================================================================
    // Configuration and Metadata classes
    // ========================================================================

    /**
     * Model configuration for checkpointing.
     */
    public static class CheckpointConfig {
        public int B;     // batch size
        public int T;     // sequence length
        public int C;     // channels (embedding dim)
        public int L;     // number of layers
        public int NH;    // number of heads
        public int V;     // vocab size
        public int Vp;    // padded vocab size
        public int maxT;  // max sequence length
        
        public CheckpointConfig() {}
        
        public CheckpointConfig(int B, int T, int C, int L, int NH, int V, int Vp, int maxT) {
            this.B = B;
            this.T = T;
            this.C = C;
            this.L = L;
            this.NH = NH;
            this.V = V;
            this.Vp = Vp;
            this.maxT = maxT;
        }
    }
    
    /**
     * Metadata returned when loading a checkpoint.
     */
    public static class CheckpointMetadata {
        public final int step;
        public final CheckpointConfig config;
        public final boolean hasOptimizer;
        
        public CheckpointMetadata(int step, CheckpointConfig config, boolean hasOptimizer) {
            this.step = step;
            this.config = config;
            this.hasOptimizer = hasOptimizer;
        }
    }
}
