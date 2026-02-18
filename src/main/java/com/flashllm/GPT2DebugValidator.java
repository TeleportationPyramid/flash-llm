package com.flashllm;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.kernel.*;
import com.flashllm.model.*;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;

/**
 * GPT-2 Debug Validator - Compare our implementation with llm.c reference.
 * 
 * Reads gpt2_124M_debug_state.bin and compares activations layer by layer.
 * 
 * Debug state format (from llm.c test_gpt2.c):
 * - Header: 256 ints
 *   - [0] = magic (20240327)
 *   - [1] = version (2)
 *   - [2] = B (batch size)
 *   - [3] = T (sequence length)
 * - x: input tokens (B * T ints)
 * - y: target tokens (B * T ints)
 * - expected_logits: (B * T * V floats)
 * - expected_loss: (1 float)
 * - expected_grads: (num_parameters floats) - we skip this for now
 */
public class GPT2DebugValidator {

    public static void main(String[] args) {
        System.out.println("╔═══════════════════════════════════════════════════════════╗");
        System.out.println("║        GPT-2 Debug Validator - Compare with llm.c         ║");
        System.out.println("╚═══════════════════════════════════════════════════════════╝\n");

        try {
            // Find files
            String weightsPath = findFile("gpt2_124M.bin");
            String debugStatePath = findFile("gpt2_124M_debug_state.bin");

            System.out.println("Found weights: " + weightsPath);
            System.out.println("Found debug state: " + debugStatePath);

            // Load weights
            GPT2WeightLoader weightLoader = new GPT2WeightLoader();
            weightLoader.load(weightsPath);
            int V = weightLoader.vocabSize;
            int Vp = weightLoader.paddedVocabSize;
            int C = weightLoader.channels;
            int L = weightLoader.numLayers;
            int NH = weightLoader.numHeads;
            int T_max = weightLoader.maxT;

            System.out.println("\nModel config:");
            System.out.println("  V=" + V + ", Vp=" + Vp + ", C=" + C + ", L=" + L + ", NH=" + NH);

            // Read debug state
            System.out.println("\n========================================");
            System.out.println("Reading debug state...");
            System.out.println("========================================");

            RandomAccessFile raf = new RandomAccessFile(debugStatePath, "r");
            FileChannel channel = raf.getChannel();
            
            // Read header
            ByteBuffer headerBuf = ByteBuffer.allocate(256 * 4).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(headerBuf);
            headerBuf.flip();
            IntBuffer header = headerBuf.asIntBuffer();
            
            int magic = header.get(0);
            int version = header.get(1);
            int B = header.get(2);
            int T = header.get(3);
            
            System.out.println("Debug state: magic=" + magic + ", version=" + version + ", B=" + B + ", T=" + T);
            
            if (magic != 20240327) {
                System.out.println("ERROR: Bad magic number! Expected 20240327, got " + magic);
                return;
            }
            
            // Read input tokens (B * T ints)
            ByteBuffer tokensBuf = ByteBuffer.allocate(B * T * 4).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(tokensBuf);
            tokensBuf.flip();
            int[] inputTokens = new int[B * T];
            tokensBuf.asIntBuffer().get(inputTokens);
            
            System.out.println("Input tokens first 10: ");
            for (int i = 0; i < Math.min(10, B * T); i++) {
                System.out.print(inputTokens[i] + " ");
            }
            System.out.println();
            
            // Read target tokens (B * T ints)
            ByteBuffer targetsBuf = ByteBuffer.allocate(B * T * 4).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(targetsBuf);
            targetsBuf.flip();
            int[] targetTokens = new int[B * T];
            targetsBuf.asIntBuffer().get(targetTokens);
            
            System.out.println("Target tokens first 10: ");
            for (int i = 0; i < Math.min(10, B * T); i++) {
                System.out.print(targetTokens[i] + " ");
            }
            System.out.println();
            
            // Read expected logits (B * T * V floats)
            // Note: debug state uses original vocab size V, NOT padded Vp
            long logitsSize = (long) B * T * V * 4;
            System.out.println("Reading expected logits: " + (logitsSize / 1024 / 1024) + " MB (V=" + V + ", not Vp=" + Vp + ")");
            
            ByteBuffer logitsBuf = ByteBuffer.allocate((int) logitsSize).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(logitsBuf);
            logitsBuf.flip();
            float[] expectedLogits = new float[B * T * V];
            logitsBuf.asFloatBuffer().get(expectedLogits);
            
            // Stats on expected logits
            float expMin = Float.MAX_VALUE, expMax = Float.MIN_VALUE;
            for (int i = 0; i < Math.min(1000, expectedLogits.length); i++) {
                expMin = Math.min(expMin, expectedLogits[i]);
                expMax = Math.max(expMax, expectedLogits[i]);
            }
            System.out.println("Expected logits (first 1000): min=" + expMin + ", max=" + expMax);
            System.out.println("Expected logits first 5: " + expectedLogits[0] + ", " + expectedLogits[1] + ", " + 
                             expectedLogits[2] + ", " + expectedLogits[3] + ", " + expectedLogits[4]);
            
            // Read expected loss (1 float)
            ByteBuffer lossBuf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(lossBuf);
            lossBuf.flip();
            float expectedLoss = lossBuf.getFloat();
            System.out.println("Expected loss: " + expectedLoss);
            
            channel.close();
            raf.close();
            
            // ========================================
            // Now run our implementation
            // ========================================
            System.out.println("\n========================================");
            System.out.println("Running our implementation...");
            System.out.println("========================================");
            
            FlashBackend backend = FlashBackend.getInstance();
            CudaDevice device = backend.getDevice();
            
            // Allocate GPU tensors
            CudaTensor wte = backend.allocateF32(Vp * C);
            CudaTensor wpe = backend.allocateF32(T_max * C);
            
            // Load embeddings
            TensorUtils.copyFromHost(device, weightLoader.getWte(), wte);
            TensorUtils.copyFromHost(device, weightLoader.getWpe(), wpe);
            
            // Encode input tokens
            CudaTensor encoded = backend.allocateF32(B * T * C);
            Encoder.forward(encoded, inputTokens, wte, wpe, B, T, C);
            
            // Check encoded
            float[] encodedData = encoded.toFloatArray();
            float encMin = Float.MAX_VALUE, encMax = Float.MIN_VALUE;
            for (float v : encodedData) { encMin = Math.min(encMin, v); encMax = Math.max(encMax, v); }
            System.out.println("Our encoded: min=" + encMin + ", max=" + encMax);
            System.out.println("Our encoded first 5: " + encodedData[0] + ", " + encodedData[1] + ", " + 
                             encodedData[2] + ", " + encodedData[3] + ", " + encodedData[4]);
            
            // Run through transformer blocks
            CudaTensor[] residual = new CudaTensor[L + 1];
            residual[0] = encoded;
            
            TransformerBlock[] blocks = new TransformerBlock[L];
            CudaTensor blockInput = encoded;
            
            for (int l = 0; l < L; l++) {
                blocks[l] = new TransformerBlock(l, B, T, C, NH);
                residual[l + 1] = backend.allocateF32(B * T * C);
                
                // Load layer weights
                CudaTensor ln1w = backend.allocateF32(C);
                CudaTensor ln1b = backend.allocateF32(C);
                CudaTensor qkvw = backend.allocateF32(3 * C * C);
                CudaTensor qkvb = backend.allocateF32(3 * C);
                CudaTensor attprojw = backend.allocateF32(C * C);
                CudaTensor attprojb = backend.allocateF32(C);
                CudaTensor ln2w = backend.allocateF32(C);
                CudaTensor ln2b = backend.allocateF32(C);
                CudaTensor fcw = backend.allocateF32(4 * C * C);
                CudaTensor fcb = backend.allocateF32(4 * C);
                CudaTensor fcprojw = backend.allocateF32(4 * C * C);
                CudaTensor fcprojb = backend.allocateF32(C);
                
                TensorUtils.copyFromHost(device, weightLoader.getLn1w(l), ln1w);
                TensorUtils.copyFromHost(device, weightLoader.getLn1b(l), ln1b);
                TensorUtils.copyFromHost(device, weightLoader.getQkvw(l), qkvw);
                TensorUtils.copyFromHost(device, weightLoader.getQkvb(l), qkvb);
                TensorUtils.copyFromHost(device, weightLoader.getAttprojw(l), attprojw);
                TensorUtils.copyFromHost(device, weightLoader.getAttprojb(l), attprojb);
                TensorUtils.copyFromHost(device, weightLoader.getLn2w(l), ln2w);
                TensorUtils.copyFromHost(device, weightLoader.getLn2b(l), ln2b);
                TensorUtils.copyFromHost(device, weightLoader.getFcw(l), fcw);
                TensorUtils.copyFromHost(device, weightLoader.getFcb(l), fcb);
                TensorUtils.copyFromHost(device, weightLoader.getFcprojw(l), fcprojw);
                TensorUtils.copyFromHost(device, weightLoader.getFcprojb(l), fcprojb);
                
                // Allocate activation tensors
                CudaTensor ln1 = backend.allocateF32(B * T * C);
                CudaTensor ln1Mean = backend.allocateF32(B * T);
                CudaTensor ln1Rstd = backend.allocateF32(B * T);
                CudaTensor qkv = backend.allocateF32(B * T * 3 * C);
                CudaTensor atty = backend.allocateF32(B * T * C);
                CudaTensor attLse = backend.allocateF32(B * NH * T);
                CudaTensor attnOut = backend.allocateF32(B * T * C);
                CudaTensor ln2 = backend.allocateF32(B * T * C);
                CudaTensor ln2Mean = backend.allocateF32(B * T);
                CudaTensor ln2Rstd = backend.allocateF32(B * T);
                CudaTensor fch = backend.allocateF32(B * T * 4 * C);
                CudaTensor fchGelu = backend.allocateF32(B * T * 4 * C);
                
                // Forward
                blocks[l].forward(
                    residual[l + 1], blockInput,
                    ln1w, ln1b, qkvw, qkvb, attprojw, attprojb,
                    ln2w, ln2b, fcw, fcb, fcprojw, fcprojb,
                    ln1, ln1Mean, ln1Rstd, qkv, atty, attLse, attnOut,
                    ln2, ln2Mean, ln2Rstd, fch, fchGelu
                );
                
                // Check output
                float[] resData = residual[l + 1].toFloatArray();
                float resMin = Float.MAX_VALUE, resMax = Float.MIN_VALUE;
                for (float v : resData) { resMin = Math.min(resMin, v); resMax = Math.max(resMax, v); }
                System.out.printf("Block[%d] output: min=%.4f, max=%.4f%n", l, resMin, resMax);
                
                if (l == L - 1) {
                    System.out.println("Block[" + l + "] first 5: " + resData[0] + ", " + resData[1] + ", " + 
                                     resData[2] + ", " + resData[3] + ", " + resData[4]);
                }
                
                // Next block's input is this block's output
                blockInput = residual[l + 1];
            }
            
            // Final LayerNorm
            CudaTensor lnfw = backend.allocateF32(C);
            CudaTensor lnfb = backend.allocateF32(C);
            float[] lnfwData = weightLoader.getLnfw();
            float[] lnfbData = weightLoader.getLnfb();
            
            // Debug lnfw/lnfb
            float lnfwMin = Float.MAX_VALUE, lnfwMax = Float.MIN_VALUE;
            float lnfbMin = Float.MAX_VALUE, lnfbMax = Float.MIN_VALUE;
            for (float v : lnfwData) { lnfwMin = Math.min(lnfwMin, v); lnfwMax = Math.max(lnfwMax, v); }
            for (float v : lnfbData) { lnfbMin = Math.min(lnfbMin, v); lnfbMax = Math.max(lnfbMax, v); }
            System.out.println("\nlnfw: min=" + lnfwMin + ", max=" + lnfwMax + ", first5=[" + 
                lnfwData[0] + "," + lnfwData[1] + "," + lnfwData[2] + "," + lnfwData[3] + "," + lnfwData[4] + "]");
            System.out.println("lnfb: min=" + lnfbMin + ", max=" + lnfbMax + ", first5=[" + 
                lnfbData[0] + "," + lnfbData[1] + "," + lnfbData[2] + "," + lnfbData[3] + "," + lnfbData[4] + "]");
            
            TensorUtils.copyFromHost(device, lnfwData, lnfw);
            TensorUtils.copyFromHost(device, lnfbData, lnfb);
            
            CudaTensor lnf = backend.allocateF32(B * T * C);
            CudaTensor lnfMean = backend.allocateF32(B * T);
            CudaTensor lnfRstd = backend.allocateF32(B * T);
            
            // residual[L] is the output of the last transformer block (Block L-1)
            LayerNorm.forward(lnf, lnfMean, lnfRstd, residual[L], lnfw, lnfb, B * T, C);
            
            float[] lnfData = lnf.toFloatArray();
            float lnfMin = Float.MAX_VALUE, lnfMax = Float.MIN_VALUE;
            for (float v : lnfData) { lnfMin = Math.min(lnfMin, v); lnfMax = Math.max(lnfMax, v); }
            System.out.println("\nOur lnf: min=" + lnfMin + ", max=" + lnfMax);
            System.out.println("Our lnf first 5: " + lnfData[0] + ", " + lnfData[1] + ", " + 
                             lnfData[2] + ", " + lnfData[3] + ", " + lnfData[4]);
            
            // Compute logits using forwardTransposed
            // We compute with Vp but only use V for loss calculation
            CudaTensor logits = backend.allocateF32(B * T * Vp);
            Matmul.forwardTransposed(logits, lnf, wte, B * T, Vp, C);
            
            float[] ourLogits = logits.toFloatArray();
            float ourMin = Float.MAX_VALUE, ourMax = Float.MIN_VALUE;
            for (int i = 0; i < Math.min(1000, ourLogits.length); i++) {
                ourMin = Math.min(ourMin, ourLogits[i]);
                ourMax = Math.max(ourMax, ourLogits[i]);
            }
            System.out.println("\nOur logits (first 1000): min=" + ourMin + ", max=" + ourMax);
            System.out.println("Our logits first 5: " + ourLogits[0] + ", " + ourLogits[1] + ", " + 
                             ourLogits[2] + ", " + ourLogits[3] + ", " + ourLogits[4]);
            
            // Check padded region
            System.out.println("\nPadded region logits [" + V + " to " + (V+5) + "]: " + 
                             ourLogits[V] + ", " + ourLogits[V+1] + ", " + ourLogits[V+2]);
            
            // ========================================
            // Compare logits
            // ========================================
            System.out.println("\n========================================");
            System.out.println("Comparison with expected (first token):");
            System.out.println("========================================");
            
            System.out.println("\nFirst 10 logits comparison:");
            System.out.println("Index\tExpected\t\tOurs\t\t\tDiff");
            for (int i = 0; i < 10; i++) {
                float diff = Math.abs(expectedLogits[i] - ourLogits[i]);
                System.out.printf("%d\t%.6f\t\t%.6f\t\t%.6f%n", i, expectedLogits[i], ourLogits[i], diff);
            }
            
            // Calculate mean absolute error for first token
            double mae = 0;
            for (int i = 0; i < V; i++) {
                mae += Math.abs(expectedLogits[i] - ourLogits[i]);
            }
            mae /= V;
            System.out.printf("\nMean Absolute Error (first token, all %d logits): %.6f%n", V, mae);
            
            // ========================================
            // Compute loss using V (not Vp) - CPU implementation
            // ========================================
            System.out.println("\n========================================");
            System.out.println("Loss calculation (using V=" + V + ", not Vp=" + Vp + "):");
            System.out.println("========================================");
            
            // Compute softmax and cross-entropy on CPU with only V logits
            float ourLoss = computeLossCPU(ourLogits, targetTokens, B, T, V, Vp);
            
            System.out.printf("Expected loss: %.6f%n", expectedLoss);
            System.out.printf("Our loss:      %.6f%n", ourLoss);
            System.out.printf("Difference:    %.6f%n", Math.abs(expectedLoss - ourLoss));
            
            // Cleanup
            backend.close();
            System.out.println("\nValidation complete!");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    /**
     * Compute cross-entropy loss on CPU, using only the first V logits (ignoring padded region).
     * 
     * @param logits  Full logits array (B * T * Vp)
     * @param targets Target tokens (B * T)
     * @param B       Batch size
     * @param T       Sequence length
     * @param V       Original vocab size (50257)
     * @param Vp      Padded vocab size (50304)
     * @return Mean cross-entropy loss
     */
    private static float computeLossCPU(float[] logits, int[] targets, int B, int T, int V, int Vp) {
        double totalLoss = 0;
        
        for (int i = 0; i < B * T; i++) {
            int offset = i * Vp;  // logits are stored with Vp stride
            
            // Find max for numerical stability (only in V range)
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int v = 0; v < V; v++) {
                maxVal = Math.max(maxVal, logits[offset + v]);
            }
            
            // Compute sum of exp(x - max) (only in V range)
            double sumExp = 0;
            for (int v = 0; v < V; v++) {
                sumExp += Math.exp(logits[offset + v] - maxVal);
            }
            
            // Compute log-softmax for target
            int target = targets[i];
            if (target < 0 || target >= V) {
                throw new IllegalArgumentException("Target " + target + " out of range [0, " + V + ")");
            }
            
            // loss = -log(softmax[target]) = -log(exp(x[target] - max) / sum) 
            //      = -(x[target] - max) + log(sum)
            double logSoftmax = (logits[offset + target] - maxVal) - Math.log(sumExp);
            totalLoss += -logSoftmax;
        }
        
        return (float) (totalLoss / (B * T));
    }
    
    private static String findFile(String filename) {
        String[] paths = {
            "src/main/resources/gpt2/" + filename,
            filename,
            "models/" + filename,
            "../llm.c/" + filename,
            "D:/gitHub/llm.c/" + filename
        };
        for (String path : paths) {
            if (new File(path).exists()) {
                return path;
            }
        }
        throw new RuntimeException("Cannot find " + filename);
    }
}
