package com.flashllm;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.kernel.*;
import com.flashllm.model.*;

import java.io.*;
import java.nio.*;
import java.nio.channels.FileChannel;

/**
 * GPT-2 Gradient Validator - Compare backward pass with llm.c reference.
 * 
 * Reads gpt2_124M_debug_state.bin which contains expected gradients,
 * runs forward+backward, and compares gradients layer by layer.
 */
public class GPT2GradientValidator {

    public static void main(String[] args) {
        System.out.println("╔═══════════════════════════════════════════════════════════╗");
        System.out.println("║     GPT-2 Gradient Validator - Compare with llm.c         ║");
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

            System.out.println("\nModel config: V=" + V + ", Vp=" + Vp + ", C=" + C + ", L=" + L + ", NH=" + NH);

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
            
            // Read input tokens
            ByteBuffer tokensBuf = ByteBuffer.allocate(B * T * 4).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(tokensBuf);
            tokensBuf.flip();
            int[] inputTokens = new int[B * T];
            tokensBuf.asIntBuffer().get(inputTokens);
            
            // Read target tokens
            ByteBuffer targetsBuf = ByteBuffer.allocate(B * T * 4).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(targetsBuf);
            targetsBuf.flip();
            int[] targetTokens = new int[B * T];
            targetsBuf.asIntBuffer().get(targetTokens);
            
            // Read expected logits (B * T * V floats) - skip for gradient validation
            long logitsSize = (long) B * T * V * 4;
            channel.position(channel.position() + logitsSize);
            
            // Read expected loss
            ByteBuffer lossBuf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN);
            channel.read(lossBuf);
            lossBuf.flip();
            float expectedLoss = lossBuf.getFloat();
            System.out.println("Expected loss: " + expectedLoss);
            
            // Read expected gradients (same layout as weights)
            System.out.println("\nReading expected gradients...");
            
            // Gradient sizes (same as param_sizes in llm.c)
            int wteSize = Vp * C;
            int wpeSize = T_max * C;
            int ln1wSize = L * C;
            int ln1bSize = L * C;
            int qkvwSize = L * 3 * C * C;
            int qkvbSize = L * 3 * C;
            int attprojwSize = L * C * C;
            int attprojbSize = L * C;
            int ln2wSize = L * C;
            int ln2bSize = L * C;
            int fcwSize = L * 4 * C * C;
            int fcbSize = L * 4 * C;
            int fcprojwSize = L * 4 * C * C;
            int fcprojbSize = L * C;
            int lnfwSize = C;
            int lnfbSize = C;
            
            // Read each gradient
            float[] expDwte = readFloats(channel, wteSize);
            float[] expDwpe = readFloats(channel, wpeSize);
            float[] expDln1w = readFloats(channel, ln1wSize);
            float[] expDln1b = readFloats(channel, ln1bSize);
            float[] expDqkvw = readFloats(channel, qkvwSize);
            float[] expDqkvb = readFloats(channel, qkvbSize);
            float[] expDattprojw = readFloats(channel, attprojwSize);
            float[] expDattprojb = readFloats(channel, attprojbSize);
            float[] expDln2w = readFloats(channel, ln2wSize);
            float[] expDln2b = readFloats(channel, ln2bSize);
            float[] expDfcw = readFloats(channel, fcwSize);
            float[] expDfcb = readFloats(channel, fcbSize);
            float[] expDfcprojw = readFloats(channel, fcprojwSize);
            float[] expDfcprojb = readFloats(channel, fcprojbSize);
            float[] expDlnfw = readFloats(channel, lnfwSize);
            float[] expDlnfb = readFloats(channel, lnfbSize);
            
            channel.close();
            raf.close();
            
            System.out.println("Expected gradients loaded!");
            System.out.println("  dwte: " + stats(expDwte));
            System.out.println("  dwpe: " + stats(expDwpe));
            System.out.println("  dlnfw: " + stats(expDlnfw));
            System.out.println("  dlnfb: " + stats(expDlnfb));
            
            // ========================================
            // Run our forward + backward
            // ========================================
            System.out.println("\n========================================");
            System.out.println("Running our forward + backward...");
            System.out.println("========================================");
            
            FlashBackend backend = FlashBackend.getInstance();
            CudaDevice device = backend.getDevice();
            
            int BT = B * T;
            int BNH = B * NH;
            
            // Allocate parameters
            CudaTensor wte = backend.allocateF32(Vp * C);
            CudaTensor wpe = backend.allocateF32(T_max * C);
            CudaTensor lnfw = backend.allocateF32(C);
            CudaTensor lnfb = backend.allocateF32(C);
            
            TensorUtils.copyFromHost(device, weightLoader.getWte(), wte);
            TensorUtils.copyFromHost(device, weightLoader.getWpe(), wpe);
            TensorUtils.copyFromHost(device, weightLoader.getLnfw(), lnfw);
            TensorUtils.copyFromHost(device, weightLoader.getLnfb(), lnfb);
            
            // Per-layer weights
            CudaTensor[] ln1w = new CudaTensor[L], ln1b = new CudaTensor[L];
            CudaTensor[] qkvw = new CudaTensor[L], qkvb = new CudaTensor[L];
            CudaTensor[] attprojw = new CudaTensor[L], attprojb = new CudaTensor[L];
            CudaTensor[] ln2w = new CudaTensor[L], ln2b = new CudaTensor[L];
            CudaTensor[] fcw = new CudaTensor[L], fcb = new CudaTensor[L];
            CudaTensor[] fcprojw = new CudaTensor[L], fcprojb = new CudaTensor[L];
            
            for (int l = 0; l < L; l++) {
                ln1w[l] = backend.allocateF32(C);
                ln1b[l] = backend.allocateF32(C);
                qkvw[l] = backend.allocateF32(3 * C * C);
                qkvb[l] = backend.allocateF32(3 * C);
                attprojw[l] = backend.allocateF32(C * C);
                attprojb[l] = backend.allocateF32(C);
                ln2w[l] = backend.allocateF32(C);
                ln2b[l] = backend.allocateF32(C);
                fcw[l] = backend.allocateF32(4 * C * C);
                fcb[l] = backend.allocateF32(4 * C);
                fcprojw[l] = backend.allocateF32(4 * C * C);
                fcprojb[l] = backend.allocateF32(C);
                
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
            
            // Allocate activations
            CudaTensor encoded = backend.allocateF32(BT * C);
            CudaTensor[] ln1 = new CudaTensor[L], ln1Mean = new CudaTensor[L], ln1Rstd = new CudaTensor[L];
            CudaTensor[] qkv = new CudaTensor[L], atty = new CudaTensor[L], attLse = new CudaTensor[L], attnOut = new CudaTensor[L];
            CudaTensor[] ln2 = new CudaTensor[L], ln2Mean = new CudaTensor[L], ln2Rstd = new CudaTensor[L];
            CudaTensor[] fch = new CudaTensor[L], fchGelu = new CudaTensor[L];
            CudaTensor[] residual = new CudaTensor[L];
            
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
            CudaTensor logits = backend.allocateF32(BT * Vp);
            CudaTensor probs = backend.allocateF32(BT * Vp);
            CudaTensor losses = backend.allocateF32(BT);
            
            // Allocate gradients
            CudaTensor dwte = backend.allocateF32(Vp * C);
            CudaTensor dwpe = backend.allocateF32(T_max * C);
            CudaTensor dlnfw = backend.allocateF32(C);
            CudaTensor dlnfb = backend.allocateF32(C);
            CudaTensor dlogits = backend.allocateF32(BT * Vp);
            CudaTensor dlnf = backend.allocateF32(BT * C);
            CudaTensor dresidual = backend.allocateF32(BT * C);
            CudaTensor dencoded = backend.allocateF32(BT * C);
            
            CudaTensor[] dln1w = new CudaTensor[L], dln1b = new CudaTensor[L];
            CudaTensor[] dqkvw = new CudaTensor[L], dqkvb = new CudaTensor[L];
            CudaTensor[] dattprojw = new CudaTensor[L], dattprojb = new CudaTensor[L];
            CudaTensor[] dln2w = new CudaTensor[L], dln2b = new CudaTensor[L];
            CudaTensor[] dfcw = new CudaTensor[L], dfcb = new CudaTensor[L];
            CudaTensor[] dfcprojw = new CudaTensor[L], dfcprojb = new CudaTensor[L];
            
            for (int l = 0; l < L; l++) {
                dln1w[l] = backend.allocateF32(C);
                dln1b[l] = backend.allocateF32(C);
                dqkvw[l] = backend.allocateF32(3 * C * C);
                dqkvb[l] = backend.allocateF32(3 * C);
                dattprojw[l] = backend.allocateF32(C * C);
                dattprojb[l] = backend.allocateF32(C);
                dln2w[l] = backend.allocateF32(C);
                dln2b[l] = backend.allocateF32(C);
                dfcw[l] = backend.allocateF32(4 * C * C);
                dfcb[l] = backend.allocateF32(4 * C);
                dfcprojw[l] = backend.allocateF32(4 * C * C);
                dfcprojb[l] = backend.allocateF32(C);
            }
            
            // Zero gradients
            backend.zeroFill(dwte);
            backend.zeroFill(dwpe);
            backend.zeroFill(dlnfw);
            backend.zeroFill(dlnfb);
            for (int l = 0; l < L; l++) {
                backend.zeroFill(dln1w[l]);
                backend.zeroFill(dln1b[l]);
                backend.zeroFill(dqkvw[l]);
                backend.zeroFill(dqkvb[l]);
                backend.zeroFill(dattprojw[l]);
                backend.zeroFill(dattprojb[l]);
                backend.zeroFill(dln2w[l]);
                backend.zeroFill(dln2b[l]);
                backend.zeroFill(dfcw[l]);
                backend.zeroFill(dfcb[l]);
                backend.zeroFill(dfcprojw[l]);
                backend.zeroFill(dfcprojb[l]);
            }
            
            // Create transformer blocks
            TransformerBlock[] blocks = new TransformerBlock[L];
            for (int l = 0; l < L; l++) {
                blocks[l] = new TransformerBlock(l, B, T, C, NH);
            }
            
            // ===== FORWARD PASS =====
            Encoder.forward(encoded, inputTokens, wte, wpe, B, T, C);
            
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
            Softmax.crossEntropyForward(losses, probs, targetTokens, B, T, V, Vp);
            
            float ourLoss = Softmax.meanLoss(losses, B, T);
            System.out.println("Our loss: " + ourLoss + " (expected: " + expectedLoss + ")");
            
            // ===== BACKWARD PASS =====
            Softmax.crossEntropySoftmaxBackward(dlogits, probs, targetTokens, B, T, V, Vp);
            Matmul.backwardTransposed(dlnf, dwte, null, dlogits, lnf, wte, BT, Vp, C);
            LayerNorm.backward(dresidual, dlnfw, dlnfb, dlnf, residual[L-1], lnfw, lnfMean, lnfRstd, BT, C);
            
            CudaTensor dout = dresidual;
            CudaTensor dinp = backend.allocateF32(BT * C);
            
            for (int l = L - 1; l >= 0; l--) {
                CudaTensor inp = (l == 0) ? encoded : residual[l - 1];
                
                blocks[l].backward(
                    dinp, dout, inp,
                    ln1w[l], qkvw[l], attprojw[l],
                    ln2w[l], fcw[l], fcprojw[l],
                    ln1[l], ln1Mean[l], ln1Rstd[l],
                    qkv[l], atty[l], attLse[l], attnOut[l],
                    ln2[l], ln2Mean[l], ln2Rstd[l],
                    fch[l], fchGelu[l],
                    dln1w[l], dln1b[l], dqkvw[l], dqkvb[l], dattprojw[l], dattprojb[l],
                    dln2w[l], dln2b[l], dfcw[l], dfcb[l], dfcprojw[l], dfcprojb[l]
                );
                dout = dinp;
            }
            
            // dout now contains gradient w.r.t. encoded (from block 0)
            Encoder.backward(dwte, dwpe, dout, inputTokens, B, T, C);
            
            // ========================================
            // Compare gradients
            // ========================================
            System.out.println("\n========================================");
            System.out.println("Gradient comparison:");
            System.out.println("========================================");
            
            // Get our gradients
            float[] ourDwte = dwte.toFloatArray();
            float[] ourDwpe = dwpe.toFloatArray();
            float[] ourDlnfw = dlnfw.toFloatArray();
            float[] ourDlnfb = dlnfb.toFloatArray();
            
            // Compare key gradients
            System.out.println("\ndlnfw (final LayerNorm weight gradient):");
            compareGradients(expDlnfw, ourDlnfw, "dlnfw", 5);
            
            System.out.println("\ndlnfb (final LayerNorm bias gradient):");
            compareGradients(expDlnfb, ourDlnfb, "dlnfb", 5);
            
            System.out.println("\ndwte (embedding gradient, first 1000):");
            compareGradients(expDwte, ourDwte, "dwte", 5, 1000);
            
            System.out.println("\ndwpe (position embedding gradient):");
            compareGradients(expDwpe, ourDwpe, "dwpe", 5);
            
            // Compare per-layer gradients for layer 0
            System.out.println("\n--- Layer 0 gradients ---");
            
            float[] ourDln1w0 = dln1w[0].toFloatArray();
            float[] ourDln1b0 = dln1b[0].toFloatArray();
            float[] ourDqkvw0 = dqkvw[0].toFloatArray();
            float[] ourDfcw0 = dfcw[0].toFloatArray();
            
            // Extract expected layer 0 gradients
            float[] expDln1w0 = extractLayer(expDln1w, 0, C, L);
            float[] expDln1b0 = extractLayer(expDln1b, 0, C, L);
            float[] expDqkvw0 = extractLayer(expDqkvw, 0, 3*C*C, L);
            float[] expDfcw0 = extractLayer(expDfcw, 0, 4*C*C, L);
            
            System.out.println("\ndln1w[0]:");
            compareGradients(expDln1w0, ourDln1w0, "dln1w[0]", 5);
            
            System.out.println("\ndln1b[0]:");
            compareGradients(expDln1b0, ourDln1b0, "dln1b[0]", 5);
            
            System.out.println("\ndqkvw[0] (first 100):");
            compareGradients(expDqkvw0, ourDqkvw0, "dqkvw[0]", 5, 100);
            
            System.out.println("\ndfcw[0] (first 100):");
            compareGradients(expDfcw0, ourDfcw0, "dfcw[0]", 5, 100);
            
            // Cleanup
            backend.close();
            System.out.println("\nGradient validation complete!");
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    private static float[] readFloats(FileChannel channel, int count) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(count * 4).order(ByteOrder.LITTLE_ENDIAN);
        channel.read(buf);
        buf.flip();
        float[] data = new float[count];
        buf.asFloatBuffer().get(data);
        return data;
    }
    
    private static String stats(float[] arr) {
        float min = Float.MAX_VALUE, max = Float.MIN_VALUE, sum = 0;
        for (float v : arr) {
            min = Math.min(min, v);
            max = Math.max(max, v);
            sum += Math.abs(v);
        }
        return String.format("len=%d, min=%.6f, max=%.6f, absSum=%.6f", arr.length, min, max, sum);
    }
    
    private static void compareGradients(float[] expected, float[] ours, String name, int showFirst) {
        compareGradients(expected, ours, name, showFirst, expected.length);
    }
    
    private static void compareGradients(float[] expected, float[] ours, String name, int showFirst, int maxCompare) {
        int n = Math.min(Math.min(expected.length, ours.length), maxCompare);
        
        double mae = 0;
        double maxDiff = 0;
        int maxDiffIdx = 0;
        
        for (int i = 0; i < n; i++) {
            double diff = Math.abs(expected[i] - ours[i]);
            mae += diff;
            if (diff > maxDiff) {
                maxDiff = diff;
                maxDiffIdx = i;
            }
        }
        mae /= n;
        
        System.out.printf("  MAE: %.8f, MaxDiff: %.8f at idx %d%n", mae, maxDiff, maxDiffIdx);
        System.out.printf("  Expected first %d: ", showFirst);
        for (int i = 0; i < showFirst && i < expected.length; i++) {
            System.out.printf("%.6f ", expected[i]);
        }
        System.out.println();
        System.out.printf("  Ours first %d:     ", showFirst);
        for (int i = 0; i < showFirst && i < ours.length; i++) {
            System.out.printf("%.6f ", ours[i]);
        }
        System.out.println();
    }
    
    private static float[] extractLayer(float[] all, int layer, int sizePerLayer, int L) {
        float[] result = new float[sizePerLayer];
        System.arraycopy(all, layer * sizePerLayer, result, 0, sizePerLayer);
        return result;
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
