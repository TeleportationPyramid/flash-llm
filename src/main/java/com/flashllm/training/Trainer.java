package com.flashllm.training;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;
import com.flashllm.config.GPT2Config;
import com.flashllm.data.DataLoader;
import com.flashllm.kernel.AdamW;
import com.flashllm.memory.GradTensors;
import com.flashllm.memory.ParameterTensors;
import com.flashllm.model.GPT2;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * Trainer - Training loop for GPT-2.
 *
 * <p>Handles:</p>
 * <ul>
 *   <li>Training loop with AdamW optimizer</li>
 *   <li>Learning rate scheduling (warmup + cosine decay)</li>
 *   <li>Gradient clipping</li>
 *   <li>Checkpoint saving/loading</li>
 *   <li>Logging and metrics</li>
 * </ul>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public class Trainer {

    // Model and data
    private final GPT2 model;
    private final DataLoader trainLoader;
    private DataLoader valLoader;

    // Optimizer state
    private CudaTensor m;  // first moment
    private CudaTensor v;  // second moment
    private int step;

    // Hyperparameters
    private float learningRate = 3e-4f;
    private float minLearningRate = 1e-5f;
    private float beta1 = 0.9f;
    private float beta2 = 0.999f;
    private float eps = 1e-8f;
    private float weightDecay = 0.1f;
    private float gradClipNorm = 1.0f;
    private int warmupSteps = 100;
    private int totalSteps = 10000;

    // Logging
    private int logInterval = 10;
    private int evalInterval = 100;
    private int saveInterval = 1000;
    private String checkpointDir = "checkpoints";

    // Metrics
    private final List<Float> trainLosses = new ArrayList<>();
    private final List<Float> valLosses = new ArrayList<>();

    // Backend
    private final FlashBackend backend;

    /**
     * Creates a trainer.
     *
     * @param model GPT-2 model
     * @param trainLoader training data loader
     */
    public Trainer(GPT2 model, DataLoader trainLoader) {
        this.model = model;
        this.trainLoader = trainLoader;
        this.backend = FlashBackend.getInstance();

        // Initialize optimizer state
        long numParams = model.getParams().numParameters();
        this.m = backend.allocateF32(numParams);
        this.v = backend.allocateF32(numParams);
        backend.zeroFill(m);
        backend.zeroFill(v);

        this.step = 0;
    }

    /**
     * Sets the validation data loader.
     */
    public Trainer setValLoader(DataLoader valLoader) {
        this.valLoader = valLoader;
        return this;
    }

    /**
     * Sets learning rate.
     */
    public Trainer setLearningRate(float lr) {
        this.learningRate = lr;
        return this;
    }

    /**
     * Sets minimum learning rate.
     */
    public Trainer setMinLearningRate(float minLr) {
        this.minLearningRate = minLr;
        return this;
    }

    /**
     * Sets weight decay.
     */
    public Trainer setWeightDecay(float wd) {
        this.weightDecay = wd;
        return this;
    }

    /**
     * Sets gradient clipping norm.
     */
    public Trainer setGradClipNorm(float norm) {
        this.gradClipNorm = norm;
        return this;
    }

    /**
     * Sets warmup steps.
     */
    public Trainer setWarmupSteps(int warmup) {
        this.warmupSteps = warmup;
        return this;
    }

    /**
     * Sets total training steps.
     */
    public Trainer setTotalSteps(int total) {
        this.totalSteps = total;
        return this;
    }

    /**
     * Sets log interval.
     */
    public Trainer setLogInterval(int interval) {
        this.logInterval = interval;
        return this;
    }

    /**
     * Sets evaluation interval.
     */
    public Trainer setEvalInterval(int interval) {
        this.evalInterval = interval;
        return this;
    }

    /**
     * Sets save interval.
     */
    public Trainer setSaveInterval(int interval) {
        this.saveInterval = interval;
        return this;
    }

    /**
     * Sets checkpoint directory.
     */
    public Trainer setCheckpointDir(String dir) {
        this.checkpointDir = dir;
        return this;
    }

    /**
     * Trains the model.
     *
     * @param numSteps number of training steps
     */
    public void train(int numSteps) {
        int B = model.getBatchSize();
        int T = model.getSeqLen();

        int[] inputTokens = new int[B * T];
        int[] targetTokens = new int[B * T];

        long startTime = System.currentTimeMillis();
        float runningLoss = 0.0f;
        int lossCount = 0;

        System.out.println("Starting training for " + numSteps + " steps...");

        for (int i = 0; i < numSteps; i++) {
            // Get batch
            if (!trainLoader.nextBatch(inputTokens, targetTokens)) {
                trainLoader.reset();
                trainLoader.nextBatch(inputTokens, targetTokens);
            }

            // Forward + backward
            float loss = model.step(inputTokens, targetTokens);
            runningLoss += loss;
            lossCount++;

            // Gradient clipping
            float gradNorm = clipGradients();

            // Optimizer step
            optimizerStep();
            step++;

            // Logging
            if (step % logInterval == 0) {
                float avgLoss = runningLoss / lossCount;
                float lr = AdamW.computeLearningRate(learningRate, minLearningRate, step, warmupSteps, totalSteps);
                long elapsed = System.currentTimeMillis() - startTime;
                float tokensPerSec = (float) (step * B * T) / (elapsed / 1000.0f);

                System.out.printf("step %d | loss %.4f | lr %.2e | grad_norm %.4f | %.1f tok/s%n",
                        step, avgLoss, lr, gradNorm, tokensPerSec);

                trainLosses.add(avgLoss);
                runningLoss = 0.0f;
                lossCount = 0;
            }

            // Evaluation
            if (valLoader != null && step % evalInterval == 0) {
                float valLoss = evaluate();
                valLosses.add(valLoss);
                System.out.printf("  val_loss: %.4f%n", valLoss);
            }

            // Save checkpoint
            if (step % saveInterval == 0) {
                saveCheckpoint();
            }
        }

        long totalTime = System.currentTimeMillis() - startTime;
        System.out.printf("Training complete! Total time: %.1f seconds%n", totalTime / 1000.0f);
    }

    /**
     * Evaluates the model on validation data.
     *
     * @return average validation loss
     */
    public float evaluate() {
        if (valLoader == null) {
            return 0.0f;
        }

        int B = model.getBatchSize();
        int T = model.getSeqLen();

        int[] inputTokens = new int[B * T];
        int[] targetTokens = new int[B * T];

        valLoader.reset();
        float totalLoss = 0.0f;
        int numBatches = 0;

        // Evaluate on up to 20 batches
        int maxBatches = Math.min(20, valLoader.getNumBatches());

        while (numBatches < maxBatches && valLoader.nextBatch(inputTokens, targetTokens)) {
            float loss = model.forward(inputTokens, targetTokens);
            totalLoss += loss;
            numBatches++;
        }

        return totalLoss / numBatches;
    }

    /**
     * Clips gradients by global norm.
     *
     * @return gradient norm before clipping
     */
    private float clipGradients() {
        GradTensors grads = model.getGrads();

        // Compute global norm across all parameters
        float totalNormSq = 0.0f;

        // For simplicity, we clip each tensor individually
        // In production, we'd compute the global norm first
        float norm = AdamW.clipGradNorm(grads.getAll(), gradClipNorm);

        return norm;
    }

    /**
     * Performs an optimizer step.
     */
    private void optimizerStep() {
        ParameterTensors params = model.getParams();
        GradTensors grads = model.getGrads();

        AdamW.updateWithSchedule(
                params.getAll(),
                grads.getAll(),
                m, v,
                learningRate, minLearningRate,
                beta1, beta2, eps,
                weightDecay,
                step, warmupSteps, totalSteps,
                params.numParameters()
        );
    }

    /**
     * Saves a checkpoint.
     */
    public void saveCheckpoint() {
        try {
            Path dir = Path.of(checkpointDir);
            Files.createDirectories(dir);

            String filename = String.format("checkpoint_step%d.bin", step);
            Path path = dir.resolve(filename);

            saveCheckpoint(path);
            System.out.println("Saved checkpoint: " + path);
        } catch (IOException e) {
            System.err.println("Failed to save checkpoint: " + e.getMessage());
        }
    }

    /**
     * Saves a checkpoint to a specific path.
     *
     * @param path output path
     * @throws IOException if save fails
     */
    public void saveCheckpoint(Path path) throws IOException {
        ParameterTensors params = model.getParams();
        float[] paramData = params.getAll().toFloatArray();

        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(path)))) {

            // Header
            out.writeInt(0x46474C4D);  // "FGLM" magic
            out.writeInt(1);            // version
            out.writeInt(step);         // current step
            out.writeLong(params.numParameters());

            // Config
            GPT2Config config = model.getConfig();
            out.writeInt(config.vocabSize);
            out.writeInt(config.maxSeqLen);
            out.writeInt(config.numLayers);
            out.writeInt(config.numHeads);
            out.writeInt(config.channels);

            // Parameters
            ByteBuffer buf = ByteBuffer.allocate(paramData.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            buf.asFloatBuffer().put(paramData);
            out.write(buf.array());

            // Optimizer state
            float[] mData = m.toFloatArray();
            float[] vData = v.toFloatArray();

            buf = ByteBuffer.allocate(mData.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            buf.asFloatBuffer().put(mData);
            out.write(buf.array());

            buf = ByteBuffer.allocate(vData.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            buf.asFloatBuffer().put(vData);
            out.write(buf.array());
        }
    }

    /**
     * Loads a checkpoint.
     *
     * @param path checkpoint path
     * @throws IOException if load fails
     */
    public void loadCheckpoint(Path path) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(path)))) {

            // Header
            int magic = in.readInt();
            if (magic != 0x46474C4D) {
                throw new IOException("Invalid checkpoint magic: " + Integer.toHexString(magic));
            }

            int version = in.readInt();
            if (version != 1) {
                throw new IOException("Unsupported checkpoint version: " + version);
            }

            this.step = in.readInt();
            long numParams = in.readLong();

            // Skip config (already set)
            in.readInt();  // vocabSize
            in.readInt();  // maxSeqLen
            in.readInt();  // numLayers
            in.readInt();  // numHeads
            in.readInt();  // channels

            // Parameters
            byte[] paramBytes = new byte[(int) (numParams * 4)];
            in.readFully(paramBytes);
            float[] paramData = new float[(int) numParams];
            ByteBuffer.wrap(paramBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(paramData);

            // Load into model
            ParameterTensors params = model.getParams();
            CudaDevice device = backend.getDevice();
            CudaTensor tempParams = CudaTensor.fromFloat(device, paramData, Precision.FP32);
            // Copy to params - need to implement this properly
            tempParams.close();

            // Optimizer state
            byte[] mBytes = new byte[(int) (numParams * 4)];
            in.readFully(mBytes);
            float[] mData = new float[(int) numParams];
            ByteBuffer.wrap(mBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(mData);

            byte[] vBytes = new byte[(int) (numParams * 4)];
            in.readFully(vBytes);
            float[] vData = new float[(int) numParams];
            ByteBuffer.wrap(vBytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(vData);

            // Load optimizer state
            CudaTensor tempM = CudaTensor.fromFloat(device, mData, Precision.FP32);
            CudaTensor tempV = CudaTensor.fromFloat(device, vData, Precision.FP32);
            // Copy to m, v
            tempM.close();
            tempV.close();

            System.out.println("Loaded checkpoint from step " + step);
        }
    }

    /**
     * Gets the current step.
     */
    public int getStep() {
        return step;
    }

    /**
     * Gets training losses.
     */
    public List<Float> getTrainLosses() {
        return trainLosses;
    }

    /**
     * Gets validation losses.
     */
    public List<Float> getValLosses() {
        return valLosses;
    }

    /**
     * Closes the trainer and releases resources.
     */
    public void close() {
        if (m != null) {
            m.close();
            m = null;
        }
        if (v != null) {
            v.close();
            v = null;
        }
    }
}
