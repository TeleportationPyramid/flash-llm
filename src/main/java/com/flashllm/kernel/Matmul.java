package com.flashllm.kernel;

import io.github.teleportationpyramid.flash.*;
import com.flashllm.backend.FlashBackend;

/**
 * Matmul kernel - Matrix Multiplication with optional bias.
 *
 * <p>Corresponds to llm.c's matmul_forward/backward.</p>
 *
 * <h2>llm.c convention:</h2>
 * <p>llm.c stores weights as (OC, IC) and computes out = inp @ weight^T</p>
 * <p>Use forwardTransposed/backwardTransposed for llm.c weight layout.</p>
 *
 * <h2>Forward (standard):</h2>
 * <pre>out = inp @ weight + bias</pre>
 * <p>where weight is (IC, OC)</p>
 *
 * <h2>Forward (transposed):</h2>
 * <pre>out = inp @ weight^T + bias</pre>
 * <p>where weight is (OC, IC) - llm.c layout</p>
 *
 * @author flash-llm
 * @since 1.0.0
 */
public final class Matmul {

    private Matmul() {}

    /**
     * Matmul forward pass with bias.
     *
     * <p>Computes: out = inp @ weight + bias</p>
     *
     * @param out    output tensor (N, OC)
     * @param inp    input tensor (N, IC)
     * @param weight weight tensor (IC, OC)
     * @param bias   bias tensor (OC,) - can be null
     * @param N      number of samples (B * T)
     * @param OC     output channels
     * @param IC     input channels
     */
    public static void forward(
            CudaTensor out,
            CudaTensor inp,
            CudaTensor weight,
            CudaTensor bias,
            int N, int OC, int IC
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaBlas blas = backend.getBlas();
        CudaDevice device = backend.getDevice();

        // out = inp @ weight
        // inp: (N, IC), weight: (IC, OC), out: (N, OC)
        // GEMM: C = alpha * A @ B + beta * C
        // m = N, n = OC, k = IC
        blas.gemm(N, OC, IC, 1.0, inp, weight, 0.0, out);

        // Add bias if present
        if (bias != null) {
            addBias(backend, out, bias, N, OC);
        }
    }

    /**
     * Matmul forward pass without bias.
     */
    public static void forward(
            CudaTensor out,
            CudaTensor inp,
            CudaTensor weight,
            int N, int OC, int IC
    ) {
        forward(out, inp, weight, null, N, OC, IC);
    }

    /**
     * Matmul forward pass with weight transposed (llm.c layout).
     * 
     * <p>Computes: out = inp @ weight^T + bias</p>
     * <p>Use this when weight is stored as (OC, IC) instead of (IC, OC).</p>
     * <p>This is the llm.c convention for all weight matrices.</p>
     * 
     * @param out    output tensor (N, OC)
     * @param inp    input tensor (N, IC)
     * @param weight weight tensor (OC, IC) - will be transposed
     * @param bias   bias tensor (OC,) - can be null
     * @param N      number of samples (B * T)
     * @param OC     output channels
     * @param IC     input channels
     */
    public static void forwardTransposed(
            CudaTensor out,
            CudaTensor inp,
            CudaTensor weight,
            CudaTensor bias,
            int N, int OC, int IC
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaBlasExt blasExt = backend.getBlasExt();

        // out = inp @ weight^T
        // inp: (N, IC), weight: (OC, IC) -> weight^T: (IC, OC)
        // out: (N, OC)
        // m = N, n = OC, k = IC
        // Use gemmEx with transB = true
        blasExt.gemmEx(false, true, N, OC, IC, 1.0, inp, weight, 0.0, out);

        // Add bias if present
        if (bias != null) {
            addBias(backend, out, bias, N, OC);
        }
    }

    /**
     * Matmul forward pass with weight transposed, no bias.
     */
    public static void forwardTransposed(
            CudaTensor out,
            CudaTensor inp,
            CudaTensor weight,
            int N, int OC, int IC
    ) {
        forwardTransposed(out, inp, weight, null, N, OC, IC);
    }

    /**
     * Matmul backward pass (standard weight layout).
     *
     * <p>Computes gradients for input, weight, and bias.</p>
     * <p>weight is (IC, OC)</p>
     *
     * @param dinp    gradient for input (N, IC) - output
     * @param dweight gradient for weight (IC, OC) - accumulated
     * @param dbias   gradient for bias (OC,) - accumulated, can be null
     * @param dout    gradient from upstream (N, OC)
     * @param inp     original input (N, IC)
     * @param weight  weight tensor (IC, OC)
     * @param N       number of samples
     * @param OC      output channels
     * @param IC      input channels
     */
    public static void backward(
            CudaTensor dinp,
            CudaTensor dweight,
            CudaTensor dbias,
            CudaTensor dout,
            CudaTensor inp,
            CudaTensor weight,
            int N, int OC, int IC
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaBlasExt blasExt = backend.getBlasExt();

        // dinp = dout @ weight^T
        // dout: (N, OC), weight: (IC, OC) -> weight^T: (OC, IC)
        // dinp: (N, IC)
        // m = N, n = IC, k = OC
        blasExt.gemmEx(false, true, N, IC, OC, 1.0, dout, weight, 0.0, dinp);

        // dweight = inp^T @ dout
        // inp: (N, IC) -> inp^T: (IC, N)
        // dout: (N, OC)
        // dweight: (IC, OC)
        // m = IC, n = OC, k = N
        blasExt.gemmEx(true, false, IC, OC, N, 1.0, inp, dout, 1.0, dweight);

        // dbias = sum(dout, axis=0)
        if (dbias != null) {
            sumBias(backend, dbias, dout, N, OC);
        }
    }

    /**
     * Matmul backward pass for transposed weight layout (llm.c).
     * 
     * <p>When forward used forwardTransposed with weight (OC, IC),
     * backward needs to handle the transposed weight correctly.</p>
     *
     * @param dinp    gradient for input (N, IC) - output
     * @param dweight gradient for weight (OC, IC) - accumulated (transposed layout)
     * @param dbias   gradient for bias (OC,) - accumulated, can be null
     * @param dout    gradient from upstream (N, OC)
     * @param inp     original input (N, IC)
     * @param weight  weight tensor (OC, IC) - transposed layout
     * @param N       number of samples
     * @param OC      output channels
     * @param IC      input channels
     */
    public static void backwardTransposed(
            CudaTensor dinp,
            CudaTensor dweight,
            CudaTensor dbias,
            CudaTensor dout,
            CudaTensor inp,
            CudaTensor weight,
            int N, int OC, int IC
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaBlasExt blasExt = backend.getBlasExt();

        // dinp = dout @ weight (no transpose because weight is already (OC, IC))
        // dout: (N, OC), weight: (OC, IC)
        // dinp: (N, IC)
        // m = N, n = IC, k = OC
        blasExt.gemmEx(false, false, N, IC, OC, 1.0, dout, weight, 0.0, dinp);

        // dweight = dout^T @ inp  (to get (OC, IC) layout)
        // dout: (N, OC) -> dout^T: (OC, N)
        // inp: (N, IC)
        // dweight: (OC, IC)
        // m = OC, n = IC, k = N
        blasExt.gemmEx(true, false, OC, IC, N, 1.0, dout, inp, 1.0, dweight);

        // dbias = sum(dout, axis=0)
        if (dbias != null) {
            sumBias(backend, dbias, dout, N, OC);
        }
    }

    /**
     * Matmul backward for input only (no weight/bias gradients).
     *
     * <p>Used when weight is shared (e.g., tied embeddings).</p>
     */
    public static void backwardInput(
            CudaTensor dinp,
            CudaTensor dout,
            CudaTensor weight,
            int N, int OC, int IC
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaBlasExt blasExt = backend.getBlasExt();

        // dinp = dout @ weight^T
        blasExt.gemmEx(false, true, N, IC, OC, 1.0, dout, weight, 0.0, dinp);
    }

    /**
     * Matmul backward for weight only.
     */
    public static void backwardWeight(
            CudaTensor dweight,
            CudaTensor dout,
            CudaTensor inp,
            int N, int OC, int IC
    ) {
        FlashBackend backend = FlashBackend.getInstance();
        CudaBlasExt blasExt = backend.getBlasExt();

        // dweight = inp^T @ dout (accumulated)
        blasExt.gemmEx(true, false, IC, OC, N, 1.0, inp, dout, 1.0, dweight);
    }

    /**
     * Adds bias to output: out[i, :] += bias[:]
     */
    private static void addBias(FlashBackend backend, CudaTensor out, CudaTensor bias, int N, int OC) {
        CudaDevice device = backend.getDevice();

        // Get data and add bias
        float[] outData = out.toFloatArray();
        float[] biasData = bias.toFloatArray();

        for (int i = 0; i < N; i++) {
            int offset = i * OC;
            for (int j = 0; j < OC; j++) {
                outData[offset + j] += biasData[j];
            }
        }

        // Copy back
        TensorUtils.copyFromHost(device, outData, out);
    }

    /**
     * Sums dout across rows to compute dbias: dbias += sum(dout, axis=0)
     */
    private static void sumBias(FlashBackend backend, CudaTensor dbias, CudaTensor dout, int N, int OC) {
        CudaDevice device = backend.getDevice();

        // Get data
        float[] dbiasData = dbias.toFloatArray();
        float[] doutData = dout.toFloatArray();

        // Sum across N dimension
        for (int i = 0; i < N; i++) {
            int offset = i * OC;
            for (int j = 0; j < OC; j++) {
                dbiasData[j] += doutData[offset + j];
            }
        }

        // Copy back
        TensorUtils.copyFromHost(device, dbiasData, dbias);
    }
}
