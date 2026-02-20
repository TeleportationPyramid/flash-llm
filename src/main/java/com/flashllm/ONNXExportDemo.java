package com.flashllm;

import com.flashllm.export.ONNXExporter;
import com.flashllm.model.GPT2WeightLoader;

import java.io.File;

/**
 * Demo for ONNX export.
 *
 * <p>Exports GPT-2 model to ONNX format for deployment.</p>
 *
 * @author flash-llm
 * @since 2.0.0
 */
public class ONNXExportDemo {

    public static void main(String[] args) {
        System.out.println("================================================================");
        System.out.println("     GPT-2 124M ONNX Export Demo");
        System.out.println("     Phase 2.2: ONNX Export");
        System.out.println("================================================================\n");

        try {
            // Find weights file
            String weightsPath = findFile("gpt2_124M.bin", new String[]{
                "src/main/resources/gpt2/gpt2_124M.bin",
                "gpt2_124M.bin",
                "models/gpt2_124M.bin"
            });

            // Load weights
            GPT2WeightLoader weights = new GPT2WeightLoader();
            weights.load(weightsPath);

            // Create exporter
            ONNXExporter exporter = new ONNXExporter(weights);

            // Export options
            System.out.println("\n========================================");
            System.out.println("Export Options:");
            System.out.println("========================================\n");

            // 1. Export weights only (for external ONNX graph construction)
            System.out.println("1. Exporting weights only...");
            exporter.exportWeightsOnly("exports/gpt2_weights");
            System.out.println("   Done! Weights saved to: exports/gpt2_weights/\n");

            // 2. Export standard ONNX model
            System.out.println("2. Exporting standard ONNX model...");
            exporter.export("exports/gpt2_124M.onnx");
            System.out.println("   Done!\n");

            // 3. Export ONNX model with KV cache
            System.out.println("3. Exporting ONNX model with KV cache...");
            exporter.exportWithKVCache("exports/gpt2_124M_kv.onnx");
            System.out.println("   Done!\n");

            System.out.println("========================================");
            System.out.println("Export Complete!");
            System.out.println("========================================\n");

            System.out.println("Exported files:");
            System.out.println("  - exports/gpt2_weights/       (individual weight files)");
            System.out.println("  - exports/gpt2_124M.onnx      (standard model)");
            System.out.println("  - exports/gpt2_124M_kv.onnx   (with KV cache)");

            System.out.println("\nTo use with ONNX Runtime (Python):");
            System.out.println("  import onnxruntime as ort");
            System.out.println("  session = ort.InferenceSession('exports/gpt2_124M.onnx')");
            System.out.println("  outputs = session.run(None, {'input_ids': tokens})");

        } catch (Exception e) {
            System.err.println("ONNX export failed:");
            e.printStackTrace();
            System.exit(1);
        }
    }

    static String findFile(String name, String[] paths) {
        for (String path : paths) {
            File f = new File(path);
            if (f.exists()) {
                System.out.println("Found " + name + " at: " + path);
                return path;
            }
        }
        throw new RuntimeException("Could not find " + name + "!");
    }
}
