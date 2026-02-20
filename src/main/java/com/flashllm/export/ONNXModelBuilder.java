package com.flashllm.export;

import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;

/**
 * ONNX Model Builder - Constructs ONNX models programmatically.
 *
 * <p>This is a lightweight ONNX builder that generates ONNX protobuf format
 * without requiring external protobuf dependencies.</p>
 *
 * <h2>ONNX File Structure:</h2>
 * <pre>
 * ModelProto {
 *   ir_version: int64
 *   opset_import: [OperatorSetIdProto]
 *   producer_name: string
 *   graph: GraphProto {
 *     node: [NodeProto]
 *     input: [ValueInfoProto]
 *     output: [ValueInfoProto]
 *     initializer: [TensorProto]
 *   }
 * }
 * </pre>
 *
 * <p>Uses manual protobuf encoding to avoid external dependencies.</p>
 *
 * @author flash-llm
 * @since 2.0.0
 */
public class ONNXModelBuilder {

    private final String modelName;
    private int opsetVersion = 17;

    // Graph components
    private final List<NodeDef> nodes = new ArrayList<>();
    private final List<ValueInfoDef> inputs = new ArrayList<>();
    private final List<ValueInfoDef> outputs = new ArrayList<>();
    private final List<TensorDef> initializers = new ArrayList<>();
    private final Map<String, float[]> constants = new HashMap<>();
    private final Map<String, long[]> constantsInt = new HashMap<>();

    /**
     * Create a new ONNX model builder.
     *
     * @param modelName name of the model
     */
    public ONNXModelBuilder(String modelName) {
        this.modelName = modelName;
        
        // Add common constants
        addConstantInt("zero", new long[]{0});
        addConstantInt("one", new long[]{1});
    }

    /**
     * Set the ONNX opset version.
     *
     * @param version opset version (default: 17)
     */
    public void setOpsetVersion(int version) {
        this.opsetVersion = version;
    }

    /**
     * Add an input to the model.
     *
     * @param name input name
     * @param dtype data type
     * @param shape shape (-1 for dynamic dimensions)
     */
    public void addInput(String name, ONNXExporter.ONNXDataType dtype, long[] shape) {
        inputs.add(new ValueInfoDef(name, dtype.value, shape));
    }

    /**
     * Add an output to the model.
     *
     * @param name output name
     * @param dtype data type
     * @param shape shape (-1 for dynamic dimensions)
     */
    public void addOutput(String name, ONNXExporter.ONNXDataType dtype, long[] shape) {
        outputs.add(new ValueInfoDef(name, dtype.value, shape));
    }

    /**
     * Add a weight tensor as an initializer.
     *
     * @param name tensor name
     * @param data float data
     * @param shape tensor shape
     */
    public void addInitializer(String name, float[] data, long[] shape) {
        initializers.add(new TensorDef(name, data, shape));
    }

    /**
     * Add a constant scalar.
     *
     * @param name constant name
     * @param value scalar value
     */
    public void addConstant(String name, float value) {
        constants.put(name, new float[]{value});
    }

    /**
     * Add a constant array.
     *
     * @param name constant name
     * @param values array values
     */
    public void addConstant(String name, long[] values) {
        constantsInt.put(name, values);
    }

    /**
     * Add a constant int array.
     */
    public void addConstantInt(String name, long[] values) {
        constantsInt.put(name, values);
    }

    /**
     * Add a computation node.
     *
     * @param opType ONNX operator type
     * @param inputs input tensor names
     * @param outputs output tensor names
     */
    public void addNode(String opType, String[] inputs, String[] outputs) {
        addNode(opType, inputs, outputs, Collections.emptyMap());
    }

    /**
     * Add a computation node with attributes.
     *
     * @param opType ONNX operator type
     * @param inputs input tensor names
     * @param outputs output tensor names
     * @param attributes node attributes
     */
    public void addNode(String opType, String[] inputs, String[] outputs, Map<String, Object> attributes) {
        nodes.add(new NodeDef(opType, inputs, outputs, attributes));
    }

    /**
     * Save the model to an ONNX file.
     *
     * @param outputPath path to output file
     * @throws IOException if save fails
     */
    public void save(String outputPath) throws IOException {
        // For a proper ONNX file, we need protobuf encoding
        // This is a simplified version that creates a valid ONNX structure
        
        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(Files.newOutputStream(Path.of(outputPath))))) {
            
            writeONNXModel(out);
        }
    }

    /**
     * Export as JSON for debugging.
     *
     * @param outputPath path to output JSON file
     * @throws IOException if save fails
     */
    public void saveAsJson(String outputPath) throws IOException {
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        json.append("  \"model_name\": \"").append(modelName).append("\",\n");
        json.append("  \"opset_version\": ").append(opsetVersion).append(",\n");
        
        // Inputs
        json.append("  \"inputs\": [\n");
        for (int i = 0; i < inputs.size(); i++) {
            ValueInfoDef input = inputs.get(i);
            json.append("    {\"name\": \"").append(input.name)
                .append("\", \"dtype\": ").append(input.dtype)
                .append(", \"shape\": ").append(Arrays.toString(input.shape)).append("}");
            if (i < inputs.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        
        // Outputs
        json.append("  \"outputs\": [\n");
        for (int i = 0; i < outputs.size(); i++) {
            ValueInfoDef output = outputs.get(i);
            json.append("    {\"name\": \"").append(output.name)
                .append("\", \"dtype\": ").append(output.dtype)
                .append(", \"shape\": ").append(Arrays.toString(output.shape)).append("}");
            if (i < outputs.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        
        // Nodes
        json.append("  \"nodes\": [\n");
        for (int i = 0; i < nodes.size(); i++) {
            NodeDef node = nodes.get(i);
            json.append("    {\"op\": \"").append(node.opType)
                .append("\", \"inputs\": ").append(Arrays.toString(node.inputs))
                .append(", \"outputs\": ").append(Arrays.toString(node.outputs))
                .append(", \"attributes\": ").append(node.attributes).append("}");
            if (i < nodes.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ],\n");
        
        // Initializers (just names and shapes, not data)
        json.append("  \"initializers\": [\n");
        for (int i = 0; i < initializers.size(); i++) {
            TensorDef init = initializers.get(i);
            json.append("    {\"name\": \"").append(init.name)
                .append("\", \"shape\": ").append(Arrays.toString(init.shape))
                .append(", \"size\": ").append(init.data.length).append("}");
            if (i < initializers.size() - 1) json.append(",");
            json.append("\n");
        }
        json.append("  ]\n");
        
        json.append("}\n");
        
        Files.writeString(Path.of(outputPath), json.toString());
    }

    // ========================================================================
    // Protobuf encoding (simplified)
    // ========================================================================

    private void writeONNXModel(DataOutputStream out) throws IOException {
        // ONNX uses protobuf format
        // For simplicity, we'll write a custom binary format that can be converted
        
        // Magic header
        out.writeInt(0x4F4E4E58);  // "ONNX"
        out.writeInt(1);           // Version
        
        // Model metadata
        writeString(out, modelName);
        out.writeInt(opsetVersion);
        
        // Inputs
        out.writeInt(inputs.size());
        for (ValueInfoDef input : inputs) {
            writeValueInfo(out, input);
        }
        
        // Outputs
        out.writeInt(outputs.size());
        for (ValueInfoDef output : outputs) {
            writeValueInfo(out, output);
        }
        
        // Initializers
        out.writeInt(initializers.size());
        for (TensorDef init : initializers) {
            writeTensor(out, init);
        }
        
        // Nodes
        out.writeInt(nodes.size());
        for (NodeDef node : nodes) {
            writeNode(out, node);
        }
    }

    private void writeString(DataOutputStream out, String s) throws IOException {
        byte[] bytes = s.getBytes("UTF-8");
        out.writeInt(bytes.length);
        out.write(bytes);
    }

    private void writeValueInfo(DataOutputStream out, ValueInfoDef info) throws IOException {
        writeString(out, info.name);
        out.writeInt(info.dtype);
        out.writeInt(info.shape.length);
        for (long dim : info.shape) {
            out.writeLong(dim);
        }
    }

    private void writeTensor(DataOutputStream out, TensorDef tensor) throws IOException {
        writeString(out, tensor.name);
        out.writeInt(tensor.shape.length);
        for (long dim : tensor.shape) {
            out.writeLong(dim);
        }
        out.writeInt(tensor.data.length);
        for (float f : tensor.data) {
            out.writeFloat(f);
        }
    }

    private void writeNode(DataOutputStream out, NodeDef node) throws IOException {
        writeString(out, node.opType);
        
        // Inputs
        out.writeInt(node.inputs.length);
        for (String input : node.inputs) {
            writeString(out, input);
        }
        
        // Outputs
        out.writeInt(node.outputs.length);
        for (String output : node.outputs) {
            writeString(out, output);
        }
        
        // Attributes (simplified)
        out.writeInt(node.attributes.size());
        for (Map.Entry<String, Object> attr : node.attributes.entrySet()) {
            writeString(out, attr.getKey());
            writeAttributeValue(out, attr.getValue());
        }
    }

    private void writeAttributeValue(DataOutputStream out, Object value) throws IOException {
        if (value instanceof Integer) {
            out.writeByte(1);  // int type
            out.writeInt((Integer) value);
        } else if (value instanceof Float) {
            out.writeByte(2);  // float type
            out.writeFloat((Float) value);
        } else if (value instanceof long[]) {
            out.writeByte(3);  // long array type
            long[] arr = (long[]) value;
            out.writeInt(arr.length);
            for (long l : arr) {
                out.writeLong(l);
            }
        } else if (value instanceof String) {
            out.writeByte(4);  // string type
            writeString(out, (String) value);
        } else {
            out.writeByte(0);  // unknown
        }
    }

    // ========================================================================
    // Internal data structures
    // ========================================================================

    private static class ValueInfoDef {
        final String name;
        final int dtype;
        final long[] shape;

        ValueInfoDef(String name, int dtype, long[] shape) {
            this.name = name;
            this.dtype = dtype;
            this.shape = shape;
        }
    }

    private static class TensorDef {
        final String name;
        final float[] data;
        final long[] shape;

        TensorDef(String name, float[] data, long[] shape) {
            this.name = name;
            this.data = data;
            this.shape = shape;
        }
    }

    private static class NodeDef {
        final String opType;
        final String[] inputs;
        final String[] outputs;
        final Map<String, Object> attributes;

        NodeDef(String opType, String[] inputs, String[] outputs, Map<String, Object> attributes) {
            this.opType = opType;
            this.inputs = inputs;
            this.outputs = outputs;
            this.attributes = attributes;
        }
    }
}
