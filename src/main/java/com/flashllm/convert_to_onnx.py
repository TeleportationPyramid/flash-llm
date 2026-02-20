#!/usr/bin/env python3
"""
GPT-2 to ONNX Converter

Converts GPT-2 weights from llm.c format to ONNX format.

Usage:
    python convert_to_onnx.py gpt2_124M.bin gpt2_124M.onnx
    python convert_to_onnx.py gpt2_124M.bin gpt2_124M_kv.onnx --with-kv-cache

Requirements:
    pip install torch onnx onnxruntime numpy

Author: flash-llm
"""

import argparse
import struct
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not found. Some features may be limited.")

try:
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("Warning: ONNX not found. Install with: pip install onnx")


def load_gpt2_weights(filepath):
    """Load GPT-2 weights from llm.c binary format."""
    print(f"Loading weights from: {filepath}")
    
    with open(filepath, 'rb') as f:
        # Read header
        header = struct.unpack('256i', f.read(256 * 4))
        
        magic = header[0]
        version = header[1]
        
        if magic != 20240326:
            raise ValueError(f"Invalid magic number: {magic}")
        
        max_seq_len = header[2]
        vocab_size = header[3]
        num_layers = header[4]
        num_heads = header[5]
        channels = header[6]
        padded_vocab_size = header[7]
        
        config = {
            'max_seq_len': max_seq_len,
            'vocab_size': vocab_size,
            'padded_vocab_size': padded_vocab_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'channels': channels,
            'head_dim': channels // num_heads
        }
        
        print(f"Model config:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        # Calculate parameter counts
        C = channels
        L = num_layers
        Vp = padded_vocab_size
        maxT = max_seq_len
        
        # Read all weights
        weights = {}
        
        # wte: (Vp, C)
        weights['wte'] = np.frombuffer(f.read(Vp * C * 4), dtype=np.float32).reshape(Vp, C)
        
        # wpe: (maxT, C)
        weights['wpe'] = np.frombuffer(f.read(maxT * C * 4), dtype=np.float32).reshape(maxT, C)
        
        # Per-layer weights
        for l in range(L):
            prefix = f'layer_{l}_'
            
            # ln1
            weights[prefix + 'ln1_w'] = np.frombuffer(f.read(C * 4), dtype=np.float32)
            weights[prefix + 'ln1_b'] = np.frombuffer(f.read(C * 4), dtype=np.float32)
            
            # qkv (3C, C)
            weights[prefix + 'qkv_w'] = np.frombuffer(f.read(3 * C * C * 4), dtype=np.float32).reshape(3 * C, C)
            weights[prefix + 'qkv_b'] = np.frombuffer(f.read(3 * C * 4), dtype=np.float32)
            
            # attn_proj (C, C)
            weights[prefix + 'attn_proj_w'] = np.frombuffer(f.read(C * C * 4), dtype=np.float32).reshape(C, C)
            weights[prefix + 'attn_proj_b'] = np.frombuffer(f.read(C * 4), dtype=np.float32)
            
            # ln2
            weights[prefix + 'ln2_w'] = np.frombuffer(f.read(C * 4), dtype=np.float32)
            weights[prefix + 'ln2_b'] = np.frombuffer(f.read(C * 4), dtype=np.float32)
            
            # fc (4C, C)
            weights[prefix + 'fc_w'] = np.frombuffer(f.read(4 * C * C * 4), dtype=np.float32).reshape(4 * C, C)
            weights[prefix + 'fc_b'] = np.frombuffer(f.read(4 * C * 4), dtype=np.float32)
            
            # fc_proj (C, 4C)
            weights[prefix + 'fc_proj_w'] = np.frombuffer(f.read(C * 4 * C * 4), dtype=np.float32).reshape(C, 4 * C)
            weights[prefix + 'fc_proj_b'] = np.frombuffer(f.read(C * 4), dtype=np.float32)
        
        # Final layer norm
        weights['ln_f_w'] = np.frombuffer(f.read(C * 4), dtype=np.float32)
        weights['ln_f_b'] = np.frombuffer(f.read(C * 4), dtype=np.float32)
        
    print(f"Loaded {len(weights)} weight tensors")
    return config, weights


def create_onnx_model(config, weights, with_kv_cache=False):
    """Create ONNX model from weights."""
    if not HAS_ONNX:
        raise ImportError("ONNX library required. Install with: pip install onnx")
    
    C = config['channels']
    L = config['num_layers']
    NH = config['num_heads']
    HS = config['head_dim']
    V = config['vocab_size']
    Vp = config['padded_vocab_size']
    maxT = config['max_seq_len']
    
    # Create graph inputs
    inputs = []
    inputs.append(helper.make_tensor_value_info('input_ids', TensorProto.INT64, ['batch', 'seq_len']))
    
    if with_kv_cache:
        for l in range(L):
            inputs.append(helper.make_tensor_value_info(
                f'past_key_{l}', TensorProto.FLOAT, ['batch', NH, 'past_len', HS]))
            inputs.append(helper.make_tensor_value_info(
                f'past_value_{l}', TensorProto.FLOAT, ['batch', NH, 'past_len', HS]))
    
    # Create graph outputs
    outputs = []
    outputs.append(helper.make_tensor_value_info('logits', TensorProto.FLOAT, ['batch', 'seq_len', V]))
    
    if with_kv_cache:
        for l in range(L):
            outputs.append(helper.make_tensor_value_info(
                f'present_key_{l}', TensorProto.FLOAT, ['batch', NH, 'total_len', HS]))
            outputs.append(helper.make_tensor_value_info(
                f'present_value_{l}', TensorProto.FLOAT, ['batch', NH, 'total_len', HS]))
    
    # Create initializers (weights)
    initializers = []
    
    # Embedding weights
    initializers.append(numpy_helper.from_array(weights['wte'].astype(np.float32), 'wte'))
    initializers.append(numpy_helper.from_array(weights['wpe'].astype(np.float32), 'wpe'))
    
    # Per-layer weights
    for l in range(L):
        prefix = f'layer_{l}_'
        
        # Transpose weight matrices for ONNX MatMul (A @ B where B is weight)
        # ONNX MatMul: [M, K] @ [K, N] = [M, N]
        # Our weights are stored as [out_features, in_features]
        # So we need to transpose to [in_features, out_features]
        
        initializers.append(numpy_helper.from_array(weights[prefix + 'ln1_w'], prefix + 'ln1_weight'))
        initializers.append(numpy_helper.from_array(weights[prefix + 'ln1_b'], prefix + 'ln1_bias'))
        
        # QKV: [3C, C] -> transpose to [C, 3C]
        initializers.append(numpy_helper.from_array(weights[prefix + 'qkv_w'].T.astype(np.float32), prefix + 'qkv_weight'))
        initializers.append(numpy_helper.from_array(weights[prefix + 'qkv_b'], prefix + 'qkv_bias'))
        
        # Attn proj: [C, C] -> transpose
        initializers.append(numpy_helper.from_array(weights[prefix + 'attn_proj_w'].T.astype(np.float32), prefix + 'attn_proj_weight'))
        initializers.append(numpy_helper.from_array(weights[prefix + 'attn_proj_b'], prefix + 'attn_proj_bias'))
        
        initializers.append(numpy_helper.from_array(weights[prefix + 'ln2_w'], prefix + 'ln2_weight'))
        initializers.append(numpy_helper.from_array(weights[prefix + 'ln2_b'], prefix + 'ln2_bias'))
        
        # FC: [4C, C] -> transpose to [C, 4C]
        initializers.append(numpy_helper.from_array(weights[prefix + 'fc_w'].T.astype(np.float32), prefix + 'fc_weight'))
        initializers.append(numpy_helper.from_array(weights[prefix + 'fc_b'], prefix + 'fc_bias'))
        
        # FC proj: [C, 4C] -> transpose to [4C, C]
        initializers.append(numpy_helper.from_array(weights[prefix + 'fc_proj_w'].T.astype(np.float32), prefix + 'fc_proj_weight'))
        initializers.append(numpy_helper.from_array(weights[prefix + 'fc_proj_b'], prefix + 'fc_proj_bias'))
    
    # Final layer norm
    initializers.append(numpy_helper.from_array(weights['ln_f_w'], 'ln_f_weight'))
    initializers.append(numpy_helper.from_array(weights['ln_f_b'], 'ln_f_bias'))
    
    # Create nodes
    nodes = []
    
    # Token embeddings: Gather(wte, input_ids)
    nodes.append(helper.make_node('Gather', ['wte', 'input_ids'], ['token_emb'], axis=0))
    
    # Position embeddings
    # Get sequence length and create position indices
    nodes.append(helper.make_node('Shape', ['input_ids'], ['input_shape']))
    
    # Constants
    initializers.append(numpy_helper.from_array(np.array([1], dtype=np.int64), 'const_1'))
    initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), 'const_0'))
    initializers.append(numpy_helper.from_array(np.array(1, dtype=np.int64), 'const_1_scalar'))
    initializers.append(numpy_helper.from_array(np.array(0, dtype=np.int64), 'const_0_scalar'))
    initializers.append(numpy_helper.from_array(np.array([0], dtype=np.int64), 'squeeze_axes'))
    
    nodes.append(helper.make_node('Gather', ['input_shape', 'const_1'], ['seq_len_raw'], axis=0))
    nodes.append(helper.make_node('Squeeze', ['seq_len_raw', 'squeeze_axes'], ['seq_len']))
    nodes.append(helper.make_node('Range', ['const_0_scalar', 'seq_len', 'const_1_scalar'], ['position_ids']))
    nodes.append(helper.make_node('Gather', ['wpe', 'position_ids'], ['pos_emb'], axis=0))
    
    # Need to unsqueeze position embeddings to match batch dimension
    # pos_emb: [seq_len, C] -> [1, seq_len, C] -> broadcast to [batch, seq_len, C]
    nodes.append(helper.make_node('Add', ['token_emb', 'pos_emb'], ['hidden_states']))
    
    # Transformer blocks
    hidden = 'hidden_states'
    for l in range(L):
        prefix = f'layer_{l}_'
        hidden = create_transformer_block_nodes(nodes, initializers, hidden, l, config, with_kv_cache)
    
    # Final layer norm
    nodes.append(helper.make_node(
        'LayerNormalization',
        [hidden, 'ln_f_weight', 'ln_f_bias'],
        ['ln_f_output'],
        axis=-1,
        epsilon=1e-5
    ))
    
    # Output projection: logits = ln_f_output @ wte.T
    # Transpose wte for final matmul
    initializers.append(numpy_helper.from_array(weights['wte'][:V].T.astype(np.float32), 'wte_T'))
    nodes.append(helper.make_node('MatMul', ['ln_f_output', 'wte_T'], ['logits']))
    
    # Create graph
    graph = helper.make_graph(
        nodes,
        'gpt2_graph',
        inputs,
        outputs,
        initializers
    )
    
    # Create model
    model = helper.make_model(
        graph,
        producer_name='flash-llm',
        opset_imports=[helper.make_opsetid('', 17)]
    )
    
    return model


def create_transformer_block_nodes(nodes, initializers, input_name, layer_idx, config, with_kv_cache):
    """Create nodes for a single transformer block."""
    L = layer_idx
    prefix = f'layer_{L}_'
    C = config['channels']
    NH = config['num_heads']
    HS = config['head_dim']
    
    # LayerNorm 1
    ln1_out = prefix + 'ln1_out'
    nodes.append(helper.make_node(
        'LayerNormalization',
        [input_name, prefix + 'ln1_weight', prefix + 'ln1_bias'],
        [ln1_out],
        axis=-1,
        epsilon=1e-5
    ))
    
    # QKV projection
    qkv_out = prefix + 'qkv_out'
    nodes.append(helper.make_node('MatMul', [ln1_out, prefix + 'qkv_weight'], [prefix + 'qkv_mm']))
    nodes.append(helper.make_node('Add', [prefix + 'qkv_mm', prefix + 'qkv_bias'], [qkv_out]))
    
    # Split Q, K, V
    # Use split input (compatible with opset 13+) instead of num_outputs attribute
    initializers.append(numpy_helper.from_array(
        np.array([C, C, C], dtype=np.int64), prefix + 'split_sizes'))
    
    nodes.append(helper.make_node(
        'Split',
        [qkv_out, prefix + 'split_sizes'],
        [prefix + 'q', prefix + 'k', prefix + 'v'],
        axis=-1
    ))
    
    # Reshape Q, K, V for multi-head attention
    # [batch, seq, C] -> [batch, seq, NH, HS] -> [batch, NH, seq, HS]
    initializers.append(numpy_helper.from_array(
        np.array([0, 0, NH, HS], dtype=np.int64), prefix + 'mh_shape'))
    
    for name in ['q', 'k', 'v']:
        full_name = prefix + name
        nodes.append(helper.make_node('Reshape', [full_name, prefix + 'mh_shape'], [full_name + '_4d']))
        nodes.append(helper.make_node('Transpose', [full_name + '_4d'], [full_name + '_mh'], perm=[0, 2, 1, 3]))
    
    # Handle KV Cache if enabled
    if with_kv_cache:
        # Concatenate past K/V with current K/V
        # past_key_L: [batch, NH, past_len, HS]
        # k_mh: [batch, NH, seq_len, HS]
        # present_key_L: [batch, NH, past_len + seq_len, HS]
        
        nodes.append(helper.make_node(
            'Concat',
            [f'past_key_{L}', prefix + 'k_mh'],
            [f'present_key_{L}'],
            axis=2
        ))
        nodes.append(helper.make_node(
            'Concat',
            [f'past_value_{L}', prefix + 'v_mh'],
            [f'present_value_{L}'],
            axis=2
        ))
        
        # Use present (concatenated) K/V for attention
        k_for_attn = f'present_key_{L}'
        v_for_attn = f'present_value_{L}'
    else:
        k_for_attn = prefix + 'k_mh'
        v_for_attn = prefix + 'v_mh'
    
    # Scaled dot-product attention
    # scores = Q @ K^T / sqrt(HS)
    nodes.append(helper.make_node('Transpose', [k_for_attn], [prefix + 'k_T'], perm=[0, 1, 3, 2]))
    nodes.append(helper.make_node('MatMul', [prefix + 'q_mh', prefix + 'k_T'], [prefix + 'scores']))
    
    scale = 1.0 / np.sqrt(HS)
    initializers.append(numpy_helper.from_array(np.array(scale, dtype=np.float32), prefix + 'scale'))
    nodes.append(helper.make_node('Mul', [prefix + 'scores', prefix + 'scale'], [prefix + 'scaled_scores']))
    
    # Softmax
    nodes.append(helper.make_node('Softmax', [prefix + 'scaled_scores'], [prefix + 'attn_weights'], axis=-1))
    
    # Attention output (use v_for_attn which may include past values)
    nodes.append(helper.make_node('MatMul', [prefix + 'attn_weights', v_for_attn], [prefix + 'attn_out_mh']))
    
    # Reshape back
    # [batch, NH, seq, HS] -> [batch, seq, NH, HS] -> [batch, seq, C]
    nodes.append(helper.make_node('Transpose', [prefix + 'attn_out_mh'], [prefix + 'attn_out_4d'], perm=[0, 2, 1, 3]))
    initializers.append(numpy_helper.from_array(np.array([0, 0, C], dtype=np.int64), prefix + 'c_shape'))
    nodes.append(helper.make_node('Reshape', [prefix + 'attn_out_4d', prefix + 'c_shape'], [prefix + 'attn_out']))
    
    # Attention projection
    nodes.append(helper.make_node('MatMul', [prefix + 'attn_out', prefix + 'attn_proj_weight'], [prefix + 'attn_proj_mm']))
    nodes.append(helper.make_node('Add', [prefix + 'attn_proj_mm', prefix + 'attn_proj_bias'], [prefix + 'attn_proj']))
    
    # Residual 1
    residual1 = prefix + 'residual1'
    nodes.append(helper.make_node('Add', [input_name, prefix + 'attn_proj'], [residual1]))
    
    # LayerNorm 2
    ln2_out = prefix + 'ln2_out'
    nodes.append(helper.make_node(
        'LayerNormalization',
        [residual1, prefix + 'ln2_weight', prefix + 'ln2_bias'],
        [ln2_out],
        axis=-1,
        epsilon=1e-5
    ))
    
    # MLP
    nodes.append(helper.make_node('MatMul', [ln2_out, prefix + 'fc_weight'], [prefix + 'fc_mm']))
    nodes.append(helper.make_node('Add', [prefix + 'fc_mm', prefix + 'fc_bias'], [prefix + 'fc_out']))
    
    # GELU activation - implemented using standard ONNX operators
    # GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # This is the "gelu_new" / "gelu_tanh" approximation used by GPT-2
    
    # Constants for GELU
    initializers.append(numpy_helper.from_array(np.array(0.5, dtype=np.float32), prefix + 'gelu_half'))
    initializers.append(numpy_helper.from_array(np.array(0.044715, dtype=np.float32), prefix + 'gelu_coef'))
    initializers.append(numpy_helper.from_array(np.array(np.sqrt(2.0 / np.pi), dtype=np.float32), prefix + 'gelu_sqrt'))
    initializers.append(numpy_helper.from_array(np.array(1.0, dtype=np.float32), prefix + 'gelu_one'))
    initializers.append(numpy_helper.from_array(np.array(3.0, dtype=np.float32), prefix + 'gelu_three'))
    
    # x^3
    nodes.append(helper.make_node('Pow', [prefix + 'fc_out', prefix + 'gelu_three'], [prefix + 'x_cubed']))
    # 0.044715 * x^3
    nodes.append(helper.make_node('Mul', [prefix + 'x_cubed', prefix + 'gelu_coef'], [prefix + 'x_cubed_scaled']))
    # x + 0.044715 * x^3
    nodes.append(helper.make_node('Add', [prefix + 'fc_out', prefix + 'x_cubed_scaled'], [prefix + 'inner_sum']))
    # sqrt(2/pi) * (x + 0.044715 * x^3)
    nodes.append(helper.make_node('Mul', [prefix + 'inner_sum', prefix + 'gelu_sqrt'], [prefix + 'tanh_input']))
    # tanh(...)
    nodes.append(helper.make_node('Tanh', [prefix + 'tanh_input'], [prefix + 'tanh_out']))
    # 1 + tanh(...)
    nodes.append(helper.make_node('Add', [prefix + 'tanh_out', prefix + 'gelu_one'], [prefix + 'one_plus_tanh']))
    # x * (1 + tanh(...))
    nodes.append(helper.make_node('Mul', [prefix + 'fc_out', prefix + 'one_plus_tanh'], [prefix + 'x_times_tanh']))
    # 0.5 * x * (1 + tanh(...))
    nodes.append(helper.make_node('Mul', [prefix + 'x_times_tanh', prefix + 'gelu_half'], [prefix + 'gelu_out']))
    
    # MLP projection
    nodes.append(helper.make_node('MatMul', [prefix + 'gelu_out', prefix + 'fc_proj_weight'], [prefix + 'fc_proj_mm']))
    nodes.append(helper.make_node('Add', [prefix + 'fc_proj_mm', prefix + 'fc_proj_bias'], [prefix + 'mlp_out']))
    
    # Residual 2
    output = prefix + 'output'
    nodes.append(helper.make_node('Add', [residual1, prefix + 'mlp_out'], [output]))
    
    return output


def verify_onnx_model(model_path):
    """Verify the ONNX model is valid."""
    if not HAS_ONNX:
        print("Skipping verification (ONNX not available)")
        return
    
    print(f"\nVerifying ONNX model: {model_path}")
    model = onnx.load(model_path)
    
    try:
        onnx.checker.check_model(model)
        print("  ✓ Model passed ONNX validation")
    except Exception as e:
        print(f"  ✗ Model validation failed: {e}")
        return
    
    # Print model info
    print(f"  IR version: {model.ir_version}")
    print(f"  Opset version: {model.opset_import[0].version}")
    print(f"  Inputs: {[i.name for i in model.graph.input]}")
    print(f"  Outputs: {[o.name for o in model.graph.output]}")
    print(f"  Nodes: {len(model.graph.node)}")
    print(f"  Initializers: {len(model.graph.initializer)}")


def test_inference(model_path):
    """Test inference with ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("\nSkipping inference test (onnxruntime not available)")
        return
    
    print(f"\nTesting inference with ONNX Runtime...")
    
    session = ort.InferenceSession(model_path)
    
    # Check if this is a KV cache model
    input_names = [inp.name for inp in session.get_inputs()]
    has_kv_cache = any('past_key' in name for name in input_names)
    
    # Create dummy input
    batch_size = 1
    seq_len = 10
    input_ids = np.random.randint(0, 50256, size=(batch_size, seq_len)).astype(np.int64)
    
    print(f"  Input shape: {input_ids.shape}")
    print(f"  KV Cache model: {has_kv_cache}")
    
    if has_kv_cache:
        # For KV cache model, need to provide past_key and past_value for each layer
        # Use empty past (past_len=0) for initial inference
        num_heads = 12
        head_dim = 64
        past_len = 0
        
        feed_dict = {'input_ids': input_ids}
        
        # Count layers from input names
        num_layers = sum(1 for name in input_names if 'past_key_' in name)
        print(f"  Num layers: {num_layers}")
        
        for l in range(num_layers):
            # Empty past KV cache: [batch, num_heads, past_len=0, head_dim]
            feed_dict[f'past_key_{l}'] = np.zeros((batch_size, num_heads, past_len, head_dim), dtype=np.float32)
            feed_dict[f'past_value_{l}'] = np.zeros((batch_size, num_heads, past_len, head_dim), dtype=np.float32)
        
        outputs = session.run(None, feed_dict)
        logits = outputs[0]
        
        print(f"  Output logits shape: {logits.shape}")
        print(f"  Present KV shapes: {outputs[1].shape} (per layer)")
        
    else:
        # Standard model
        outputs = session.run(None, {'input_ids': input_ids})
        logits = outputs[0]
        print(f"  Output shape: {logits.shape}")
    
    print(f"  Output dtype: {logits.dtype}")
    print(f"  ✓ Inference successful!")


def main():
    parser = argparse.ArgumentParser(description='Convert GPT-2 weights to ONNX format')
    parser.add_argument('input', help='Input weights file (llm.c format)')
    parser.add_argument('output', help='Output ONNX file')
    parser.add_argument('--with-kv-cache', action='store_true', help='Include KV cache support')
    parser.add_argument('--verify', action='store_true', help='Verify the output model')
    parser.add_argument('--test', action='store_true', help='Test inference')
    
    args = parser.parse_args()
    
    # Load weights
    config, weights = load_gpt2_weights(args.input)
    
    # Create ONNX model
    print(f"\nCreating ONNX model (with_kv_cache={args.with_kv_cache})...")
    model = create_onnx_model(config, weights, args.with_kv_cache)
    
    # Save model
    print(f"Saving to: {args.output}")
    onnx.save(model, args.output)
    
    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")
    
    # Verify
    if args.verify:
        verify_onnx_model(args.output)
    
    # Test
    if args.test:
        test_inference(args.output)
    
    print("\n✓ ONNX export complete!")


if __name__ == '__main__':
    main()
