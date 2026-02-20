# flash-llm

A Java implementation of GPT-2 training and inference using GPU acceleration via the [Flash](https://github.com/teleportation-pyramid/flash) library.

This project is inspired by and references [Andrej Karpathy's llm.c](https://github.com/karpathy/llm.c), implementing the same GPT-2 124M fine-tuning workflow in Java with CUDA acceleration.

## Features

- **GPT-2 124M Training**: Full parameter fine-tuning on Shakespeare dataset
- **GPU Acceleration**: CUDA-accelerated training via Flash library
- **BF16 Mixed-Precision Training**: BF16 compute with FP32 master weights — no loss scaling needed
- **FP32 Baseline Training**: Reference implementation matching llm.c output
- **llm.c Compatibility**: Loads weights and data in llm.c format
- **Advanced Sampling**: Text generation with top-k, top-p (nucleus), and temperature sampling
- **Flash Attention**: FP32 flash attention with O(n) memory
- **Training Utilities**: LR scheduling, gradient clipping (optional)
- **ONNX Export**: Export trained models for deployment with ONNX Runtime, TensorRT, OpenVINO

## Training Modes

### BF16 Training (Recommended)

BF16 has the same dynamic range as FP32 (8-bit exponent), eliminating the numerical instability issues of FP16:

| Feature | FP16 Mixed-Precision | BF16 |
|---------|---------------------|------|
| Loss Scaling | Required | **Not needed** |
| Overflow Detection | Required | **Not needed** |
| Gradient Unscaling | Required | **Not needed** |
| Dynamic Range | 6e-5 ~ 65504 | **1e-38 ~ 3.4e38** |
| Initial Loss (GPT-2) | 13.08 ❌ | **10.39** ✅ |
| Final Loss (20 steps) | 6.92 | **3.70** |
| Step Time | ~2500 ms | **~1900 ms** |
| GPU Requirement | Any with FP16 Tensor Core | Ampere+ (RTX 30xx, A100) |

```bash
# Run BF16 training
mvn exec:java -Dexec.mainClass="com.flashllm.model.BF16TrainingDemo"
```

Expected output:
```
========================================
BF16 GPT-2 Training Demo
========================================
  No LossScaler needed (BF16 range = FP32)
  No overflow detection needed

  Step   0: loss=10.3932, time=1575ms
  Step   5: loss=9.3131, time=1455ms
  Step  10: loss=6.3478, time=2320ms
  Step  15: loss=4.7677, time=1959ms
  Step  19: loss=3.7040, time=1883ms

  Avg time/step: 1898 ms
  Overflow skips: 0 (BF16 never overflows)
```

### FP32 Baseline Training

Reference implementation that matches llm.c output exactly:

```bash
mvn exec:java -Dexec.mainClass="com.flashllm.GPT2PretrainedTraining"
```

Expected output:
```
val loss 5.251912
step 0: train loss 5.356083 (took 1089 ms)
step 1: train loss 4.282791 (took 1056 ms)
...
step 39: train loss 4.014056 (took 938 ms)
val loss 4.134809

generating (top-k=50, top-p=0.95, temp=0.80):
---
Is this the most noble thing to be said?
---
```

## Quick Start

### Prerequisites

- Java 22+ (with Foreign Function & Memory API)
- NVIDIA GPU with CUDA support (BF16 requires Ampere+: RTX 30xx / A100 / RTX 40xx)
- Maven
- [Flash](https://github.com/teleportation-pyramid/flash) v0.1.10+

### 1. Clone the Repository

```bash
git clone https://github.com/YourUsername/flash-llm.git
cd flash-llm
```

### 2. Prepare Data Files

You need the following files in `src/main/resources/gpt2/`:

| File | Description |
|------|-------------|
| `gpt2_124M.bin` | GPT-2 124M pretrained weights |
| `gpt2_tokenizer.bin` | GPT-2 tokenizer |
| `tiny_shakespeare_train.bin` | Training data |
| `tiny_shakespeare_val.bin` | Validation data |
| `gpt2_124M_debug_state.bin` | Debug state for validation (optional) |

See [Data Preparation](#data-preparation) section below for how to generate these files.

### 3. Build and Run

```bash
# Build
mvn clean compile

# Run BF16 training (recommended)
mvn exec:java -Dexec.mainClass="com.flashllm.model.BF16TrainingDemo"

# Run FP32 baseline training
mvn exec:java -Dexec.mainClass="com.flashllm.GPT2PretrainedTraining"
```

## Project Structure

```
flash-llm/
├── src/main/java/com/flashllm/
│   ├── GPT2PretrainedTraining.java    # FP32 baseline training
│   ├── GPT2WeightLoader.java          # Load llm.c format weights
│   ├── GPT2DebugValidator.java        # Validate forward pass
│   ├── GPT2GradientValidator.java     # Validate backward pass
│   ├── backend/
│   │   └── FlashBackend.java          # GPU backend wrapper
│   ├── config/
│   │   └── GPT2Config.java            # Model configuration
│   ├── kernel/
│   │   ├── Encoder.java               # Token + position embedding
│   │   ├── LayerNorm.java             # Layer normalization
│   │   ├── Matmul.java                # Matrix multiplication
│   │   ├── Attention.java             # Multi-head attention (Flash Attention)
│   │   ├── Gelu.java                  # GELU activation
│   │   ├── Softmax.java               # Softmax + cross-entropy
│   │   ├── AdamW.java                 # AdamW optimizer
│   │   └── TensorUtils.java           # Tensor utilities
│   ├── model/
│   │   ├── BF16GPT2.java             # BF16 mixed-precision model
│   │   ├── BF16TrainingDemo.java      # BF16 training demo
│   │   ├── MixedPrecisionGPT2.java    # FP16 mixed-precision (legacy)
│   │   └── TransformerBlock.java      # FP32 Transformer block
│   ├── export/
│   │   ├── ONNXExporter.java          # ONNX format export
│   │   └── ONNXModelBuilder.java      # ONNX graph builder
│   ├── tokenizer/
│   │   └── GPT2TokenizerLoader.java   # Load GPT-2 tokenizer
│   └── training/
│       ├── Generate.java              # Text generation with sampling
│       ├── GradientClipper.java        # Gradient clipping
│       └── LossScaler.java            # Loss scaling (FP16 only)
└── src/main/resources/gpt2/
    ├── gpt2_124M.bin
    ├── gpt2_tokenizer.bin
    ├── tiny_shakespeare_train.bin
    ├── tiny_shakespeare_val.bin
    └── gpt2_124M_debug_state.bin
├── scripts/
│   └── convert_to_onnx.py             # Python ONNX conversion
```

## ONNX Export

Export trained GPT-2 models to ONNX format for deployment:

### Java Export

```java
GPT2WeightLoader weights = new GPT2WeightLoader();
weights.load("gpt2_124M.bin");

ONNXExporter exporter = new ONNXExporter(weights);

// Standard export
exporter.export("gpt2_124M.onnx");

// With KV cache for autoregressive generation
exporter.exportWithKVCache("gpt2_124M_kv.onnx");

// Weights only (for custom graph construction)
exporter.exportWeightsOnly("gpt2_weights/");
```

### Python Conversion (alternative)

```bash
# Convert llm.c weights to ONNX via PyTorch
python convert_to_onnx.py --input gpt2_124M.bin --output gpt2_124M.onnx
```

### ONNX Model I/O

| Model | Input | Output |
|-------|-------|--------|
| Standard | `input_ids` [batch, seq] int64 | `logits` [batch, seq, 50257] float32 |
| KV Cache | `input_ids` + `past_key/value_N` | `logits` + `present_key/value_N` |

### Deployment Targets

- **ONNX Runtime**: CPU/GPU inference
- **TensorRT**: NVIDIA GPU optimized inference
- **OpenVINO**: Intel hardware inference
- **CoreML**: Apple devices (via onnx-coreml)

## Data Preparation

All data files are generated using scripts from [llm.c](https://github.com/karpathy/llm.c).

### Step 1: Clone llm.c

```bash
git clone https://github.com/karpathy/llm.c.git
cd llm.c
```

### Step 2: Install Python Dependencies

```bash
pip install torch numpy tiktoken
```

### Step 3: Generate GPT-2 Weights and Tokenizer

```bash
# This downloads GPT-2 124M from HuggingFace and converts to llm.c format
python dev/data/download_starter_pack.py

# Or generate manually:
python train_gpt2.py
```

This creates:
- `gpt2_124M.bin` - GPT-2 124M weights (548 MB)
- `gpt2_tokenizer.bin` - Tokenizer data (371 KB)

### Step 4: Generate Shakespeare Dataset

```bash
# Download and tokenize Shakespeare dataset
python dev/data/tinyshakespeare.py
```

This creates in `dev/data/tinyshakespeare/`:
- `tiny_shakespeare_train.bin` - Training tokens
- `tiny_shakespeare_val.bin` - Validation tokens

### Step 5: Generate Debug State (Optional)

```bash
python train_gpt2.py --write_debug_state
```

This creates:
- `gpt2_124M_debug_state.bin` - Contains input tokens, targets, logits, loss, and gradients for validation

### Step 6: Copy Files to flash-llm

```bash
mkdir -p /path/to/flash-llm/src/main/resources/gpt2/
cp gpt2_124M.bin /path/to/flash-llm/src/main/resources/gpt2/
cp gpt2_tokenizer.bin /path/to/flash-llm/src/main/resources/gpt2/
cp dev/data/tinyshakespeare/tiny_shakespeare_train.bin /path/to/flash-llm/src/main/resources/gpt2/
cp dev/data/tinyshakespeare/tiny_shakespeare_val.bin /path/to/flash-llm/src/main/resources/gpt2/
cp gpt2_124M_debug_state.bin /path/to/flash-llm/src/main/resources/gpt2/  # Optional
```

## Configuration

Training parameters can be modified in `BF16TrainingDemo.java`:

```java
// Training config
int B = 4;           // Batch size
int T = 64;          // Sequence length
int numSteps = 20;   // Training steps

// Optimizer config
float lr = 3e-4f;
float beta1 = 0.9f;
float beta2 = 0.999f;
float eps = 1e-8f;
float weightDecay = 0.01f;
```

## Technical Details

### Weight Layout

llm.c GPT-2 weights use PyTorch Conv1D layout:
- `qkvw`: (3C, C) — forward computes `out = inp @ w^T`
- `attprojw`: (C, C)
- `fcw`: (4C, C)
- `fcprojw`: (C, 4C)
- `wte`: (V, C)

### BF16 Training Pipeline

```
Forward:
  BF16 embedding → BF16 layernorm → BF16 matmul → FP32 flash attention
  → BF16 matmul → BF16 gelu → BF16 matmul → FP32 softmax + CE

Backward:
  FP32 dlogits → BF16 matmul → BF16 layernorm backward
  → FP32 flash attention backward → BF16 matmul → BF16 gelu backward

Update:
  BF16 grads → adamwUpdateBf16 → FP32 master weights → BF16 compute weights
```

Key design decisions:
- **Softmax + CrossEntropy**: Always FP32 (precision requirement, not range)
- **Flash Attention**: FP32 internally (QKV converted BF16→FP32→BF16)
- **LayerNorm stats** (mean, rstd): FP32
- **Adam state** (m, v): FP32
- **Master weights**: FP32 (updated by Adam, then cast to BF16)

### Vocabulary Padding

- Real vocabulary size: **50257**
- Padded vocabulary size: **50304** (for CUDA efficiency)
- Softmax computed only over real vocabulary range

## Performance

### FP32 Baseline (vs llm.c)

| Metric | flash-llm | llm.c |
|--------|-----------|-------|
| Initial val loss | 5.25 | 5.27 |
| Step 1 train loss | 4.28 | 4.30 |
| Step 39 train loss | 4.01 | 3.97 |
| Final val loss | 4.13 | 4.11 |
| Speed | ~1000 ms/step | ~1300 ms/step |

### BF16 Training

| Step | Loss | Time |
|------|------|------|
| 0 | 10.39 | 1575 ms |
| 5 | 9.31 | 1455 ms |
| 10 | 6.35 | 2320 ms |
| 15 | 4.77 | 1959 ms |
| 19 | 3.70 | 1883 ms |

Average: **1898 ms/step**, zero overflow skips.

## Validation

To validate your implementation against llm.c:

```bash
# Validate forward pass (logits and loss)
mvn exec:java -Dexec.mainClass="com.flashllm.GPT2DebugValidator"

# Validate backward pass (gradients)
mvn exec:java -Dexec.mainClass="com.flashllm.GPT2GradientValidator"
```

Expected output:
```
Validation PASSED!
- Logits match within tolerance
- Loss matches: 5.2699...
- Gradients match within tolerance
```

## References

- [llm.c](https://github.com/karpathy/llm.c) - Andrej Karpathy's LLM training in pure C/CUDA
- [Flash](https://github.com/teleportation-pyramid/flash) - Java GPU acceleration library
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Language Models are Unsupervised Multitask Learners
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) - Karpathy's course

## License

Apache 2.0 - see [LICENSE](LICENSE) for details.

## Acknowledgments

Special thanks to:
- Andrej Karpathy for the incredible [llm.c](https://github.com/karpathy/llm.c) project and educational content
