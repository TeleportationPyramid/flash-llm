# flash-llm

A Java implementation of GPT-2 training and inference using GPU acceleration via the [Flash](https://github.com/teleportation-pyramid/flash) library.

This project is inspired by and references [Andrej Karpathy's llm.c](https://github.com/karpathy/llm.c), implementing the same GPT-2 124M fine-tuning workflow in Java with CUDA acceleration.

## Features

- **GPT-2 124M Training**: Full parameter fine-tuning on Shakespeare dataset
- **GPU Acceleration**: CUDA-accelerated training via Flash library
- **llm.c Compatibility**: Loads weights and data in llm.c format
- **Advanced Sampling**: Text generation with top-k, top-p (nucleus), and temperature sampling
- **Training Utilities**: LR scheduling, gradient clipping (optional)

## Quick Start

### Prerequisites

- Java 22+ (with Foreign Function & Memory API)
- NVIDIA GPU with CUDA support
- Maven

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

# Run training
mvn exec:java -Dexec.mainClass="com.flashllm.GPT2PretrainedTraining"
```

### Expected Output

```
================================================================
     GPT-2 124M Pretrained Fine-tuning on Shakespeare
     Phase 1.5: Full Training with Advanced Features
================================================================

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

Training complete!
```

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

The debug state file is used to validate forward/backward pass correctness.

```bash
python train_gpt2.py --write_debug_state
```

This creates:
- `gpt2_124M_debug_state.bin` - Contains input tokens, targets, logits, loss, and gradients for validation

### Step 6: Copy Files to flash-llm

```bash
# Create directory
mkdir -p /path/to/flash-llm/src/main/resources/gpt2/

# Copy files
cp gpt2_124M.bin /path/to/flash-llm/src/main/resources/gpt2/
cp gpt2_tokenizer.bin /path/to/flash-llm/src/main/resources/gpt2/
cp dev/data/tinyshakespeare/tiny_shakespeare_train.bin /path/to/flash-llm/src/main/resources/gpt2/
cp dev/data/tinyshakespeare/tiny_shakespeare_val.bin /path/to/flash-llm/src/main/resources/gpt2/
cp gpt2_124M_debug_state.bin /path/to/flash-llm/src/main/resources/gpt2/  # Optional
```

## Project Structure

```
flash-llm/
├── src/main/java/com/flashllm/
│   ├── GPT2PretrainedTraining.java    # Main training script
│   ├── GPT2WeightLoader.java          # Load llm.c format weights
│   ├── GPT2DebugValidator.java        # Validate forward pass
│   ├── GPT2GradientValidator.java     # Validate backward pass
│   ├── backend/
│   │   └── FlashBackend.java          # GPU backend wrapper
│   ├── kernel/
│   │   ├── Encoder.java               # Token + position embedding
│   │   ├── LayerNorm.java             # Layer normalization
│   │   ├── Matmul.java                # Matrix multiplication
│   │   ├── Attention.java             # Multi-head attention
│   │   ├── Gelu.java                  # GELU activation
│   │   ├── Softmax.java               # Softmax + cross-entropy
│   │   ├── AdamW.java                 # AdamW optimizer
│   │   └── TensorUtils.java           # Tensor utilities
│   ├── model/
│   │   └── TransformerBlock.java      # Transformer block
│   ├── tokenizer/
│   │   └── GPT2TokenizerLoader.java   # Load GPT-2 tokenizer
│   └── training/
│       ├── Generate.java              # Text generation with sampling
│       ├── LRScheduler.java           # Learning rate scheduling
│       ├── GradientClipper.java       # Gradient clipping
│       └── Checkpoint.java            # Model checkpointing
└── src/main/resources/gpt2/
    ├── gpt2_124M.bin
    ├── gpt2_tokenizer.bin
    ├── tiny_shakespeare_train.bin
    ├── tiny_shakespeare_val.bin
    └── gpt2_124M_debug_state.bin
```

## Configuration

Training parameters can be modified in `GPT2PretrainedTraining.java`:

```java
// Training config
private static final int BATCH_SIZE = 4;
private static final int SEQ_LEN = 64;
private static final int NUM_STEPS = 40;

// Optimizer config (matching llm.c defaults)
private static final float LEARNING_RATE = 1e-4f;
private static final float WEIGHT_DECAY = 0.0f;

// Generation config
private static final int GEN_TOP_K = 50;
private static final float GEN_TOP_P = 0.95f;
private static final float GEN_TEMPERATURE = 0.8f;
```

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

## About Flash

[Flash](https://github.com/teleportation-pyramid/flash) is a Java library for GPU-accelerated tensor operations using CUDA. It provides:

- Direct CUDA kernel execution from Java
- Foreign Function & Memory API integration (Java 22+)
- High-performance matrix operations (cuBLAS)
- LLM-specific kernels (attention, layer norm, etc.)

## References

- [llm.c](https://github.com/karpathy/llm.c) - Andrej Karpathy's LLM training in pure C/CUDA
- [Flash](https://github.com/teleportation-pyramid/flash) - Java GPU acceleration library
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Language Models are Unsupervised Multitask Learners

## License

MIT License

## Acknowledgments

Special thanks to:
- Andrej Karpathy for the incredible [llm.c](https://github.com/karpathy/llm.c) project and educational content
