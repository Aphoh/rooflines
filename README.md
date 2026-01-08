# KV Cache Calculator

A web-based tool to calculate KV cache memory requirements for LLM models. Simply enter a Hugging Face model path and get the KV cache bytes per token for different data types.

**[Live Demo](https://yourusername.github.io/kv-cache-calculator/)**

## Features

- **Automatic Config Fetching**: Fetches model configuration directly from Hugging Face
- **MLA Support**: Full support for Multi-head Latent Attention (MLA) used in DeepSeek-V2/V3/R1 and Kimi K2 models
- **Sliding Window Attention**: Shows both linear growth and bounded memory for SWA models
- **Multiple Data Types**: Calculates KV cache for BF16 and FP8
- **Tensor Parallelism**: Accounts for KV head distribution across GPUs
- **Custom Context Length**: Override default context length for memory planning
- **No Dependencies**: Fully self-contained HTML/JS, no server required

## Supported Architectures

### MLA (Multi-head Latent Attention)
Models using compressed KV cache:
- **DeepSeek-V2 / V2-Lite / V3 / R1** - kv_lora_rank=512, qk_rope_head_dim=64
- **Kimi K2** - Uses MLA with similar compression
- **MiniCPM3** - Smaller MLA model
- **Mistral Large 3** - Also uses MLA

### Standard MHA (Multi-Head Attention)
- **Llama** (all versions including 3.1)
- **Qwen** (all versions)
- **GLM-4 / GLM-4.5**
- **MiniMax-M1/M2**
- **MiMo**
- And most other transformer models

### Sliding Window Attention (SWA)
Models with bounded KV cache after window size:
- **Mistral 7B** - 4096 token window
- **Mixtral 8x7B** - 4096 token window
- **Gemma 2** - Alternating full/sliding attention

## How It Works

### Standard MHA Formula
```
KV Cache per token = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
```

Where:
- `2` accounts for both K and V tensors
- `num_layers` is the number of transformer layers
- `num_kv_heads` is the number of key-value heads (can differ from query heads in GQA)
- `head_dim` is the dimension of each attention head
- `bytes_per_element` is 2 for BF16/FP16, 1 for FP8

### MLA Formula
```
KV Cache per token = num_layers × (kv_lora_rank + qk_rope_head_dim) × bytes_per_element
```

MLA compresses the KV cache into a latent representation, significantly reducing memory:
- `kv_lora_rank` is the dimension of the compressed KV representation (e.g., 512 for DeepSeek-V3)
- `qk_rope_head_dim` is the rotary position embedding dimension (e.g., 64)

### Sliding Window Attention
For models with sliding window attention:
- **Before window size**: KV cache grows linearly (bytes/token × seq_len)
- **After window size**: KV cache is bounded (bytes/token × window_size)

## Example Calculations

| Model | Architecture | Layers | KV Heads | Head Dim | BF16 bytes/tok | FP8 bytes/tok |
|-------|-------------|--------|----------|----------|----------------|---------------|
| DeepSeek-V3/R1 | MLA | 61 | - | - | 70,272 | 35,136 |
| Kimi K2 | MLA | 61 | - | - | 70,272 | 35,136 |
| Llama 3.1 8B | MHA (GQA) | 32 | 8 | 128 | 131,072 | 65,536 |
| Llama 3.1 70B | MHA (GQA) | 80 | 8 | 128 | 327,680 | 163,840 |
| Mistral 7B | MHA + SWA | 32 | 8 | 128 | 131,072* | 65,536* |
| Gemma 2 9B | MHA + SWA | 42 | 8 | 256 | 344,064* | 172,032* |

\* Bounded at sliding window size (typically 4096 tokens)

### Memory at 128K Context

| Model | BF16 Total | FP8 Total | Notes |
|-------|------------|-----------|-------|
| DeepSeek-V3 | 8.6 GB | 4.3 GB | MLA compression |
| Llama 3.1 8B | 16 GB | 8 GB | |
| Llama 3.1 70B | 40 GB | 20 GB | |
| Mistral 7B | 512 MB | 256 MB | Bounded at 4K window |

## Usage

### Option 1: GitHub Pages
Visit the hosted version at: `https://yourusername.github.io/kv-cache-calculator/`

### Option 2: Local
Simply open `index.html` in your browser. No server required!

### Option 3: Python HTTP Server
```bash
cd kv-cache-calculator
python -m http.server 8080
# Open http://localhost:8080
```

## Testing

Open `test.html` in your browser to run the test suite. This verifies the calculations against known model configurations.

```bash
# Or serve and open
python -m http.server 8080
# Open http://localhost:8080/test.html
```

## Deployment to GitHub Pages

1. Create a new repository on GitHub
2. Push this code:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/kv-cache-calculator.git
git push -u origin main
```

3. Go to Settings → Pages
4. Select "main" branch and "/" (root) folder
5. Save and wait for deployment

## License

MIT License - feel free to use and modify as needed.
