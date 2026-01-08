# KV Cache Calculator

A web-based tool to compare KV cache memory requirements across LLM models with interactive charts.

**[Live Demo](https://yourusername.github.io/kv-cache-calculator/)**

## Features

- **Pre-loaded Model Configs**: Compare popular models without hitting HuggingFace API
- **Interactive Charts**: Bar chart comparison and log-log sequence length plot
- **MLA Support**: Multi-head Latent Attention (DeepSeek-V2/V3/R1, Kimi K2)
- **Sliding Window**: Shows bounded memory for SWA models (Mistral, Gemma)
- **Custom Models**: Add any HuggingFace model for comparison
- **Data Type Toggle**: Compare BF16 vs FP8 KV cache

## Pre-loaded Models

| Model | Type | Layers | KV Heads | BF16 B/tok | 128K BF16 |
|-------|------|--------|----------|------------|-----------|
| DeepSeek-V3 | MLA | 61 | - | 70,272 | 8.58 GB |
| DeepSeek-R1 | MLA | 61 | - | 70,272 | 8.58 GB |
| Kimi-K2 | MLA | 61 | - | 70,272 | 8.58 GB |
| GLM-4.5 | MHA | 92 | 8 | 376,832 | 46.00 GB |
| MiniMax-M2.1 | MHA | 62 | 8 | 253,952 | 31.00 GB |
| Qwen3-235B-A22B | MHA | 94 | 4 | 192,512 | 23.50 GB |
| Qwen2.5-72B | MHA | 80 | 8 | 327,680 | 40.00 GB |
| Qwen2.5-7B | MHA | 28 | 4 | 57,344 | 7.00 GB |
| Mixtral-8x7B | MHA | 32 | 8 | 131,072 | 16.00 GB |
| Mistral-7B | SWA | 32 | 8 | 131,072 | 512 MB* |
| Llama-3.1-70B | MHA | 80 | 8 | 327,680 | 40.00 GB |
| Llama-3.1-8B | MHA | 32 | 8 | 131,072 | 16.00 GB |

\* Bounded by sliding window (4096 tokens)

## Key Insights

```
• MLA models average 70,272 bytes/tok vs MHA average 224,768 bytes/tok
  → MLA saves ~69% KV cache memory on average

• Most efficient: Qwen2.5-7B at 57,344 bytes/tok (BF16)
• Least efficient: GLM-4.5 at 376,832 bytes/tok (BF16)
```

## Formulas

### Standard MHA
```
KV Cache = 2 × num_layers × num_kv_heads × head_dim × bytes_per_element
```

### MLA (Multi-head Latent Attention)
```
KV Cache = num_layers × (kv_lora_rank + qk_rope_head_dim) × bytes_per_element
```

### Sliding Window Attention
```
- seq_len ≤ window_size: KV grows linearly
- seq_len > window_size: KV bounded at (bytes_per_token × window_size)
```

## Usage

### Open Locally
```bash
open index.html
# Or with a server:
python3 -m http.server 8080
```

### Run Tests
```bash
python3 test.py
```

## Deploy to GitHub Pages

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/kv-cache-calculator.git
git push -u origin main
```

Then enable Pages in Settings → Pages → Select "main" branch.

## Files

| File | Purpose |
|------|---------|
| `index.html` | Main comparison tool with charts |
| `test.py` | Python test suite |
| `test.html` | Browser-based tests |
| `test.js` | Node.js tests |

## License

MIT
