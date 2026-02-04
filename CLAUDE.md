# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KV Cache Calculator - a single-page web tool that compares KV cache memory requirements across LLM models. Deployed to GitHub Pages.

## Commands

**Run locally:**
```bash
open index.html
# Or with a server:
python3 -m http.server 8080
```

**Run tests:**
```bash
python3 test.py    # Python test suite with verification
node test.js       # Node.js test suite
```

## Architecture

This is a self-contained single-file web application (`index.html`) with no build system. All JavaScript logic is embedded in the HTML file.

### Key Concepts

**Attention Types:**
- **MHA (Multi-Head Attention):** Standard KV cache - `2 × layers × kv_heads × head_dim × bytes`
- **MLA (Multi-head Latent Attention):** Compressed KV cache (DeepSeek, Kimi) - `layers × (kv_lora_rank + qk_rope_head_dim) × bytes`
- **SWA (Sliding Window Attention):** Memory bounded at window size (Mistral, Gemma)
- **Hybrid-SWA:** Mix of full and sliding attention layers (GPT-OSS) - some layers grow with sequence length, others bounded

**Detection Logic:**
- MLA: config has `kv_lora_rank` OR architecture in `MLA_ARCHITECTURES` list
- SWA: config has `sliding_window` AND `use_sliding_window !== false`
- Hybrid: config has `layer_types` array with mix of `"full_attention"` and `"sliding_attention"`

### Code Structure

- `BUILTIN_MODELS` - Pre-loaded model configurations with architecture parameters
- `calculateKVCache(config)` - Core calculation returning bytes/token for bf16/fp8
- `getKVCacheAtSeqLen(result, seqLen, dtype)` - Memory at specific sequence length (handles SWA bounding)
- Four Chart.js visualizations: bar chart, scatter plot (size vs KV), line chart (seq length scaling), requests chart
- Custom models stored in localStorage

### Test Files

`test.py` and `test.js` contain the same calculation logic with expected values for verification. The Python tests include `EXPECTED` dict with manually computed values to catch regressions.
