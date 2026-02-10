# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

KV Cache Calculator - a web tool that compares KV cache memory requirements across LLM models. Deployed to GitHub Pages from `dist/`. Also supports blog posts written in Typst.

## Commands

**Run locally:**
```bash
open dist/index.html
# Or with a server:
cd dist && python3 -m http.server 8080
```

**Build:**
```bash
bash build.sh    # Compiles posts, assembles dist/
```

**Run tests:**
```bash
python3 test.py    # Python test suite with verification
node test.js       # Node.js test suite
```

## Directory Structure

```
rooflines/
├── index.html                # Source calculator page (has <!-- POSTS_LIST --> placeholder)
├── posts/                    # Typst source files for blog posts
│   └── .gitkeep
├── build.sh                  # Build script: compiles posts, assembles dist/
├── .pre-commit-config.yaml   # prek config (runs build on commit)
├── dist/                     # Built output (committed, deployed to GitHub Pages)
│   ├── index.html            # Calculator with post links injected
│   └── posts/                # Compiled typst HTML wrapped with site chrome
├── test.py, test.js
└── .github/workflows/pages.yml
```

## Build Process

`build.sh` does the following:
1. Cleans and creates `dist/` and `dist/posts/`
2. Copies `index.html` to `dist/index.html`
3. For each `posts/*.typ` file:
   - Creates two temp typst wrappers in `posts/` that `#include` the original with different page widths
   - Compiles to SVG at two widths: desktop (500pt, 40pt margins) and mobile (350pt, 20pt margins), both with `height: auto`
   - Embeds both SVGs inline in an HTML wrapper with CSS media queries — desktop shown above 600px, mobile at 600px and below
4. Generates HTML post links and replaces `<!-- POSTS_LIST -->` in `dist/index.html`
5. Runs `git add dist/` to stage built files

Posts are rendered as SVG (not typst HTML export) so that math and all typst features work correctly. The dual-width approach handles responsive layout: each width gets its own SVG with appropriate text reflow, and CSS swaps between them.

A prek pre-commit hook runs `build.sh` automatically when `.html` or `.typ` files change.

## Adding a Blog Post

1. Create `posts/my-post.typ` with a `= Title` heading (the first `= ` heading becomes the page title)
2. Run `bash build.sh` (or just commit — the pre-commit hook will run it)
3. The post appears at `dist/posts/my-post.html` and is linked from the index
4. Write standard typst — math, figures, etc. all work since output is SVG

## Architecture

The calculator is a self-contained single-file web application (`index.html`) with all JavaScript embedded. `dist/` is the deployable output.

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
- Models are defined in `BUILTIN_MODELS` (no custom model UI)

### Test Files

`test.py` and `test.js` contain the same calculation logic with expected values for verification. The Python tests include `EXPECTED` dict with manually computed values to catch regressions.
