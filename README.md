# Rooflines

A static inference roofline tool for KV-cache capacity and benchmark overlays.

The app is built from TypeScript and generated JSON. It can optionally pull Artificial Analysis benchmark data at build time, then ships static files from the generated `dist/` directory.

## Features

- KV-cache memory comparisons across manually curated LLM architectures
- BF16 and FP8 KV-cache calculations
- MLA, sliding-window, hybrid sliding, and hybrid linear-attention support
- Max concurrent request estimates with weights and KV/state perfectly sharded across selected GPUs
- Intelligence vs max request scatter plot using cached Artificial Analysis data
- Typst-backed posts compiled into static HTML under `dist/posts/`

## Build

```bash
bun run build
```

This runs `scripts/build.ts`, which:

- builds `dist/data/models.json` from `data/models.manual.json` and optional `data/aa-cache.json`
- bundles `src/app.ts` to `dist/assets/app.js`
- copies `index.html` to `dist/index.html`
- compiles Typst posts into `dist/posts/`

`dist/` is generated output and is not checked in.

## Artificial Analysis Data

Do not put API keys in source files or `dist/`.

```bash
cp .env.example .env
# set ARTIFICIAL_ANALYSIS_API_KEY in .env
bun run fetch-aa
bun run build
```

`bun run fetch-aa` writes `data/aa-cache.json`. The static app reads only the normalized benchmark fields emitted into `dist/data/models.json`.

## Local Development

```bash
bun run build
bun run serve
```

Open `http://localhost:8080`. Serving over HTTP is preferred because the app fetches `data/models.json`.

## Tests

```bash
bun test
bunx tsc --noEmit
```

The TypeScript tests cover KV-cache math and the sharded capacity model.

## Capacity Model

For the selected GPU count `N`, the app assumes one perfectly sharded model replica:

```text
max reqs = floor((gpu_memory * utilization - weight_bytes / N) / (request_kv_state_bytes / N))
```

This is not modeling independent data-parallel replicas; both weights and KV/state are divided across the selected GPUs.
