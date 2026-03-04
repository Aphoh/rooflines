= Attention Mechanisms & KV Cache Math

A reference for understanding how different attention mechanisms affect KV cache memory in LLM inference.

== Notation

#table(
  columns: (auto, auto),
  stroke: 0.5pt + gray,
  inset: 6pt,
  [*Symbol*], [*Meaning*],
  [$L$], [Number of layers],
  [$n_"kv"$], [Number of KV heads],
  [$d_h$], [Head dimension],
  [$b$], [Bytes per element (2 for BF16, 1 for FP8)],
  [$s$], [Sequence length (tokens)],
  [$w$], [Sliding window size],
  [$r_"kv"$], [KV LoRA rank (MLA latent dim)],
  [$d_"rope"$], [RoPE head dimension (MLA)],
)

== MHA / GQA (Multi-Head / Grouped-Query Attention)

Standard attention stores separate K and V vectors per layer, per head:

$ "bytes/token" = 2 dot L dot n_"kv" dot d_h dot b $

The factor of 2 is for K and V. GQA reduces $n_"kv"$ below the number of query heads — e.g. Llama 3.1 70B uses 8 KV heads with 64 query heads.

*Total KV cache* at sequence length $s$:

$ "KV memory" = 2 dot L dot n_"kv" dot d_h dot b dot s $

*Examples:*
- Llama 3.1 8B (32 layers, 8 KV heads, $d_h=128$): $2 times 32 times 8 times 128 times 2 = 131,072$ bytes/token
- Llama 3.1 70B (80 layers, 8 KV heads, $d_h=128$): $2 times 80 times 8 times 128 times 2 = 327,680$ bytes/token

== MLA (Multi-head Latent Attention)

Used by DeepSeek V2/V3/R1, Kimi K2, MiniCPM3, GLM-5.

MLA compresses the KV cache into a low-rank latent vector per token. With weight absorption, the per-head K/V projections are folded into the query side and output side, so the cache stores only the compressed latent plus a small RoPE component:

$ "bytes/token" = L dot (r_"kv" + d_"rope") dot b $

No factor of 2 — the latent jointly encodes both K and V.

*Total KV cache:*

$ "KV memory" = L dot (r_"kv" + d_"rope") dot b dot s $

*Example:*
- DeepSeek V3 (61 layers, $r_"kv"=512$, $d_"rope"=64$): $61 times 576 times 2 = 70,272$ bytes/token

*GLM-5 (MLA + NSA indexer):* GLM-5 uses MLA but also has a DSA indexer that stores FP8-quantized keys per token per layer. The indexer adds $L times 132$ bytes/token (128 for the key + 4 for scales, always uint8):
- MLA: $78 times 576 times 2 = 89,856$ bytes/token
- Indexer: $78 times 132 = 10,296$ bytes/token
- Total: $100,152$ bytes/token BF16, $55,224$ bytes/token FP8

Compared to an equivalent MHA model, MLA typically saves 50-70% KV cache memory.

== SWA (Sliding Window Attention)

Each layer only attends to the most recent $w$ tokens, bounding the KV cache:

$ "KV memory" = 2 dot L dot n_"kv" dot d_h dot b dot min(s, w) $

Some models (GPT-OSS) use a *hybrid* with $L_F$ full attention layers and $L_S$ sliding layers. Beyond the window, only full layers continue to grow:

$ "KV memory" = 2 dot L_F dot n_"kv" dot d_h dot b dot s + 2 dot L_S dot n_"kv" dot d_h dot b dot min(s, w) $

== Hybrid-Linear (Full GQA + Linear Attention)

Used by Qwen3.5. $L_F$ layers use standard GQA, $L_L$ layers use linear attention (GatedDeltaNet). Only full layers contribute per-token KV cache. Linear layers have a fixed recurrent state per request (~4.07 MB/layer: temporal state in FP32 + conv state in BF16).

$ "KV memory" = underbrace(2 dot L_F dot n_"kv" dot d_h dot b dot s, "per-token (full layers)") + underbrace(L_L dot 4.07 "MB", "fixed per-request (linear layers)") $

*Examples:*
- Qwen3.5-397B (15F+45L, 2 KV heads, $d_h=256$): $30,720$ bytes/token + $approx 183$ MB/req fixed
- Qwen3.5-122B (12F+36L, 2 KV heads, $d_h=256$): $24,576$ bytes/token + $approx 147$ MB/req fixed

A 397B model with less bytes/token than a 7B Llama. The fixed cost matters at short contexts but is negligible at long ones.

== Summary

#table(
  columns: (auto, 1fr, auto),
  stroke: 0.5pt + gray,
  inset: 6pt,
  [*Mechanism*], [*Bytes/token formula*], [*Scales with $s$?*],
  [MHA/GQA], [$2 L n_"kv" d_h b$], [Yes, linearly],
  [MLA], [$L(r_"kv" + d_"rope") b$ (+indexer if NSA)], [Yes, linearly],
  [SWA / Hybrid-SWA], [$2 L n_"kv" d_h b dot min(s, w)$], [Bounded],
  [Hybrid-Linear], [$2 L_F n_"kv" d_h b$ + ~4 MB/linear layer fixed], [Yes, + fixed cost],
)

The key insight: reducing KV cache per token directly translates to more concurrent requests in production. A model with half the bytes/token can serve twice as many users from the same GPU memory.
