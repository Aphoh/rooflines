#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/cetz-plot:0.1.3": plot

= DP Attention vs TP in GQA

#align(center)[
  #block(
    fill: luma(235),
    inset: 12pt,
    radius: 4pt,
  )[
    *TL;DR* — For GQA models in memory-bound attention, DP wins over TP
    when

    $ B Q gt.tilde (H + D_k) dot R / M $

    where $B$ is batch size, $Q$ is query tokens per sequence (1 for
    decode, $L$ for prefill, $1 + n_"draft"$ for speculative decoding),
    $H$ is the hidden dimension, $D_k$ is the KV dimension,
    $R$ is the realised AllReduce bandwidth, and $M$ is HBM bandwidth.

    The crossover is *precision-independent*. In practice, prefill
    favors DP attention; decode favors TP up to reasonably large batch
    sizes. Speculative decoding reduces the crossover by
    $1 slash (1 + n_"draft")$.
  ]
]

We derive the crossover between tensor-parallel (TP) and data-parallel (DP)
attention in the memory-bound decode regime. All derivations are symbolic;
numbers appear only at the end.

== Notation

One attention layer with grouped-query attention (GQA) on $P$ GPUs, decoding
a batch of $B$ total tokens at context length $L$.

#table(
  columns: (auto, auto),
  inset: 8pt,
  stroke: 0.5pt,
  table.header([*Symbol*], [*Definition*]),
  [$n_q, n_k$], [query and KV head counts],
  [$d$], [head dimension],
  [$H = n_q d$], [hidden dimension],
  [$D_k = n_k d$], [KV dimension],
  [$b$], [bytes per element (weights, activations, KV cache)],
  [$M$], [HBM bandwidth (bytes/s)],
  [$R$], [realised AllReduce ICI bandwidth (bytes/s)],
)

The Q, K, V, O projections have total weight bytes:

$ W = b (underbrace(H^2, Q) + underbrace(H D_k, K) + underbrace(H D_k, V) + underbrace(H^2, O)) = 2 b H (H + D_k) $

== Per-rank costs

=== Weight reads

TP shards weights across $P$ ranks; DP replicates fully:

$ "TP:" quad W / P #h(3em) "DP:" quad W $

=== KV cache reads

TP processes all $B$ tokens but only $n_k slash P$ heads per rank.
DP processes $B slash P$ tokens but all $n_k$ heads:

$ "TP:" quad B dot L dot n_k / P dot d dot 2 b &= (2 b B L D_k) / P \
  "DP:" quad B / P dot L dot n_k dot d dot 2 b &= (2 b B L D_k) / P $

*Identical.* Define $K = 2 b B L D_k slash P$. The KV cache drops out of any
TP--DP comparison.

=== Communication

After the output projection, TP requires an AllReduce of $2 b B H$ bytes.
At realised ICI bandwidth $R$:

$ T_"ar" = (2 b B H) / R $

DP requires no communication.

== Decode latency

In the memory-bound regime, time is dominated by HBM reads:

$ T_"TP" &= (W slash P + K) / M + (2 b B H) / R \
  T_"DP" &= (W + K) / M $

The identical $K slash M$ terms cancel in the difference:

$ T_"DP" - T_"TP" = underbrace((W (P - 1)) / (P M), "weight-read savings")
  - underbrace((2 b B H) / R, "AllReduce cost") $

TP is faster whenever this is positive.

== Breakeven batch size

Setting $T_"TP" = T_"DP"$ and solving for $B$:

$ (2 b B^* H) / R &= (W (P - 1)) / (P M) $

Substituting $W = 2 b H (H + D_k)$:

$ (2 b B^* H) / R &= (2 b H (H + D_k)(P - 1)) / (P M) $

The $2 b H$ terms cancel on both sides:

$ B^* / R &= ((H + D_k)(P - 1)) / (P M) $

#align(center)[
  #block(
    fill: luma(235),
    inset: 12pt,
    radius: 4pt,
  )[
    $ B^* = ((H + D_k)(P - 1) R) / (P M) $
  ]
]

Note that $b$ cancels entirely: the crossover is *precision-independent* (assuming weights and activations share the same dtype).

== Two constants

The breakeven separates cleanly into a hardware constant and a model constant:

$ B^* = underbrace((H + D_k), "model") dot underbrace((P - 1) / P, "deployment") dot underbrace(R / M, "hardware") $

Below $B^*$, TP is faster. Above $B^*$, DP is faster.

=== Hardware constant: $R slash M$

The ratio of realised AllReduce bandwidth to HBM bandwidth determines how
cheap communication is relative to weight reads.

#{
  let hw = (
    ("H100 SXM", 3.35, 0.9),
    ("GB200",     8.0,  1.8),
  )
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    stroke: 0.5pt,
    table.header([*Hardware*], [$M$ (HBM)], [$R$ (AllReduce)], [$R slash M$]),
    ..hw.map(((name, m, r)) => (
      [#name],
      [#{ calc.round(m, digits: 2) } TB/s],
      [#{ calc.round(r * 1000, digits: 0) } GB/s],
      [#{ calc.round(r / m, digits: 3) }],
    )).flatten(),
  )
}

GB200 has a _lower_ $R slash M$ than H100 despite being faster in absolute
terms---HBM bandwidth improved more than NVLink. This makes TP even more
attractive on GB200: the weight-read savings are worth relatively more.

=== Model constant: $(H + D_k)(P - 1) slash P$

Since $H + D_k = d(n_q + n_k)$, the model factor depends only on head counts
and head dimension. The deployment factor $(P - 1) slash P$ saturates quickly
(${7 slash 8}$ at $P = 8$).

#align(center)[
  #block(
    fill: luma(235),
    inset: 12pt,
    radius: 4pt,
  )[
    *DP attention wins when* $ B > (H + D_k) dot R / M $
  ]
]

If your decode batch exceeds $(H + D_k)$ scaled by $R slash M$, the
AllReduce cost outweighs the weight-read savings.

At small batch ($B approx 0$), TP achieves nearly $P times$ speedup---it reads
$P times$ fewer weights. As $B$ grows, the AllReduce cost erodes this advantage
linearly until crossover at $B^*$.

== Breakeven across models

Using $P = 8$ and the exact formula $B^* = d(n_q + n_k) dot (P - 1) slash P dot R slash M$
for non-MLA models. The "max $L$" column shows the longest context per request
at batch $B^*$, given each GPU's memory, FP8 weights, and FP8 KV cache:

#{
  let P = 8
  let pf = (P - 1) / P
  // (name, M in TB/s, R in TB/s, GPU memory in GB)
  let hw = (
    ("H100", 3.35, 0.9, 80),
    ("GB200", 8.0, 1.8, 192),
  )
  // (name, n_q, n_k, d, layers, total_params_b)
  let models = (
    ("Qwen2.5-7B",     28, 4,  128, 28,  7),
    ("Llama-3.1-8B",   32, 8,  128, 32,  8),
    ("MiniMax-M2.1",   48, 8,  128, 62,  229),
    ("Qwen3-235B",     64, 4,  128, 94,  235),
    ("Llama-3.1-70B",  64, 8,  128, 80,  70),
    ("GLM-4.5",        96, 8,  128, 92,  358),
  )
  table(
    columns: (auto, auto, ..hw.map(_ => (auto, auto)).flatten()),
    inset: 6pt,
    stroke: 0.5pt,
    table.header(
      [*Model*], [$d(n_q + n_k)$],
      ..hw.map(((name, ..)) => ([*#name* $B^*$], [max $L$])).flatten(),
    ),
    ..models.map(((name, nq, nk, d, layers, params_b)) => {
      let dim = d * (nq + nk)
      let kv_per_tok = 2 * layers * nk * d  // fp8 bytes per token
      (
        [#name],
        [#dim],
        ..hw.map(((_, m, r, gpu_gb)) => {
          let gpu_bytes = gpu_gb * 1000000000.0 * 0.9
          let weight_per_gpu = params_b * 1000000000.0 / P
          let avail = gpu_bytes - weight_per_gpu
          let max_tok_per_gpu = avail / kv_per_tok
          let bstar = calc.round(dim * pf * r / m)
          let max_l = calc.round(max_tok_per_gpu * P / bstar)
          let max_l_k = calc.round(max_l / 1000, digits: 1)
          ([#bstar], [#{max_l_k}K])
        }).flatten(),
      )
    }).flatten(),
  )
}

#text(size: 0.85em)[_H100: 80 GB/GPU, GB200: 192 GB/GPU. FP8 weights (1 B/param), FP8 KV cache, $P = 8$._]

For a typical large model ($d(n_q + n_k) approx 9 thin 000$), the breakeven
is around $2 thin 000$ tokens on H100 and $1 thin 800$ on GB200. At those
batch sizes, the per-request context fits comfortably in memory (tens of
thousands of tokens). In practice, memory-bound decode batches rarely reach
$B^*$---by that point you are approaching the compute-bound regime anyway.

== Example: GLM-4.5 on GB200

#{
  let nq = 96
  let nk = 8
  let d = 128
  let P = 8
  let M = 8.0
  let R = 1.8
  let dim = d * (nq + nk)
  let pf = (P - 1) / P
  let bstar = calc.round(dim * pf * R / M)
  let speedup(b) = calc.round(1 / (1.0/P + b * M / (R * dim)), digits: 1)

  [GLM-4.5 ($n_q = #nq$, $n_k = #nk$, $d = #d$) on $P = #P$ GB200s
  ($M = #M$ TB/s, $R = #R$ TB/s):]

  [$ d(n_q + n_k) = #d times #(nq + nk) = #dim $]
  [$ B^* = #dim dot #{calc.round(pf, digits: 3)} dot #{calc.round(R / M, digits: 3)} approx #bstar "tokens" $]

  [At $B = 1$, TP is $#{ speedup(1) } times$ faster (nearly $P times$---the
  full weight-sharding benefit). At $B = 256$, TP is still
  $#{ speedup(256) } times$ faster. DP only catches up at
  $B^* approx #bstar$.]
}

The breakeven scales linearly with realised AllReduce bandwidth. As your
collective becomes more efficient, DP needs a larger batch to win:

#set text(size: 10pt)

#{
  let P = 8
  let pf = (P - 1) / P
  let M = 8.0    // TB/s (GB200)
  let R_peak = 1.8  // TB/s peak NVLink

  let models = (
    ("Qwen2.5-7B", 4096),
    ("Llama-3.1-8B", 5120),
    ("Llama-3.1-70B", 9216),
    ("GLM-4.5", 13312),
  )

  canvas({
    import draw: *

    set-style(
      axes: (stroke: .5pt, tick: (stroke: .5pt)),
      legend: (stroke: none, orientation: ttb, item: (spacing: .3), scale: 80%),
    )

    plot.plot(
      size: (10, 6),
      x-label: [Realised AllReduce (% of 1.8 TB/s peak NVLink)],
      y-label: [Breakeven batch $B^*$ (tokens)],
      x-tick-step: 20,
      y-tick-step: 500,
      x-min: 0, x-max: 100,
      y-min: 0, y-max: 3000,
      legend: "inner-north-west",
      {
        for (name, dim) in models {
          let bstar_max = dim * pf * R_peak / M
          plot.add(
            ((0, 0), (100, bstar_max)),
            label: [#name],
          )
        }
      },
    )
  })
}

#text(size: 0.85em)[_GB200, $P = 8$. Each line shows $B^*$ at a given AllReduce
efficiency; below the line TP wins, above it DP wins._]

#set text(size: 1em)

== Beyond single-token decode

Everything above assumed *decode*: each step produces 1 token per sequence,
so the AllReduce payload is $2 b B H$. But whenever a rank processes $Q > 1$
query tokens per sequence, the AllReduce payload grows by a factor of $Q$:

- *Prefill:* $Q = L$ (the full context length)
- *Speculative decoding:* $Q = 1 + n_"draft"$ (the original token plus draft tokens verified in parallel)

The weight reads are unchanged (you still read every weight once), but the
AllReduce becomes $2 b B Q H$:

$ T_"ar" = (2 b B Q H) / R $

Substituting into the breakeven with $B Q$ in place of $B$:

$ (2 b B^* Q H) / R = (W (P - 1)) / (P M) $

The $2 b H$ cancel exactly as before:

#align(center)[
  #block(
    fill: luma(235),
    inset: 12pt,
    radius: 4pt,
  )[
    $ B^* Q = ((H + D_k)(P - 1) R) / (P M) $
  ]
]

The right-hand side is the same decode $B^*$. With $Q > 1$ query tokens per
sequence, the batch-size breakeven shrinks by a factor of $Q$.

=== GLM-4.5 on GB200

#{
  let nq = 96
  let nk = 8
  let d = 128
  let P = 8
  let M = 8.0
  let R = 1.8
  let dim = d * (nq + nk)
  let pf = (P - 1) / P
  let bstar_decode = calc.round(dim * pf * R / M)

  [With $B^*_(Q=1) approx #bstar_decode$:]

  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    stroke: 0.5pt,
    table.header([*Scenario*], [$Q$], [$B^*$], [*DP wins at batch...*]),
    ..(
      ([Decode], 1),
      ([Spec-dec ($n_"draft" = 4$)], 5),
      ([Spec-dec ($n_"draft" = 16$)], 17),
      ([Prefill 1K], 1024),
      ([Prefill 4K], 4096),
      ([Prefill 32K], 32768),
    ).map(((label, q)) => {
      let bp = calc.round(bstar_decode / q, digits: 1)
      (
        label,
        [#q],
        [#bp],
        [$ >= #{ calc.ceil(bp) }$],
      )
    }).flatten(),
  )
}

Speculative decoding with $n_"draft" = 4$ already cuts $B^*$ by $5 times$,
from $tilde 2600$ to $tilde 520$ — well within typical serving batch sizes.
At $n_"draft" = 16$, the breakeven drops to $tilde 154$.

For prefill, even at context length 1K, the breakeven is about 2.6 requests.
At 4K context, a single request exceeds $B^*$.
In either regime, *DP attention dominates* once $Q$ is appreciable.
