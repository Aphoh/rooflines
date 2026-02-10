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
  set text(size: 0.7em)
  table(
    columns: (auto, auto, ..hw.map(_ => (auto, auto)).flatten()),
    inset: 4pt,
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

== Compute-bound regime

The derivation above assumes *memory-bound* projections. But at high batch
sizes, projection compute dominates. Let's derive the complete time formulas
and systematically analyze each case.

=== Complete latency formulas

*Notation:* $B$ is the *global batch size* (total sequences across all GPUs).
- DP: each GPU processes $B slash P$ sequences (batch sharded)
- TP: each GPU processes all $B$ sequences cooperatively (head sharded)

For $B$ sequences with $Q$ query tokens each:
- Projection weights: $W = 2 b H (H + D_k)$ bytes
- Total FLOPs: $4 B Q H (H + D_k)$
- Per-GPU FLOPs (both approaches): $4 B Q H (H + D_k) slash P$
- Peak compute per GPU: $C$ FLOPs/s

*Per-GPU latency:*

#align(center)[
  #block(
    fill: luma(235),
    inset: 12pt,
    radius: 4pt,
  )[
    $ T_"DP" &= max(underbrace(W / M, "load"), underbrace((4 B Q H (H + D_k)) / (P C), "compute")) \
      T_"TP" &= max(underbrace(W / (P M), "load"), underbrace((4 B Q H (H + D_k)) / (P C), "compute")) + underbrace((2 b B Q H) / R, "AllReduce") $
  ]
]

*DP:* Each GPU loads all $W$ weights and computes on $B slash P$ sequences
($4 (B slash P) Q H (H + D_k)$ FLOPs per GPU).

*TP:* Each GPU loads $W slash P$ weights (sharded) and computes on all $B$
sequences but only $n_k slash P$ heads ($4 B Q H (H + D_k) slash P$ FLOPs per GPU).

Both have identical per-GPU compute. The AllReduce communicates the full $B Q H$
output (all GPUs cooperate on all sequences).

=== Case analysis

The two $max()$ expressions create four potential cases:

#table(
  columns: (auto, auto, auto),
  inset: 8pt,
  stroke: 0.5pt,
  table.header([*Case*], [*DP regime*], [*TP regime*]),
  [1], [memory-bound], [memory-bound],
  [2], [memory-bound], [compute-bound],
  [3], [compute-bound], [compute-bound],
  [4], [compute-bound], [memory-bound],
)

*Case 4 is impossible:* TP reads fewer weights ($W slash P < W$), so if DP
is compute-bound (load hidden), TP must also be compute-bound.

==== Case 1: Both memory-bound

When both $W slash M$ and $W slash (P M)$ dominate:

$ T_"DP" &= W / M \
  T_"TP" &= W / (P M) + (2 b B Q H) / R $

This is the regime analyzed earlier. Setting $T_"TP" = T_"DP"$:

$ W / (P M) + (2 b B Q H) / R = W / M $

Solving for $B Q$:

$ B Q^* = (W (P - 1) R) / (2 b H P M) = ((H + D_k)(P - 1) R) / (P M) $

*However:* This crossover is only valid when $B Q < B Q_"cb,TP"$ (TP still
memory-bound). For typical hardware, $B Q^*$ exceeds $B Q_"cb,TP"$, so this
crossover *never happens*. TP becomes compute-bound before reaching the
memory-bound crossover point.

==== Case 2: DP memory-bound, TP compute-bound

When DP is memory-bound but TP has become compute-bound ($B Q > B Q_"cb,TP"$):

$ T_"DP" &= W / M \
  T_"TP" &= (4 B Q H (H + D_k)) / (P C) + (2 b B Q H) / R $

*This is where the actual crossover happens* in practice. Both compute and
AllReduce terms grow linearly with $B Q$. Setting $T_"TP" = T_"DP"$:

$ (4 B Q H (H + D_k)) / (P C) + (2 b B Q H) / R = W / M $

Factor out $B Q H$ and solve:

$ B Q^* = (W P C R) / (M (4 H (H + D_k) R + 2 b H P C)) $

Substitute $W = 2 b H (H + D_k)$ and simplify:

$ B Q^* = (C P R b (H + D_k)) / (M (C P b + 2 R (H + D_k))) $

Rewrite with each constant in minimal factors (divide by $P C b$):

$ B Q^* = 1 / (underbrace(M / (R (H + D_k)), "communication") + underbrace(2 M / (P C b), "compute")) $

This is the *harmonic mean* of two bottlenecks! The denominator is
$2 slash "HM"$ where $"HM" = "harmonic mean"(R (H + D_k), P C b slash 2)$:

#align(center)[
  #block(
    fill: luma(235),
    inset: 12pt,
    radius: 4pt,
  )[
    $ B Q^* = 1 / (2 M) times "harmonic mean"(underbrace(R (H + D_k), "communication"), underbrace(P C b slash 2, "compute")) $
  ]
]

The crossover batch size is inversely proportional to memory bandwidth $M$ and
proportional to the harmonic mean of:
- *Communication bandwidth* $times$ *model dimension*: $R (H + D_k)$
- *Compute capacity*: $P C b slash 2$

The harmonic mean naturally balances these two bottlenecks. As $P$ increases,
the compute term grows and $B Q^*$ grows.

Valid when $B Q_"cb,TP" < B Q < B Q_"cb,DP"$. For typical models/hardware,
the crossover falls in this regime.

==== Case 3: Both compute-bound

When both approaches hide weight loads behind compute ($B Q > B_"cb,DP" Q$):

$ T_"DP" &= (4 B Q H (H + D_k)) / (P C) \
  T_"TP" &= (4 B Q H (H + D_k)) / (P C) + (2 b B Q H) / R $

*DP always wins* in this regime---identical compute, but TP pays AllReduce cost.

=== Compute-bound thresholds

TP becomes compute-bound when load time equals compute time:

$ W / (P M) = (4 B Q_"cb,TP" H (H + D_k)) / (P C) $

$ B Q_"cb,TP" = (W C) / (4 H (H + D_k) M) = (b C) / (2 M) $

Similarly for DP:

$ B Q_"cb,DP" = (W C P) / (4 H (H + D_k) M) = (b C P) / (2 M) = P dot B Q_"cb,TP" $

Note that $B Q_"cb,TP"$ is *hardware-dependent only* (independent of model
architecture). This is why all models have the same compute-bound threshold on
given hardware.

=== Example: GLM-4.5 decode on GB200

#{
  let P = 8
  let M = 8.0  // TB/s
  let R = 1.8  // TB/s
  let C = 4000.0  // TFLOPS
  let b_bytes = 2  // bf16
  let nq = 96
  let nk = 8
  let d = 128
  let H = nq * d
  let D_k = nk * d
  let dim = d * (nq + nk)
  let W_bytes = 2 * b_bytes * H * (H + D_k)
  let Q = 1  // decode

  // Convert everything to microseconds for the plot
  let M_us = M * 1e12 / 1e6  // TB/s -> B/us
  let R_us = R * 1e12 / 1e6
  let C_us = C * 1e12 / 1e6  // TFLOPS -> FLOP/us

  let t_dp(BQ) = {
    let load = W_bytes / M_us
    let compute = 4 * BQ * H * (H + D_k) / (P * C_us)
    calc.max(load, compute)
  }

  let t_tp(BQ) = {
    let load = W_bytes / (P * M_us)
    let compute = 4 * BQ * H * (H + D_k) / (P * C_us)
    let ar = 2 * b_bytes * BQ * H / R_us
    calc.max(load, compute) + ar
  }

  // Compute key thresholds
  let BQ_cb_tp = calc.round(b_bytes * C * 1e12 / (2 * M * 1e12))
  let BQ_cb_dp = calc.round(b_bytes * C * P * 1e12 / (2 * M * 1e12))
  let BQ_star_mem = calc.round(dim * (P - 1) * R / (P * M))

  [GLM-4.5 ($n_q = #nq$, $n_k = #nk$, $d = #d$) on $P = #P$ GB200s, decode ($Q = #Q$).]

  [*Thresholds:*]
  [- $B Q_"cb,TP" = #BQ_cb_tp$ (TP becomes compute-bound)]
  [- $B Q^*_"mem" = #BQ_star_mem$ (memory-bound crossover, *never reached*)]
  [- $B Q_"cb,DP" = #BQ_cb_dp$ (DP becomes compute-bound)]

  [Since $B Q_"cb,TP" < B Q^*_"mem"$, TP becomes compute-bound before the
  memory-bound crossover. The actual crossover occurs in Case 2 (mixed regime).]

  canvas({
    import draw: *

    set-style(
      axes: (stroke: .5pt, tick: (stroke: .5pt)),
      legend: (stroke: none, orientation: ttb, item: (spacing: .3), scale: 80%),
    )

    plot.plot(
      size: (12, 7),
      x-label: [Total query tokens $B Q$ (decode: $Q = 1$)],
      y-label: [Latency per layer (μs)],
      x-tick-step: 1000,
      y-tick-step: 20,
      x-min: 0, x-max: 6000,
      y-min: 0, y-max: 120,
      legend: "inner-north-west",
      {
        // Sample points for the plot (BQ from 0 to 6000)
        let samples = range(0, 6001, step: 100)
        let dp_points = samples.map(BQ => (BQ, t_dp(BQ)))
        let tp_points = samples.map(BQ => (BQ, t_tp(BQ)))

        plot.add(
          dp_points,
          style: (stroke: (paint: blue, thickness: 1.5pt)),
          label: [DP],
        )
        plot.add(
          tp_points,
          style: (stroke: (paint: red, thickness: 1.5pt)),
          label: [TP],
        )

        // Mark the crossover point (find it numerically)
        let crossover = none
        for i in range(1, samples.len()) {
          let BQ_prev = samples.at(i - 1)
          let BQ_curr = samples.at(i)
          if t_tp(BQ_prev) < t_dp(BQ_prev) and t_tp(BQ_curr) >= t_dp(BQ_curr) {
            crossover = BQ_curr
            break
          }
        }

        if crossover != none {
          // Draw vertical line at crossover
          plot.add(
            ((crossover, 0), (crossover, 120)),
            style: (stroke: (paint: gray, thickness: 0.5pt, dash: "dashed")),
          )
        }

        // Mark BQ_cb,TP threshold
        plot.add(
          ((BQ_cb_tp, 0), (BQ_cb_tp, 120)),
          style: (stroke: (paint: red, thickness: 0.5pt, dash: "dotted")),
        )

        // Mark BQ_cb,DP threshold
        plot.add(
          ((BQ_cb_dp, 0), (BQ_cb_dp, 120)),
          style: (stroke: (paint: blue, thickness: 0.5pt, dash: "dotted")),
        )
      },
    )
  })

  [The plot shows:]
  [- TP (red) starts lower due to $W slash P$ vs $W$]
  [- At $B Q = #BQ_cb_tp$ (red dotted), TP becomes compute-bound and curve steepens]
  [- DP (blue) stays flat (memory-bound) until $B Q = #BQ_cb_dp$ (blue dotted)]
  [- Crossover (gray dashed) occurs in Case 2, where TP is compute-bound but DP is memory-bound]
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
    inset: 4pt,
    stroke: 0.5pt,
    table.header([*Scenario*], [$Q$], [$B^*$], [*DP wins at batch...*]),
    ..(
      ([#text(size: 0.7em)[Decode]], 1),
      ([#text(size: 0.7em)[Spec-dec ($n_"draft" = 4$)]], 5),
      ([#text(size: 0.7em)[Spec-dec ($n_"draft" = 16$)]], 17),
      ([#text(size: 0.7em)[Prefill 1K]], 1024),
      ([#text(size: 0.7em)[Prefill 4K]], 4096),
      ([#text(size: 0.7em)[Prefill 32K]], 32768),
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

*Caveat:* The analysis above assumes memory-bound attention. At high batch
sizes, both approaches become compute-bound. See § Compute-bound regime for
complete latency formulas with $max("load", "compute")$. However, DP requires
evenly distributed work; when batches are imbalanced, TP is preferable.
