#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/cetz-plot:0.1.3": plot

= DP Attention vs TP in GQA

#align(center)[
  #block(
    fill: luma(235),
    inset: 12pt,
    radius: 4pt,
  )[
    *TL;DR* — DP vs TP crossover for GQA attention:

    $ B Q^* = "harmonic mean"(R / (2 M) (H + D_k), P / 4 dot.c C / M) $

    where $B Q$ is total query tokens, $M$ is HBM bandwidth, $R$ is AllReduce
    bandwidth, $H$ is hidden dimension, $D_k$ is KV dimension, $P$ is parallelism
    degree, and $C$ is compute (FLOPs/s). Assumes FP8 weights and KV cache.
    The crossover is the harmonic mean of bandwidth ratios ($R slash M$, $C slash M$)
    scaled by model and deployment parameters.

    For decode ($Q = 1$), TP wins at low batch; DP wins at high batch.
    For prefill ($Q = L$), DP wins at very low batch sizes. Speculative decoding
    ($Q = 1 + n_"draft"$) cuts $B^*$ proportionally, making DP viable at lower batch.

    Above $B Q = P C slash (2 M)$, DP is always optimal (both compute-bound; TP pays AllReduce).
    For FP8: $C slash (2 M) approx #calc.round(2250 / (2 * 3.35))$ (Hopper), $approx #calc.round(5000 / (2 * 8.0))$ (Blackwell).
  ]
]

We derive the complete crossover between tensor-parallel (TP) and data-parallel
(DP) attention, accounting for both memory-bound and compute-bound regimes.

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
  [$M$], [HBM bandwidth (bytes/s)],
  [$R$], [realised AllReduce ICI bandwidth (bytes/s)],
  [$C$], [peak compute (FLOPs/s per GPU)],
)

The Q, K, V, O projections have total weight bytes (FP8, 1 byte per element):

$ W = underbrace(H^2, Q) + underbrace(H D_k, K) + underbrace(H D_k, V) + underbrace(H^2, O) = 2 H (H + D_k) $

We also add $Q$ for query tokens per sequence (1 for decode, $L$ for prefill, $1 + n_"draft"$ for spec-dec).

== Complete per-GPU latency formulas

At high batch sizes, projection compute can dominate over weight loading from HBM.
We derive complete formulas accounting for both bottlenecks.

=== Complete latency formulas

*Notation:* $B$ is the *global batch size* (total sequences across all GPUs).
- DP: each GPU processes $B slash P$ sequences (batch sharded)
- TP: each GPU processes all $B$ sequences cooperatively (head sharded)

For $B$ sequences with $Q$ query tokens each:
- Projection weights: $W = 2 H (H + D_k)$ bytes (FP8)
- Total FLOPs: $4 B Q H (H + D_k)$
- Per-GPU FLOPs (both approaches): $4 B Q H (H + D_k) slash P$

*Per-GPU latency:*

#align(center)[
  #block(
    fill: luma(235),
    inset: 12pt,
    radius: 4pt,
  )[
    $ T_"DP" &= max(underbrace(W / M, "load"), underbrace((4 B Q H (H + D_k)) / (P C), "compute")) \
      T_"TP" &= max(underbrace(W / (P M), "load"), underbrace((4 B Q H (H + D_k)) / (P C), "compute")) + underbrace((2 B Q H) / R, "AllReduce") $
  ]
]

*DP:* Each GPU loads all $W$ weights and computes on $B slash P$ sequences
($4 (B slash P) Q H (H + D_k)$ FLOPs per GPU).

*TP:* Each GPU loads $W slash P$ weights (sharded) and computes on all $B$
sequences but only $n_k slash P$ heads ($4 B Q H (H + D_k) slash P$ FLOPs per GPU).

Both have identical per-GPU compute. The AllReduce communicates the full $B Q H$
output (all GPUs cooperate on all sequences).

=== Three regimes

The two $max()$ expressions create three cases (Case 4 where DP is compute-bound
but TP is memory-bound is impossible since TP reads fewer weights):

#table(
  columns: (auto, auto, auto, auto),
  inset: 8pt,
  stroke: 0.5pt,
  table.header([*Case*], [*DP regime*], [*TP regime*], [*Winner*]),
  [1], [memory-bound], [memory-bound], [TP (except very small models)],
  [2], [memory-bound], [compute-bound], [DP when $B Q > B Q^*$],
  [3], [compute-bound], [compute-bound], [DP],
)

==== Case 1: Both memory-bound

When both $W slash M$ and $W slash (P M)$ dominate:

$ T_"DP" &= W / M \
  T_"TP" &= W / (P M) + (2 B Q H) / R $

This is the regime analyzed earlier. Setting $T_"TP" = T_"DP"$:

$ W / (P M) + (2 B Q H) / R = W / M $

Solving for $B Q$:

$ B Q^* = (W (P - 1) R) / (2 H P M) = ((H + D_k)(P - 1) R) / (P M) $

*However:* This crossover is only valid when $B Q < B Q_"cb,TP"$ (TP still
memory-bound). TP becomes compute-bound when $B Q = C slash (2 M)$. For
Case 1 to reach its crossover, we need:

$ (H + D_k)(P - 1) R / (P M) < C / (2 M) $

Rearranging: $(H + D_k)(P - 1) slash P < C slash (2 R)$. For typical hardware,
$C slash (2 R) approx #calc.round(2250 / (2 * 0.9))$ (Hopper) or $approx #calc.round(5000 / (2 * 1.8))$ (Blackwell), but most models have
$(H + D_k)(P - 1) slash P > 4000$. Therefore, this crossover *never happens*
in practice---TP becomes compute-bound first.

==== Case 2: DP memory-bound, TP compute-bound

When DP is memory-bound but TP has become compute-bound ($B Q > B Q_"cb,TP"$):

$ T_"DP" &= W / M \
  T_"TP" &= (4 B Q H (H + D_k)) / (P C) + (2 B Q H) / R $

*This is where the actual crossover happens* in practice. Both compute and
AllReduce terms grow linearly with $B Q$. Setting $T_"TP" = T_"DP"$:

$ (4 B Q H (H + D_k)) / (P C) + (2 B Q H) / R = W / M $

Factor out $B Q H$ and solve:

$ B Q^* = (W P C R) / (M (4 H (H + D_k) R + 2 H P C)) $

Substitute $W = 2 H (H + D_k)$ and simplify:

$ B Q^* = (C P R (H + D_k)) / (M (C P + 2 R (H + D_k))) $

Rewrite with each constant in minimal factors. Divide numerator and denominator
by $2 M$ to expose bandwidth ratios:

$ B Q^* = 1 / (underbrace(M / (R (H + D_k)), "communication") + underbrace(2 M / (P C), "compute")) $

This is the *harmonic mean* with hardware ratios separated from model/deployment!

#align(center)[
  #block(
    fill: luma(235),
    inset: 12pt,
    radius: 4pt,
  )[
    $ B Q^* = "harmonic mean"(underbrace(R / (2 M), "hardware") (H + D_k), P / 4 dot.c underbrace(C / M, "hardware")) $
  ]
]

The crossover batch size is the harmonic mean of:
- *AllReduce ratio* $times$ *model dimension*: $R slash (2 M) dot.c (H + D_k)$
- *Deployment factor* $times$ *compute ratio*: $P slash 4 dot.c C slash M$

This form cleanly separates hardware ratios ($R slash M$, $C slash M$) from
model dimension ($H + D_k$) and deployment ($P$). For GB200: $R slash M = #calc.round(1.8 / 8.0, digits: 3)$
and $C slash M = #calc.round(5000 / 8.0)$ (FP8).

Valid when $B Q_"cb,TP" < B Q < B Q_"cb,DP"$. For typical models/hardware,
the crossover falls in this regime.

==== Case 3: Both compute-bound

When both approaches hide weight loads behind compute (at very high $B Q$):

$ T_"DP" &= (4 B Q H (H + D_k)) / (P C) \
  T_"TP" &= (4 B Q H (H + D_k)) / (P C) + (2 B Q H) / R $

*DP always wins*---identical compute per rank, but TP pays AllReduce cost.

=== Example: GLM-4.5 decode on GB200

#{
  let P = 8
  let M = 8.0  // TB/s
  let R = 1.8  // TB/s
  let C = 5000.0  // TFLOPS (FP8)
  let nq = 96
  let nk = 8
  let d = 128
  let H = nq * d
  let D_k = nk * d
  let dim = d * (nq + nk)
  let W_bytes = 2 * H * (H + D_k)  // FP8
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
    let ar = 2 * BQ * H / R_us  // FP8
    calc.max(load, compute) + ar
  }

  // Compute key thresholds
  let BQ_cb_tp = calc.round(C * 1e12 / (2 * M * 1e12))
  let BQ_cb_dp = calc.round(C * P * 1e12 / (2 * M * 1e12))

  // Find crossover numerically (where TP and DP curves intersect)
  let crossover_BQ = {
    let found = none
    for BQ in range(0, 6001, step: 10) {
      if t_tp(BQ) < t_dp(BQ) and t_tp(BQ + 10) >= t_dp(BQ + 10) {
        found = BQ + 10
        break
      }
    }
    found
  }

  [GLM-4.5 ($n_q = #nq$, $n_k = #nk$, $d = #d$) on $P = #P$ GB200s, decode ($Q = #Q$).]

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

        // Mark the crossover point (Case 2)
        if crossover_BQ != none {
          plot.add(
            ((crossover_BQ, 0), (crossover_BQ, 120)),
            style: (stroke: (paint: gray, thickness: 1pt, dash: "dashed")),
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

  [The plot shows three regimes:]
  [- *Case 1* ($B Q < #BQ_cb_tp$): Both memory-bound, TP (except very small models)]
  [- *Case 2* ($#BQ_cb_tp < B Q < #BQ_cb_dp$): TP compute-bound, DP memory-bound. DP when $B Q > B Q^*$]
  [- *Case 3* ($B Q > #BQ_cb_dp$): Both compute-bound, DP]
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
