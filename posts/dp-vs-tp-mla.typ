#import "@preview/cetz:0.4.2": canvas, draw
#import "@preview/cetz-plot:0.1.3": plot

= DP Attention vs TP in MLA

In MLA (Multi-head Latent Attention), the KV projection is factored into a
_compress_ stage and an _expand_ stage. With weight absorption, the expand
matrix (`kv_b_proj`) is never applied to the KV cache. Instead its K and V
parts are absorbed as per-head BMMs on the Q side and output side. Attention
runs entirely in the compressed latent space.

For TP, the compress projection is *replicated* (every GPU loads the full
weight and processes all tokens), while the expand projections and output are
*sharded* by heads. This creates two asymmetries vs GQA:

+ TP loads more weight per GPU than $W slash P$ (the replicated part doesn't divide)
+ TP does more compute per GPU than DP (redundant work on the replicated projection)

== DeepSeek V3 decode latency

#{
  // DeepSeek V3 MLA dimensions
  let H = 7168
  let n_q = 128
  let d_nope = 128
  let d_rope = 64
  let d_v = 128
  let c_q = 1536
  let c_kv = 512
  let d_qk = d_nope + d_rope

  // Weight bytes per layer (FP8, 1 byte/element)
  let W_rep = H * (c_q + c_kv + d_rope)
  let W_shard = c_q * n_q * d_qk + n_q * d_nope * c_kv + n_q * c_kv * d_v + n_q * d_v * H
  let W_total = W_rep + W_shard

  // GB200
  let M = 8.0     // TB/s HBM
  let R = 1.8     // TB/s AllReduce
  let C = 5000.0  // TFLOPS FP8

  let M_us = M * 1e6
  let R_us = R * 1e6
  let C_us = C * 1e6

  // Parameterised latency functions
  let w_tp(P) = W_rep + W_shard / P

  let t_dp(BQ, P) = {
    let load = W_total / M_us
    let compute = 2.0 * BQ * W_total / (P * C_us)
    calc.max(load, compute)
  }

  let t_tp(BQ, P) = {
    let wt = w_tp(P)
    let load = wt / M_us
    let compute = 2.0 * BQ * wt / C_us
    let ar = 2.0 * BQ * H / R_us
    calc.max(load, compute) + ar
  }

  let find_crossover(P) = {
    let found = none
    for BQ in range(1, 10001, step: 5) {
      if t_tp(BQ, P) >= t_dp(BQ, P) {
        found = BQ
        break
      }
    }
    found
  }

  let W_rep_mb = calc.round(W_rep / 1e6, digits: 1)
  let W_q_mb = calc.round(c_q * n_q * d_qk / 1e6, digits: 1)
  let W_kc_mb = calc.round(n_q * d_nope * c_kv / 1e6, digits: 1)
  let W_vc_mb = calc.round(n_q * c_kv * d_v / 1e6, digits: 1)
  let W_o_mb = calc.round(n_q * d_v * H / 1e6, digits: 1)
  let W_total_mb = calc.round(W_total / 1e6, digits: 1)
  let W_tp8_mb = calc.round(w_tp(8) / 1e6, digits: 1)
  let W_tp32_mb = calc.round(w_tp(32) / 1e6, digits: 1)

  [DeepSeek V3 ($H = #H$, $n_q = #n_q$, $c_q = #c_q$, $c_(k v) = #c_kv$)
   on GB200s, decode ($Q = 1$).]

  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    stroke: 0.5pt,
    table.header([*Component*], [*Bytes (MB)*], [*TP sharding*]),
    [Compress (`fused_qkv_a`)],
      [#W_rep_mb],
      [Replicated],
    [Q expand (`q_b_proj`)],
      [#W_q_mb],
      [$div P$],
    [K absorb BMM (`w_kc`)],
      [#W_kc_mb],
      [$div P$],
    [V absorb BMM (`w_vc`)],
      [#W_vc_mb],
      [$div P$],
    [Output (`o_proj`)],
      [#W_o_mb],
      [$div P$],
    table.hline(),
    [*Total*],
      [*#W_total_mb*],
      [],
    [*TP per-GPU (P=8)*],
      [*#W_tp8_mb*],
      [],
    [*TP per-GPU (P=32)*],
      [*#W_tp32_mb*],
      [],
  )

  let cross8 = find_crossover(8)
  let cross32 = find_crossover(32)
  let cross8_pg = if cross8 != none { calc.round(cross8 / 8) } else { none }
  let cross32_pg = if cross32 != none { calc.round(cross32 / 32) } else { none }
  let BQ_cb_tp = calc.round(C / (2.0 * M))

  canvas({
    import draw: *

    set-style(
      axes: (stroke: .5pt, tick: (stroke: .5pt)),
      legend: (stroke: none, orientation: ttb, item: (spacing: .3), scale: 80%),
    )

    plot.plot(
      size: (12, 7),
      x-label: [Batch size $B$ (decode: $B Q = B$)],
      y-label: [Latency per layer (μs)],
      x-tick-step: 1000,
      y-tick-step: 20,
      x-min: 0, x-max: 4000,
      y-min: 0, y-max: 140,
      legend: "inner-north-west",
      legend-style: (item: (spacing: 0.15), scale: 60%),
      {
        let samples = range(0, 4001, step: 50)

        // P=8
        plot.add(
          samples.map(BQ => (BQ, t_dp(BQ, 8))),
          style: (stroke: (paint: blue, thickness: 1.5pt)),
          label: [DP P=8],
        )
        plot.add(
          samples.map(BQ => (BQ, t_tp(BQ, 8))),
          style: (stroke: (paint: red, thickness: 1.5pt)),
          label: [TP P=8],
        )

        // P=32
        plot.add(
          samples.map(BQ => (BQ, t_dp(BQ, 32))),
          style: (stroke: (paint: blue, thickness: 1.5pt, dash: "dashed")),
          label: [DP P=32],
        )
        plot.add(
          samples.map(BQ => (BQ, t_tp(BQ, 32))),
          style: (stroke: (paint: red, thickness: 1.5pt, dash: "dashed")),
          label: [TP P=32],
        )

        // Crossover lines
        if cross8 != none {
          plot.add(
            ((cross8, 0), (cross8, 140)),
            style: (stroke: (paint: gray, thickness: 0.75pt, dash: "dotted")),
            label: [$B^*_8 slash P approx #cross8_pg$],
          )
        }
        if cross32 != none {
          plot.add(
            ((cross32, 0), (cross32, 140)),
            style: (stroke: (paint: gray, thickness: 0.75pt, dash: "loosely-dotted")),
            label: [$B^*_32 slash P approx #cross32_pg$],
          )
        }

        // TP compute-bound threshold (same for all P)
        plot.add(
          ((BQ_cb_tp, 0), (BQ_cb_tp, 140)),
          style: (stroke: (paint: orange, thickness: 0.5pt, dash: "dotted")),
          label: [$C slash (2 M) = #BQ_cb_tp$],
        )
      },
    )
  })

  let tp8_extra = calc.round(100.0 * (w_tp(8) * 8 / W_total - 1.0), digits: 0)
  let tp32_extra = calc.round(100.0 * (w_tp(32) * 32 / W_total - 1.0), digits: 0)

  [Crossover: TP wins below $B^*$, DP wins above. Per-GPU batch size at crossover:]
  [- $P = 8$: $B^* slash P approx #cross8_pg$ (TP loads #W_tp8_mb MB/GPU,
     #tp8_extra\% extra compute)]
  [- $P = 32$: $B^* slash P approx #cross32_pg$ (TP loads #W_tp32_mb MB/GPU,
     #tp32_extra\% extra compute)]

  [At $P = 32$, DP stays memory-bound across the entire plot
   ($P C slash (2 M) = #calc.round(32 * C / (2.0 * M))$) while TP's
   AllReduce ($2 B H slash R$, independent of $P$) dominates.
   More GPUs push the per-GPU crossover later --- but not by much, because the
   AllReduce cost is the same either way.]

  [*Per-GPU latency formulas:*]

  [$ T_"DP" = max(W_"total" / M, 2 B Q W_"total" / (P C)) $]

  [$ T_"TP" = max(W_"rep" / M + W_"shard" / (P M), 2 B Q (W_"rep" + W_"shard" / P) / C) + (2 B Q H) / R $]

  [where $W_"rep" = H (c_q + c_(k v) + d_r)$ is the replicated compress weight and
  $W_"shard"$ is the head-sharded expand + output weight.]
}
