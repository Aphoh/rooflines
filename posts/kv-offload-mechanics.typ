#import "@preview/cetz:0.4.2": canvas, draw

= KV Offload Mechanics

We have multiple tiers of KV cache storage in LLM serving:

- G1: GPU memory
- G2: CPU memory
- G3: NVMe disk
- G4: Remote disk

When working with models that have large KVs, G1 fills up quickly and to serve many concurrent requests offloading may become optimal. 

#let num_params = 229e9
#let gpu_memory = 80e9
#let n_gpus = 32
#let w_per_gpu = 1.1 * num_params/ n_gpus 
#let bytes_per_kv = 128000
#let kv_cache_size = 0.9 * (gpu_memory - w_per_gpu)
#let g2_cache_size = 256e9
#let g3_cache_size = 3.84e12
#let tokens_per_gpu = kv_cache_size / bytes_per_kv
#let total_seqlen = 32000
#let trace_s = 3.82
#let trace_g = 2.59
#let trace_paused_frac = trace_g / (trace_s + trace_g)
#let trace_l_avg = 131000
#let trace_state_size = trace_l_avg * bytes_per_kv
#let trace_paused_bytes_per_live = trace_paused_frac * trace_state_size
#let g1_full_trajectories = kv_cache_size / trace_paused_bytes_per_live
#let g12_full_trajectories = (kv_cache_size + g2_cache_size) / trace_paused_bytes_per_live
#let g123_full_trajectories = (kv_cache_size + g2_cache_size + g3_cache_size) / trace_paused_bytes_per_live
#let g3_paused_states = g3_cache_size / trace_state_size
#let g123_paused_states = (kv_cache_size + g2_cache_size + g3_cache_size) / trace_state_size
#let r(n) = calc.round(n, digits: 2)

For example, let's model Minimax M2.1 on #(n_gpus)xH100. At 229B parameters FP8, this leaves $approx #r(tokens_per_gpu/1e6)$ million tokens per GPU, or $approx #r(tokens_per_gpu/total_seqlen)$ requests each taking #total_seqlen tokens.

Requests are processed at some rate $T_"decode"$ tokens/sec. For some input seqlen $L_"in"$ and output seqlen $L_"out"$, the per request throughput is $R_"decode" = T_"decode" / (L_"out" - L_"in")$.

For example, consider the case of a very short OSL, say 300 tokens, and an extremely well optimized deployment, running at 2000 tokens/sec/gpu. This gives a per request throughput of $R_"decode" = #r(2000 / 300)$ requests/sec.

We can think of our GPU as a queue which both takes in and completes requests at a rate of $R_"decode"$ requests/sec/gpu.

#{
  let g1-color = rgb("#1565c0")
  let g1-fill = rgb("#bbdefb")
  let tier-color = rgb("#6a1b9a")
  let tier-fill = rgb("#e1bee7")
  let arrow-color = rgb("#424242")
  let box-width = 2.2
  let box-height = 1.1

  let tier-box(pos, label, stroke-color, fill-color) = {
    let (x, y) = pos
    draw.rect(
      (x - box-width / 2, y - box-height / 2),
      (x + box-width / 2, y + box-height / 2),
      stroke: (paint: stroke-color, thickness: 1.6pt),
      fill: fill-color,
      radius: 8pt,
    )
    draw.content((x, y), text(weight: "bold", fill: stroke-color)[#label])
  }

  align(center)[
    #canvas({
      tier-box((0, 1.4), "G1", g1-color, g1-fill)
      tier-box((-3.1, -1.2), "G2", tier-color, tier-fill)
      tier-box((0, -1.2), "G3", tier-color, tier-fill)
      tier-box((3.1, -1.2), "G4", tier-color, tier-fill)

      draw.line(
        (-5.2, 1.4),
        (-box-width / 2, 1.4),
        stroke: (paint: arrow-color, thickness: 1.4pt),
        mark: (end: "stealth", fill: arrow-color, scale: 0.7),
      )
      draw.content((-5.35, 1.4), anchor: "mid-east", text(weight: "bold")[Prefill])
      draw.content((-3.0, 1.85), text(size: 8pt, fill: arrow-color)[$R_"prefill"$])

      draw.line(
        (box-width / 2, 1.4),
        (5.2, 1.4),
        stroke: (paint: arrow-color, thickness: 1.4pt),
        mark: (end: "stealth", fill: arrow-color, scale: 0.7),
      )
      draw.content((3.15, 1.85), text(size: 8pt, fill: arrow-color)[$R_"decode"$])

      draw.line(
        (-3.1, -0.65),
        (0, 0.35),
        stroke: (paint: arrow-color, thickness: 1.4pt),
      )
      draw.line(
        (0, -0.65),
        (0, 0.35),
        stroke: (paint: arrow-color, thickness: 1.4pt),
      )
      draw.line(
        (3.1, -0.65),
        (0, 0.35),
        stroke: (paint: arrow-color, thickness: 1.4pt),
      )
      draw.line(
        (0, 0.35),
        (0, box-height / 2),
        stroke: (paint: arrow-color, thickness: 1.4pt),
        mark: (end: "stealth", fill: arrow-color, scale: 0.7),
      )
      draw.content((0.65, 0.55), anchor: "mid-west", text(size: 8pt, fill: arrow-color)[$R_"cache"$])
    })
  ]
}

[_Queueing view of G1 with prefill input, decode output, and cache refill from lower tiers._]

Now let's think about G2. Imagine we have many users, and each user submits requests seldom enough that we never get cache hits on G1. For an extremely fast GPU, we could become bottlenecked by the rate we can move requests from G2 to G1 for processing.

= Trajectory timing

The unit we cache is not really a request, but a trajectory state. A trajectory alternates between model execution and external work:
$
  "model generation" -> "tool / environment work" -> "model generation" -> dots
$

Let $S$ be the model generation service time and $G$ be the time between generations while the trajectory is waiting on tool calls, environment work, or orchestration. In our traces, the tool-call gap is tightly concentrated enough that we can treat $G_i$ as a constant $G$ for a first-order model. For the Qwen3.5 122B / Slime+Dynamo / SweBench-pro trace:

- Between-generation tool time: p50 $approx 2.16$s, p95 $approx 2.87$s.
- Non-terminal model generation time: mean $approx 3.82$s, p50 $approx 2.92$s, p95 $approx 10.06$s.
- Summed over non-terminal cycles, time is roughly $60%$ model-running and $40%$ tool-calling. Filtering timeout-like tails gives roughly $63%$ model-running and $37%$ tool-calling.

For a fixed number of live trajectories $N$, the expected number currently running on the model is
$
  N_"run" approx N S / (S + G),
$
and the expected number paused with reusable KV state is
$
  N_"cache" approx N G / (S + G).
$
These are the states competing for offload cache capacity. Since $G_i approx G$, all trajectories have approximately the same resume rate, so popularity is roughly uniform at the trajectory level.

This is the core reason offloading matters for trajectory-like workloads: paused KV is idle from the GPU's point of view, but hot from the trajectory's point of view. Throwing it away forces a large re-prefill on the next generation; keeping all of it in G1 spends scarce HBM on work that is not currently running.


There are three seqlens we care about. From smallest to largest:
- $L_"cached"$: the seqlen of the finished, cached request
- $L_"in"$: the seqlen of the incoming request (our prefill output size would be $L_"in" - L_"cached"$)
- $L_"out"$: the seqlen of the outgoing request (our decode output size would be $L_"out" - L_"in"$)

The request rates of input feeding the system we have are
- The rate of prefill processing new tokens
  - Prefill can generate at the rate $R_"prefill,gen" = T_"prefill" / (L_"in" - L_"cached")$
  - Prefill can transfer at the rate $R_"disagg" = W_"ici" / (D (L_"in" - L_"cached"))$
  - This gives us a maximum prefill rate of $R_"prefill" = min(R_"prefill,gen", R_"disagg")$
- The rate of decode loading tokens from cache
  - Decode can load from G2 at the rate $R_"g2" = W_"g2" / (D L_"cached")$, and similarly for G3 and G4.
  - These loads depend on the hit rate $h_i$ for tier $i$.
  - Assuming we can always hit _some_ tier, we can set $h_4 = 1$
  - This gives us a rate $R_"cache" = h_2 R_"g2" + (1-h_2) h_3 R_"g3" + (1-h_2) (1-h_3) R_"g4"$

The overall request rate of the system is 
$
  R_"system" = min(R_"decode", R_"prefill", R_"cache")
$

= Modeling the cache hit rate

Let's assume our caches behave like standard LRU caches. Even though our object sizes grow over time, we can assume within batch these balance out, and the average object size is $L_"avg" D$.

Che's approximation says the hit probability for object $i$ is $h_i approx 1 - e^(-q_i t_C)$, where $t_C$ is chosen so the expected occupancy equals cache capacity:
$
  C = sum_i^N (1-e^(-q_i t_C)).
$

For $N$ equal-size trajectory states with uniform popularity, $q_i = 1 / N$, so:
$
  C &= N (1 - e^(-t_C / N)) \
  t_C &= -N log(1 - C / N) \
  h &= 1 - e^(-t_C / N) \
    &= 1 - e^(log(1 - C / N)) \
    &= C / N.
$
So in the equal-size, uniform-popularity case, the hit rate is just the fraction of the trajectory working set that fits in cache, capped at $1$.

At the trajectory level, the relevant working set is the paused KV set $N_"cache"$ rather than the total live trajectory count $N$. If the average cached trajectory state has length $L_"avg"$, then for a physical cache capacity $B$ bytes:
$
  W_"KV"(N) = N_"cache" D L_"avg" = N G/(S+G) D L_"avg",
$
and $h_B(N) approx min(B / W_"KV"(N), 1)$.

For two tiers, the cumulative hit rates are
$
  h_"G1"(N) &= min(B_"G1" / W_"KV"(N), 1) \
  h_"G1+G2"(N) &= min((B_"G1" + B_"G2") / W_"KV"(N), 1).
$
Adding G3 gives the next cumulative hit-rate curve:
$
  h_"G1+G2+G3"(N) &= min((B_"G1" + B_"G2" + B_"G3") / W_"KV"(N), 1).
$
The incremental fraction served by G2 is $h_"G1+G2" - h_"G1"$, and the incremental fraction served by G3 is $h_"G1+G2+G3" - h_"G1+G2"$.

Using the trace values $S approx #trace_s "s"$, $G approx #trace_g "s"$, $G / (S + G) approx #r(100 * trace_paused_frac) "%"$, $L_"avg" approx #trace_l_avg " tokens"$, and $D = #r(bytes_per_kv / 1000) "KB/token"$, each live trajectory contributes about $#r(trace_paused_bytes_per_live / 1e9) "GB"$ of paused KV working set. For our H100 Minimax example, the G1 budget is $B_"G1" approx #r(kv_cache_size / 1e9) "GB/GPU"$, G2 is $B_"G2" approx #r(g2_cache_size / 1e9) "GB/GPU"$, and G3 is $B_"G3" approx #r(g3_cache_size / 1e12) "TB/GPU"$ from $8 times 3.84 "TB"$ of node-local NVMe across 8 GPUs, so:

$
  N_"full,G1" approx #r(g1_full_trajectories) " live trajectories/GPU" \
  N_"full,G1+G2" approx #r(g12_full_trajectories) " live trajectories/GPU" \
  N_"full,G1+G2+G3" approx #r(g123_full_trajectories) " live trajectories/GPU".
$

The G3 capacity alone holds about $#r(g3_paused_states)$ paused trajectory states/GPU. The larger live-trajectory number comes from the fact that only about $#r(100 * trace_paused_frac)%$ of live trajectories are paused at a time; cumulatively, G1+G2+G3 holds about $#r(g123_paused_states)$ paused states/GPU, which corresponds to about $#r(g123_full_trajectories)$ live trajectories/GPU.

#figure(
  image("assets/cache_hit_rate_vs_trajectories.png", width: 100%),
  caption: [Estimated cumulative G1, G1+G2, and G1+G2+G3 hit rate as live trajectory concurrency grows. The curves assume uniform trajectory popularity, average reusable sequence length from the trace, and per-GPU cache capacity.],
)

This gives the main offload argument: as concurrency grows past the G1 full-cache point, G1 alone starts dropping hot trajectory state. Adding G2 extends the full-hit regime by roughly
$
  times #r(g12_full_trajectories / g1_full_trajectories),
$
and adding G3 extends it by roughly
$
  times #r(g123_full_trajectories / g1_full_trajectories),
$
converting otherwise repeated prefill into a lower-tier KV load.
