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
#let tokens_per_gpu = kv_cache_size / bytes_per_kv
#let total_seqlen = 32000
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

Let's assume our caches behave like standard LRU caches. Even though our object sizes grow over time, we can assume within batch these balance out, and the average size of the objects is $L_"avg" D$.
For a cache of capacity $C$, we can use Che's approximation:
$
  h(n) &approx 1-e^(-q(n)t_C) \
$

For popularity $q(n)$, where $t_C$ is the root of $C &= sum_i^N (1-e^(-q(n)t))$. We have a uniform popularity $q(n) = 1$, so $C = N (1-e^(-t))$, so $t = -log(1-C/N)$ and $h(n) = 1-e^(log(1-C/N)) = C/N$, meaning our hit rate is directly proportional to our cache size v.s. the number of concurrent users we have.

For our H100 minimax example, each GPU gets (2TB per node)/(8 gpus per node) = 256GB of cache. If we have $approx 16$ req/s/gpu, and users take 20s between requests, this gives us $approx 320$ users/gpu, for which storing all the requests in cache would take $approx #r(320 * 32000 * 128000 / 1e9)$ GB. 
This gives us a poor hit rate of $tilde 10%$ if we only use G2, meaning we likely need to offload to G3 or G4.


