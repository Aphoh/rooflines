"""Symbolic verification and numerical tables for dp-vs-tp-attention.typ"""
from sympy import *

H, D_k, P, B, Q, b, M, R, C = symbols('H D_k P B Q b M R C', positive=True)

# Use BQ as total query tokens for cleaner notation
BQ = symbols('BQ', positive=True)

# --- Symbolic verification ---

# Attention weight bytes (Q + K + V + O)
W = 2 * b * H * (H + D_k)

# Load times
T_load_DP = W / M
T_load_TP = W / (P * M)

# Compute time (using BQ for total query tokens)
T_compute = 4 * BQ * H * (H + D_k) / (P * C)

# AllReduce time
T_ar = 2 * b * BQ * H / R

# Complete formulas
T_DP_complete = Max(T_load_DP, T_compute)
T_TP_complete = Max(T_load_TP, T_compute) + T_ar

print("=== Complete formulas ===")
print("T_DP =", T_DP_complete)
print("T_TP =", T_TP_complete)

# Compute-bound thresholds (when load = compute)
BQ_cb_TP = solve(Eq(T_load_TP, T_compute), BQ)[0]
BQ_cb_DP = solve(Eq(T_load_DP, T_compute), BQ)[0]
print("\n=== Compute-bound thresholds ===")
print("BQ_cb,TP =", simplify(BQ_cb_TP))
print("BQ_cb,DP =", simplify(BQ_cb_DP))
print("Ratio: BQ_cb,DP / BQ_cb,TP =", simplify(BQ_cb_DP / BQ_cb_TP))

# Case 1: Both memory-bound (BQ < BQ_cb,TP)
print("\n=== Case 1: Both memory-bound ===")
T_DP_case1 = T_load_DP
T_TP_case1 = T_load_TP + T_ar
BQ_star_case1 = solve(Eq(T_TP_case1, T_DP_case1), BQ)[0]
print("BQ* (memory-bound) =", simplify(BQ_star_case1))
print("Valid when: BQ < BQ_cb,TP")

# Check: is BQ* > BQ_cb,TP? (i.e., does memory-bound crossover happen?)
# If so, the memory-bound formula is invalid at the crossover
BQ_star_simplified = simplify(BQ_star_case1)
BQ_cb_TP_simplified = simplify(BQ_cb_TP)
comparison = simplify(BQ_star_simplified - BQ_cb_TP_simplified)
print(f"\nBQ* - BQ_cb,TP = {comparison}")
print("If positive, TP becomes compute-bound before memory-bound crossover")

# Case 2: DP memory-bound, TP compute-bound (BQ_cb,TP < BQ < BQ_cb,DP)
print("\n=== Case 2: DP memory-bound, TP compute-bound ===")
T_DP_case2 = T_load_DP
T_TP_case2 = T_compute + T_ar
BQ_star_case2 = solve(Eq(T_TP_case2, T_DP_case2), BQ)[0]
print("BQ* (with W) =", simplify(BQ_star_case2))

# Substitute W = 2bH(H+D_k) and simplify
BQ_star_case2_sub = BQ_star_case2.subs(W, 2*b*H*(H + D_k))
BQ_star_case2_simplified = simplify(BQ_star_case2_sub)
print("BQ* (expanded) =", BQ_star_case2_simplified)

# Try factor/cancel
BQ_star_case2_factored = cancel(BQ_star_case2_sub)
print("BQ* (factored) =", BQ_star_case2_factored)

# Factor out (D_k + H) from denominator: CPb + 2(D_k+H)R
BQ_star_case2_manual = C*P*R*b*(D_k + H) / (M*(C*P*b + 2*(D_k + H)*R))
print("BQ* (cleanest) =", simplify(BQ_star_case2_manual))

# Try to separate into model × hardware × parallelism
print("\nFactorization attempts:")

# Factor 1: Pull out (H+D_k) and P from numerator
print("1. (H+D_k) × P × [CRb / (M(CPb + 2R(H+D_k)))]")

# Factor 2: Note that denominator mixes everything
# Try dividing by P to see per-rank batch
BQ_per_rank = C*R*b*(D_k + H) / (M*(C*P*b + 2*(D_k + H)*R))
print("2. BQ*/P (per-rank) =", simplify(BQ_per_rank))

# Factor 3: Get (H+D_k) to appear only once
# Multiply num and denom by P/(H+D_k)
BQ_star_single_model = C*R*b*P / (M*(C*b*P/(D_k + H) + 2*R))
print("3. Factor to isolate (H+D_k):")
print("   BQ* = PCRb / (M(PCb/(H+D_k) + 2R))")
print("   Simplified:", simplify(BQ_star_single_model))

# Verify this equals the original
BQ_star_original = C*P*R*b*(D_k + H) / (M*(C*P*b + 2*R*(D_k + H)))
difference = simplify(BQ_star_single_model - BQ_star_original)
print("   Verification (should be 0):", difference)

print("\n   This separates:")
print("   - Model: (H+D_k)^(-1) in denominator only")
print("   - Hardware: C, R, M, b")
print("   - Parallelism: P")

# Factor 4: Also get R to appear only once (divide by R)
BQ_star_single_R = P*C*b / (M*(P*C*b/(R*(D_k + H)) + 2))
print("\n4. Also isolate R (divide by R):")
print("   BQ* = PCb / (M(PCb/(R(H+D_k)) + 2))")
print("   Simplified:", simplify(BQ_star_single_R))
difference_R = simplify(BQ_star_single_R - BQ_star_original)
print("   Verification (should be 0):", difference_R)

print("\n   Now each hardware constant appears once:")
print("   - (H+D_k): denominator only")
print("   - R: denominator only")
print("   - C, M, b: appear in multiple places")
print("   - P: numerator and denominator")

# Factor 5: Divide by PCb to minimize where each constant appears
BQ_star_minimal = 1 / (M/(R*(D_k + H)) + 2*M/(P*C*b))
print("\n5. Divide by PCb to minimize factor occurrences:")
print("   BQ* = 1 / (M/(R(H+D_k)) + 2M/(PCb))")
print("   Simplified:", simplify(BQ_star_minimal))
difference_minimal = simplify(BQ_star_minimal - BQ_star_original)
print("   Verification (should be 0):", difference_minimal)

print("\n   Each constant appears in MINIMAL factors:")
print("   - (H+D_k): 1st denominator term only")
print("   - R: 1st denominator term only")
print("   - M: coefficient of both denominator terms")
print("   - P: 2nd denominator term only")
print("   - C: 2nd denominator term only")
print("   - b: 2nd denominator term only")

# Factor 6: Recognize as harmonic mean!
# HM(a,b) = 2ab/(a+b), and 1/HM = (1/a + 1/b)/2
# Our expression: 1/(R(H+D_k)) + 2/(PCb) = 1/a + 1/b where a=R(H+D_k), b=PCb/2
# So this is 2/HM where HM is harmonic mean of R(H+D_k) and PCb/2
a_term = R*(D_k + H)
b_term = P*C*b/2
harmonic_mean = 2*a_term*b_term / (a_term + b_term)
BQ_star_harmonic = harmonic_mean / (2*M)
print("\n6. Recognize as harmonic mean:")
print("   Let a = R(H+D_k), b = PCb/2")
print("   HM(a,b) = 2ab/(a+b) =", simplify(harmonic_mean))
print("   BQ* = HM/(2M) =", simplify(BQ_star_harmonic))
difference_harmonic = simplify(BQ_star_harmonic - BQ_star_original)
print("   Verification (should be 0):", difference_harmonic)
print("\n   Interpretation: BQ* = (1/(2M)) × harmonic_mean(R(H+D_k), PCb/2)")
print("   Crossover scales with harmonic mean of:")
print("   - Communication: R(H+D_k) [AllReduce bandwidth × model dimension]")
print("   - Compute: PCb/2 [parallelism × compute × precision / 2]")

print("\nValid when: BQ_cb,TP < BQ < BQ_cb,DP")

# Case 3: Both compute-bound (BQ > BQ_cb,DP)
print("\n=== Case 3: Both compute-bound ===")
T_DP_case3 = T_compute
T_TP_case3 = T_compute + T_ar
print("T_DP =", T_DP_case3)
print("T_TP =", T_TP_case3)
print("DP always wins: T_TP - T_DP =", simplify(T_TP_case3 - T_DP_case3), "> 0")

# --- Hardware constants ---

hardware = {
    "H100 SXM":  {"M": 3.35e12, "R": 900e9,  "gpu_gb": 80,  "C": 2000e12},  # 2000 TFLOPS FP8
    "GB200":     {"M": 8.0e12,  "R": 1.8e12, "gpu_gb": 192, "C": 4000e12},  # 4000 TFLOPS FP8
}

print("\n--- Hardware constants ---")
for name, hw in hardware.items():
    ratio = hw["R"] / hw["M"]
    inv = hw["M"] / hw["R"]
    print(f"{name}: R/M = {ratio:.4f}, M/R = {inv:.2f}, C = {hw['C']/1e12:.0f} TFLOPS")

# --- Complete latency formulas ---

C, Q = symbols('C Q', positive=True)

# Projection compute time
T_compute = 4 * B * Q * H * (H + D_k) / (P * C)

# Weight load times
T_load_DP = W / M
T_load_TP = W / (P * M)

# AllReduce time (TP only)
T_ar = 2 * b * B * Q * H / R

# Complete formulas
T_DP = Max(T_load_DP, T_compute)
T_TP = Max(T_load_TP, T_compute) + T_ar

print("\n--- Complete latency formulas (symbolic) ---")
print("T_DP =", T_DP)
print("T_TP =", T_TP)

# Compute-bound thresholds: when load time = compute time
# TP: W/(PM) = 4BQH(H+D_k)/(PC)
BQ_cb_TP = solve(Eq(T_load_TP, T_compute), B * Q)[0]
BQ_cb_TP = simplify(BQ_cb_TP)
print("\n--- Compute-bound thresholds (when load = compute) ---")
print("B_cb,TP * Q =", BQ_cb_TP)

# DP: W/M = 4BQH(H+D_k)/(PC)
BQ_cb_DP = solve(Eq(T_load_DP, T_compute), B * Q)[0]
BQ_cb_DP = simplify(BQ_cb_DP)
print("B_cb,DP * Q =", BQ_cb_DP)

# --- Models (matching typst table) ---
# (name, n_q, n_k, d, layers, total_params_b)

models = {
    "Qwen2.5-7B":      {"n_q": 28, "n_k": 4,  "d": 128, "layers": 28,  "params_b": 7,   "H": 3584},
    "Llama-3.1-8B":     {"n_q": 32, "n_k": 8,  "d": 128, "layers": 32,  "params_b": 8,   "H": 4096},
    "MiniMax-M2.1":     {"n_q": 48, "n_k": 8,  "d": 128, "layers": 62,  "params_b": 229, "H": 3072},
    "Qwen3-235B":       {"n_q": 64, "n_k": 4,  "d": 128, "layers": 94,  "params_b": 235, "H": 4096},
    "Llama-3.1-70B":    {"n_q": 64, "n_k": 8,  "d": 128, "layers": 80,  "params_b": 70,  "H": 8192},
    "GLM-4.5":          {"n_q": 96, "n_k": 8,  "d": 128, "layers": 92,  "params_b": 358, "H": 5120},
}

P_val = 8
p_factor = (P_val - 1) / P_val

# Numerical verification: is BQ* > BQ_cb,TP?
print("\n=== Verification: Is memory-bound crossover valid? ===")
print("For each model, check if BQ* (memory-bound) > BQ_cb,TP")
print("If BQ* > BQ_cb,TP: TP becomes compute-bound BEFORE memory-bound crossover")
print()

for hw_name, hw in hardware.items():
    print(f"\n{hw_name}:")
    BQ_cb_TP_val = 2 * hw["C"] / (2 * hw["M"])  # Cb/(2M)
    print(f"  BQ_cb,TP = {BQ_cb_TP_val:.0f}")

    all_invalid = True
    for model_name, m in models.items():
        dim = m["d"] * (m["n_q"] + m["n_k"])
        BQ_star_mem = dim * (P_val - 1) * hw["R"] / (P_val * hw["M"])  # (H+D_k)(P-1)R/(PM)

        if BQ_star_mem > BQ_cb_TP_val:
            status = "✓ TP compute-bound before crossover"
        else:
            status = "✗ Memory-bound crossover IS valid"
            all_invalid = False

        print(f"  {model_name:<18} BQ* = {BQ_star_mem:>6.0f}  {status}")

    if all_invalid:
        print(f"  => Memory-bound crossover NEVER happens on {hw_name}")
        print(f"  => Actual crossover occurs in Case 2 (DP memory-bound, TP compute-bound)")

print(f"\n--- Breakeven B* and max seq len at P={P_val} ---")
print(f"{'Model':<18} {'d(nq+nk)':>10}", end="")
for hw_name in hardware:
    print(f"  {'B* '+hw_name:>14} {'max L':>8}", end="")
print()

for name, m in models.items():
    dim = m["d"] * (m["n_q"] + m["n_k"])
    kv_per_tok = 2 * m["layers"] * m["n_k"] * m["d"]  # FP8: 1 byte per element

    print(f"{name:<18} {dim:>10}", end="")
    for hw_name, hw in hardware.items():
        gpu_bytes = hw["gpu_gb"] * 1e9 * 0.9
        weight_per_gpu = m["params_b"] * 1e9 / P_val   # FP8: 1 byte per param
        avail = gpu_bytes - weight_per_gpu
        max_tok_per_gpu = avail / kv_per_tok
        b_star = round(dim * p_factor * hw["R"] / hw["M"])
        max_l = round(max_tok_per_gpu * P_val / b_star)
        max_l_k = round(max_l / 1000, 1)
        print(f"  {b_star:>14} {max_l_k:>7.1f}K", end="")
    print()

# --- Compute-bound threshold table ---

print(f"\n--- Compute-bound thresholds at P={P_val} (for decode Q=1) ---")
print(f"{'Model':<18} {'d(nq+nk)':>10}", end="")
for hw_name in hardware:
    print(f"  {hw_name+' BQ_cb,TP':>16} {'BQ_cb,DP':>10} {'B*Q':>10}", end="")
print()

for name, m in models.items():
    dim = m["d"] * (m["n_q"] + m["n_k"])
    H_val = m["H"]
    D_k_val = m["n_k"] * m["d"]

    print(f"{name:<18} {dim:>10}", end="")
    for hw_name, hw in hardware.items():
        # Weight bytes (bf16, 2 bytes per element)
        W_val = 2 * 2 * H_val * (H_val + D_k_val)

        # BQ_cb,TP = (W*C) / (4*H*(H+D_k)*M - 2*b*H*P*C/R)
        numerator_tp = W_val * hw["C"]
        denominator_tp = 4 * H_val * (H_val + D_k_val) * hw["M"] - 2 * 2 * H_val * P_val * hw["C"] / hw["R"]
        bq_cb_tp = numerator_tp / denominator_tp if denominator_tp > 0 else float('inf')

        # BQ_cb,DP = (W*C*P) / (4*H*(H+D_k)*M)
        numerator_dp = W_val * hw["C"] * P_val
        denominator_dp = 4 * H_val * (H_val + D_k_val) * hw["M"]
        bq_cb_dp = numerator_dp / denominator_dp

        # Memory-bound breakeven (B*Q)
        bq_star = round(dim * p_factor * hw["R"] / hw["M"])

        print(f"  {int(round(bq_cb_tp)):>16} {int(round(bq_cb_dp)):>10} {bq_star:>10}", end="")
    print()

print(f"\n--- Complete latency formulas: GLM-4.5 decode on GB200 ---")
m = models["GLM-4.5"]
hw = hardware["GB200"]
H_val = m["H"]
D_k_val = m["n_k"] * m["d"]
W_val = 2 * 2 * H_val * (H_val + D_k_val)
Q_val = 1  # decode

def t_dp(B):
    """DP time in microseconds"""
    load = W_val / hw["M"] * 1e6
    compute = 4 * B * Q_val * H_val * (H_val + D_k_val) / (P_val * hw["C"]) * 1e6
    return max(load, compute)

def t_tp(B):
    """TP time in microseconds"""
    load = W_val / (P_val * hw["M"]) * 1e6
    compute = 4 * B * Q_val * H_val * (H_val + D_k_val) / (P_val * hw["C"]) * 1e6
    ar = 2 * 2 * B * Q_val * H_val / hw["R"] * 1e6
    return max(load, compute) + ar

# Compute-bound thresholds
bq_cb_tp = W_val * hw["C"] / (4 * H_val * (H_val + D_k_val) * hw["M"])
bq_cb_dp = W_val * hw["C"] * P_val / (4 * H_val * (H_val + D_k_val) * hw["M"])
b_cb_tp = bq_cb_tp / Q_val
b_cb_dp = bq_cb_dp / Q_val

print(f"TP becomes compute-bound at B = {b_cb_tp:.0f}")
print(f"DP becomes compute-bound at B = {b_cb_dp:.0f}")
print(f"\nSample latencies:")
print(f"{'B':<8} {'T_TP (μs)':<12} {'T_DP (μs)':<12} {'Winner':<8}")
for B in [1, 100, 500, 1000, 2000, 3000, 4000, 5000]:
    ttp = t_tp(B)
    tdp = t_dp(B)
    winner = "TP" if ttp < tdp else "DP"
    print(f"{B:<8} {ttp:<12.1f} {tdp:<12.1f} {winner:<8}")

# Find crossover
for B in range(1, 6000):
    if t_tp(B) >= t_dp(B):
        print(f"\nCrossover at B ≈ {B} (TP time = {t_tp(B):.1f} μs, DP time = {t_dp(B):.1f} μs)")
        break

# --- Detailed timing table for one model (Llama-3.1-70B) at P=8 ---

print("\n--- Timing table: Llama-3.1-70B, P=8 ---")
m = models["Llama-3.1-70B"]
dim = m["d"] * (m["n_q"] + m["n_k"])
H_val = m["H"]
b_val = 2  # bf16

for hw_name, hw in hardware.items():
    W_val = 2 * b_val * H_val * dim  # note: H_in * (n_q*d + n_k*d) when H != n_q*d
    # For Llama, H = n_q*d = 8192, so this is correct
    t_dp = W_val / hw["M"] * 1e6  # us
    ar_per_tok = 2 * b_val * H_val / hw["R"] * 1e6  # us/token
    b_star = dim * p_factor * hw["R"] / hw["M"]

    print(f"\n{hw_name}: W={W_val/1e6:.1f} MB, T_DP={t_dp:.1f} us, AR/tok={ar_per_tok:.4f} us")
    print(f"  B*={b_star:.0f}")
    for batch in [1, 256, 1024, int(round(b_star))]:
        t_tp = W_val / (P_val * hw["M"]) * 1e6 + ar_per_tok * batch
        ratio = t_dp / t_tp if t_tp > 0 else float('inf')
        print(f"  B={batch:>6}: T_TP={t_tp:>8.1f} us, T_DP={t_dp:>8.1f} us, ratio={ratio:.2f}x")

# --- GLM-4.5 on GB200 example ---

print("\n--- Example: GLM-4.5 on GB200 ---")
m = models["GLM-4.5"]
dim = m["d"] * (m["n_q"] + m["n_k"])
hw = hardware["GB200"]
b_star = round(dim * p_factor * hw["R"] / hw["M"])

# TP speedup = T_DP / T_TP = 1 / (1/P + B*M/(R*dim))
def speedup(batch):
    return 1 / (1/P_val + batch * hw["M"] / (hw["R"] * dim))

print(f"d(nq+nk) = {dim}")
print(f"B* = {dim} * {p_factor:.3f} * {hw['R']/hw['M']:.3f} = {b_star}")
for batch in [1, 256]:
    s = speedup(batch)
    print(f"  B={batch}: TP speedup = {s:.1f}x")

# --- Multi-query breakeven (prefill & spec-dec) ---

print("\n--- B* / Q: GLM-4.5 on GB200 ---")
print(f"B*(Q=1) = {b_star}")
for label, Q in [("spec-dec n=4", 5), ("spec-dec n=16", 17),
                 ("prefill 1K", 1024), ("prefill 4K", 4096), ("prefill 32K", 32768)]:
    bp = b_star / Q
    print(f"  {label:<16} Q={Q:>5}: B* = {b_star}/{Q} = {bp:.1f}  (DP wins at batch >= {-(-b_star // Q)})")
