"""Symbolic verification and numerical tables for dp-vs-tp-attention.typ"""
from sympy import *

H, D_k, P, B, b, M, R = symbols('H D_k P B b M R', positive=True)

# --- Symbolic ---

# Attention weight bytes (Q + K + V + O)
W = 2 * b * H * (H + D_k)

# AllReduce: 2 * data_size / realised_allreduce_ici_bandwidth
T_ar = 2 * b * B * H / R

# Time expressions (K/M identical on both sides, cancels)
T_TP = W / (P * M) + T_ar
T_DP = W / M

# Breakeven
B_star = solve(Eq(T_TP, T_DP), B)[0]
B_star = simplify(B_star)
print("B* =", B_star)

# --- Hardware constants ---

hardware = {
    "H100 SXM":  {"M": 3.35e12, "R": 900e9,  "gpu_gb": 80},
    "GB200":     {"M": 8.0e12,  "R": 1.8e12, "gpu_gb": 192},
}

print("\n--- Hardware constants ---")
for name, hw in hardware.items():
    ratio = hw["R"] / hw["M"]
    inv = hw["M"] / hw["R"]
    print(f"{name}: R/M = {ratio:.4f}, M/R = {inv:.2f}")

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
