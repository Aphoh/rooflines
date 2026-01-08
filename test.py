#!/usr/bin/env python3
"""
KV Cache Calculator - Test Suite
Run with: python test.py
"""

import math

# MLA architectures that use compressed KV cache
MLA_ARCHITECTURES = [
    'DeepseekV2ForCausalLM', 'DeepseekV32ForCausalLM', 'DeepseekV3ForCausalLM',
    'DeepseekV3ForCausalLMNextN', 'DeepseekVL2ForCausalLM', 'LongcatFlashForCausalLM',
    'MistralLarge3ForCausalLM', 'PixtralForConditionalGeneration', 'MiniCPM3ForCausalLM',
    'KimiVLForConditionalGeneration', 'KimiLinearForCausalLM'
]

BYTES_PER_DTYPE = {'bf16': 2, 'fp16': 2, 'fp8': 1}


def format_bytes(num_bytes):
    if num_bytes == 0:
        return '0 B'
    k = 1024
    sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    i = int(math.floor(math.log(num_bytes) / math.log(k)))
    return f"{num_bytes / (k ** i):.2f} {sizes[i]}"


def is_mla(config):
    archs = config.get('architectures', [])
    if config.get('kv_lora_rank'):
        return True
    return any(a in MLA_ARCHITECTURES for a in archs)


def get_sliding_window(config):
    sw = config.get('sliding_window')
    if isinstance(sw, list):
        sw = next((v for v in sw if v is not None), None)
    if config.get('use_sliding_window') is False:
        return None
    return sw


def calculate_kv_cache(config, tp_size=1):
    num_layers = config.get('num_hidden_layers', 32)
    use_mla = is_mla(config)
    sliding_window = get_sliding_window(config)

    if use_mla:
        kv_lora_rank = config.get('kv_lora_rank', 512)
        qk_rope_head_dim = config.get('qk_rope_head_dim', 64)
        latent_dim = kv_lora_rank + qk_rope_head_dim

        bf16 = num_layers * latent_dim * BYTES_PER_DTYPE['bf16']
        fp8 = num_layers * latent_dim * BYTES_PER_DTYPE['fp8']
    else:
        num_kv_heads = config.get('num_key_value_heads') or config.get('num_attention_heads', 32)
        hidden_size = config.get('hidden_size', 4096)
        num_attention_heads = config.get('num_attention_heads', 32)
        default_head_dim = hidden_size // num_attention_heads if num_attention_heads else 128
        head_dim = config.get('head_dim', default_head_dim)
        kv_heads_per_tp = max(1, num_kv_heads // tp_size)

        bf16 = 2 * num_layers * kv_heads_per_tp * head_dim * BYTES_PER_DTYPE['bf16']
        fp8 = 2 * num_layers * kv_heads_per_tp * head_dim * BYTES_PER_DTYPE['fp8']

    return {'bf16': bf16, 'fp8': fp8, 'use_mla': use_mla, 'sliding_window': sliding_window}


def get_kv_at_seq_len(result, seq_len, dtype='bf16'):
    bytes_per_token = result['fp8'] if dtype == 'fp8' else result['bf16']
    sw = result['sliding_window']
    if sw and seq_len > sw:
        return bytes_per_token * sw
    return bytes_per_token * seq_len


# Real model configurations (fetched from HuggingFace)
MODELS = {
    "DeepSeek-V3": {
        "architectures": ["DeepseekV3ForCausalLM"],
        "num_hidden_layers": 61,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "num_attention_heads": 128,
        "hidden_size": 7168,
        "max_position_embeddings": 163840
    },
    "DeepSeek-R1": {
        "architectures": ["DeepseekV3ForCausalLM"],
        "num_hidden_layers": 61,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "num_attention_heads": 128,
        "hidden_size": 7168,
        "max_position_embeddings": 163840
    },
    "Kimi-K2": {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "kimi_k2",
        "num_hidden_layers": 61,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "num_attention_heads": 64,
        "hidden_size": 7168,
        "max_position_embeddings": 131072
    },
    "GLM-4.5": {
        "architectures": ["Glm4MoeForCausalLM"],
        "num_hidden_layers": 92,
        "num_attention_heads": 96,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 5120,
        "max_position_embeddings": 131072
    },
    "MiniMax-M2.1": {
        "architectures": ["MiniMaxM2ForCausalLM"],
        "num_hidden_layers": 62,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 3072,
        "max_position_embeddings": 196608
    },
    "Qwen3-235B-A22B": {
        "architectures": ["Qwen3MoeForCausalLM"],
        "num_hidden_layers": 94,
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_size": 4096,
        "max_position_embeddings": 40960
    },
    "Qwen2.5-72B": {
        "architectures": ["Qwen2ForCausalLM"],
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 8192,
        "max_position_embeddings": 32768,
        "sliding_window": 131072,
        "use_sliding_window": False
    },
    "Qwen2.5-7B": {
        "architectures": ["Qwen2ForCausalLM"],
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_size": 3584,
        "max_position_embeddings": 32768
    },
    "Mixtral-8x7B": {
        "architectures": ["MixtralForCausalLM"],
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "max_position_embeddings": 32768
    },
    "Mistral-7B-SWA": {
        "architectures": ["MistralForCausalLM"],
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "max_position_embeddings": 32768,
        "sliding_window": 4096
    },
    "Llama-3.1-70B": {
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 8192,
        "max_position_embeddings": 131072
    },
    "Llama-3.1-8B": {
        "architectures": ["LlamaForCausalLM"],
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "hidden_size": 4096,
        "max_position_embeddings": 131072
    }
}

# Expected values for verification
EXPECTED = {
    "DeepSeek-V3": {"bf16": 61 * 576 * 2, "fp8": 61 * 576, "use_mla": True},       # 70272, 35136
    "DeepSeek-R1": {"bf16": 61 * 576 * 2, "fp8": 61 * 576, "use_mla": True},       # 70272, 35136
    "Kimi-K2": {"bf16": 61 * 576 * 2, "fp8": 61 * 576, "use_mla": True},           # 70272, 35136
    "GLM-4.5": {"bf16": 2 * 92 * 8 * 128 * 2, "fp8": 2 * 92 * 8 * 128, "use_mla": False},  # 376832, 188416
    "MiniMax-M2.1": {"bf16": 2 * 62 * 8 * 128 * 2, "fp8": 2 * 62 * 8 * 128, "use_mla": False},  # 253952, 126976
    "Qwen3-235B-A22B": {"bf16": 2 * 94 * 4 * 128 * 2, "fp8": 2 * 94 * 4 * 128, "use_mla": False},  # 192512, 96256
    "Qwen2.5-72B": {"bf16": 2 * 80 * 8 * 128 * 2, "fp8": 2 * 80 * 8 * 128, "use_mla": False},  # 327680, 163840
    "Qwen2.5-7B": {"bf16": 2 * 28 * 4 * 128 * 2, "fp8": 2 * 28 * 4 * 128, "use_mla": False},   # 57344, 28672
    "Mixtral-8x7B": {"bf16": 2 * 32 * 8 * 128 * 2, "fp8": 2 * 32 * 8 * 128, "use_mla": False}, # 131072, 65536
    "Mistral-7B-SWA": {"bf16": 2 * 32 * 8 * 128 * 2, "fp8": 2 * 32 * 8 * 128, "use_mla": False, "sliding_window": 4096},
    "Llama-3.1-70B": {"bf16": 2 * 80 * 8 * 128 * 2, "fp8": 2 * 80 * 8 * 128, "use_mla": False},  # 327680, 163840
    "Llama-3.1-8B": {"bf16": 2 * 32 * 8 * 128 * 2, "fp8": 2 * 32 * 8 * 128, "use_mla": False},   # 131072, 65536
}


def main():
    print("╔" + "═" * 78 + "╗")
    print("║" + "KV Cache Calculator - Model Comparison".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Print header
    print(f"{'Model':<20} {'Type':<6} {'Layers':<7} {'KV/Tok BF16':>12} {'KV/Tok FP8':>12} {'128K BF16':>12} {'128K FP8':>12}")
    print("─" * 90)

    results = []
    for name, config in MODELS.items():
        result = calculate_kv_cache(config)
        results.append((name, config, result))

    # Sort by BF16 bytes/token
    results.sort(key=lambda x: x[2]['bf16'])

    passed = 0
    failed = 0
    ctx_len = 131072  # 128K

    for name, config, result in results:
        type_str = "MLA" if result['use_mla'] else ("SWA" if result['sliding_window'] else "MHA")

        # Calculate 128K memory (bounded for SWA)
        bf16_128k = get_kv_at_seq_len(result, ctx_len, 'bf16')
        fp8_128k = get_kv_at_seq_len(result, ctx_len, 'fp8')
        bounded = result['sliding_window'] and ctx_len > result['sliding_window']

        print(f"{name:<20} {type_str:<6} {config.get('num_hidden_layers', 32):<7} "
              f"{result['bf16']:>12,} {result['fp8']:>12,} "
              f"{format_bytes(bf16_128k):>12}{'*' if bounded else ' '} {format_bytes(fp8_128k):>12}{'*' if bounded else ' '}")

        # Verify against expected if available
        if name in EXPECTED:
            exp = EXPECTED[name]
            if result['bf16'] == exp['bf16'] and result['fp8'] == exp['fp8'] and result['use_mla'] == exp['use_mla']:
                passed += 1
            else:
                failed += 1
                print(f"  ❌ MISMATCH: expected bf16={exp['bf16']}, fp8={exp['fp8']}, mla={exp['use_mla']}")

    print("─" * 90)
    print("* Memory bounded by sliding window size")
    print()

    if EXPECTED:
        print(f"\nVerification: {passed}/{len(EXPECTED)} tests passed")
        if failed > 0:
            print(f"❌ {failed} test(s) failed")
            return 1
        else:
            print("✅ All tests passed!")

    # Summary comparisons
    print("\n" + "═" * 90)
    print("KEY INSIGHTS:")
    print("═" * 90)

    mla_models = [(n, r) for n, _, r in results if r['use_mla']]
    mha_models = [(n, r) for n, _, r in results if not r['use_mla'] and not r['sliding_window']]

    if mla_models and mha_models:
        avg_mla = sum(r['bf16'] for _, r in mla_models) / len(mla_models)
        avg_mha = sum(r['bf16'] for _, r in mha_models) / len(mha_models)
        print(f"• MLA models average {avg_mla:,.0f} bytes/tok vs MHA average {avg_mha:,.0f} bytes/tok")
        print(f"  → MLA saves ~{(1 - avg_mla/avg_mha)*100:.0f}% KV cache memory on average")

    # Find best and worst
    best = results[0]
    worst = results[-1]
    print(f"\n• Most efficient: {best[0]} at {best[2]['bf16']:,} bytes/tok (BF16)")
    print(f"• Least efficient: {worst[0]} at {worst[2]['bf16']:,} bytes/tok (BF16)")

    return 0


if __name__ == "__main__":
    exit(main())
