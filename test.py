#!/usr/bin/env python3
"""
KV Cache Calculator - Test Suite
Run with: python test.py
"""

import math

# MLA architectures that use compressed KV cache
MLA_ARCHITECTURES = [
    'DeepseekV2ForCausalLM',
    'DeepseekV32ForCausalLM', 
    'DeepseekV3ForCausalLM',
    'DeepseekV3ForCausalLMNextN',
    'DeepseekVL2ForCausalLM',
    'LongcatFlashForCausalLM',
    'LongcatFlashForCausalLMNextN',
    'DotsVLMForCausalLM',
    'MistralLarge3ForCausalLM',
    'PixtralForConditionalGeneration',
    'MistralLarge3ForCausalLMEagle',
    'MiniCPM3ForCausalLM',
    'KimiVLForConditionalGeneration',
    'KimiLinearForCausalLM',
]

BYTES_PER_DTYPE = {'bf16': 2, 'fp16': 2, 'fp8': 1, 'int8': 1, 'fp32': 4}


def format_bytes(num_bytes):
    if num_bytes == 0:
        return '0 B'
    k = 1024
    sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    i = int(math.floor(math.log(num_bytes) / math.log(k)))
    return f"{num_bytes / (k ** i):.2f} {sizes[i]}"


def is_mla(architectures):
    if not architectures:
        return False
    return any(arch in MLA_ARCHITECTURES for arch in architectures)


def get_sliding_window(config):
    sw = config.get('sliding_window')
    if config.get('text_config'):
        sw = sw or config['text_config'].get('sliding_window')
    if isinstance(sw, list):
        sw = next((v for v in sw if v is not None), None)
    return sw


def get_text_config(config):
    if config.get('text_config'):
        return {**config, **config['text_config']}
    return config


def calculate_kv_cache(config, tp_size=1):
    text_config = get_text_config(config)
    architectures = config.get('architectures', [])
    use_mla = is_mla(architectures)
    sliding_window = get_sliding_window(config)
    
    num_layers = text_config.get('num_hidden_layers') or text_config.get('n_layer', 32)
    num_kv_heads = text_config.get('num_key_value_heads') or text_config.get('num_attention_heads', 32)
    hidden_size = text_config.get('hidden_size', 4096)
    num_attention_heads = text_config.get('num_attention_heads', 32)
    default_head_dim = hidden_size // num_attention_heads if num_attention_heads else 128
    head_dim = text_config.get('head_dim', default_head_dim)
    
    if use_mla:
        kv_lora_rank = text_config.get('kv_lora_rank', 512)
        qk_rope_head_dim = text_config.get('qk_rope_head_dim', 64)
        latent_dim = kv_lora_rank + qk_rope_head_dim
        
        bf16 = num_layers * latent_dim * BYTES_PER_DTYPE['bf16']
        fp8 = num_layers * latent_dim * BYTES_PER_DTYPE['fp8']
    else:
        kv_heads_per_tp = max(1, num_kv_heads // tp_size)
        bf16 = 2 * num_layers * kv_heads_per_tp * head_dim * BYTES_PER_DTYPE['bf16']
        fp8 = 2 * num_layers * kv_heads_per_tp * head_dim * BYTES_PER_DTYPE['fp8']
    
    return {'bf16': bf16, 'fp8': fp8, 'use_mla': use_mla, 'sliding_window': sliding_window}


# Test configurations
TEST_CASES = [
    {
        'name': "DeepSeek-V3 (MLA)",
        'config': {
            'architectures': ["DeepseekV3ForCausalLM"],
            'num_hidden_layers': 61,
            'kv_lora_rank': 512,
            'qk_rope_head_dim': 64,
            'num_attention_heads': 128,
            'hidden_size': 7168
        },
        'expected': {'bf16': 70272, 'fp8': 35136, 'use_mla': True, 'sliding_window': None}
    },
    {
        'name': "DeepSeek-R1 (MLA)",
        'config': {
            'architectures': ["DeepseekV3ForCausalLM"],
            'num_hidden_layers': 61,
            'kv_lora_rank': 512,
            'qk_rope_head_dim': 64,
            'num_attention_heads': 128,
            'hidden_size': 7168
        },
        'expected': {'bf16': 70272, 'fp8': 35136, 'use_mla': True, 'sliding_window': None}
    },
    {
        'name': "DeepSeek-V2-Lite (MLA)",
        'config': {
            'architectures': ["DeepseekV2ForCausalLM"],
            'num_hidden_layers': 27,
            'kv_lora_rank': 512,
            'qk_rope_head_dim': 64,
            'num_attention_heads': 16,
            'hidden_size': 2048
        },
        'expected': {'bf16': 31104, 'fp8': 15552, 'use_mla': True, 'sliding_window': None}
    },
    {
        'name': "Kimi K2 (MLA)",
        'config': {
            'architectures': ["KimiLinearForCausalLM"],
            'num_hidden_layers': 61,
            'kv_lora_rank': 512,
            'qk_rope_head_dim': 64,
            'num_attention_heads': 64,
            'hidden_size': 6144
        },
        'expected': {'bf16': 70272, 'fp8': 35136, 'use_mla': True, 'sliding_window': None}
    },
    {
        'name': "Llama 3.1 8B (MHA, GQA)",
        'config': {
            'architectures': ["LlamaForCausalLM"],
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'hidden_size': 4096
        },
        'expected': {'bf16': 131072, 'fp8': 65536, 'use_mla': False, 'sliding_window': None}
    },
    {
        'name': "Llama 3.1 70B (MHA, GQA)",
        'config': {
            'architectures': ["LlamaForCausalLM"],
            'num_hidden_layers': 80,
            'num_attention_heads': 64,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'hidden_size': 8192
        },
        'expected': {'bf16': 327680, 'fp8': 163840, 'use_mla': False, 'sliding_window': None}
    },
    {
        'name': "Qwen2.5 72B (MHA, GQA)",
        'config': {
            'architectures': ["Qwen2ForCausalLM"],
            'num_hidden_layers': 80,
            'num_attention_heads': 64,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'hidden_size': 8192
        },
        'expected': {'bf16': 327680, 'fp8': 163840, 'use_mla': False, 'sliding_window': None}
    },
    {
        'name': "Mistral 7B (SWA)",
        'config': {
            'architectures': ["MistralForCausalLM"],
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'hidden_size': 4096,
            'sliding_window': 4096
        },
        'expected': {'bf16': 131072, 'fp8': 65536, 'use_mla': False, 'sliding_window': 4096}
    },
    {
        'name': "Mixtral 8x7B (SWA)",
        'config': {
            'architectures': ["MixtralForCausalLM"],
            'num_hidden_layers': 32,
            'num_attention_heads': 32,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'hidden_size': 4096,
            'sliding_window': 4096
        },
        'expected': {'bf16': 131072, 'fp8': 65536, 'use_mla': False, 'sliding_window': 4096}
    },
    {
        'name': "Gemma 2 9B (SWA)",
        'config': {
            'architectures': ["Gemma2ForCausalLM"],
            'num_hidden_layers': 42,
            'num_attention_heads': 16,
            'num_key_value_heads': 8,
            'head_dim': 256,
            'hidden_size': 3584,
            'sliding_window': 4096
        },
        'expected': {'bf16': 344064, 'fp8': 172032, 'use_mla': False, 'sliding_window': 4096}
    },
    {
        'name': "GLM-4 9B (MHA, Heavy GQA)",
        'config': {
            'architectures': ["ChatGLMForCausalLM"],
            'num_hidden_layers': 40,
            'num_attention_heads': 32,
            'num_key_value_heads': 2,
            'head_dim': 128,
            'hidden_size': 4096
        },
        'expected': {'bf16': 40960, 'fp8': 20480, 'use_mla': False, 'sliding_window': None}
    },
    {
        'name': "MiMo 7B (MHA)",
        'config': {
            'architectures': ["MiMoForCausalLM"],
            'num_hidden_layers': 28,
            'num_attention_heads': 28,
            'num_key_value_heads': 4,
            'head_dim': 128,
            'hidden_size': 3584
        },
        'expected': {'bf16': 57344, 'fp8': 28672, 'use_mla': False, 'sliding_window': None}
    },
    {
        'name': "Llama 3.1 70B with TP=8",
        'config': {
            'architectures': ["LlamaForCausalLM"],
            'num_hidden_layers': 80,
            'num_attention_heads': 64,
            'num_key_value_heads': 8,
            'head_dim': 128,
            'hidden_size': 8192
        },
        'tp_size': 8,
        'expected': {'bf16': 40960, 'fp8': 20480, 'use_mla': False, 'sliding_window': None}
    },
    {
        'name': "MiniCPM3 (MLA)",
        'config': {
            'architectures': ["MiniCPM3ForCausalLM"],
            'num_hidden_layers': 62,
            'kv_lora_rank': 512,
            'qk_rope_head_dim': 64,
            'num_attention_heads': 40,
            'hidden_size': 2560
        },
        'expected': {'bf16': 71424, 'fp8': 35712, 'use_mla': True, 'sliding_window': None}
    }
]


def main():
    print("╔" + "═" * 78 + "╗")
    print("║" + "KV Cache Calculator - Test Suite".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    passed = 0
    failed = 0
    
    for test_case in TEST_CASES:
        tp_size = test_case.get('tp_size', 1)
        result = calculate_kv_cache(test_case['config'], tp_size)
        expected = test_case['expected']
        
        checks = [
            ('BF16', result['bf16'], expected['bf16']),
            ('FP8', result['fp8'], expected['fp8']),
            ('MLA', result['use_mla'], expected['use_mla']),
            ('SWA', result['sliding_window'], expected['sliding_window'])
        ]
        
        all_pass = all(actual == exp for _, actual, exp in checks)
        if all_pass:
            passed += 1
        else:
            failed += 1
        
        status = '✅' if all_pass else '❌'
        tp_note = f" (TP={tp_size})" if tp_size > 1 else ""
        print(f"{status} {test_case['name']}{tp_note}")
        
        if not all_pass:
            for name, actual, exp in checks:
                if actual != exp:
                    print(f"   ❌ {name}: expected {exp}, got {actual}")
        
        # Show memory info
        ctx_len = 131072  # 128K
        sw = result['sliding_window']
        effective_len = min(ctx_len, sw) if sw and ctx_len > sw else ctx_len
        bounded = sw is not None and ctx_len > sw
        
        total_bf16 = result['bf16'] * effective_len
        total_fp8 = result['fp8'] * effective_len
        bounded_str = " (bounded)" if bounded else ""
        
        print(f"   BF16: {result['bf16']:,} bytes/tok → {format_bytes(total_bf16)} @ 128K{bounded_str}")
        print(f"   FP8:  {result['fp8']:,} bytes/tok → {format_bytes(total_fp8)} @ 128K{bounded_str}")
        print()
    
    print("═" * 80)
    print(f"\nResults: {passed}/{passed + failed} tests passed\n")
    
    if failed > 0:
        print(f"❌ {failed} test(s) failed")
        return 1
    else:
        print("✅ All tests passed!")
        return 0


if __name__ == "__main__":
    exit(main())
