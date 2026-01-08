#!/usr/bin/env node
/**
 * KV Cache Calculator - Test Suite
 * Run with: node test.js
 */

// MLA architectures that use compressed KV cache
const MLA_ARCHITECTURES = [
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
];

const BYTES_PER_DTYPE = { bf16: 2, fp16: 2, fp8: 1, int8: 1, fp32: 4 };

function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function isMLA(architectures) {
    if (!architectures) return false;
    return architectures.some(arch => MLA_ARCHITECTURES.includes(arch));
}

function getSlidingWindow(config) {
    let sw = config.sliding_window;
    if (config.text_config) sw = sw || config.text_config.sliding_window;
    if (Array.isArray(sw)) sw = sw.find(v => v !== null && v !== undefined);
    return sw || null;
}

function getTextConfig(config) {
    if (config.text_config) return { ...config, ...config.text_config };
    return config;
}

function calculateKVCache(config, tpSize = 1) {
    const textConfig = getTextConfig(config);
    const architectures = config.architectures || [];
    const useMLA = isMLA(architectures);
    const slidingWindow = getSlidingWindow(config);
    
    const numLayers = textConfig.num_hidden_layers || textConfig.n_layer || 32;
    const numKVHeads = textConfig.num_key_value_heads || textConfig.num_attention_heads || 32;
    const defaultHeadDim = textConfig.hidden_size ? Math.floor(textConfig.hidden_size / (textConfig.num_attention_heads || 32)) : 128;
    const headDim = textConfig.head_dim || defaultHeadDim;
    
    let bf16, fp8;
    
    if (useMLA) {
        const kvLoraRank = textConfig.kv_lora_rank || 512;
        const qkRopeHeadDim = textConfig.qk_rope_head_dim || 64;
        const latentDim = kvLoraRank + qkRopeHeadDim;
        
        bf16 = numLayers * latentDim * BYTES_PER_DTYPE.bf16;
        fp8 = numLayers * latentDim * BYTES_PER_DTYPE.fp8;
    } else {
        const kvHeadsPerTP = Math.max(1, Math.floor(numKVHeads / tpSize));
        bf16 = 2 * numLayers * kvHeadsPerTP * headDim * BYTES_PER_DTYPE.bf16;
        fp8 = 2 * numLayers * kvHeadsPerTP * headDim * BYTES_PER_DTYPE.fp8;
    }
    
    return { bf16, fp8, useMLA, slidingWindow };
}

// Test configurations
const TEST_CASES = [
    {
        name: "DeepSeek-V3 (MLA)",
        config: {
            architectures: ["DeepseekV3ForCausalLM"],
            num_hidden_layers: 61,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            num_attention_heads: 128,
            hidden_size: 7168
        },
        expected: { bf16: 70272, fp8: 35136, useMLA: true, slidingWindow: null }
    },
    {
        name: "DeepSeek-R1 (MLA)",
        config: {
            architectures: ["DeepseekV3ForCausalLM"],
            num_hidden_layers: 61,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            num_attention_heads: 128,
            hidden_size: 7168
        },
        expected: { bf16: 70272, fp8: 35136, useMLA: true, slidingWindow: null }
    },
    {
        name: "DeepSeek-V2-Lite (MLA)",
        config: {
            architectures: ["DeepseekV2ForCausalLM"],
            num_hidden_layers: 27,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            num_attention_heads: 16,
            hidden_size: 2048
        },
        expected: { bf16: 31104, fp8: 15552, useMLA: true, slidingWindow: null }
    },
    {
        name: "Kimi K2 (MLA)",
        config: {
            architectures: ["KimiLinearForCausalLM"],
            num_hidden_layers: 61,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            num_attention_heads: 64,
            hidden_size: 6144
        },
        expected: { bf16: 70272, fp8: 35136, useMLA: true, slidingWindow: null }
    },
    {
        name: "Llama 3.1 8B (MHA, GQA)",
        config: {
            architectures: ["LlamaForCausalLM"],
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            hidden_size: 4096
        },
        expected: { bf16: 131072, fp8: 65536, useMLA: false, slidingWindow: null }
    },
    {
        name: "Llama 3.1 70B (MHA, GQA)",
        config: {
            architectures: ["LlamaForCausalLM"],
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            hidden_size: 8192
        },
        expected: { bf16: 327680, fp8: 163840, useMLA: false, slidingWindow: null }
    },
    {
        name: "Qwen2.5 72B (MHA, GQA)",
        config: {
            architectures: ["Qwen2ForCausalLM"],
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            hidden_size: 8192
        },
        expected: { bf16: 327680, fp8: 163840, useMLA: false, slidingWindow: null }
    },
    {
        name: "Mistral 7B (SWA)",
        config: {
            architectures: ["MistralForCausalLM"],
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            hidden_size: 4096,
            sliding_window: 4096
        },
        expected: { bf16: 131072, fp8: 65536, useMLA: false, slidingWindow: 4096 }
    },
    {
        name: "Mixtral 8x7B (SWA)",
        config: {
            architectures: ["MixtralForCausalLM"],
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            hidden_size: 4096,
            sliding_window: 4096
        },
        expected: { bf16: 131072, fp8: 65536, useMLA: false, slidingWindow: 4096 }
    },
    {
        name: "Gemma 2 9B (SWA)",
        config: {
            architectures: ["Gemma2ForCausalLM"],
            num_hidden_layers: 42,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 256,
            hidden_size: 3584,
            sliding_window: 4096
        },
        expected: { bf16: 344064, fp8: 172032, useMLA: false, slidingWindow: 4096 }
    },
    {
        name: "GLM-4 9B (MHA, Heavy GQA)",
        config: {
            architectures: ["ChatGLMForCausalLM"],
            num_hidden_layers: 40,
            num_attention_heads: 32,
            num_key_value_heads: 2,
            head_dim: 128,
            hidden_size: 4096
        },
        expected: { bf16: 40960, fp8: 20480, useMLA: false, slidingWindow: null }
    },
    {
        name: "MiMo 7B (MHA)",
        config: {
            architectures: ["MiMoForCausalLM"],
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: 4,
            head_dim: 128,
            hidden_size: 3584
        },
        expected: { bf16: 57344, fp8: 28672, useMLA: false, slidingWindow: null }
    },
    {
        name: "Llama 3.1 70B with TP=8",
        config: {
            architectures: ["LlamaForCausalLM"],
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            head_dim: 128,
            hidden_size: 8192
        },
        tpSize: 8,
        expected: { bf16: 40960, fp8: 20480, useMLA: false, slidingWindow: null }
    },
    {
        name: "MiniCPM3 (MLA)",
        config: {
            architectures: ["MiniCPM3ForCausalLM"],
            num_hidden_layers: 62,
            kv_lora_rank: 512,
            qk_rope_head_dim: 64,
            num_attention_heads: 40,
            hidden_size: 2560
        },
        expected: { bf16: 71424, fp8: 35712, useMLA: true, slidingWindow: null }
    }
];

// Run tests
console.log("╔══════════════════════════════════════════════════════════════════════════════╗");
console.log("║                        KV Cache Calculator - Test Suite                       ║");
console.log("╚══════════════════════════════════════════════════════════════════════════════╝\n");

let passed = 0;
let failed = 0;

for (const testCase of TEST_CASES) {
    const result = calculateKVCache(testCase.config, testCase.tpSize || 1);
    const checks = [
        { name: 'BF16', actual: result.bf16, expected: testCase.expected.bf16 },
        { name: 'FP8', actual: result.fp8, expected: testCase.expected.fp8 },
        { name: 'MLA', actual: result.useMLA, expected: testCase.expected.useMLA },
        { name: 'SWA', actual: result.slidingWindow, expected: testCase.expected.slidingWindow }
    ];
    
    const allPass = checks.every(c => c.actual === c.expected);
    if (allPass) passed++; else failed++;
    
    const status = allPass ? '✅' : '❌';
    const tpNote = testCase.tpSize ? ` (TP=${testCase.tpSize})` : '';
    console.log(`${status} ${testCase.name}${tpNote}`);
    
    if (!allPass) {
        for (const c of checks) {
            if (c.actual !== c.expected) {
                console.log(`   ❌ ${c.name}: expected ${c.expected}, got ${c.actual}`);
            }
        }
    }
    
    // Show memory info
    const ctxLen = 131072; // 128K
    const totalBf16 = result.slidingWindow 
        ? Math.min(ctxLen, result.slidingWindow) * result.bf16 
        : ctxLen * result.bf16;
    const bounded = result.slidingWindow && ctxLen > result.slidingWindow;
    console.log(`   BF16: ${result.bf16.toLocaleString()} bytes/tok → ${formatBytes(totalBf16)} @ 128K${bounded ? ' (bounded)' : ''}`);
    console.log(`   FP8:  ${result.fp8.toLocaleString()} bytes/tok → ${formatBytes(result.fp8 * (bounded ? result.slidingWindow : ctxLen))} @ 128K${bounded ? ' (bounded)' : ''}`);
    console.log('');
}

console.log("═".repeat(80));
console.log(`\nResults: ${passed}/${passed + failed} tests passed\n`);

if (failed > 0) {
    console.log(`❌ ${failed} test(s) failed`);
    process.exit(1);
} else {
    console.log("✅ All tests passed!");
    process.exit(0);
}
