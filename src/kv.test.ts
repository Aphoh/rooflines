import { describe, expect, test } from "bun:test";
import { calculateKVCache, getCapacity, getKVCacheAtSeqLen } from "./kv";
import type { ModelArchitecture } from "./types";

describe("KV cache math", () => {
  test("calculates MLA bytes per token", () => {
    const result = calculateKVCache({
      architectures: ["DeepseekV3ForCausalLM"],
      num_hidden_layers: 61,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
    });

    expect(result.bf16).toBe(70272);
    expect(result.fp8).toBe(35136);
  });

  test("calculates DeepSeek V4 Pro compressed cache and fixed raw cache bytes", () => {
    const architecture: ModelArchitecture = {
      architectures: ["DeepseekV4ForCausalLM"],
      model_type: "deepseek_v4",
      num_hidden_layers: 61,
      num_attention_heads: 128,
      num_key_value_heads: 1,
      head_dim: 512,
      index_head_dim: 128,
      qk_rope_head_dim: 64,
      sliding_window: 128,
      max_position_embeddings: 1048576,
      compress_ratios: Array.from({ length: 61 }, (_, index) =>
        index < 2 || index % 2 === 1 ? 128 : 4,
      ),
    };

    const result = calculateKVCache(architecture);

    expect(result.deepseekV4).toMatchObject({
      c4Layers: 30,
      c128Layers: 31,
      swaOnlyLayers: 0,
      mainCacheBytesPerToken: 4596.75,
      indexerCacheBytesPerToken: 1012.5,
      fixedSwaBytesPerRequest: 4567680,
      fixedCompressorBytesPerRequest: 18772992,
    });
    expect(result.fp8).toBe(5609.25);
    expect(result.bf16).toBe(5609.25);
    expect(result.fixedPerRequest).toBe(23340672);
    expect(getKVCacheAtSeqLen(result, 100000, "fp8")).toBe(584265672);
  });

  test("calculates DeepSeek V4 Flash compressed cache and fixed raw cache bytes", () => {
    const architecture: ModelArchitecture = {
      architectures: ["DeepseekV4ForCausalLM"],
      model_type: "deepseek_v4",
      num_hidden_layers: 43,
      num_attention_heads: 64,
      num_key_value_heads: 1,
      head_dim: 512,
      index_head_dim: 128,
      qk_rope_head_dim: 64,
      sliding_window: 128,
      max_position_embeddings: 1048576,
      compress_ratios: Array.from({ length: 43 }, (_, index) =>
        index < 2 ? 0 : index % 2 === 0 ? 4 : 128,
      ),
    };

    const result = calculateKVCache(architecture);

    expect(result.deepseekV4).toMatchObject({
      c4Layers: 21,
      c128Layers: 20,
      swaOnlyLayers: 2,
      mainCacheBytesPerToken: 3206.25,
      indexerCacheBytesPerToken: 708.75,
      fixedSwaBytesPerRequest: 3219840,
      fixedCompressorBytesPerRequest: 12248064,
    });
    expect(result.fp8).toBe(3915);
    expect(result.bf16).toBe(3915);
    expect(result.fixedPerRequest).toBe(15467904);
    expect(getKVCacheAtSeqLen(result, 100000, "fp8")).toBe(406967904);
  });

  test("calculates MLA bytes only for full-attention layers in Kimi Linear style hybrids", () => {
    const result = calculateKVCache({
      architectures: ["KimiLinearForCausalLM"],
      num_hidden_layers: 4,
      kv_lora_rank: 512,
      qk_rope_head_dim: 64,
      layer_types: ["linear_attention", "full_attention", "linear_attention", "full_attention"],
      linear_state_per_layer: 1000,
    });

    expect(result.bf16).toBe(2 * (512 + 64) * 2);
    expect(result.fp8).toBe(2 * (512 + 64));
    expect(result.fixedPerRequest).toBe(2000);
    expect(getKVCacheAtSeqLen(result, 100, "fp8")).toBe(2 * (512 + 64) * 100 + 2000);
  });

  test("bounds hybrid sliding layers at the sliding window", () => {
    const result = calculateKVCache({
      num_hidden_layers: 4,
      num_attention_heads: 4,
      num_key_value_heads: 2,
      head_dim: 8,
      sliding_window: 128,
      layer_types: ["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    });

    const bytesPerLayerTokenFp8 = 2 * 2 * 8;
    const expected =
      2 * bytesPerLayerTokenFp8 * 128 +
      2 * bytesPerLayerTokenFp8 * 1024;

    expect(getKVCacheAtSeqLen(result, 1024, "fp8")).toBe(expected);
  });

  test("calculates asymmetric K/V bytes for MiMo V2 style hybrid attention", () => {
    const result = calculateKVCache({
      architectures: ["MiMoV2ForCausalLM"],
      model_type: "mimo_v2",
      num_hidden_layers: 70,
      num_attention_heads: 128,
      num_key_value_heads: 8,
      head_dim: 192,
      v_head_dim: 128,
      hidden_size: 6144,
      sliding_window_size: 128,
      hybrid_layer_pattern: [
        0, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1,
        0,
      ],
    });

    expect(result.numFullLayers).toBe(10);
    expect(result.numSlidingLayers).toBe(60);
    expect(result.vHeadDim).toBe(128);
    expect(result.fp8).toBe(70 * 8 * (192 + 128));
    expect(getKVCacheAtSeqLen(result, 131072, "fp8")).toBe(3375104000);
  });

  test("calculates Gemma 4 mixed full and sliding attention cache bytes", () => {
    const result = calculateKVCache({
      architectures: ["Gemma4ForConditionalGeneration"],
      model_type: "gemma4_text",
      num_hidden_layers: 60,
      num_attention_heads: 32,
      num_key_value_heads: 16,
      num_global_key_value_heads: 4,
      head_dim: 256,
      global_head_dim: 512,
      max_position_embeddings: 262144,
      sliding_window: 1024,
      attention_k_eq_v: true,
      layer_pattern: {
        sequence: [
          "sliding_attention",
          "sliding_attention",
          "sliding_attention",
          "sliding_attention",
          "sliding_attention",
          "full_attention",
        ],
        repeat: 10,
      },
    });

    expect(result.numFullLayers).toBe(10);
    expect(result.numSlidingLayers).toBe(50);
    expect(result.fp8).toBe(450560);
    expect(getKVCacheAtSeqLen(result, 100000, "fp8")).toBe(4515430400);
    expect(getKVCacheAtSeqLen(result, 131072, "fp8")).toBe(5788139520);
  });

  test("calculates Nemotron-H attention KV and Mamba2 fixed request state", () => {
    const result = calculateKVCache({
      architectures: ["NemotronHForCausalLM"],
      model_type: "nemotron_h",
      num_hidden_layers: 88,
      num_attention_heads: 32,
      num_key_value_heads: 2,
      head_dim: 128,
      hybrid_override_pattern:
        "MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME",
      mamba_num_heads: 128,
      mamba_head_dim: 64,
      ssm_state_size: 128,
      conv_kernel: 4,
      n_groups: 8,
    });

    expect(result.numFullLayers).toBe(8);
    expect(result.numLinearLayers).toBe(40);
    expect(result.numNoAttentionLayers).toBe(40);
    expect(result.fp8).toBe(4096);
    expect(result.bf16).toBe(8192);
    expect(result.fixedPerRequestByDtype?.fp8).toBe(43171840);
    expect(result.fixedPerRequestByDtype?.bf16).toBe(86343680);
    expect(getKVCacheAtSeqLen(result, 100000, "fp8")).toBe(452771840);
    expect(getKVCacheAtSeqLen(result, 131072, "fp8")).toBe(580042752);
  });

  test("shards both weights and KV across the selected GPUs", () => {
    const architecture: ModelArchitecture = {
      num_hidden_layers: 1,
      num_attention_heads: 1,
      num_key_value_heads: 1,
      head_dim: 500,
      total_params_b: 0,
    };
    const result = calculateKVCache(architecture);

    const oneGpu = getCapacity(architecture, result, {
      seqLen: 1000,
      gpuMemoryGB: 1,
      numGpus: 1,
      kvDtype: "fp8",
      weightDtype: "fp8",
      memoryUtilization: 1,
    });
    const eightGpus = getCapacity(architecture, result, {
      seqLen: 1000,
      gpuMemoryGB: 1,
      numGpus: 8,
      kvDtype: "fp8",
      weightDtype: "fp8",
      memoryUtilization: 1,
    });

    expect(oneGpu).not.toBeNull();
    expect(eightGpus).not.toBeNull();
    expect(eightGpus!.kvBytesPerRequestPerGpu).toBe(oneGpu!.kvBytesPerRequestTotal / 8);
    expect(eightGpus!.maxRequests).toBeGreaterThanOrEqual(oneGpu!.maxRequests * 8);
  });

  test("weight sharding can make a model fit when one GPU cannot hold weights", () => {
    const architecture: ModelArchitecture = {
      num_hidden_layers: 1,
      num_attention_heads: 1,
      num_key_value_heads: 1,
      head_dim: 500,
      total_params_b: 8,
    };
    const result = calculateKVCache(architecture);

    const oneGpu = getCapacity(architecture, result, {
      seqLen: 1000,
      gpuMemoryGB: 2,
      numGpus: 1,
      kvDtype: "fp8",
      weightDtype: "fp8",
      memoryUtilization: 1,
    });
    const eightGpus = getCapacity(architecture, result, {
      seqLen: 1000,
      gpuMemoryGB: 2,
      numGpus: 8,
      kvDtype: "fp8",
      weightDtype: "fp8",
      memoryUtilization: 1,
    });

    expect(oneGpu!.maxRequests).toBe(0);
    expect(eightGpus!.weightBytesPerGpu).toBe(1e9);
    expect(eightGpus!.maxRequests).toBeGreaterThan(0);
  });
});
