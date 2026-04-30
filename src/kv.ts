import type {
  CapacityResult,
  CapacityScenario,
  KvCacheResult,
  KvDtype,
  LayerType,
  ModelArchitecture,
  WeightDtype,
} from "./types";

export const BYTES_PER_DTYPE: Record<KvDtype, number> = { bf16: 2, fp8: 1 };
export const WEIGHT_BYTES: Record<WeightDtype, number> = { fp16: 2, fp8: 1, nvfp4: 0.56 };

const MLA_ARCHITECTURES = new Set([
  "DeepseekV4ForCausalLM",
  "DeepseekV2ForCausalLM",
  "DeepseekV32ForCausalLM",
  "DeepseekV3ForCausalLM",
  "DeepseekV3ForCausalLMNextN",
  "DeepseekVL2ForCausalLM",
  "LongcatFlashForCausalLM",
  "MistralLarge3ForCausalLM",
  "PixtralForConditionalGeneration",
  "MiniCPM3ForCausalLM",
  "KimiK25ForConditionalGeneration",
  "KimiVLForConditionalGeneration",
  "KimiLinearForCausalLM",
  "GlmMoeDsaForCausalLM",
]);

const DEEPSEEK_V4_ARCHITECTURE = "DeepseekV4ForCausalLM";
const DEEPSEEK_V4_DEFAULT_CACHE_BLOCK_SIZE = 256;
const DEEPSEEK_V4_DEFAULT_CACHE_TOKEN_BYTES = 584;
const DEEPSEEK_V4_DEFAULT_CACHE_ALIGNMENT = 576;
const DEEPSEEK_V4_DEFAULT_SWA_BLOCK_SIZE = 64;
const DEEPSEEK_V4_C4_COMPRESSOR_BLOCK_SIZE = 4;
const DEEPSEEK_V4_C128_COMPRESSOR_BLOCK_SIZE = 8;
const FP32_BYTES = 4;

function dereferenceTextConfig(config: ModelArchitecture): ModelArchitecture {
  if (!config.text_config) return { ...config };

  const { text_config: textConfig, ...parentConfig } = config;
  return {
    ...parentConfig,
    ...textConfig,
    architectures: textConfig.architectures ?? config.architectures,
    total_params_b: textConfig.total_params_b ?? config.total_params_b,
    active_params_b: textConfig.active_params_b ?? config.active_params_b,
  };
}

function mapHybridOverrideType(type: string): LayerType {
  if (type === "*") return "full_attention";
  if (type === "M") return "linear_attention";
  return "mlp";
}

export function expandLayerPattern(rawConfig: ModelArchitecture): ModelArchitecture {
  const config = dereferenceTextConfig(rawConfig);
  if (config.layer_types) return { ...config };

  if (config.hybrid_layer_pattern) {
    return {
      ...config,
      layer_types: config.hybrid_layer_pattern.map((type) =>
        type === 1 ? "sliding_attention" : "full_attention",
      ),
    };
  }

  if (config.hybrid_override_pattern) {
    return {
      ...config,
      layer_types: [...config.hybrid_override_pattern].map(mapHybridOverrideType),
    };
  }

  if (!config.layer_pattern) return { ...config };

  const layerTypes: LayerType[] = [];
  for (let i = 0; i < config.layer_pattern.repeat; i += 1) {
    layerTypes.push(...config.layer_pattern.sequence);
  }

  return {
    ...config,
    layer_types: layerTypes,
  };
}

export function isMLA(config: ModelArchitecture): boolean {
  if (config.kv_lora_rank) return true;
  return (config.architectures || []).some((architecture) => MLA_ARCHITECTURES.has(architecture));
}

function isDeepSeekV4(config: ModelArchitecture): boolean {
  return (
    config.model_type === "deepseek_v4" ||
    (config.architectures || []).includes(DEEPSEEK_V4_ARCHITECTURE)
  );
}

function isGemma4(config: ModelArchitecture): boolean {
  return (
    config.model_type === "gemma4" ||
    config.model_type === "gemma4_text" ||
    (config.architectures || []).some((architecture) => architecture.startsWith("Gemma4"))
  );
}

function isNemotronH(config: ModelArchitecture): boolean {
  return (
    config.model_type === "nemotron_h" ||
    (config.architectures || []).includes("NemotronHForCausalLM")
  );
}

function roundUp(value: number, alignment: number): number {
  return Math.ceil(value / alignment) * alignment;
}

export function getSlidingWindow(config: ModelArchitecture): number | null {
  let slidingWindow: number | Array<number | null> | null | undefined = config.sliding_window;
  if (Array.isArray(slidingWindow)) {
    slidingWindow = slidingWindow.find((value): value is number => value !== null) ?? null;
  }
  slidingWindow ??= config.sliding_window_size;
  if (config.use_sliding_window === false) return null;
  return typeof slidingWindow === "number" ? slidingWindow : null;
}

function calculateDeepSeekV4KVCache(config: ModelArchitecture): KvCacheResult {
  const numLayers = config.num_hidden_layers || 0;
  const headDim = config.head_dim || 512;
  const indexHeadDim = config.index_head_dim || 128;
  const slidingWindow = getSlidingWindow(config) || 128;
  const cacheBlockSize = config.deepseek_v4?.cache_block_size || DEEPSEEK_V4_DEFAULT_CACHE_BLOCK_SIZE;
  const cacheTokenBytes =
    config.deepseek_v4?.cache_token_bytes || DEEPSEEK_V4_DEFAULT_CACHE_TOKEN_BYTES;
  const cacheAlignment =
    config.deepseek_v4?.cache_alignment || DEEPSEEK_V4_DEFAULT_CACHE_ALIGNMENT;
  const includeIndexerCache = config.deepseek_v4?.include_indexer_cache ?? true;
  const includeFixedRawCaches = config.deepseek_v4?.include_fixed_raw_caches ?? true;

  const ratios = Array.from({ length: numLayers }, (_, index) =>
    Math.max(1, config.compress_ratios?.[index] ?? 1),
  );
  const c4Layers = ratios.filter((ratio) => ratio === 4).length;
  const c128Layers = ratios.filter((ratio) => ratio === 128).length;
  const swaOnlyLayers = ratios.filter((ratio) => ratio <= 1).length;

  const pageBytes = (bytes: number) => roundUp(bytes, cacheAlignment);
  const mainCacheBytesPerToken = ratios.reduce((total, ratio) => {
    if (ratio <= 1) return total;
    return total + pageBytes((cacheBlockSize / ratio) * cacheTokenBytes) / cacheBlockSize;
  }, 0);

  const indexerTokenBytes = indexHeadDim + Math.floor(indexHeadDim / 128) * FP32_BYTES;
  const indexerCacheBytesPerToken =
    includeIndexerCache && c4Layers > 0
      ? c4Layers * pageBytes((cacheBlockSize / 4) * indexerTokenBytes) / cacheBlockSize
      : 0;

  let fixedSwaBytesPerRequest = 0;
  let fixedCompressorBytesPerRequest = 0;
  if (includeFixedRawCaches) {
    const swaPageBytes = pageBytes(DEEPSEEK_V4_DEFAULT_SWA_BLOCK_SIZE * cacheTokenBytes);
    fixedSwaBytesPerRequest =
      numLayers * Math.ceil(slidingWindow / DEEPSEEK_V4_DEFAULT_SWA_BLOCK_SIZE) * swaPageBytes;

    const c4MainStateDim = 2 * 2 * headDim;
    const c4MainStatePageBytes = pageBytes(
      DEEPSEEK_V4_C4_COMPRESSOR_BLOCK_SIZE * c4MainStateDim * FP32_BYTES,
    );
    const c4MainFixedBytes =
      c4Layers *
      Math.ceil((2 * 4) / DEEPSEEK_V4_C4_COMPRESSOR_BLOCK_SIZE) *
      c4MainStatePageBytes;

    const c4IndexerStateDim = 2 * 2 * indexHeadDim;
    const c4IndexerStatePageBytes = pageBytes(
      DEEPSEEK_V4_C4_COMPRESSOR_BLOCK_SIZE * c4IndexerStateDim * FP32_BYTES,
    );
    const c4IndexerFixedBytes =
      includeIndexerCache && c4Layers > 0
        ? c4Layers *
          Math.ceil((2 * 4) / DEEPSEEK_V4_C4_COMPRESSOR_BLOCK_SIZE) *
          c4IndexerStatePageBytes
        : 0;

    const c128MainStateDim = 2 * headDim;
    const c128MainStatePageBytes = pageBytes(
      DEEPSEEK_V4_C128_COMPRESSOR_BLOCK_SIZE * c128MainStateDim * FP32_BYTES,
    );
    const c128MainFixedBytes =
      c128Layers *
      Math.ceil(128 / DEEPSEEK_V4_C128_COMPRESSOR_BLOCK_SIZE) *
      c128MainStatePageBytes;

    fixedCompressorBytesPerRequest =
      c4MainFixedBytes + c4IndexerFixedBytes + c128MainFixedBytes;
  }

  const fp8 = mainCacheBytesPerToken + indexerCacheBytesPerToken;

  return {
    // vLLM DeepSeek V4 currently forces fp8_ds_mla KV storage, so BF16 and FP8
    // scenarios use the same physical cache budget here.
    bf16: fp8,
    fp8,
    useMLA: true,
    slidingWindow: null,
    numLayers,
    kvHeads: 1,
    headDim: cacheTokenBytes,
    vHeadDim: cacheTokenBytes,
    numFullLayers: c4Layers + c128Layers,
    numSlidingLayers: swaOnlyLayers,
    numLinearLayers: 0,
    numNoAttentionLayers: 0,
    hasHybrid: true,
    hasHybridLinear: false,
    maxCtx: config.max_position_embeddings || 1048576,
    fixedPerRequest: fixedSwaBytesPerRequest + fixedCompressorBytesPerRequest,
    deepseekV4: {
      c4Layers,
      c128Layers,
      swaOnlyLayers,
      mainCacheBytesPerToken,
      indexerCacheBytesPerToken,
      fixedSwaBytesPerRequest,
      fixedCompressorBytesPerRequest,
    },
  };
}

function calculateGemma4KVCache(config: ModelArchitecture): KvCacheResult {
  const numLayers = config.num_hidden_layers || config.layer_types?.length || 60;
  const layerTypes = config.layer_types || [];
  const numFullLayers =
    layerTypes.length > 0 ? layerTypes.filter((type) => type === "full_attention").length : numLayers;
  const numSlidingLayers = layerTypes.filter((type) => type === "sliding_attention").length;
  const numLinearLayers = layerTypes.filter((type) => type === "linear_attention").length;
  const numNoAttentionLayers = layerTypes.filter((type) => type === "mlp").length;

  const slidingKVHeads = config.num_key_value_heads || config.num_attention_heads || 16;
  const slidingHeadDim = config.head_dim || 256;
  const slidingVHeadDim = config.v_head_dim || slidingHeadDim;

  const fullKVHeads =
    config.attention_k_eq_v && config.num_global_key_value_heads
      ? config.num_global_key_value_heads
      : slidingKVHeads;
  const fullHeadDim = config.global_head_dim || slidingHeadDim;
  const fullVHeadDim = fullHeadDim;

  const fullLayerBf16 = fullKVHeads * (fullHeadDim + fullVHeadDim) * BYTES_PER_DTYPE.bf16;
  const fullLayerFp8 = fullKVHeads * (fullHeadDim + fullVHeadDim) * BYTES_PER_DTYPE.fp8;
  const slidingLayerBf16 =
    slidingKVHeads * (slidingHeadDim + slidingVHeadDim) * BYTES_PER_DTYPE.bf16;
  const slidingLayerFp8 =
    slidingKVHeads * (slidingHeadDim + slidingVHeadDim) * BYTES_PER_DTYPE.fp8;

  return {
    bf16: numFullLayers * fullLayerBf16 + numSlidingLayers * slidingLayerBf16,
    fp8: numFullLayers * fullLayerFp8 + numSlidingLayers * slidingLayerFp8,
    useMLA: false,
    slidingWindow: getSlidingWindow(config),
    numLayers,
    kvHeads: Math.max(fullKVHeads, slidingKVHeads),
    headDim: Math.max(fullHeadDim, slidingHeadDim),
    vHeadDim: Math.max(fullVHeadDim, slidingVHeadDim),
    numFullLayers,
    numSlidingLayers,
    numLinearLayers,
    numNoAttentionLayers,
    hasHybrid: numSlidingLayers > 0 || numLinearLayers > 0 || numNoAttentionLayers > 0,
    hasHybridLinear: numLinearLayers > 0,
    maxCtx: config.max_position_embeddings || 262144,
    fixedPerRequest: 0,
    perLayerKVBytes: {
      full: { bf16: fullLayerBf16, fp8: fullLayerFp8 },
      sliding: { bf16: slidingLayerBf16, fp8: slidingLayerFp8 },
    },
  };
}

function calculateMamba2StateBytesPerLayer(config: ModelArchitecture, bytesPerElement: number): number {
  const numHeads = config.mamba_num_heads || 128;
  const headDim = config.mamba_head_dim || 64;
  const stateSize = config.ssm_state_size || 128;
  const nGroups = config.n_groups || 1;
  const convKernel = config.conv_kernel || 4;
  const intermediateSize = numHeads * headDim;

  const convDim = intermediateSize + 2 * nGroups * stateSize;
  const convStateElements = convDim * Math.max(0, convKernel - 1);
  const temporalStateElements = numHeads * headDim * stateSize;
  return (convStateElements + temporalStateElements) * bytesPerElement;
}

function getLinearStatePerLayer(
  config: ModelArchitecture,
  dtype: KvDtype,
): number | undefined {
  const explicit =
    dtype === "fp8" ? config.linear_state_per_layer_fp8 : config.linear_state_per_layer_bf16;
  if (explicit !== undefined) return explicit;
  if (config.linear_state_per_layer !== undefined) return config.linear_state_per_layer;
  if (isNemotronH(config)) return calculateMamba2StateBytesPerLayer(config, BYTES_PER_DTYPE[dtype]);
  return undefined;
}

export function calculateKVCache(rawConfig: ModelArchitecture): KvCacheResult {
  const config = expandLayerPattern(rawConfig);
  if (isDeepSeekV4(config)) return calculateDeepSeekV4KVCache(config);
  if (isGemma4(config)) return calculateGemma4KVCache(config);

  const numLayers = config.num_hidden_layers || 32;
  const useMLA = isMLA(config);
  const slidingWindow = getSlidingWindow(config);
  const layerTypes = config.layer_types || null;

  let bf16: number;
  let fp8: number;
  let kvHeads: number;
  let headDim: number;
  let vHeadDim: number;
  let numFullLayers = numLayers;
  let numSlidingLayers = 0;
  let numLinearLayers = 0;
  let numNoAttentionLayers = 0;

  if (layerTypes) {
    numFullLayers = layerTypes.filter((type) => type === "full_attention").length;
    numSlidingLayers = layerTypes.filter((type) => type === "sliding_attention").length;
    numLinearLayers = layerTypes.filter((type) => type === "linear_attention").length;
    numNoAttentionLayers = layerTypes.filter((type) => type === "mlp").length;
  }

  const tokenKVLayers = layerTypes ? numFullLayers + numSlidingLayers : numLayers;

  if (useMLA) {
    const kvLoraRank = config.kv_lora_rank || 512;
    const qkRopeHeadDim = config.qk_rope_head_dim || 64;
    const latentDim = kvLoraRank + qkRopeHeadDim;
    bf16 = tokenKVLayers * latentDim * BYTES_PER_DTYPE.bf16;
    fp8 = tokenKVLayers * latentDim * BYTES_PER_DTYPE.fp8;
    kvHeads = 1;
    headDim = latentDim;
    vHeadDim = latentDim;
  } else {
    const numKVHeads = config.num_key_value_heads || config.num_attention_heads || 32;
    const defaultHeadDim = config.hidden_size
      ? Math.floor(config.hidden_size / (config.num_attention_heads || 32))
      : 128;
    headDim = config.head_dim || defaultHeadDim;
    vHeadDim = config.v_head_dim || headDim;
    bf16 = tokenKVLayers * numKVHeads * (headDim + vHeadDim) * BYTES_PER_DTYPE.bf16;
    fp8 = tokenKVLayers * numKVHeads * (headDim + vHeadDim) * BYTES_PER_DTYPE.fp8;
    kvHeads = numKVHeads;
  }

  const nsaIndexBytes = config.nsa_index_bytes_per_layer || 0;
  bf16 += numLayers * nsaIndexBytes;
  fp8 += numLayers * nsaIndexBytes;

  const fixedPerLayerBf16 = getLinearStatePerLayer(config, "bf16");
  const fixedPerLayerFp8 = getLinearStatePerLayer(config, "fp8");
  const fixedPerRequestByDtype =
    numLinearLayers > 0
      ? {
          bf16: fixedPerLayerBf16 !== undefined ? numLinearLayers * fixedPerLayerBf16 : 0,
          fp8: fixedPerLayerFp8 !== undefined ? numLinearLayers * fixedPerLayerFp8 : 0,
        }
      : undefined;
  const fixedPerRequest = fixedPerRequestByDtype?.fp8 ?? fixedPerRequestByDtype?.bf16 ?? 0;

  return {
    bf16,
    fp8,
    useMLA,
    slidingWindow,
    numLayers,
    kvHeads,
    headDim,
    vHeadDim,
    numFullLayers,
    numSlidingLayers,
    numLinearLayers,
    numNoAttentionLayers,
    hasHybrid: layerTypes !== null && (numSlidingLayers > 0 || numLinearLayers > 0),
    hasHybridLinear: layerTypes !== null && numLinearLayers > 0,
    maxCtx: config.max_position_embeddings || 32768,
    fixedPerRequest,
    fixedPerRequestByDtype,
  };
}

function getFixedPerRequest(result: KvCacheResult, dtype: KvDtype): number {
  return result.fixedPerRequestByDtype?.[dtype] ?? result.fixedPerRequest ?? 0;
}

export function getKVCacheAtSeqLen(result: KvCacheResult, seqLen: number, dtype: KvDtype): number {
  const bytesPerElement = BYTES_PER_DTYPE[dtype];
  const fixed = getFixedPerRequest(result, dtype);

  if (result.hasHybrid && result.slidingWindow) {
    const uniformBytesPerLayerPerToken =
      result.kvHeads * (result.headDim + result.vHeadDim) * bytesPerElement;
    const fullBytesPerLayerPerToken =
      result.perLayerKVBytes?.full?.[dtype] ?? uniformBytesPerLayerPerToken;
    const slidingBytesPerLayerPerToken =
      result.perLayerKVBytes?.sliding?.[dtype] ?? uniformBytesPerLayerPerToken;

    const slidingContrib =
      result.numSlidingLayers * slidingBytesPerLayerPerToken * Math.min(seqLen, result.slidingWindow);
    const fullContrib = result.numFullLayers * fullBytesPerLayerPerToken * seqLen;
    return slidingContrib + fullContrib + fixed;
  }

  const bytesPerToken = dtype === "fp8" ? result.fp8 : result.bf16;
  if (result.slidingWindow && seqLen > result.slidingWindow) {
    return bytesPerToken * result.slidingWindow + fixed;
  }
  return bytesPerToken * seqLen + fixed;
}

export function getCapacity(
  config: ModelArchitecture,
  result: KvCacheResult,
  scenario: CapacityScenario,
): CapacityResult | null {
  if (config.total_params_b === undefined) return null;

  const numGpus = Math.max(1, scenario.numGpus);
  const gpuMemoryBytes = scenario.gpuMemoryGB * 1024 ** 3;
  const usableGpuMemoryBytes = gpuMemoryBytes * scenario.memoryUtilization;
  const weightBytesTotal = config.total_params_b * 1e9 * WEIGHT_BYTES[scenario.weightDtype];
  const weightBytesPerGpu = weightBytesTotal / numGpus;
  const kvBytesPerRequestTotal = getKVCacheAtSeqLen(result, scenario.seqLen, scenario.kvDtype);
  const kvBytesPerRequestPerGpu = kvBytesPerRequestTotal / numGpus;
  const availableBytesPerGpu = Math.max(0, usableGpuMemoryBytes - weightBytesPerGpu);
  const maxRequests =
    kvBytesPerRequestPerGpu > 0 ? Math.floor(availableBytesPerGpu / kvBytesPerRequestPerGpu) : 0;

  return {
    maxRequests,
    gpuMemoryBytes,
    usableGpuMemoryBytes,
    weightBytesTotal,
    weightBytesPerGpu,
    kvBytesPerRequestTotal,
    kvBytesPerRequestPerGpu,
    availableBytesPerGpu,
  };
}

export function getModelType(result: KvCacheResult): { label: string; className: string } {
  if (result.deepseekV4) return { label: "DSV4", className: "badge-mla" };
  if (result.hasHybridLinear) return { label: "Hybrid-Lin", className: "badge-hyb" };
  if (result.useMLA) return { label: "MLA", className: "badge-mla" };
  if (result.hasHybrid) return { label: "Hybrid-SWA", className: "badge-swa" };
  if (result.slidingWindow) return { label: "SWA", className: "badge-swa" };
  return { label: "MHA", className: "badge-mha" };
}
