// src/format.ts
function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0)
    return "0 B";
  const unit = 1024;
  const sizes = ["B", "KB", "MB", "GB", "TB", "PB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(unit)), sizes.length - 1);
  const value = bytes / unit ** index;
  return `${Number(value.toFixed(2))} ${sizes[index]}`;
}
function formatNumber(value) {
  if (!Number.isFinite(value))
    return "-";
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}
function formatMetric(value, suffix = "") {
  if (value === undefined || value === null || !Number.isFinite(value))
    return "-";
  return `${Number(value.toFixed(2)).toLocaleString()}${suffix}`;
}
function escapeHtml(value) {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#039;");
}

// src/kv.ts
var BYTES_PER_DTYPE = { bf16: 2, fp8: 1 };
var WEIGHT_BYTES = { fp16: 2, fp8: 1, nvfp4: 0.56 };
var MLA_ARCHITECTURES = new Set([
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
  "GlmMoeDsaForCausalLM"
]);
var DEEPSEEK_V4_ARCHITECTURE = "DeepseekV4ForCausalLM";
var DEEPSEEK_V4_DEFAULT_CACHE_BLOCK_SIZE = 256;
var DEEPSEEK_V4_DEFAULT_CACHE_TOKEN_BYTES = 584;
var DEEPSEEK_V4_DEFAULT_CACHE_ALIGNMENT = 576;
var DEEPSEEK_V4_DEFAULT_SWA_BLOCK_SIZE = 64;
var DEEPSEEK_V4_C4_COMPRESSOR_BLOCK_SIZE = 4;
var DEEPSEEK_V4_C128_COMPRESSOR_BLOCK_SIZE = 8;
var FP32_BYTES = 4;
function dereferenceTextConfig(config) {
  if (!config.text_config)
    return { ...config };
  const { text_config: textConfig, ...parentConfig } = config;
  return {
    ...parentConfig,
    ...textConfig,
    architectures: textConfig.architectures ?? config.architectures,
    total_params_b: textConfig.total_params_b ?? config.total_params_b,
    active_params_b: textConfig.active_params_b ?? config.active_params_b
  };
}
function mapHybridOverrideType(type) {
  if (type === "*")
    return "full_attention";
  if (type === "M")
    return "linear_attention";
  return "mlp";
}
function expandLayerPattern(rawConfig) {
  const config = dereferenceTextConfig(rawConfig);
  if (config.layer_types)
    return { ...config };
  if (config.hybrid_layer_pattern) {
    return {
      ...config,
      layer_types: config.hybrid_layer_pattern.map((type) => type === 1 ? "sliding_attention" : "full_attention")
    };
  }
  if (config.hybrid_override_pattern) {
    return {
      ...config,
      layer_types: [...config.hybrid_override_pattern].map(mapHybridOverrideType)
    };
  }
  if (!config.layer_pattern)
    return { ...config };
  const layerTypes = [];
  for (let i = 0;i < config.layer_pattern.repeat; i += 1) {
    layerTypes.push(...config.layer_pattern.sequence);
  }
  return {
    ...config,
    layer_types: layerTypes
  };
}
function isMLA(config) {
  if (config.kv_lora_rank)
    return true;
  return (config.architectures || []).some((architecture) => MLA_ARCHITECTURES.has(architecture));
}
function isDeepSeekV4(config) {
  return config.model_type === "deepseek_v4" || (config.architectures || []).includes(DEEPSEEK_V4_ARCHITECTURE);
}
function isGemma4(config) {
  return config.model_type === "gemma4" || config.model_type === "gemma4_text" || (config.architectures || []).some((architecture) => architecture.startsWith("Gemma4"));
}
function isNemotronH(config) {
  return config.model_type === "nemotron_h" || (config.architectures || []).includes("NemotronHForCausalLM");
}
function roundUp(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}
function getSlidingWindow(config) {
  let slidingWindow = config.sliding_window;
  if (Array.isArray(slidingWindow)) {
    slidingWindow = slidingWindow.find((value) => value !== null) ?? null;
  }
  slidingWindow ??= config.sliding_window_size;
  if (config.use_sliding_window === false)
    return null;
  return typeof slidingWindow === "number" ? slidingWindow : null;
}
function calculateDeepSeekV4KVCache(config) {
  const numLayers = config.num_hidden_layers || 0;
  const headDim = config.head_dim || 512;
  const indexHeadDim = config.index_head_dim || 128;
  const slidingWindow = getSlidingWindow(config) || 128;
  const cacheBlockSize = config.deepseek_v4?.cache_block_size || DEEPSEEK_V4_DEFAULT_CACHE_BLOCK_SIZE;
  const cacheTokenBytes = config.deepseek_v4?.cache_token_bytes || DEEPSEEK_V4_DEFAULT_CACHE_TOKEN_BYTES;
  const cacheAlignment = config.deepseek_v4?.cache_alignment || DEEPSEEK_V4_DEFAULT_CACHE_ALIGNMENT;
  const includeIndexerCache = config.deepseek_v4?.include_indexer_cache ?? true;
  const includeFixedRawCaches = config.deepseek_v4?.include_fixed_raw_caches ?? true;
  const ratios = Array.from({ length: numLayers }, (_, index) => Math.max(1, config.compress_ratios?.[index] ?? 1));
  const c4Layers = ratios.filter((ratio) => ratio === 4).length;
  const c128Layers = ratios.filter((ratio) => ratio === 128).length;
  const swaOnlyLayers = ratios.filter((ratio) => ratio <= 1).length;
  const pageBytes = (bytes) => roundUp(bytes, cacheAlignment);
  const mainCacheBytesPerToken = ratios.reduce((total, ratio) => {
    if (ratio <= 1)
      return total;
    return total + pageBytes(cacheBlockSize / ratio * cacheTokenBytes) / cacheBlockSize;
  }, 0);
  const indexerTokenBytes = indexHeadDim + Math.floor(indexHeadDim / 128) * FP32_BYTES;
  const indexerCacheBytesPerToken = includeIndexerCache && c4Layers > 0 ? c4Layers * pageBytes(cacheBlockSize / 4 * indexerTokenBytes) / cacheBlockSize : 0;
  let fixedSwaBytesPerRequest = 0;
  let fixedCompressorBytesPerRequest = 0;
  if (includeFixedRawCaches) {
    const swaPageBytes = pageBytes(DEEPSEEK_V4_DEFAULT_SWA_BLOCK_SIZE * cacheTokenBytes);
    fixedSwaBytesPerRequest = numLayers * Math.ceil(slidingWindow / DEEPSEEK_V4_DEFAULT_SWA_BLOCK_SIZE) * swaPageBytes;
    const c4MainStateDim = 2 * 2 * headDim;
    const c4MainStatePageBytes = pageBytes(DEEPSEEK_V4_C4_COMPRESSOR_BLOCK_SIZE * c4MainStateDim * FP32_BYTES);
    const c4MainFixedBytes = c4Layers * Math.ceil(2 * 4 / DEEPSEEK_V4_C4_COMPRESSOR_BLOCK_SIZE) * c4MainStatePageBytes;
    const c4IndexerStateDim = 2 * 2 * indexHeadDim;
    const c4IndexerStatePageBytes = pageBytes(DEEPSEEK_V4_C4_COMPRESSOR_BLOCK_SIZE * c4IndexerStateDim * FP32_BYTES);
    const c4IndexerFixedBytes = includeIndexerCache && c4Layers > 0 ? c4Layers * Math.ceil(2 * 4 / DEEPSEEK_V4_C4_COMPRESSOR_BLOCK_SIZE) * c4IndexerStatePageBytes : 0;
    const c128MainStateDim = 2 * headDim;
    const c128MainStatePageBytes = pageBytes(DEEPSEEK_V4_C128_COMPRESSOR_BLOCK_SIZE * c128MainStateDim * FP32_BYTES);
    const c128MainFixedBytes = c128Layers * Math.ceil(128 / DEEPSEEK_V4_C128_COMPRESSOR_BLOCK_SIZE) * c128MainStatePageBytes;
    fixedCompressorBytesPerRequest = c4MainFixedBytes + c4IndexerFixedBytes + c128MainFixedBytes;
  }
  const fp8 = mainCacheBytesPerToken + indexerCacheBytesPerToken;
  return {
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
      fixedCompressorBytesPerRequest
    }
  };
}
function calculateGemma4KVCache(config) {
  const numLayers = config.num_hidden_layers || config.layer_types?.length || 60;
  const layerTypes = config.layer_types || [];
  const numFullLayers = layerTypes.length > 0 ? layerTypes.filter((type) => type === "full_attention").length : numLayers;
  const numSlidingLayers = layerTypes.filter((type) => type === "sliding_attention").length;
  const numLinearLayers = layerTypes.filter((type) => type === "linear_attention").length;
  const numNoAttentionLayers = layerTypes.filter((type) => type === "mlp").length;
  const slidingKVHeads = config.num_key_value_heads || config.num_attention_heads || 16;
  const slidingHeadDim = config.head_dim || 256;
  const slidingVHeadDim = config.v_head_dim || slidingHeadDim;
  const fullKVHeads = config.attention_k_eq_v && config.num_global_key_value_heads ? config.num_global_key_value_heads : slidingKVHeads;
  const fullHeadDim = config.global_head_dim || slidingHeadDim;
  const fullVHeadDim = fullHeadDim;
  const fullLayerBf16 = fullKVHeads * (fullHeadDim + fullVHeadDim) * BYTES_PER_DTYPE.bf16;
  const fullLayerFp8 = fullKVHeads * (fullHeadDim + fullVHeadDim) * BYTES_PER_DTYPE.fp8;
  const slidingLayerBf16 = slidingKVHeads * (slidingHeadDim + slidingVHeadDim) * BYTES_PER_DTYPE.bf16;
  const slidingLayerFp8 = slidingKVHeads * (slidingHeadDim + slidingVHeadDim) * BYTES_PER_DTYPE.fp8;
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
      sliding: { bf16: slidingLayerBf16, fp8: slidingLayerFp8 }
    }
  };
}
function calculateMamba2StateBytesPerLayer(config, bytesPerElement) {
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
function getLinearStatePerLayer(config, dtype) {
  const explicit = dtype === "fp8" ? config.linear_state_per_layer_fp8 : config.linear_state_per_layer_bf16;
  if (explicit !== undefined)
    return explicit;
  if (config.linear_state_per_layer !== undefined)
    return config.linear_state_per_layer;
  if (isNemotronH(config))
    return calculateMamba2StateBytesPerLayer(config, BYTES_PER_DTYPE[dtype]);
  return;
}
function calculateKVCache(rawConfig) {
  const config = expandLayerPattern(rawConfig);
  if (isDeepSeekV4(config))
    return calculateDeepSeekV4KVCache(config);
  if (isGemma4(config))
    return calculateGemma4KVCache(config);
  const numLayers = config.num_hidden_layers || 32;
  const useMLA = isMLA(config);
  const slidingWindow = getSlidingWindow(config);
  const layerTypes = config.layer_types || null;
  let bf16;
  let fp8;
  let kvHeads;
  let headDim;
  let vHeadDim;
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
    const defaultHeadDim = config.hidden_size ? Math.floor(config.hidden_size / (config.num_attention_heads || 32)) : 128;
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
  const fixedPerRequestByDtype = numLinearLayers > 0 ? {
    bf16: fixedPerLayerBf16 !== undefined ? numLinearLayers * fixedPerLayerBf16 : 0,
    fp8: fixedPerLayerFp8 !== undefined ? numLinearLayers * fixedPerLayerFp8 : 0
  } : undefined;
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
    fixedPerRequestByDtype
  };
}
function getFixedPerRequest(result, dtype) {
  return result.fixedPerRequestByDtype?.[dtype] ?? result.fixedPerRequest ?? 0;
}
function getKVCacheAtSeqLen(result, seqLen, dtype) {
  const bytesPerElement = BYTES_PER_DTYPE[dtype];
  const fixed = getFixedPerRequest(result, dtype);
  if (result.hasHybrid && result.slidingWindow) {
    const uniformBytesPerLayerPerToken = result.kvHeads * (result.headDim + result.vHeadDim) * bytesPerElement;
    const fullBytesPerLayerPerToken = result.perLayerKVBytes?.full?.[dtype] ?? uniformBytesPerLayerPerToken;
    const slidingBytesPerLayerPerToken = result.perLayerKVBytes?.sliding?.[dtype] ?? uniformBytesPerLayerPerToken;
    const slidingContrib = result.numSlidingLayers * slidingBytesPerLayerPerToken * Math.min(seqLen, result.slidingWindow);
    const fullContrib = result.numFullLayers * fullBytesPerLayerPerToken * seqLen;
    return slidingContrib + fullContrib + fixed;
  }
  const bytesPerToken = dtype === "fp8" ? result.fp8 : result.bf16;
  if (result.slidingWindow && seqLen > result.slidingWindow) {
    return bytesPerToken * result.slidingWindow + fixed;
  }
  return bytesPerToken * seqLen + fixed;
}
function getCapacity(config, result, scenario) {
  if (config.total_params_b === undefined)
    return null;
  const numGpus = Math.max(1, scenario.numGpus);
  const gpuMemoryBytes = scenario.gpuMemoryGB * 1024 ** 3;
  const usableGpuMemoryBytes = gpuMemoryBytes * scenario.memoryUtilization;
  const weightBytesTotal = config.total_params_b * 1e9 * WEIGHT_BYTES[scenario.weightDtype];
  const weightBytesPerGpu = weightBytesTotal / numGpus;
  const kvBytesPerRequestTotal = getKVCacheAtSeqLen(result, scenario.seqLen, scenario.kvDtype);
  const kvBytesPerRequestPerGpu = kvBytesPerRequestTotal / numGpus;
  const availableBytesPerGpu = Math.max(0, usableGpuMemoryBytes - weightBytesPerGpu);
  const maxRequests = kvBytesPerRequestPerGpu > 0 ? Math.floor(availableBytesPerGpu / kvBytesPerRequestPerGpu) : 0;
  return {
    maxRequests,
    gpuMemoryBytes,
    usableGpuMemoryBytes,
    weightBytesTotal,
    weightBytesPerGpu,
    kvBytesPerRequestTotal,
    kvBytesPerRequestPerGpu,
    availableBytesPerGpu
  };
}
function getModelType(result) {
  if (result.deepseekV4)
    return { label: "DSV4", className: "badge-mla" };
  if (result.hasHybridLinear)
    return { label: "Hybrid-Lin", className: "badge-hyb" };
  if (result.useMLA)
    return { label: "MLA", className: "badge-mla" };
  if (result.hasHybrid)
    return { label: "Hybrid-SWA", className: "badge-swa" };
  if (result.slidingWindow)
    return { label: "SWA", className: "badge-swa" };
  return { label: "MHA", className: "badge-mha" };
}

// src/app.ts
var COLORS = [
  "#e6194b",
  "#3cb44b",
  "#4363d8",
  "#f58231",
  "#911eb4",
  "#42d4f4",
  "#f032e6",
  "#bfef45",
  "#fabed4",
  "#469990",
  "#dcbeff",
  "#9A6324",
  "#800000",
  "#aaffc3",
  "#808000",
  "#000075"
];
var HEATMAP_SEQ_LENS = [1024, 4096, 16384, 32768, 65536, 1e5, 131072];
var HEATMAP_SEQ_LABELS = ["1K", "4K", "16K", "32K", "64K", "100K", "128K"];
var BYTES_PER_TOKEN_REFERENCE_SEQ_LEN = 1e5;
var dataset = null;
var viewModels = [];
var barChart = null;
var scatterChartActive = null;
var scatterChartTotal = null;
var benchmarkChart = null;
var kvSortState = { col: HEATMAP_SEQ_LENS.length - 1, asc: true };
var batchSortState = { col: HEATMAP_SEQ_LENS.length - 1, asc: false };
var lastModelData = [];
function getEl(id) {
  const element = document.getElementById(id);
  if (!element)
    throw new Error(`Missing element: ${id}`);
  return element;
}
function getSelectValue(id) {
  return getEl(id).value;
}
function getNumericInput(id) {
  return Number(getEl(id).value);
}
function getScenario(seqLen) {
  if (!dataset) {
    throw new Error("Dataset has not loaded.");
  }
  return {
    seqLen: seqLen ?? Math.max(1, Math.floor(getNumericInput("benchmark-seq-len"))),
    gpuMemoryGB: Math.max(1, Math.floor(getNumericInput("batch-gpu-memory"))),
    numGpus: Math.max(1, Number(getEl("num-gpus").value)),
    kvDtype: getSelectValue("dtype"),
    weightDtype: getSelectValue("weight-dtype"),
    memoryUtilization: dataset.defaults.memoryUtilization
  };
}
function updateScenarioLabels() {
  const gpuMemoryGB = Math.max(1, Math.floor(getNumericInput("batch-gpu-memory")));
  const seqLen = Math.max(1, Math.floor(getNumericInput("benchmark-seq-len")));
  getEl("batch-gpu-memory-value").textContent = `${gpuMemoryGB} GB`;
  getEl("benchmark-seq-len-value").textContent = seqLen.toLocaleString();
}
function setGpuMemory(gb) {
  getEl("batch-gpu-memory").value = String(gb);
  updateScenarioLabels();
  render();
}
function setBenchmarkSeqLen(seqLen) {
  getEl("benchmark-seq-len").value = String(seqLen);
  updateScenarioLabels();
  render();
}
function bytesPerToken(model, dtype) {
  return dtype === "fp8" ? model.result.fp8 : model.result.bf16;
}
function effectiveBytesPerToken(model, dtype, seqLen) {
  return getKVCacheAtSeqLen(model.result, seqLen, dtype) / seqLen;
}
function modelColor(model) {
  if (model.result.useMLA)
    return "#c084fc";
  if (model.result.hasHybridLinear)
    return "#60a5fa";
  if (model.result.slidingWindow)
    return "#fbbf24";
  return "#4ade80";
}
function heatColor(value, min, max) {
  if (value <= 0 || min <= 0 || max <= 0 || !Number.isFinite(value))
    return "#f3f4f6";
  if (min === max)
    return "rgba(80, 180, 60, 0.2)";
  const logVal = Math.log(value);
  const logMin = Math.log(min);
  const logMax = Math.log(max);
  const t = Math.max(0, Math.min(1, (logVal - logMin) / (logMax - logMin)));
  const r = t < 0.5 ? Math.round(80 + 175 * (t * 2)) : 255;
  const g = t < 0.5 ? 180 : Math.round(180 - 140 * ((t - 0.5) * 2));
  return `rgba(${r}, ${g}, 60, 0.25)`;
}
function heatColorInverse(value, min, max) {
  if (value <= 0)
    return "#fee2e2";
  if (min === max)
    return "rgba(80, 180, 60, 0.2)";
  return heatColor(1 / value, 1 / max, 1 / min);
}
function pointLabelPlugin(pluginId) {
  return {
    id: pluginId,
    afterDatasetsDraw(chart) {
      const ctx = chart.ctx;
      const meta = chart.getDatasetMeta(0);
      const data = chart.data.datasets[0]?.data || [];
      ctx.save();
      ctx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillStyle = "#333";
      const labels = data.map((point, index) => {
        const element = meta.data[index];
        const width = ctx.measureText(point.name).width;
        return {
          name: point.name,
          x: element?.x ?? 0,
          y: (element?.y ?? 0) - 14,
          width,
          height: 12
        };
      });
      for (let iteration = 0;iteration < 10; iteration += 1) {
        for (let i = 0;i < labels.length; i += 1) {
          for (let j = i + 1;j < labels.length; j += 1) {
            const a = labels[i];
            const b = labels[j];
            const overlapX = Math.abs(a.x - b.x) < (a.width + b.width) / 2 + 4;
            const overlapY = Math.abs(a.y - b.y) < (a.height + b.height) / 2 + 2;
            if (overlapX && overlapY) {
              if (a.y < b.y) {
                a.y -= 6;
                b.y += 6;
              } else {
                a.y += 6;
                b.y -= 6;
              }
            }
          }
        }
      }
      labels.forEach((label, index) => {
        const element = meta.data[index];
        if (!element)
          return;
        const originalY = element.y - 14;
        if (Math.abs(label.y - originalY) > 10) {
          ctx.beginPath();
          ctx.strokeStyle = "#999";
          ctx.lineWidth = 0.5;
          ctx.moveTo(element.x, element.y - 8);
          ctx.lineTo(label.x, label.y + 4);
          ctx.stroke();
        }
        ctx.textAlign = "center";
        ctx.fillText(label.name, label.x, label.y);
      });
      ctx.restore();
    }
  };
}
function downloadChart(canvasId, filename, title) {
  const canvas = getEl(canvasId);
  const titleHeight = 60;
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = canvas.width;
  exportCanvas.height = canvas.height + titleHeight;
  const ctx = exportCanvas.getContext("2d");
  if (!ctx)
    return;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
  ctx.fillStyle = "#333333";
  ctx.font = 'bold 32px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  ctx.textAlign = "center";
  ctx.fillText(title, exportCanvas.width / 2, titleHeight - 16);
  ctx.drawImage(canvas, 0, titleHeight);
  ctx.fillStyle = "#999999";
  ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
  ctx.textAlign = "right";
  ctx.fillText("roofline.cc", exportCanvas.width - 10, exportCanvas.height - 10);
  const link = document.createElement("a");
  link.download = `${filename}.png`;
  link.href = exportCanvas.toDataURL("image/png");
  link.click();
}
function renderStatus() {
  if (!dataset)
    return;
  const meta = dataset.artificialAnalysis;
  const generated = dataset.generatedAt ? new Date(dataset.generatedAt).toLocaleString() : "manual-only data";
  const fetched = meta.fetchedAt ? new Date(meta.fetchedAt).toLocaleString() : "no cache";
  const aaText = meta.modelCount > 0 ? `Artificial Analysis cache: ${meta.modelCount.toLocaleString()} models fetched ${fetched}; ${meta.matchedModelCount}/${dataset.models.length} local models matched.` : "Artificial Analysis cache: none. Run bun run fetch-aa with ARTIFICIAL_ANALYSIS_API_KEY, then bun run build.";
  getEl("data-status").textContent = `Data snapshot: ${generated}. ${aaText}`;
  getEl("aa-attribution").innerHTML = `Benchmark data source: <a href="${escapeHtml(meta.attributionUrl)}">Artificial Analysis</a>.`;
}
function render() {
  if (!dataset)
    return;
  updateScenarioLabels();
  const dtype = getSelectValue("dtype");
  const modelData = [...viewModels].sort((a, b) => effectiveBytesPerToken(a, dtype, BYTES_PER_TOKEN_REFERENCE_SEQ_LEN) - effectiveBytesPerToken(b, dtype, BYTES_PER_TOKEN_REFERENCE_SEQ_LEN));
  lastModelData = modelData;
  renderBarChart(modelData, dtype);
  renderScatterChart(modelData, dtype, "active", "scatterChartActive");
  renderScatterChart(modelData, dtype, "total", "scatterChartTotal");
  renderKVHeatmap(modelData, dtype);
  renderBatchHeatmap(modelData);
  renderBenchmarkChart(modelData);
  renderTable(modelData);
  renderStatus();
}
function renderBarChart(modelData, dtype) {
  const ctx = getEl("barChart").getContext("2d");
  if (!ctx)
    return;
  if (barChart)
    barChart.destroy();
  barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: modelData.map((model) => model.name),
      datasets: [
        {
          data: modelData.map((model) => effectiveBytesPerToken(model, dtype, BYTES_PER_TOKEN_REFERENCE_SEQ_LEN)),
          backgroundColor: modelData.map(modelColor)
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => items[0]?.label || "",
            label: (context) => {
              const model = modelData[context.dataIndex];
              const type = getModelType(model.result).label;
              return [
                `${formatNumber(context.raw)} effective bytes/token at ${BYTES_PER_TOKEN_REFERENCE_SEQ_LEN.toLocaleString()} tokens (${type})`,
                `Raw append cost: ${formatNumber(bytesPerToken(model, dtype))} bytes/token`
              ];
            }
          }
        }
      },
      scales: {
        x: { title: { display: true, text: "Effective Bytes/Token" } },
        y: { ticks: { font: { size: 10 } } }
      }
    }
  });
}
function renderScatterChart(modelData, dtype, paramType, canvasId) {
  const ctx = getEl(canvasId).getContext("2d");
  if (!ctx)
    return;
  const modelsWithSize = modelData.filter((model) => model.architecture.total_params_b !== undefined && model.architecture.active_params_b !== undefined);
  const points = modelsWithSize.map((model) => ({
    x: paramType === "total" ? model.architecture.total_params_b : model.architecture.active_params_b,
    y: effectiveBytesPerToken(model, dtype, BYTES_PER_TOKEN_REFERENCE_SEQ_LEN),
    name: model.name,
    model
  }));
  if (paramType === "active" && scatterChartActive)
    scatterChartActive.destroy();
  if (paramType === "total" && scatterChartTotal)
    scatterChartTotal.destroy();
  const chart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          data: points,
          backgroundColor: modelsWithSize.map(modelColor),
          pointRadius: 8,
          pointHoverRadius: 10
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 30 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => {
              const point = context.raw;
              const type = getModelType(point.model.result).label;
              return [
                `${point.name}: ${point.x}B params, ${formatNumber(point.y)} effective bytes/token (${type})`,
                `Raw append cost: ${formatNumber(bytesPerToken(point.model, dtype))} bytes/token`
              ];
            }
          }
        }
      },
      scales: {
        x: {
          type: "logarithmic",
          title: {
            display: true,
            text: paramType === "total" ? "Total Parameters (B)" : "Active Parameters (B)"
          },
          ticks: {
            callback: (value) => [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].includes(value) ? `${value}B` : ""
          }
        },
        y: {
          type: "logarithmic",
          title: {
            display: true,
            text: `Effective KV Bytes/Token at ${BYTES_PER_TOKEN_REFERENCE_SEQ_LEN.toLocaleString()}`
          },
          ticks: { callback: (value) => value.toLocaleString() }
        }
      }
    },
    plugins: [pointLabelPlugin(`${canvasId}-labels`)]
  });
  if (paramType === "active")
    scatterChartActive = chart;
  else
    scatterChartTotal = chart;
}
function renderBenchmarkChart(modelData) {
  const ctx = getEl("benchmarkChart").getContext("2d");
  if (!ctx)
    return;
  if (benchmarkChart)
    benchmarkChart.destroy();
  const scenario = getScenario();
  const rows = modelData.map((model) => ({
    model,
    aa: model.artificialAnalysis,
    capacity: getCapacity(model.architecture, model.result, scenario)
  })).filter((row) => row.aa !== null && row.aa.intelligenceIndex !== undefined && row.capacity !== null && row.capacity.maxRequests > 0);
  if (rows.length === 0) {
    getEl("benchmark-status").textContent = "No matched Artificial Analysis intelligence data with nonzero capacity in this build.";
    return;
  }
  getEl("benchmark-status").textContent = `Showing ${rows.length} models at ${scenario.seqLen.toLocaleString()} tokens, ${scenario.gpuMemoryGB} GB/GPU, ${scenario.numGpus} perfectly sharded GPUs.`;
  benchmarkChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          data: rows.map((row) => ({
            x: row.capacity.maxRequests,
            y: row.aa.intelligenceIndex,
            name: row.model.name,
            row
          })),
          backgroundColor: rows.map((row) => modelColor(row.model)),
          pointRadius: 8,
          pointHoverRadius: 10
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 30 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context) => {
              const row = context.raw.row;
              return [
                `${row.model.name}: ${context.raw.y} intelligence, ${context.raw.x.toLocaleString()} reqs`,
                `KV/req/GPU: ${formatBytes(row.capacity.kvBytesPerRequestPerGpu)}`,
                `Weights/GPU: ${formatBytes(row.capacity.weightBytesPerGpu)}`,
                `Speed: ${formatMetric(row.aa.medianOutputTokensPerSecond, " tok/s")}`,
                `Price: ${formatMetric(row.aa.priceBlendedUsdPer1MTokens, " USD/1M")}`
              ];
            }
          }
        }
      },
      scales: {
        x: {
          type: "logarithmic",
          title: { display: true, text: "Max Concurrent Requests" },
          ticks: { callback: (value) => value.toLocaleString() }
        },
        y: {
          title: { display: true, text: "Artificial Analysis Intelligence Index" }
        }
      }
    },
    plugins: [pointLabelPlugin("benchmark-labels")]
  });
}
function renderKVHeatmap(modelData, dtype) {
  const sortSeqLen = HEATMAP_SEQ_LENS[kvSortState.col];
  const sorted = [...modelData].sort((a, b) => {
    const aValue = getKVCacheAtSeqLen(a.result, sortSeqLen, dtype);
    const bValue = getKVCacheAtSeqLen(b.result, sortSeqLen, dtype);
    return kvSortState.asc ? aValue - bValue : bValue - aValue;
  });
  const colRanges = HEATMAP_SEQ_LENS.map((seqLen) => {
    const values = sorted.map((model) => getKVCacheAtSeqLen(model.result, seqLen, dtype)).filter((value) => value > 0);
    return { min: Math.min(...values), max: Math.max(...values) };
  });
  let html = '<table class="heatmap"><thead><tr><th>Model</th><th>Type</th>';
  HEATMAP_SEQ_LABELS.forEach((label, index) => {
    const arrow = index === kvSortState.col ? kvSortState.asc ? " ^" : " v" : "";
    html += `<th onclick="kvSortCol(${index})">${label}<span class="sort-arrow">${arrow}</span></th>`;
  });
  html += "</tr></thead><tbody>";
  sorted.forEach((model) => {
    const type = getModelType(model.result);
    html += `<tr><td>${escapeHtml(model.name)}</td><td><span class="badge ${type.className}">${type.label}</span></td>`;
    HEATMAP_SEQ_LENS.forEach((seqLen, colIndex) => {
      const value = getKVCacheAtSeqLen(model.result, seqLen, dtype);
      const bg = heatColor(value, colRanges[colIndex].min, colRanges[colIndex].max);
      html += `<td style="background:${bg}">${formatBytes(value)}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  getEl("kvHeatmap").innerHTML = html;
}
function kvSortCol(colIdx) {
  if (kvSortState.col === colIdx)
    kvSortState.asc = !kvSortState.asc;
  else {
    kvSortState.col = colIdx;
    kvSortState.asc = true;
  }
  renderKVHeatmap(lastModelData, getSelectValue("dtype"));
}
function renderBatchHeatmap(modelData) {
  const scenario = getScenario();
  const modelsWithParams = modelData.filter((model) => model.architecture.total_params_b !== undefined);
  const getMaxReqs = (model, seqLen) => getCapacity(model.architecture, model.result, { ...scenario, seqLen })?.maxRequests ?? 0;
  const sortSeqLen = HEATMAP_SEQ_LENS[batchSortState.col];
  const sorted = [...modelsWithParams].sort((a, b) => {
    const aValue = getMaxReqs(a, sortSeqLen);
    const bValue = getMaxReqs(b, sortSeqLen);
    return batchSortState.asc ? aValue - bValue : bValue - aValue;
  });
  const colRanges = HEATMAP_SEQ_LENS.map((seqLen) => {
    const values = sorted.map((model) => getMaxReqs(model, seqLen)).filter((value) => value > 0);
    return { min: values.length ? Math.min(...values) : 1, max: values.length ? Math.max(...values) : 1 };
  });
  let html = '<table class="heatmap"><thead><tr><th>Model</th>';
  HEATMAP_SEQ_LABELS.forEach((label, index) => {
    const arrow = index === batchSortState.col ? batchSortState.asc ? " ^" : " v" : "";
    html += `<th onclick="batchSortCol(${index})">${label}<span class="sort-arrow">${arrow}</span></th>`;
  });
  html += "</tr></thead><tbody>";
  sorted.forEach((model) => {
    html += `<tr><td>${escapeHtml(model.name)}</td>`;
    HEATMAP_SEQ_LENS.forEach((seqLen, colIndex) => {
      const reqs = getMaxReqs(model, seqLen);
      const bg = reqs === 0 ? "#fee2e2" : heatColorInverse(reqs, colRanges[colIndex].min, colRanges[colIndex].max);
      const text = reqs === 0 ? "-" : reqs.toLocaleString();
      html += `<td style="background:${bg}">${text}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  getEl("batchHeatmap").innerHTML = html;
}
function batchSortCol(colIdx) {
  if (batchSortState.col === colIdx)
    batchSortState.asc = !batchSortState.asc;
  else {
    batchSortState.col = colIdx;
    batchSortState.asc = false;
  }
  renderBatchHeatmap(lastModelData);
}
function renderTable(modelData) {
  const tbody = getEl("model-tbody");
  const contextLength = 131072;
  tbody.innerHTML = modelData.map((model) => {
    const result = model.result;
    const type = getModelType(result);
    const bf16PerTokenAt100k = effectiveBytesPerToken(model, "bf16", BYTES_PER_TOKEN_REFERENCE_SEQ_LEN);
    const fp8PerTokenAt100k = effectiveBytesPerToken(model, "fp8", BYTES_PER_TOKEN_REFERENCE_SEQ_LEN);
    const bf16At128k = getKVCacheAtSeqLen(result, contextLength, "bf16");
    const fp8At128k = getKVCacheAtSeqLen(result, contextLength, "fp8");
    let layersInfo = result.numLayers;
    if (result.deepseekV4) {
      layersInfo = `${result.deepseekV4.c4Layers}C4+${result.deepseekV4.c128Layers}C128`;
      if (result.deepseekV4.swaOnlyLayers > 0)
        layersInfo += `+${result.deepseekV4.swaOnlyLayers}S`;
    } else if (result.hasHybridLinear) {
      layersInfo = `${result.numFullLayers}F+${result.numLinearLayers}L`;
      if (result.numNoAttentionLayers > 0)
        layersInfo += `+${result.numNoAttentionLayers}MLP`;
    } else if (result.hasHybrid)
      layersInfo = `${result.numFullLayers}F+${result.numSlidingLayers}S`;
    const bounded = result.slidingWindow && contextLength > result.slidingWindow;
    const fixedNote = result.fixedPerRequest > 0 ? ` <span class="muted">+${formatBytes(result.fixedPerRequest)}/req</span>` : "";
    const aa = model.artificialAnalysis;
    const headDimInfo = result.vHeadDim !== result.headDim ? `${result.headDim}+${result.vHeadDim}v` : result.headDim;
    return `<tr>
        <td><strong>${escapeHtml(model.name)}</strong>${fixedNote}</td>
        <td><span class="badge ${type.className}">${type.label}</span></td>
        <td>${layersInfo}</td>
        <td>${result.useMLA ? "-" : result.kvHeads}</td>
        <td>${headDimInfo}</td>
        <td>${formatNumber(bf16PerTokenAt100k)}</td>
        <td>${formatNumber(fp8PerTokenAt100k)}</td>
        <td>${formatBytes(bf16At128k)}${bounded ? "*" : ""}</td>
        <td>${formatBytes(fp8At128k)}${bounded ? "*" : ""}</td>
        <td>${formatMetric(aa?.intelligenceIndex)}</td>
        <td>${formatMetric(aa?.medianOutputTokensPerSecond)}</td>
        <td>${formatMetric(aa?.priceBlendedUsdPer1MTokens)}</td>
      </tr>`;
  }).join("");
}
async function loadDataset() {
  const response = await fetch("data/models.json", { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`Failed to load data/models.json: ${response.status}`);
  }
  return await response.json();
}
async function main() {
  try {
    dataset = await loadDataset();
    const defaults = dataset.defaults;
    getEl("dtype").value = defaults.kvDtype;
    getEl("batch-gpu-memory").value = String(defaults.gpuMemoryGB);
    getEl("num-gpus").value = String(defaults.numGpus);
    getEl("weight-dtype").value = defaults.weightDtype;
    getEl("benchmark-seq-len").value = String(defaults.seqLen);
    viewModels = dataset.models.map((model, index) => ({
      ...model,
      result: calculateKVCache(model.architecture),
      color: COLORS[index % COLORS.length]
    }));
    for (const id of ["dtype", "num-gpus", "weight-dtype"]) {
      getEl(id).addEventListener("change", render);
    }
    getEl("batch-gpu-memory").addEventListener("input", render);
    getEl("benchmark-seq-len").addEventListener("input", render);
    render();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    getEl("data-status").textContent = `${message}. Run bun run build and serve dist/ over HTTP for local testing.`;
  }
}
window.downloadChart = downloadChart;
window.setGpuMemory = setGpuMemory;
window.setBenchmarkSeqLen = setBenchmarkSeqLen;
window.kvSortCol = kvSortCol;
window.batchSortCol = batchSortCol;
main();
