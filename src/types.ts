export type KvDtype = "bf16" | "fp8";
export type WeightDtype = "fp16" | "fp8" | "nvfp4";
export type LayerType = "full_attention" | "sliding_attention" | "linear_attention" | "mlp";

export interface LayerPattern {
  sequence: LayerType[];
  repeat: number;
}

export interface ModelArchitecture {
  architectures?: string[];
  model_type?: string;
  text_config?: ModelArchitecture;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  num_global_key_value_heads?: number;
  head_dim?: number;
  global_head_dim?: number;
  v_head_dim?: number;
  hidden_size?: number;
  attention_k_eq_v?: boolean;
  index_head_dim?: number;
  index_n_heads?: number;
  index_topk?: number;
  max_position_embeddings?: number;
  sliding_window?: number | Array<number | null>;
  use_sliding_window?: boolean;
  layer_types?: LayerType[];
  layer_pattern?: LayerPattern;
  hybrid_layer_pattern?: number[];
  hybrid_override_pattern?: string;
  kv_lora_rank?: number;
  qk_rope_head_dim?: number;
  sliding_window_size?: number;
  swa_head_dim?: number;
  swa_v_head_dim?: number;
  swa_num_attention_heads?: number;
  swa_num_key_value_heads?: number;
  compress_ratios?: number[];
  deepseek_v4?: {
    cache_block_size?: number;
    cache_token_bytes?: number;
    cache_alignment?: number;
    include_indexer_cache?: boolean;
    include_fixed_raw_caches?: boolean;
  };
  nsa_index_bytes_per_layer?: number;
  linear_state_per_layer?: number;
  linear_state_per_layer_bf16?: number;
  linear_state_per_layer_fp8?: number;
  n_groups?: number;
  mamba_num_heads?: number;
  mamba_head_dim?: number;
  ssm_state_size?: number;
  conv_kernel?: number;
  total_params_b?: number;
  active_params_b?: number;
}

export interface ManualModel {
  name: string;
  aa?: {
    id?: string;
    slug?: string;
    aliases?: string[];
  };
  architecture: ModelArchitecture;
}

export interface ArtificialAnalysisBenchmark {
  id: string;
  name: string;
  slug: string;
  creator?: {
    id?: string;
    name?: string;
    slug?: string;
  };
  intelligenceIndex?: number;
  codingIndex?: number;
  mathIndex?: number;
  priceBlendedUsdPer1MTokens?: number;
  priceInputUsdPer1MTokens?: number;
  priceOutputUsdPer1MTokens?: number;
  medianOutputTokensPerSecond?: number;
  medianTimeToFirstTokenSeconds?: number;
}

export interface NormalizedModel {
  name: string;
  architecture: ModelArchitecture;
  artificialAnalysis: ArtificialAnalysisBenchmark | null;
}

export interface ModelsDataset {
  generatedAt: string | null;
  defaults: {
    seqLen: number;
    gpuMemoryGB: number;
    numGpus: number;
    kvDtype: KvDtype;
    weightDtype: WeightDtype;
    memoryUtilization: number;
  };
  artificialAnalysis: {
    source: string;
    attributionUrl: string;
    fetchedAt: string | null;
    modelCount: number;
    matchedModelCount: number;
    promptOptions?: unknown;
  };
  models: NormalizedModel[];
}

export interface KvCacheResult {
  bf16: number;
  fp8: number;
  useMLA: boolean;
  slidingWindow: number | null;
  numLayers: number;
  kvHeads: number;
  headDim: number;
  vHeadDim: number;
  numFullLayers: number;
  numSlidingLayers: number;
  numLinearLayers: number;
  numNoAttentionLayers: number;
  hasHybrid: boolean;
  hasHybridLinear: boolean;
  maxCtx: number;
  fixedPerRequest: number;
  fixedPerRequestByDtype?: Partial<Record<KvDtype, number>>;
  perLayerKVBytes?: {
    full?: Partial<Record<KvDtype, number>>;
    sliding?: Partial<Record<KvDtype, number>>;
  };
  deepseekV4?: {
    c4Layers: number;
    c128Layers: number;
    swaOnlyLayers: number;
    mainCacheBytesPerToken: number;
    indexerCacheBytesPerToken: number;
    fixedSwaBytesPerRequest: number;
    fixedCompressorBytesPerRequest: number;
  };
}

export interface CapacityScenario {
  seqLen: number;
  gpuMemoryGB: number;
  numGpus: number;
  kvDtype: KvDtype;
  weightDtype: WeightDtype;
  memoryUtilization: number;
}

export interface CapacityResult {
  maxRequests: number;
  gpuMemoryBytes: number;
  usableGpuMemoryBytes: number;
  weightBytesTotal: number;
  weightBytesPerGpu: number;
  kvBytesPerRequestTotal: number;
  kvBytesPerRequestPerGpu: number;
  availableBytesPerGpu: number;
}
