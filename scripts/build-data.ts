import { expandLayerPattern } from "../src/kv";
import type {
  ArtificialAnalysisBenchmark,
  ManualModel,
  ModelArchitecture,
  ModelsDataset,
} from "../src/types";

const MANUAL_MODELS_PATH = "data/models.manual.json";
const AA_CACHE_PATH = "data/aa-cache.json";
const OUTPUT_PATH = "dist/data/models.json";
const AA_SOURCE = "https://artificialanalysis.ai/api/v2/data/llms/models";
const AA_ATTRIBUTION_URL = "https://artificialanalysis.ai/";

interface RawArtificialAnalysisModel {
  id?: unknown;
  name?: unknown;
  slug?: unknown;
  model_creator?: {
    id?: unknown;
    name?: unknown;
    slug?: unknown;
  };
  evaluations?: Record<string, unknown>;
  pricing?: Record<string, unknown>;
  median_output_tokens_per_second?: unknown;
  median_time_to_first_token_seconds?: unknown;
  median_time_to_first_answer_token?: unknown;
}

interface ArtificialAnalysisCache {
  fetched_at?: string;
  source?: string;
  prompt_options?: unknown;
  data?: RawArtificialAnalysisModel[];
}

async function readJson<T>(path: string): Promise<T> {
  return (await Bun.file(path).json()) as T;
}

async function readOptionalJson<T>(path: string): Promise<T | null> {
  const file = Bun.file(path);
  if (!(await file.exists())) return null;
  return (await file.json()) as T;
}

function num(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function text(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}

function normalizeKey(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

function normalizeArtificialAnalysisModel(
  raw: RawArtificialAnalysisModel,
): ArtificialAnalysisBenchmark | null {
  const id = text(raw.id);
  const name = text(raw.name);
  const slug = text(raw.slug);
  if (!id || !name || !slug) return null;

  const evaluations = raw.evaluations || {};
  const pricing = raw.pricing || {};

  return {
    id,
    name,
    slug,
    creator: {
      id: text(raw.model_creator?.id),
      name: text(raw.model_creator?.name),
      slug: text(raw.model_creator?.slug),
    },
    intelligenceIndex: num(evaluations.artificial_analysis_intelligence_index),
    codingIndex: num(evaluations.artificial_analysis_coding_index),
    mathIndex: num(evaluations.artificial_analysis_math_index),
    priceBlendedUsdPer1MTokens: num(pricing.price_1m_blended_3_to_1),
    priceInputUsdPer1MTokens: num(pricing.price_1m_input_tokens),
    priceOutputUsdPer1MTokens: num(pricing.price_1m_output_tokens),
    medianOutputTokensPerSecond: num(raw.median_output_tokens_per_second),
    medianTimeToFirstTokenSeconds:
      num(raw.median_time_to_first_token_seconds) ?? num(raw.median_time_to_first_answer_token),
  };
}

function buildArtificialAnalysisIndex(models: ArtificialAnalysisBenchmark[]) {
  const byId = new Map<string, ArtificialAnalysisBenchmark>();
  const bySlug = new Map<string, ArtificialAnalysisBenchmark>();
  const byName = new Map<string, ArtificialAnalysisBenchmark>();

  for (const model of models) {
    byId.set(model.id, model);
    bySlug.set(model.slug, model);
    byName.set(normalizeKey(model.name), model);
    byName.set(normalizeKey(model.slug), model);
  }

  return { byId, bySlug, byName };
}

function matchArtificialAnalysisModel(
  manual: ManualModel,
  index: ReturnType<typeof buildArtificialAnalysisIndex>,
): ArtificialAnalysisBenchmark | null {
  if (manual.aa?.id) {
    const match = index.byId.get(manual.aa.id);
    if (match) return match;
  }

  if (manual.aa?.slug) {
    const match = index.bySlug.get(manual.aa.slug);
    if (match) return match;
  }

  const names = [manual.name, ...(manual.aa?.aliases || [])];
  for (const name of names) {
    const match = index.byName.get(normalizeKey(name));
    if (match) return match;
  }

  return null;
}

function normalizeArchitecture(architecture: ModelArchitecture): ModelArchitecture {
  const expanded = expandLayerPattern(architecture);
  const { layer_pattern: _layerPattern, ...withoutPattern } = expanded;
  return withoutPattern;
}

const manualModels = await readJson<ManualModel[]>(MANUAL_MODELS_PATH);
const aaCache = await readOptionalJson<ArtificialAnalysisCache>(AA_CACHE_PATH);
const aaModels = (aaCache?.data || [])
  .map(normalizeArtificialAnalysisModel)
  .filter((model): model is ArtificialAnalysisBenchmark => model !== null);
const aaIndex = buildArtificialAnalysisIndex(aaModels);

let matchedModelCount = 0;
const models = manualModels.map((manual) => {
  const artificialAnalysis = matchArtificialAnalysisModel(manual, aaIndex);
  if (artificialAnalysis) matchedModelCount += 1;

  return {
    name: manual.name,
    architecture: normalizeArchitecture(manual.architecture),
    artificialAnalysis,
  };
});

const dataset: ModelsDataset = {
  generatedAt: aaCache?.fetched_at || null,
  defaults: {
    seqLen: 131072,
    gpuMemoryGB: 192,
    numGpus: 8,
    kvDtype: "fp8",
    weightDtype: "nvfp4",
    memoryUtilization: 0.9,
  },
  artificialAnalysis: {
    source: aaCache?.source || AA_SOURCE,
    attributionUrl: AA_ATTRIBUTION_URL,
    fetchedAt: aaCache?.fetched_at || null,
    modelCount: aaModels.length,
    matchedModelCount,
    promptOptions: aaCache?.prompt_options,
  },
  models,
};

await Bun.$`mkdir -p dist/data`;
await Bun.write(OUTPUT_PATH, `${JSON.stringify(dataset, null, 2)}\n`);

console.log(
  `Wrote ${models.length} models to ${OUTPUT_PATH} (${matchedModelCount}/${models.length} matched to Artificial Analysis)`,
);
