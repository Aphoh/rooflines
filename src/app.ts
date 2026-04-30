import { escapeHtml, formatBytes, formatMetric, formatNumber } from "./format";
import {
  calculateKVCache,
  getCapacity,
  getKVCacheAtSeqLen,
  getModelType,
} from "./kv";
import type {
  ArtificialAnalysisBenchmark,
  CapacityScenario,
  KvCacheResult,
  KvDtype,
  ModelsDataset,
  NormalizedModel,
  WeightDtype,
} from "./types";

declare const Chart: any;

interface ViewModel extends NormalizedModel {
  result: KvCacheResult;
  color: string;
}

const COLORS = [
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
  "#000075",
];

const HEATMAP_SEQ_LENS = [1024, 4096, 16384, 32768, 65536, 100000, 131072];
const HEATMAP_SEQ_LABELS = ["1K", "4K", "16K", "32K", "64K", "100K", "128K"];
const BYTES_PER_TOKEN_REFERENCE_SEQ_LEN = 100000;

let dataset: ModelsDataset | null = null;
let viewModels: ViewModel[] = [];
let barChart: any = null;
let scatterChartActive: any = null;
let scatterChartTotal: any = null;
let benchmarkChart: any = null;
let kvSortState = { col: HEATMAP_SEQ_LENS.length - 1, asc: true };
let batchSortState = { col: HEATMAP_SEQ_LENS.length - 1, asc: false };
let lastModelData: ViewModel[] = [];

function getEl<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing element: ${id}`);
  return element as T;
}

function getSelectValue<T extends string>(id: string): T {
  return getEl<HTMLSelectElement>(id).value as T;
}

function getNumericInput(id: string): number {
  return Number(getEl<HTMLInputElement>(id).value);
}

function getScenario(seqLen?: number): CapacityScenario {
  if (!dataset) {
    throw new Error("Dataset has not loaded.");
  }

  return {
    seqLen: seqLen ?? Math.max(1, Math.floor(getNumericInput("benchmark-seq-len"))),
    gpuMemoryGB: Math.max(1, Math.floor(getNumericInput("batch-gpu-memory"))),
    numGpus: Math.max(1, Number(getEl<HTMLSelectElement>("num-gpus").value)),
    kvDtype: getSelectValue<KvDtype>("dtype"),
    weightDtype: getSelectValue<WeightDtype>("weight-dtype"),
    memoryUtilization: dataset.defaults.memoryUtilization,
  };
}

function updateScenarioLabels(): void {
  const gpuMemoryGB = Math.max(1, Math.floor(getNumericInput("batch-gpu-memory")));
  const seqLen = Math.max(1, Math.floor(getNumericInput("benchmark-seq-len")));
  getEl("batch-gpu-memory-value").textContent = `${gpuMemoryGB} GB`;
  getEl("benchmark-seq-len-value").textContent = seqLen.toLocaleString();
}

function setGpuMemory(gb: number): void {
  getEl<HTMLInputElement>("batch-gpu-memory").value = String(gb);
  updateScenarioLabels();
  render();
}

function setBenchmarkSeqLen(seqLen: number): void {
  getEl<HTMLInputElement>("benchmark-seq-len").value = String(seqLen);
  updateScenarioLabels();
  render();
}

function bytesPerToken(model: ViewModel, dtype: KvDtype): number {
  return dtype === "fp8" ? model.result.fp8 : model.result.bf16;
}

function effectiveBytesPerToken(model: ViewModel, dtype: KvDtype, seqLen: number): number {
  return getKVCacheAtSeqLen(model.result, seqLen, dtype) / seqLen;
}

function modelColor(model: ViewModel): string {
  if (model.result.useMLA) return "#c084fc";
  if (model.result.hasHybridLinear) return "#60a5fa";
  if (model.result.slidingWindow) return "#fbbf24";
  return "#4ade80";
}

function heatColor(value: number, min: number, max: number): string {
  if (value <= 0 || min <= 0 || max <= 0 || !Number.isFinite(value)) return "#f3f4f6";
  if (min === max) return "rgba(80, 180, 60, 0.2)";

  const logVal = Math.log(value);
  const logMin = Math.log(min);
  const logMax = Math.log(max);
  const t = Math.max(0, Math.min(1, (logVal - logMin) / (logMax - logMin)));
  const r = t < 0.5 ? Math.round(80 + 175 * (t * 2)) : 255;
  const g = t < 0.5 ? 180 : Math.round(180 - 140 * ((t - 0.5) * 2));
  return `rgba(${r}, ${g}, 60, 0.25)`;
}

function heatColorInverse(value: number, min: number, max: number): string {
  if (value <= 0) return "#fee2e2";
  if (min === max) return "rgba(80, 180, 60, 0.2)";
  return heatColor(1 / value, 1 / max, 1 / min);
}

function pointLabelPlugin(pluginId: string) {
  return {
    id: pluginId,
    afterDatasetsDraw(chart: any) {
      const ctx = chart.ctx;
      const meta = chart.getDatasetMeta(0);
      const data = chart.data.datasets[0]?.data || [];
      ctx.save();
      ctx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillStyle = "#333";

      const labels = data.map((point: { name: string }, index: number) => {
        const element = meta.data[index];
        const width = ctx.measureText(point.name).width;
        return {
          name: point.name,
          x: element?.x ?? 0,
          y: (element?.y ?? 0) - 14,
          width,
          height: 12,
        };
      });

      for (let iteration = 0; iteration < 10; iteration += 1) {
        for (let i = 0; i < labels.length; i += 1) {
          for (let j = i + 1; j < labels.length; j += 1) {
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

      labels.forEach((label: { name: string; x: number; y: number }, index: number) => {
        const element = meta.data[index];
        if (!element) return;
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
    },
  };
}

function downloadChart(canvasId: string, filename: string, title: string): void {
  const canvas = getEl<HTMLCanvasElement>(canvasId);
  const titleHeight = 60;
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = canvas.width;
  exportCanvas.height = canvas.height + titleHeight;
  const ctx = exportCanvas.getContext("2d");
  if (!ctx) return;

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

function renderStatus(): void {
  if (!dataset) return;
  const meta = dataset.artificialAnalysis;
  const generated = dataset.generatedAt
    ? new Date(dataset.generatedAt).toLocaleString()
    : "manual-only data";
  const fetched = meta.fetchedAt ? new Date(meta.fetchedAt).toLocaleString() : "no cache";
  const aaText =
    meta.modelCount > 0
      ? `Artificial Analysis cache: ${meta.modelCount.toLocaleString()} models fetched ${fetched}; ${meta.matchedModelCount}/${dataset.models.length} local models matched.`
      : "Artificial Analysis cache: none. Run bun run fetch-aa with ARTIFICIAL_ANALYSIS_API_KEY, then bun run build.";

  getEl("data-status").textContent = `Data snapshot: ${generated}. ${aaText}`;
  getEl("aa-attribution").innerHTML =
    `Benchmark data source: <a href="${escapeHtml(meta.attributionUrl)}">Artificial Analysis</a>.`;
}

function render(): void {
  if (!dataset) return;
  updateScenarioLabels();

  const dtype = getSelectValue<KvDtype>("dtype");
  const modelData = [...viewModels].sort(
    (a, b) =>
      effectiveBytesPerToken(a, dtype, BYTES_PER_TOKEN_REFERENCE_SEQ_LEN) -
      effectiveBytesPerToken(b, dtype, BYTES_PER_TOKEN_REFERENCE_SEQ_LEN),
  );
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

function renderBarChart(modelData: ViewModel[], dtype: KvDtype): void {
  const ctx = getEl<HTMLCanvasElement>("barChart").getContext("2d");
  if (!ctx) return;
  if (barChart) barChart.destroy();

  barChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: modelData.map((model) => model.name),
      datasets: [
        {
          data: modelData.map((model) =>
            effectiveBytesPerToken(model, dtype, BYTES_PER_TOKEN_REFERENCE_SEQ_LEN),
          ),
          backgroundColor: modelData.map(modelColor),
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items: any[]) => items[0]?.label || "",
            label: (context: any) => {
              const model = modelData[context.dataIndex];
              const type = getModelType(model.result).label;
              return [
                `${formatNumber(context.raw)} effective bytes/token at ${BYTES_PER_TOKEN_REFERENCE_SEQ_LEN.toLocaleString()} tokens (${type})`,
                `Raw append cost: ${formatNumber(bytesPerToken(model, dtype))} bytes/token`,
              ];
            },
          },
        },
      },
      scales: {
        x: { title: { display: true, text: "Effective Bytes/Token" } },
        y: { ticks: { font: { size: 10 } } },
      },
    },
  });
}

function renderScatterChart(
  modelData: ViewModel[],
  dtype: KvDtype,
  paramType: "active" | "total",
  canvasId: string,
): void {
  const ctx = getEl<HTMLCanvasElement>(canvasId).getContext("2d");
  if (!ctx) return;

  const modelsWithSize = modelData.filter(
    (model) =>
      model.architecture.total_params_b !== undefined &&
      model.architecture.active_params_b !== undefined,
  );
  const points = modelsWithSize.map((model) => ({
    x:
      paramType === "total"
        ? model.architecture.total_params_b
        : model.architecture.active_params_b,
    y: effectiveBytesPerToken(model, dtype, BYTES_PER_TOKEN_REFERENCE_SEQ_LEN),
    name: model.name,
    model,
  }));

  if (paramType === "active" && scatterChartActive) scatterChartActive.destroy();
  if (paramType === "total" && scatterChartTotal) scatterChartTotal.destroy();

  const chart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          data: points,
          backgroundColor: modelsWithSize.map(modelColor),
          pointRadius: 8,
          pointHoverRadius: 10,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 30 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context: any) => {
              const point = context.raw;
              const type = getModelType(point.model.result).label;
              return [
                `${point.name}: ${point.x}B params, ${formatNumber(point.y)} effective bytes/token (${type})`,
                `Raw append cost: ${formatNumber(bytesPerToken(point.model, dtype))} bytes/token`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          type: "logarithmic",
          title: {
            display: true,
            text: paramType === "total" ? "Total Parameters (B)" : "Active Parameters (B)",
          },
          ticks: {
            callback: (value: number) =>
              [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].includes(value)
                ? `${value}B`
                : "",
          },
        },
        y: {
          type: "logarithmic",
          title: {
            display: true,
            text: `Effective KV Bytes/Token at ${BYTES_PER_TOKEN_REFERENCE_SEQ_LEN.toLocaleString()}`,
          },
          ticks: { callback: (value: number) => value.toLocaleString() },
        },
      },
    },
    plugins: [pointLabelPlugin(`${canvasId}-labels`)],
  });

  if (paramType === "active") scatterChartActive = chart;
  else scatterChartTotal = chart;
}

function renderBenchmarkChart(modelData: ViewModel[]): void {
  const ctx = getEl<HTMLCanvasElement>("benchmarkChart").getContext("2d");
  if (!ctx) return;
  if (benchmarkChart) benchmarkChart.destroy();

  const scenario = getScenario();
  const rows = modelData
    .map((model) => ({
      model,
      aa: model.artificialAnalysis,
      capacity: getCapacity(model.architecture, model.result, scenario),
    }))
    .filter(
      (row): row is {
        model: ViewModel;
        aa: ArtificialAnalysisBenchmark;
        capacity: NonNullable<ReturnType<typeof getCapacity>>;
      } =>
        row.aa !== null &&
        row.aa.intelligenceIndex !== undefined &&
        row.capacity !== null &&
        row.capacity.maxRequests > 0,
    );

  if (rows.length === 0) {
    getEl("benchmark-status").textContent =
      "No matched Artificial Analysis intelligence data with nonzero capacity in this build.";
    return;
  }

  getEl("benchmark-status").textContent =
    `Showing ${rows.length} models at ${scenario.seqLen.toLocaleString()} tokens, ${scenario.gpuMemoryGB} GB/GPU, ${scenario.numGpus} perfectly sharded GPUs.`;

  benchmarkChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          data: rows.map((row) => ({
            x: row.capacity.maxRequests,
            y: row.aa.intelligenceIndex,
            name: row.model.name,
            row,
          })),
          backgroundColor: rows.map((row) => modelColor(row.model)),
          pointRadius: 8,
          pointHoverRadius: 10,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      layout: { padding: { top: 30 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (context: any) => {
              const row = context.raw.row;
              return [
                `${row.model.name}: ${context.raw.y} intelligence, ${context.raw.x.toLocaleString()} reqs`,
                `KV/req/GPU: ${formatBytes(row.capacity.kvBytesPerRequestPerGpu)}`,
                `Weights/GPU: ${formatBytes(row.capacity.weightBytesPerGpu)}`,
                `Speed: ${formatMetric(row.aa.medianOutputTokensPerSecond, " tok/s")}`,
                `Price: ${formatMetric(row.aa.priceBlendedUsdPer1MTokens, " USD/1M")}`,
              ];
            },
          },
        },
      },
      scales: {
        x: {
          type: "logarithmic",
          title: { display: true, text: "Max Concurrent Requests" },
          ticks: { callback: (value: number) => value.toLocaleString() },
        },
        y: {
          title: { display: true, text: "Artificial Analysis Intelligence Index" },
        },
      },
    },
    plugins: [pointLabelPlugin("benchmark-labels")],
  });
}

function renderKVHeatmap(modelData: ViewModel[], dtype: KvDtype): void {
  const sortSeqLen = HEATMAP_SEQ_LENS[kvSortState.col];
  const sorted = [...modelData].sort((a, b) => {
    const aValue = getKVCacheAtSeqLen(a.result, sortSeqLen, dtype);
    const bValue = getKVCacheAtSeqLen(b.result, sortSeqLen, dtype);
    return kvSortState.asc ? aValue - bValue : bValue - aValue;
  });

  const colRanges = HEATMAP_SEQ_LENS.map((seqLen) => {
    const values = sorted
      .map((model) => getKVCacheAtSeqLen(model.result, seqLen, dtype))
      .filter((value) => value > 0);
    return { min: Math.min(...values), max: Math.max(...values) };
  });

  let html = '<table class="heatmap"><thead><tr><th>Model</th><th>Type</th>';
  HEATMAP_SEQ_LABELS.forEach((label, index) => {
    const arrow = index === kvSortState.col ? (kvSortState.asc ? " ^" : " v") : "";
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

function kvSortCol(colIdx: number): void {
  if (kvSortState.col === colIdx) kvSortState.asc = !kvSortState.asc;
  else {
    kvSortState.col = colIdx;
    kvSortState.asc = true;
  }
  renderKVHeatmap(lastModelData, getSelectValue<KvDtype>("dtype"));
}

function renderBatchHeatmap(modelData: ViewModel[]): void {
  const scenario = getScenario();
  const modelsWithParams = modelData.filter(
    (model) => model.architecture.total_params_b !== undefined,
  );

  const getMaxReqs = (model: ViewModel, seqLen: number) =>
    getCapacity(model.architecture, model.result, { ...scenario, seqLen })?.maxRequests ?? 0;

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
    const arrow = index === batchSortState.col ? (batchSortState.asc ? " ^" : " v") : "";
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

function batchSortCol(colIdx: number): void {
  if (batchSortState.col === colIdx) batchSortState.asc = !batchSortState.asc;
  else {
    batchSortState.col = colIdx;
    batchSortState.asc = false;
  }
  renderBatchHeatmap(lastModelData);
}

function renderTable(modelData: ViewModel[]): void {
  const tbody = getEl<HTMLTableSectionElement>("model-tbody");
  const contextLength = 131072;

  tbody.innerHTML = modelData
    .map((model) => {
      const result = model.result;
      const type = getModelType(result);
      const bf16PerTokenAt100k = effectiveBytesPerToken(model, "bf16", BYTES_PER_TOKEN_REFERENCE_SEQ_LEN);
      const fp8PerTokenAt100k = effectiveBytesPerToken(model, "fp8", BYTES_PER_TOKEN_REFERENCE_SEQ_LEN);
      const bf16At128k = getKVCacheAtSeqLen(result, contextLength, "bf16");
      const fp8At128k = getKVCacheAtSeqLen(result, contextLength, "fp8");

      let layersInfo: string | number = result.numLayers;
      if (result.deepseekV4) {
        layersInfo = `${result.deepseekV4.c4Layers}C4+${result.deepseekV4.c128Layers}C128`;
        if (result.deepseekV4.swaOnlyLayers > 0) layersInfo += `+${result.deepseekV4.swaOnlyLayers}S`;
      } else if (result.hasHybridLinear) {
        layersInfo = `${result.numFullLayers}F+${result.numLinearLayers}L`;
        if (result.numNoAttentionLayers > 0) layersInfo += `+${result.numNoAttentionLayers}MLP`;
      } else if (result.hasHybrid) layersInfo = `${result.numFullLayers}F+${result.numSlidingLayers}S`;

      const bounded = result.slidingWindow && contextLength > result.slidingWindow;
      const fixedNote =
        result.fixedPerRequest > 0
          ? ` <span class="muted">+${formatBytes(result.fixedPerRequest)}/req</span>`
          : "";
      const aa = model.artificialAnalysis;

      const headDimInfo =
        result.vHeadDim !== result.headDim ? `${result.headDim}+${result.vHeadDim}v` : result.headDim;

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
    })
    .join("");
}

async function loadDataset(): Promise<ModelsDataset> {
  const response = await fetch("data/models.json", { cache: "no-cache" });
  if (!response.ok) {
    throw new Error(`Failed to load data/models.json: ${response.status}`);
  }
  return (await response.json()) as ModelsDataset;
}

async function main(): Promise<void> {
  try {
    dataset = await loadDataset();
    const defaults = dataset.defaults;
    getEl<HTMLSelectElement>("dtype").value = defaults.kvDtype;
    getEl<HTMLInputElement>("batch-gpu-memory").value = String(defaults.gpuMemoryGB);
    getEl<HTMLSelectElement>("num-gpus").value = String(defaults.numGpus);
    getEl<HTMLSelectElement>("weight-dtype").value = defaults.weightDtype;
    getEl<HTMLInputElement>("benchmark-seq-len").value = String(defaults.seqLen);

    viewModels = dataset.models.map((model, index) => ({
      ...model,
      result: calculateKVCache(model.architecture),
      color: COLORS[index % COLORS.length],
    }));

    for (const id of ["dtype", "num-gpus", "weight-dtype"]) {
      getEl(id).addEventListener("change", render);
    }
    getEl("batch-gpu-memory").addEventListener("input", render);
    getEl("benchmark-seq-len").addEventListener("input", render);

    render();
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    getEl("data-status").textContent =
      `${message}. Run bun run build and serve dist/ over HTTP for local testing.`;
  }
}

(window as any).downloadChart = downloadChart;
(window as any).setGpuMemory = setGpuMemory;
(window as any).setBenchmarkSeqLen = setBenchmarkSeqLen;
(window as any).kvSortCol = kvSortCol;
(window as any).batchSortCol = batchSortCol;

void main();
