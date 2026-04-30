const API_URL = "https://artificialanalysis.ai/api/v2/data/llms/models";
const OUTPUT_PATH = "data/aa-cache.json";

const apiKey = Bun.env.ARTIFICIAL_ANALYSIS_API_KEY;

if (!apiKey) {
  console.error("Missing ARTIFICIAL_ANALYSIS_API_KEY. Copy .env.example to .env or export it.");
  process.exit(1);
}

const response = await fetch(API_URL, {
  headers: {
    "x-api-key": apiKey,
  },
});

if (!response.ok) {
  throw new Error(`Artificial Analysis request failed: ${response.status} ${response.statusText}`);
}

const body = await response.json();

if (!body || !Array.isArray(body.data)) {
  throw new Error("Artificial Analysis response did not include a data array.");
}

await Bun.write(
  OUTPUT_PATH,
  `${JSON.stringify(
    {
      fetched_at: new Date().toISOString(),
      source: API_URL,
      prompt_options: body.prompt_options ?? null,
      data: body.data,
    },
    null,
    2,
  )}\n`,
);

console.log(`Wrote ${body.data.length} Artificial Analysis LLM records to ${OUTPUT_PATH}`);

export {};
