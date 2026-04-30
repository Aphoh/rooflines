const DIST_DIR = "dist";
const POSTS_DIR = "posts";
const POSTS_PLACEHOLDER = "<!-- POSTS_LIST -->";

const rootUrl = new URL("../", import.meta.url);
const rootPath = decodeURIComponent(rootUrl.pathname);
process.chdir(rootPath);

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function postSlug(path: string): string {
  const filename = path.split("/").pop() || path;
  return filename.replace(/\.typ$/i, "");
}

async function extractTitle(path: string): Promise<string> {
  const source = await Bun.file(path).text();
  const heading = source
    .split(/\r?\n/)
    .find((line: string) => line.startsWith("= "))
    ?.replace(/^= /, "")
    .trim();
  return heading || postSlug(path);
}

async function runTypst(input: string, output: string): Promise<void> {
  await Bun.$`typst compile ${input} ${output}`;
}

async function buildApp(): Promise<void> {
  const result = await Bun.build({
    entrypoints: ["src/app.ts"],
    outdir: `${DIST_DIR}/assets`,
    target: "browser",
    format: "esm",
  });

  if (!result.success) {
    for (const log of result.logs) console.error(log);
    throw new Error("Browser bundle failed.");
  }
}

function wrapPostHtml(title: string, desktopSvg: string, mobileSvg: string): string {
  const escapedTitle = escapeHtml(title);
  return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${escapedTitle}</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.5;
            padding: 1rem;
        }
        .container { max-width: 800px; margin: 0 auto; }
        .nav { margin-bottom: 1rem; }
        .nav a { color: #4f46e5; text-decoration: none; font-size: 0.85rem; }
        .nav a:hover { text-decoration: underline; }
        .card {
            background: white;
            border: 1px solid #ddd;
            padding: 2rem;
            margin-bottom: 1rem;
        }
        .page svg { width: 100%; height: auto; }
        .page-mobile { display: none; }
        @media (max-width: 600px) {
            .page-desktop { display: none; }
            .page-mobile { display: block; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="nav"><a href="../index.html">&larr; back</a></div>
        <div class="card">
            <div class="page page-desktop">${desktopSvg}</div>
            <div class="page page-mobile">${mobileSvg}</div>
        </div>
    </div>
</body>
</html>
`;
}

async function buildPosts(): Promise<string> {
  const glob = new Bun.Glob(`${POSTS_DIR}/*.typ`);
  const postPaths: string[] = [];
  for await (const path of glob.scan(".")) {
    if (!path.includes("/.tmp-")) postPaths.push(path);
  }

  const links: string[] = [];
  for (const path of postPaths.sort()) {
    const slug = postSlug(path);
    const title = await extractTitle(path);
    const desktopWrapper = `${POSTS_DIR}/.tmp-desktop-${slug}.typ`;
    const mobileWrapper = `${POSTS_DIR}/.tmp-mobile-${slug}.typ`;
    const desktopSvgPath = `${DIST_DIR}/posts/${slug}-desktop.svg`;
    const mobileSvgPath = `${DIST_DIR}/posts/${slug}-mobile.svg`;
    const postHtmlPath = `${DIST_DIR}/posts/${slug}.html`;

    await Bun.write(
      desktopWrapper,
      `#set page(width: 500pt, height: auto, margin: 40pt)\n#include "${slug}.typ"\n`,
    );
    await Bun.write(
      mobileWrapper,
      `#set page(width: 250pt, height: auto, margin: 0pt)\n#include "${slug}.typ"\n`,
    );

    try {
      await runTypst(desktopWrapper, desktopSvgPath);
      await runTypst(mobileWrapper, mobileSvgPath);

      const desktopSvg = await Bun.file(desktopSvgPath).text();
      const mobileSvg = await Bun.file(mobileSvgPath).text();
      await Bun.write(postHtmlPath, wrapPostHtml(title, desktopSvg, mobileSvg));
      links.push(`<li><a href="posts/${slug}.html">${escapeHtml(title)}</a></li>`);
    } finally {
      await Bun.$`rm -f ${desktopWrapper} ${mobileWrapper} ${desktopSvgPath} ${mobileSvgPath}`;
    }
  }

  if (links.length === 0) return "";
  return `<div class="card"><h2>Roofline Calculations</h2><ul style="list-style: none; padding: 0;">${links.join("")}</ul></div>`;
}

async function buildIndex(postsHtml: string): Promise<void> {
  const indexHtml = await Bun.file("index.html").text();
  await Bun.write(`${DIST_DIR}/index.html`, indexHtml.replace(POSTS_PLACEHOLDER, postsHtml));
}

await Bun.$`rm -rf ${DIST_DIR}`;
await Bun.$`mkdir -p ${DIST_DIR}/posts ${DIST_DIR}/assets ${DIST_DIR}/data`;

await Bun.$`bun scripts/build-data.ts`;
await buildApp();
const postsHtml = await buildPosts();
await buildIndex(postsHtml);

console.log(`Build complete: ${rootPath}${DIST_DIR}`);

export {};
