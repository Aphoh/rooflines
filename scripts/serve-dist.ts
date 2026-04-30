const DIST_DIR = "dist";
const PORT = Number(Bun.env.PORT || 8080);

const contentTypes: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".webp": "image/webp",
};

function contentType(pathname: string): string {
  const match = pathname.match(/\.[^.]+$/);
  return match ? contentTypes[match[0]] || "application/octet-stream" : "application/octet-stream";
}

function handler(request: Request): Promise<Response> {
  return handleRequest(request);
}

async function handleRequest(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const pathname = decodeURIComponent(url.pathname);
  if (pathname.includes("..")) {
    return new Response("Bad request", { status: 400 });
  }

  const relativePath = pathname === "/" ? "/index.html" : pathname;
  const file = Bun.file(`${DIST_DIR}${relativePath}`);
  if (!(await file.exists())) {
    return new Response("Not found", { status: 404 });
  }

  return new Response(file, {
    headers: {
      "content-type": contentType(relativePath),
    },
  });
}

let server: ReturnType<typeof Bun.serve> | null = null;
let lastError: unknown = null;

for (let offset = 0; offset <= 20; offset += 1) {
  try {
    server = Bun.serve({
      hostname: "127.0.0.1",
      port: PORT + offset,
      fetch: handler,
    });
    break;
  } catch (error) {
    lastError = error;
  }
}

if (!server) {
  throw lastError;
}

console.log(`Serving ${DIST_DIR}/ at http://${server.hostname}:${server.port}`);

export {};
