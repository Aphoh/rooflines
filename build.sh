#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DIST="$ROOT/dist"

# Clean and create output directories
rm -rf "$DIST"
mkdir -p "$DIST/posts"

# Copy source index.html to dist
cp "$ROOT/index.html" "$DIST/index.html"

# Collect post links
post_links=""

for typ_file in "$ROOT"/posts/*.typ; do
    [ -f "$typ_file" ] || continue

    basename="$(basename "$typ_file" .typ)"

    # Extract title from first '= ' heading
    title="$(grep -m1 '^= ' "$typ_file" | sed 's/^= //')"
    [ -z "$title" ] && title="$basename"

    # Compile typst to HTML
    typst compile --features html --format html "$typ_file" "$DIST/posts/${basename}.html.tmp"

    # Wrap with site chrome
    cat > "$DIST/posts/${basename}.html" <<WRAPPER
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${title}</title>
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
    </style>
</head>
<body>
    <div class="container">
        <div class="nav"><a href="../index.html">&larr; Back to KV Cache Calculator</a></div>
        <div class="card">
$(cat "$DIST/posts/${basename}.html.tmp")
        </div>
    </div>
</body>
</html>
WRAPPER

    rm "$DIST/posts/${basename}.html.tmp"

    post_links="${post_links}<li><a href=\"posts/${basename}.html\">${title}</a></li>"
done

# Replace POSTS_LIST placeholder in dist/index.html
if [ -n "$post_links" ]; then
    replacement="<div class=\"card\"><h2>Posts</h2><ul style=\"list-style: none; padding: 0;\">$(echo "$post_links" | sed 's/&/\\&/g')</ul></div>"
    sed -i '' "s|<!-- POSTS_LIST -->|${replacement}|" "$DIST/index.html"
else
    sed -i '' 's|<!-- POSTS_LIST -->||' "$DIST/index.html"
fi

# Stage dist/ so pre-commit hook includes built files
git add "$DIST"

echo "Build complete: $DIST"
