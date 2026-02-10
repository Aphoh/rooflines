#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
DIST="$ROOT/dist"
WATCH_MODE=0

usage() {
    cat <<USAGE
Usage: bash build.sh [--watch|-watch|-w]

Options:
  --watch, -watch, -w   Rebuild whenever source files change.
USAGE
}

for arg in "$@"; do
    case "$arg" in
        --watch|-watch|-w)
            WATCH_MODE=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            usage
            exit 1
            ;;
    esac
done

build_once() {
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

        # Compile at two widths: desktop (500pt) and mobile (350pt)
        # Create temp wrappers in the posts/ dir so relative includes work
        posts_dir="$(dirname "$typ_file")"
        desktop_wrapper="${posts_dir}/.tmp-desktop-${basename}.typ"
        mobile_wrapper="${posts_dir}/.tmp-mobile-${basename}.typ"
        cat > "$desktop_wrapper" <<TYP
#set page(width: 500pt, height: auto, margin: 40pt)
#include "${basename}.typ"
TYP
        cat > "$mobile_wrapper" <<TYP
#set page(width: 350pt, height: auto, margin: 20pt)
#include "${basename}.typ"
TYP

        typst compile "$desktop_wrapper" "$DIST/posts/${basename}-desktop.svg"
        typst compile "$mobile_wrapper" "$DIST/posts/${basename}-mobile.svg"
        rm "$desktop_wrapper" "$mobile_wrapper"

        # Read SVGs and remove temp files
        desktop_svg="$(cat "$DIST/posts/${basename}-desktop.svg")"
        mobile_svg="$(cat "$DIST/posts/${basename}-mobile.svg")"
        rm "$DIST/posts/${basename}-desktop.svg" "$DIST/posts/${basename}-mobile.svg"

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
        <div class="nav"><a href="../index.html">&larr; Back to KV Cache Calculator</a></div>
        <div class="card">
            <div class="page page-desktop">${desktop_svg}</div>
            <div class="page page-mobile">${mobile_svg}</div>
        </div>
    </div>
</body>
</html>
WRAPPER

        post_links="${post_links}<li><a href=\"posts/${basename}.html\">${title}</a></li>"
    done

    # Replace POSTS_LIST placeholder in dist/index.html
    if [ -n "$post_links" ]; then
        replacement="<div class=\"card\"><h2>Roofline Calculations</h2><ul style=\"list-style: none; padding: 0;\">$(echo "$post_links" | sed 's/&/\\&/g')</ul></div>"
        sed -i '' "s|<!-- POSTS_LIST -->|${replacement}|" "$DIST/index.html"
    else
        sed -i '' 's|<!-- POSTS_LIST -->||' "$DIST/index.html"
    fi

    # Stage dist/ so pre-commit hook includes built files
    git add "$DIST"

    echo "Build complete: $DIST"
}

watch_loop() {
    local sleep_s=1
    local last_sig=""

    echo "Watch mode enabled. Monitoring index.html, posts/*.typ, and build.sh"

    while true; do
        local sig
        sig="$({
            [ -f "$ROOT/index.html" ] && stat -f '%m %N' "$ROOT/index.html"
            [ -f "$ROOT/build.sh" ] && stat -f '%m %N' "$ROOT/build.sh"
            for f in "$ROOT"/posts/*.typ; do
                [ -f "$f" ] || continue
                stat -f '%m %N' "$f"
            done
        } | sort)"

        if [ "$sig" != "$last_sig" ]; then
            echo "Change detected at $(date '+%Y-%m-%d %H:%M:%S'). Rebuilding..."
            if build_once; then
                last_sig="$sig"
            else
                echo "Build failed; waiting for next change..."
            fi
        fi

        sleep "$sleep_s"
    done
}

if [ "$WATCH_MODE" -eq 1 ]; then
    watch_loop
else
    build_once
fi
