#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CLS++ Multi-Browser Extension Builder
# Generates Edge, Firefox, and Safari extensions from Chrome source
# Also creates submission-ready ZIP files for each store
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="$ROOT/extension"
VERSION=$(grep '"version"' "$SRC/manifest.json" | head -1 | sed 's/.*: *"\(.*\)".*/\1/')

echo "Building CLS++ v$VERSION for all browsers..."
echo ""

# Shared files (identical across all browsers)
SHARED_FILES=(
  "intercept.js"
  "content_chatgpt.js"
  "content_claude.js"
  "content_gemini.js"
  "popup.html"
  "icons/brain.svg"
  "icons/icon16.png"
  "icons/icon48.png"
  "icons/icon128.png"
)

# ─────────────────────────────────────────────────────────────────────────────
# EDGE — identical to Chrome (same Chromium engine)
# ─────────────────────────────────────────────────────────────────────────────
echo "1/4  Edge (Chromium)"
EDGE="$ROOT/extension-edge"
mkdir -p "$EDGE/icons"
cp "$SRC/manifest.json" "$EDGE/"
cp "$SRC/background.js" "$EDGE/"
cp "$SRC/content_common.js" "$EDGE/"
cp "$SRC/content_marker.js" "$EDGE/"
cp "$SRC/popup.js" "$EDGE/"
for f in "${SHARED_FILES[@]}"; do cp "$SRC/$f" "$EDGE/$f"; done
echo "   -> $EDGE/"

# ─────────────────────────────────────────────────────────────────────────────
# FIREFOX — MV3 with background.scripts + gecko settings
# ─────────────────────────────────────────────────────────────────────────────
echo "2/4  Firefox (Gecko)"
FF="$ROOT/extension-firefox"
mkdir -p "$FF/icons"
for f in "${SHARED_FILES[@]}"; do cp "$SRC/$f" "$FF/$f"; done
# Firefox-specific files (background.js, content_common.js, popup.js, content_marker.js)
# are maintained separately — only copy shared files here
echo "   -> $FF/"

# ─────────────────────────────────────────────────────────────────────────────
# SAFARI — MV3 web extension (needs Xcode conversion)
# ─────────────────────────────────────────────────────────────────────────────
echo "3/4  Safari (WebKit)"
SAF="$ROOT/extension-safari"
mkdir -p "$SAF/icons"
for f in "${SHARED_FILES[@]}"; do cp "$SRC/$f" "$SAF/$f"; done
# Safari-specific files are maintained separately
echo "   -> $SAF/"

# ─────────────────────────────────────────────────────────────────────────────
# ZIP — create submission-ready packages
# ─────────────────────────────────────────────────────────────────────────────
echo "4/4  Packaging ZIPs..."
DIST="$ROOT/dist"
mkdir -p "$DIST"

(cd "$ROOT/extension"       && zip -r "$DIST/clsplusplus-chrome-v$VERSION.zip" . -x '*.DS_Store' 'e2e/*' 'node_modules/*' 'package*.json' 'playwright*' 'README*')
(cd "$ROOT/extension-edge"  && zip -r "$DIST/clsplusplus-edge-v$VERSION.zip"   . -x '*.DS_Store')
(cd "$ROOT/extension-firefox" && zip -r "$DIST/clsplusplus-firefox-v$VERSION.zip" . -x '*.DS_Store')

echo ""
echo "Done! Submission packages:"
echo "  Chrome  -> dist/clsplusplus-chrome-v$VERSION.zip"
echo "  Edge    -> dist/clsplusplus-edge-v$VERSION.zip"
echo "  Firefox -> dist/clsplusplus-firefox-v$VERSION.zip"
echo "  Safari  -> Run: ./build-safari.sh (requires Xcode)"
echo ""
