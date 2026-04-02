#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# CLS++ Safari Extension Builder
# Converts the web extension to a Safari app using Apple's converter tool
# Requires: Xcode 15+ with Safari Web Extension support
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SAF_SRC="$ROOT/extension-safari"
SAF_APP="$ROOT/safari-app"

echo "Building CLS++ Safari Extension..."
echo ""

# Check for Xcode command line tools
if ! command -v xcrun &>/dev/null; then
  echo "Error: Xcode command line tools not found."
  echo "Install with: xcode-select --install"
  exit 1
fi

# Check for safari-web-extension-converter
CONVERTER=$(xcrun --find safari-web-extension-converter 2>/dev/null || true)
if [ -z "$CONVERTER" ]; then
  echo "Error: safari-web-extension-converter not found."
  echo "Make sure Xcode 15+ is installed with Safari extension support."
  echo ""
  echo "You can also convert manually:"
  echo "  1. Open Xcode"
  echo "  2. File -> New -> Project -> Safari Web Extension"
  echo "  3. Copy extension-safari/ files into the Resources folder"
  exit 1
fi

# Clean previous build
rm -rf "$SAF_APP"

# Convert web extension to Xcode project
echo "Running safari-web-extension-converter..."
"$CONVERTER" "$SAF_SRC" \
  --project-location "$SAF_APP" \
  --app-name "CLS++ Memory" \
  --bundle-identifier "com.clsplusplus.memory" \
  --swift \
  --macos-only \
  --no-open \
  --force

echo ""
echo "Safari extension Xcode project created at: $SAF_APP"
echo ""
echo "Next steps:"
echo "  1. Open $SAF_APP in Xcode"
echo "  2. Sign with your Apple Developer certificate"
echo "  3. Build and archive for distribution"
echo "  4. Submit to App Store Connect"
echo ""
echo "For local testing:"
echo "  1. Build in Xcode (Cmd+B)"
echo "  2. Enable in Safari -> Settings -> Extensions"
echo ""
