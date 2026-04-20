#!/bin/bash
# ══════════════════════════════════════════════════════════
#  Build CLS++ .dmg installer for macOS
#  Requires: build_installer.sh to have run first (creates the zip)
# ══════════════════════════════════════════════════════════

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST="$ROOT/prototype/downloads"
VERSION="1.4.0"
DMG_NAME="CLS++-macOS-v${VERSION}"
DMG_PATH="$DIST/${DMG_NAME}.dmg"
VOLUME_NAME="CLS++"

echo "Building CLS++ v${VERSION} DMG..."

# ── Step 1: Build the zip first if not present ─────────────────────
ZIP_PATH="$DIST/CLS++-macOS-v${VERSION}.zip"
if [ ! -f "$ZIP_PATH" ]; then
    echo "  ZIP not found — building it first..."
    bash "$ROOT/build_installer.sh"
fi

# ── Step 2: Extract zip to staging area ─────────────────────────────
STAGING="$DIST/_dmg_staging"
rm -rf "$STAGING"
mkdir -p "$STAGING"
unzip -q "$ZIP_PATH" -d "$STAGING"
echo "  Extracted to staging"

# ── Step 3: Create DMG ─────────────────────────────────────────────
rm -f "$DMG_PATH"

# Create a read-write DMG first
hdiutil create \
    -volname "$VOLUME_NAME" \
    -srcfolder "$STAGING/CLS++" \
    -ov -format UDRW \
    "$DIST/_rw_${DMG_NAME}.dmg" \
    > /dev/null 2>&1

echo "  Read-write DMG created"

# Mount it to customize
MOUNT_DIR=$(hdiutil attach "$DIST/_rw_${DMG_NAME}.dmg" -readwrite -noverify -noautoopen 2>/dev/null | grep "/Volumes" | awk '{print $NF}')
if [ -z "$MOUNT_DIR" ]; then
    MOUNT_DIR="/Volumes/$VOLUME_NAME"
fi

# Set custom view options via AppleScript (Finder appearance)
osascript << APPLESCRIPT 2>/dev/null || true
tell application "Finder"
    tell disk "$VOLUME_NAME"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set bounds of container window to {100, 100, 640, 460}
        set theViewOptions to the icon view options of container window
        set icon size of theViewOptions to 72
        close
    end tell
end tell
APPLESCRIPT

sleep 1

# Unmount
hdiutil detach "$MOUNT_DIR" -quiet 2>/dev/null || hdiutil detach "/Volumes/$VOLUME_NAME" -quiet 2>/dev/null || true

# Convert to compressed read-only DMG
hdiutil convert \
    "$DIST/_rw_${DMG_NAME}.dmg" \
    -format UDZO \
    -o "$DMG_PATH" \
    > /dev/null 2>&1

echo "  Compressed DMG created"

# ── Cleanup ─────────────────────────────────────────────────────
rm -rf "$STAGING"
rm -f "$DIST/_rw_${DMG_NAME}.dmg"

echo ""
echo "Done! DMG: $DMG_PATH"
ls -lh "$DMG_PATH"
