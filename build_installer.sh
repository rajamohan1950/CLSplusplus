#!/bin/bash
# ══════════════════════════════════════════════════════════
#  Build CLS++ macOS installer zip (engine + prototype server + daemon).
# ══════════════════════════════════════════════════════════

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIST="$ROOT/prototype/downloads"
SRC="$ROOT/src"
PROTO="$ROOT/prototype"
VERSION="1.3.0"

rm -rf "$DIST"
mkdir -p "$DIST"

echo "Building CLS++ v${VERSION} installers..."

# ── macOS installer ─────────────────────────────────────────────────────
echo ""
echo "=== macOS ==="

MAC_DIR="$DIST/_mac_build/CLS++"
mkdir -p "$MAC_DIR/engine"

# Copy engine source
cp -r "$SRC/clsplusplus" "$MAC_DIR/engine/"

# Copy server + daemon
cp "$PROTO/server.py"    "$MAC_DIR/"
cp "$PROTO/daemon.py"    "$MAC_DIR/"
cp "$PROTO/daemon_requirements.txt" "$MAC_DIR/"
cp "$PROTO/memory.html"  "$MAC_DIR/"
cp "$PROTO/index.html"   "$MAC_DIR/"

# Copy .env if exists
[ -f "$ROOT/.env" ] && cp "$ROOT/.env" "$MAC_DIR/"

# Create the one-click installer script
cat > "$MAC_DIR/Install CLS++.command" << 'INSTALLER'
#!/bin/bash
# ╔═══════════════════════════════════════════════════════╗
# ║  CLS++ Installer — this runs ONCE. After that,       ║
# ║  CLS++ starts automatically on login.                ║
# ╚═══════════════════════════════════════════════════════╝

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$HOME/.clspp"

clear
echo ""
echo "  🧠 Installing CLS++..."
echo ""

# ── Find Python 3 ──────────────────────────────────────────────
PYTHON=""
for p in python3 /usr/bin/python3 /usr/local/bin/python3 \
    "$HOME/.pyenv/shims/python3" "$HOME/miniforge3/bin/python3"; do
  if command -v "$p" &>/dev/null; then PYTHON="$p"; break; fi
done
if [ -z "$PYTHON" ]; then
  osascript -e 'display alert "CLS++ needs Python 3" message "Install from python.org" as critical'
  exit 1
fi
echo "  ✓ Python found: $PYTHON"

# ── Copy to ~/.clspp ──────────────────────────────────────────
echo "  Installing to ~/.clspp..."
mkdir -p "$APP_DIR"
cp -r "$DIR"/* "$APP_DIR/"
echo "  ✓ Files installed"

# ── Install dependencies ─────────────────────────────────────
echo "  Installing dependencies (one time, ~60s)..."
REQ="$DIR/daemon_requirements.txt"
if [ -f "$REQ" ]; then
  "$PYTHON" -m pip install --quiet --upgrade -r "$REQ" \
    fastapi "uvicorn[standard]" httpx python-dotenv 2>/dev/null || true
else
  "$PYTHON" -m pip install --quiet --upgrade \
    fastapi "uvicorn[standard]" httpx python-dotenv requests \
    rumps "pyobjc-framework-Quartz>=10.0" "pyobjc-framework-ApplicationServices>=10.0" \
    2>/dev/null || true
fi
echo "  ✓ Dependencies installed"

# ── Create launch script ────────────────────────────────────
cat > "$APP_DIR/launch.sh" << 'LAUNCH'
#!/bin/bash
DIR="$HOME/.clspp"
LOG="$DIR/.clspp.log"

# Kill stale
lsof -ti:8080 2>/dev/null | xargs kill -9 2>/dev/null
sleep 0.2

# Start server
PYTHONPATH="$DIR/engine" python3 "$DIR/server.py" >> "$LOG" 2>&1 &

# Wait for server
for i in $(seq 1 20); do
  curl -s http://localhost:8080/health > /dev/null 2>&1 && break
  sleep 0.5
done

# Start daemon (menubar app)
python3 "$DIR/daemon.py" >> "$LOG" 2>&1 &
LAUNCH
chmod +x "$APP_DIR/launch.sh"

# ── Create macOS LaunchAgent (auto-start on login) ──────────
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST="$PLIST_DIR/com.clspp.daemon.plist"
mkdir -p "$PLIST_DIR"

cat > "$PLIST" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.clspp.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>/bin/bash</string>
        <string>${APP_DIR}/launch.sh</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${APP_DIR}/.clspp.log</string>
    <key>StandardErrorPath</key>
    <string>${APP_DIR}/.clspp.log</string>
</dict>
</plist>
PLIST_EOF
echo "  ✓ Auto-start on login configured"

# ── Start now ────────────────────────────────────────────────
echo "  Starting CLS++..."
bash "$APP_DIR/launch.sh"
sleep 2

echo ""
echo "  ╔═══════════════════════════════════════════════════╗"
echo "  ║  🧠 CLS++ installed successfully!                ║"
echo "  ║                                                   ║"
echo "  ║  • 🧠 is in your menubar (top-right)             ║"
echo "  ║  • Starts automatically on login                  ║"
echo "  ║  • Open ChatGPT/Claude in any browser             ║"
echo "  ║  • Memory works silently in the background        ║"
echo "  ║                                                   ║"
echo "  ║  macOS will ask for Accessibility permission —    ║"
echo "  ║  click Allow so CLS++ can read AI chat windows.   ║"
echo "  ╚═══════════════════════════════════════════════════╝"
echo ""

osascript -e 'display notification "Open any AI in any browser. Memory is active." with title "🧠 CLS++ Installed" sound name "Glass"' 2>/dev/null

# Open Accessibility settings so user can grant permission
osascript -e 'tell application "System Preferences" to reveal anchor "Privacy_Accessibility" of pane id "com.apple.preference.security"' 2>/dev/null
osascript -e 'tell application "System Preferences" to activate' 2>/dev/null

echo "  You can close this window."
INSTALLER
chmod +x "$MAC_DIR/Install CLS++.command"

# Fix server.py sys.path for installed location
sed -i '' "s|sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))|sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'engine'))|" "$MAC_DIR/server.py"
sed -i '' "s|load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))|load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))|" "$MAC_DIR/server.py"

# Zip it
cd "$DIST/_mac_build"
zip -rq "$DIST/CLS++-macOS-v${VERSION}.zip" "CLS++"
echo "  ✓ CLS++-macOS-v${VERSION}.zip created ($(du -h "$DIST/CLS++-macOS-v${VERSION}.zip" | cut -f1))"

# ── Cleanup ─────────────────────────────────────────────────────
rm -rf "$DIST/_mac_build"

echo ""
echo "Done! Installers in: prototype/downloads/"
ls -lh "$DIST/"
