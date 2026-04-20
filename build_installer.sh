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

# Copy full Phase Memory Engine package (entire clsplusplus tree, including memory_phase.py)
cp -r "$SRC/clsplusplus" "$MAC_DIR/engine/"
while IFS= read -r d; do rm -rf "$d"; done < <(find "$MAC_DIR/engine" -type d -name __pycache__ 2>/dev/null)
find "$MAC_DIR/engine" -name '*.pyc' -delete 2>/dev/null || true

# Copy server + daemon + one-file dependency list (engine + server + daemon)
cp "$PROTO/server.py"    "$MAC_DIR/"
cp "$PROTO/daemon.py"    "$MAC_DIR/"
cp "$PROTO/daemon_requirements.txt" "$MAC_DIR/"
cp "$PROTO/clspp_bundle_requirements.txt" "$MAC_DIR/"
cp "$PROTO/memory.html"  "$MAC_DIR/"
cp "$PROTO/memory-activate.html" "$MAC_DIR/"
cp "$PROTO/index.html"   "$MAC_DIR/"
cp -r "$ROOT/extension"  "$MAC_DIR/extension"

# What this zip contains (for support / clarity)
cat > "$MAC_DIR/README-BUNDLE.txt" << 'BUNDLE_README'
CLS++ macOS bundle
==================
• Phase Memory Engine — full Python package in engine/clsplusplus/
  (thermodynamic phase engine: memory_phase.py plus supporting modules).
• Local API — server.py (FastAPI, default port 8080). No cloud required.
• Menu bar — daemon.py (Accessibility permission needed to attach to browser chats).
• Web UI — index.html (home), memory.html (viewer), memory-activate.html (Chrome extension helper) at http://127.0.0.1:8080/ui/
• Chrome MV3 extension — extension/ (Load unpacked in chrome://extensions)

From the web (recommended): open http://127.0.0.1:8080/ui/ and click the single "Install CLS++" button —
the server installs in the background. Or read INSTALL-FROM-BROWSER.txt. Fallback: double-click "Install CLS++.command". Python 3.9+ required.
Dependencies: see clspp_bundle_requirements.txt (installed automatically).
BUNDLE_README

cat > "$MAC_DIR/INSTALL-FROM-BROWSER.txt" << 'BROWSER_INSTALL'
================================================================================
  CLS++ — Install from your web browser (Mac)
================================================================================

RECOMMENDED (one click)
-----------------------
1. Start the CLS++ preview server (or open the page your team gives you at
   http://127.0.0.1:8080/ui/ ).
2. Click the single **Install CLS++** button. Everything else runs on your Mac
   in the background (no Terminal, no double-clicking the zip).
3. This tab may stop loading for a minute while the installer restarts the app.
   Wait up to 3 minutes, then refresh:  http://127.0.0.1:8080/ui/

If you opened the site from somewhere other than 127.0.0.1, the page may only
offer a zip download — use that folder’s "Install CLS++.command" in that case.

How to use CLS++ after install
------------------------------
• Chat in ChatGPT, Claude, or Gemini — memory is added in the background.
• Click 🧠 in the menu bar for the CLS++ menu.
• Open **Memories** from the web page (top right).
• Optional: load the Chrome extension from the same page.

More detail: README-BUNDLE.txt
BROWSER_INSTALL

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

# ── Install dependencies (full phase engine + server + daemon) ─
echo "  Installing dependencies (one time, ~1–3 min; includes FastAPI + engine stack)..."
BUNDLE_REQ="$DIR/clspp_bundle_requirements.txt"
if [ -f "$BUNDLE_REQ" ]; then
  "$PYTHON" -m pip install --quiet --upgrade -r "$BUNDLE_REQ" 2>/dev/null || true
else
  REQ="$DIR/daemon_requirements.txt"
  if [ -f "$REQ" ]; then
    "$PYTHON" -m pip install --quiet --upgrade -r "$REQ" \
      fastapi "uvicorn[standard]" httpx python-dotenv pydantic 2>/dev/null || true
  else
    "$PYTHON" -m pip install --quiet --upgrade \
      fastapi "uvicorn[standard]" httpx python-dotenv pydantic requests \
      rumps "pyobjc-framework-Quartz>=10.0" "pyobjc-framework-ApplicationServices>=10.0" \
      2>/dev/null || true
  fi
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

echo "  Opening CLS++ in your default browser (finish setup there)…"
open "http://127.0.0.1:8080/ui/" 2>/dev/null || true

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
