#!/usr/bin/env bash
# Usage: wait-for-http.sh URL [TIMEOUT_SECONDS]
# Exits 0 when URL returns HTTP 200, 1 if timeout elapses.
set -euo pipefail

URL="${1:?URL required}"
TIMEOUT="${2:-30}"

deadline=$(($(date +%s) + TIMEOUT))
while true; do
  if curl -fsS --max-time 2 "$URL" >/dev/null 2>&1; then
    echo "OK: $URL"
    exit 0
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "TIMEOUT: $URL did not respond 200 within ${TIMEOUT}s" >&2
    exit 1
  fi
  sleep 1
done
