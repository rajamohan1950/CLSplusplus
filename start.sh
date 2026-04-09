#!/bin/bash
# CLS++ Server — Start script
# Usage: ./start.sh [port]

set -euo pipefail

PORT="${1:-8181}"
DIR="$(cd "$(dirname "$0")" && pwd)"

export PYTHONPATH="$DIR/src"
export KMP_DUPLICATE_LIB_OK=TRUE
export CLS_DATABASE_URL="${CLS_DATABASE_URL:-postgresql://cls:cls@localhost:5433/cls}"
export CLS_JWT_SECRET="${CLS_JWT_SECRET:-e2e-test-secret-key-min-32-bytes!}"
export CLS_TRACK_USAGE="${CLS_TRACK_USAGE:-true}"
export CLS_ENFORCE_QUOTAS="${CLS_ENFORCE_QUOTAS:-false}"
export CLS_RATE_LIMIT_REQUESTS="${CLS_RATE_LIMIT_REQUESTS:-5000}"
export CLS_RATE_LIMIT_WINDOW_SECONDS="${CLS_RATE_LIMIT_WINDOW_SECONDS:-60}"

echo "CLS++ starting on port $PORT"
echo "Database: $CLS_DATABASE_URL"
echo "PYTHONPATH: $PYTHONPATH"

exec python3 -m uvicorn clsplusplus.api:create_app \
  --factory \
  --host 0.0.0.0 \
  --port "$PORT" \
  --reload
