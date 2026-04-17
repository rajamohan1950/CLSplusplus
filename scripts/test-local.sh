#!/usr/bin/env bash
# scripts/test-local.sh — full local verification before any prod promotion.
#
# 1. Bring up the isolated test stack (docker-compose.test.yml)
# 2. Run every test category against it, stopping on first failure
# 3. Produce a per-category pass/fail report
# 4. Leave the stack running if --keep is passed, else tear it down
#
# Usage:
#   scripts/test-local.sh            # full run, tear down stack after
#   scripts/test-local.sh --keep     # leave stack running for manual poking
#   scripts/test-local.sh --fast     # skip load / stress / dip
#
# Exit code 0 iff every category passes.
set -euo pipefail

KEEP=0
FAST=0
for arg in "$@"; do
  case "$arg" in
    --keep) KEEP=1 ;;
    --fast) FAST=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

export CLS_TEST_API_URL="${CLS_TEST_API_URL:-http://localhost:18080}"
export CLS_TEST_MODE=true

REPORT=$(mktemp)
trap 'rm -f "$REPORT"' EXIT

declare -a RESULTS

run_step() {
  local label="$1"; shift
  local logfile
  logfile=$(mktemp)
  echo ""
  echo "════════════════════════════════════════════════════════════"
  echo "▶ $label"
  echo "   $*"
  echo "════════════════════════════════════════════════════════════"
  if "$@" 2>&1 | tee "$logfile"; then
    RESULTS+=("PASS  $label")
  else
    RESULTS+=("FAIL  $label (see $logfile)")
    echo "⚠  $label FAILED — continuing with remaining categories."
  fi
}

echo "[test-local] Bringing up isolated test stack..."
make test-stack-up

# 1. Unit tests — no stack required but cheap to run here
run_step "unit"        make test-unit
run_step "regression"  make test-regression
run_step "smoke"       make test-smoke
run_step "sanity"      make test-sanity
run_step "functional"  make test-functional
run_step "blackbox"    make test-blackbox || true   # tagged overlaps fine
run_step "beta"        make test-beta
run_step "performance" make test-performance

if [ $FAST -eq 0 ]; then
  run_step "load"    make test-load
  run_step "stress"  make test-stress
  run_step "dip"     make test-dip
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Summary"
echo "════════════════════════════════════════════════════════════"
for line in "${RESULTS[@]}"; do
  echo "  $line"
done

if [ $KEEP -eq 0 ]; then
  echo ""
  echo "[test-local] Tearing down test stack..."
  make test-stack-down
else
  echo ""
  echo "[test-local] --keep set: test stack left running on :18080"
fi

# Exit non-zero if any category FAILed
for line in "${RESULTS[@]}"; do
  if [[ "$line" == FAIL* ]]; then
    exit 1
  fi
done
exit 0
