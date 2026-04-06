#!/bin/bash
# CLS++ Production Smoke Test
# Usage: ./smoke_test.sh [BASE_URL]
# Default: https://clsplusplus-api.onrender.com

BASE="${1:-https://clsplusplus-api.onrender.com}"
PASS=0
FAIL=0
TOTAL=0

check() {
  TOTAL=$((TOTAL + 1))
  local name="$1"
  local url="$2"
  local expect_status="${3:-200}"
  local expect_content="$4"

  local response
  response=$(curl -s -o /tmp/smoke_body -w "%{http_code}" --max-time 30 "$url" 2>/dev/null)

  if [ "$response" != "$expect_status" ]; then
    echo "FAIL: $name — expected HTTP $expect_status, got $response"
    FAIL=$((FAIL + 1))
    return
  fi

  if [ -n "$expect_content" ]; then
    if ! grep -q "$expect_content" /tmp/smoke_body 2>/dev/null; then
      echo "FAIL: $name — missing content: $expect_content"
      FAIL=$((FAIL + 1))
      return
    fi
  fi

  echo "PASS: $name"
  PASS=$((PASS + 1))
}

echo "============================================"
echo "CLS++ Smoke Test — $BASE"
echo "============================================"
echo ""

# 1. Website pages (served from API)
echo "--- Website Pages ---"
check "Homepage serves HTML"        "$BASE/"                    200 "CLS++"
check "Docs page"                   "$BASE/docs.html"           200 "API Documentation"
check "Integrations page"           "$BASE/integrate.html"      200 "Integrate"
check "Chat page"                   "$BASE/chat.html"           200 "CLS++ Chat"
check "Benchmark page"              "$BASE/benchmark.html"      200 "LoCoMo Benchmark"
check "Benchmark v1 page"           "$BASE/benchmark_v1_direct.html" 200 "LoCoMo Benchmark"
check "CSS loads"                   "$BASE/styles.css"          200 ""
check "demo.js loads"               "$BASE/demo.js"             200 "API_URL"
check "chat.js loads"               "$BASE/chat.js"             200 "API_URL"
check "integrations.js loads"       "$BASE/integrations.js"     200 "API_URL"
check "script.js loads"             "$BASE/script.js"           200 ""

echo ""
echo "--- API Endpoints ---"
check "Health endpoint"             "$BASE/v1/memory/health"    200 ""
check "Swagger docs"                "$BASE/docs"                200 "swagger"
check "ReDoc"                       "$BASE/redoc"               200 "redoc"
check "Demo status"                 "$BASE/v1/demo/status"      200 ""

echo ""
echo "--- API Write/Read ---"
# Write a memory
WRITE_RESP=$(curl -s -o /tmp/smoke_write -w "%{http_code}" --max-time 30 \
  -X POST "$BASE/v1/memory/write" \
  -H "Content-Type: application/json" \
  -d '{"content":"smoke test fact: sky is blue","namespace":"smoke-test"}' 2>/dev/null)
TOTAL=$((TOTAL + 1))
if [ "$WRITE_RESP" = "200" ] || [ "$WRITE_RESP" = "201" ]; then
  echo "PASS: Write memory"
  PASS=$((PASS + 1))
else
  echo "FAIL: Write memory — HTTP $WRITE_RESP"
  FAIL=$((FAIL + 1))
fi

# Read it back
READ_RESP=$(curl -s -o /tmp/smoke_read -w "%{http_code}" --max-time 30 \
  -X POST "$BASE/v1/memory/read" \
  -H "Content-Type: application/json" \
  -d '{"query":"what color is the sky","namespace":"smoke-test"}' 2>/dev/null)
TOTAL=$((TOTAL + 1))
if [ "$READ_RESP" = "200" ]; then
  echo "PASS: Read memory"
  PASS=$((PASS + 1))
else
  echo "FAIL: Read memory — HTTP $READ_RESP"
  FAIL=$((FAIL + 1))
fi

echo ""
echo "--- UI Test Runner ---"
check "Test runner page loads"      "$BASE/tests/ui_test_runner.html" 200 "CLS++ UI Tests"
check "Test JS loads"               "$BASE/tests/ui_tests.js"   200 "CLSTests"

echo ""
echo "============================================"
echo "Results: $PASS passed, $FAIL failed, $TOTAL total"
echo "============================================"

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
