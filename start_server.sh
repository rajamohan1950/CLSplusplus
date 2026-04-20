#!/bin/bash
cd /Users/rjabbala/Projects/CLSplusplus/prototype
export PYTHONPATH=../src
export KMP_DUPLICATE_LIB_OK=TRUE
echo "Starting CLS++ server on http://localhost:8080 ..."
exec python3 -m uvicorn server:app --host 0.0.0.0 --port 8080
