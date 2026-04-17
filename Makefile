.PHONY: install run \
        docker-up docker-down docker-logs \
        test-stack-up test-stack-down test-stack-logs \
        test test-unit test-regression test-smoke test-sanity \
        test-functional test-blackbox test-whitebox test-journey \
        test-performance test-load test-stress test-dip \
        test-local test-all coverage

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
install:
	pip install -e ".[server,dev]"

run:
	uvicorn clsplusplus.api:app --host 0.0.0.0 --port 8080 --reload

# -----------------------------------------------------------------------------
# Dev stack (docker-compose.yml) — for running the server locally
# -----------------------------------------------------------------------------
docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f cls-api

# -----------------------------------------------------------------------------
# Test stack (docker-compose.test.yml) — isolated, for pytest / locust
# -----------------------------------------------------------------------------
test-stack-up:
	docker compose -f docker-compose.test.yml up -d --build
	@echo "Waiting for test backend on :18080 ..."
	@./scripts/wait-for-http.sh http://localhost:18080/health 60

test-stack-down:
	docker compose -f docker-compose.test.yml down -v

test-stack-logs:
	docker compose -f docker-compose.test.yml logs -f cls-api-test

# -----------------------------------------------------------------------------
# Per-category test runs. Each target owns one marker.
# `test-unit` does NOT need the test stack up; every other category does.
# -----------------------------------------------------------------------------
test: test-unit  ## alias: unit tests only (fast loop)

test-unit:
	pytest tests/ -m "unit or not (functional or smoke or sanity or performance or load or stress or dip or beta or blackbox or whitebox or regression)" \
	  --ignore=tests/functional --ignore=tests/smoke --ignore=tests/journey --ignore=tests/load \
	  -v

test-regression:
	pytest tests/test_regression.py -v

test-smoke:
	pytest tests/smoke -m smoke -v

test-sanity:
	pytest tests/ -m sanity -v

test-functional:
	pytest tests/functional -m functional -v

test-blackbox:
	pytest tests/ -m blackbox -v

test-whitebox:
	pytest tests/ -m whitebox -v

test-journey: test-beta

test-beta:
	pytest tests/journey -m beta -v

test-performance:
	pytest tests/load/test_performance.py -m performance -v -s

test-load:
	locust -f tests/load/locustfile.py --headless --users 50 --spawn-rate 10 --run-time 2m --host $${CLS_TEST_API_URL:-http://localhost:18080}

test-stress:
	LOCUST_SHAPE=stress locust -f tests/load/locustfile.py --headless --host $${CLS_TEST_API_URL:-http://localhost:18080}

test-dip:
	LOCUST_SHAPE=dip locust -f tests/load/locustfile.py --headless --host $${CLS_TEST_API_URL:-http://localhost:18080}

# -----------------------------------------------------------------------------
# Meta targets: local verification before any prod push
# -----------------------------------------------------------------------------
test-local:
	@./scripts/test-local.sh

test-all: test-unit test-regression test-smoke test-sanity test-functional test-beta test-performance

coverage:
	pytest tests/ --ignore=tests/load --cov=src/clsplusplus --cov-report=term-missing --cov-report=html
