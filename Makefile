.PHONY: install run test docker-up docker-down

install:
	pip install -e .

run:
	uvicorn clsplusplus.api:app --host 0.0.0.0 --port 8080 --reload

test:
	pytest tests/ -v

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f cls-api
