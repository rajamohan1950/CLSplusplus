FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY src/ ./src/
COPY prototype/ ./prototype/
COPY extension/ ./extension/
COPY website/ ./website/

RUN pip install --no-cache-dir \
    fastapi>=0.109.0 \
    "uvicorn[standard]>=0.27.0" \
    httpx>=0.26.0 \
    python-dotenv>=1.0.0 \
    pydantic>=2.5.0

ENV PYTHONPATH=/app/src

EXPOSE 8080

CMD ["sh", "-c", "cd /app/prototype && uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080}"]
