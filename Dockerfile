FROM python:3.12-slim

WORKDIR /app

# Install system deps for sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -e .

COPY src/ ./src/

ENV PYTHONPATH=/app/src
ENV CLS_HOST=0.0.0.0
ENV CLS_PORT=8080

# Default to Redis/Postgres/MinIO from docker-compose
ENV CLS_REDIS_URL=redis://redis:6379
ENV CLS_DATABASE_URL=postgresql://cls:cls@postgres:5432/cls
ENV CLS_MINIO_ENDPOINT=minio:9000
ENV CLS_MINIO_ACCESS_KEY=minioadmin
ENV CLS_MINIO_SECRET_KEY=minioadmin

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "clsplusplus.api:app", "--host", "0.0.0.0", "--port", "8080"]
