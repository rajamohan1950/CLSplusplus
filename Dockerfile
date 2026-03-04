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

# Render / docker-compose inject these at runtime

EXPOSE 8080

# Render sets PORT; default to 8080 for local/Docker
CMD ["sh", "-c", "uvicorn clsplusplus.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
