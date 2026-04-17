FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-render.txt ./
RUN pip install --no-cache-dir -r requirements-render.txt

COPY src/ ./src/
COPY prototype/ ./prototype/
COPY extension/ ./extension/
# website/ archived to archive/website/ — the API no longer serves static HTML.
# archive/ is copied so the waitlist test runner still finds its fixture path.
COPY archive/ ./archive/

ENV PYTHONPATH=/app/src

EXPOSE 8080

CMD ["python", "-m", "clsplusplus.main"]
