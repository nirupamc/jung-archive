# Jung Archive backend — Fly.io / Render / Railway
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for PyMuPDF, Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml README.md* ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[api,rerank]" && pip install python-dotenv

# Runtime data — read-only index baked into image
# Keep only what the API needs; exclude raw PDFs and heavy diagnostics
COPY data/chunks ./data/chunks
COPY data/chroma ./data/chroma
COPY data/bm25 ./data/bm25
COPY data/graph ./data/graph
COPY config ./config

# Hugging Face cache dir (models download on first cold start unless baked)
ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface

EXPOSE 8000

# Fly/Render set $PORT; default to 8000 for local docker
CMD ["sh", "-c", "uvicorn jung_archive.api.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
