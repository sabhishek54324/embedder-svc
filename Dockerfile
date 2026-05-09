# ── Build args ────────────────────────────────────────────────────────────────
# Set BASE_IMAGE to nvidia/cuda image for GPU builds, python:3.11-slim for CPU
ARG BASE_IMAGE=python:3.11-slim

FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="embedder-svc" \
      org.opencontainers.image.description="GPU/CPU vector embedding microservice" \
      org.opencontainers.image.source="https://github.com/sabhishek54324/embedder-svc"

# ── System deps + Python (needed when using CUDA base image) ─────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        python3.11 \
        python3.11-dev \
        python3-pip \
        python3.11-distutils \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ── Python env ────────────────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ── Install deps first (better layer caching) ─────────────────────────────────
COPY requirements.txt .
RUN python3.11 -m pip install --upgrade pip && python3.11 -m pip install -r requirements.txt

# ── Pre-download model into image (offline inference) ─────────────────────────
ARG EMBEDDING_MODEL=multi-qa-MiniLM-L6-cos-v1
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    TOKENIZER_MODEL=sentence-transformers/${EMBEDDING_MODEL} \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface \
    TRANSFORMERS_OFFLINE=0

RUN python3.11 -c "\
from sentence_transformers import SentenceTransformer; \
from transformers import AutoTokenizer; \
SentenceTransformer('multi-qa-MiniLM-L6-cos-v1'); \
AutoTokenizer.from_pretrained('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')"

# Switch to offline mode so the container never calls home
ENV TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

# ── Copy app ──────────────────────────────────────────────────────────────────
COPY app/ .

# ── Runtime config ────────────────────────────────────────────────────────────
ENV PORT=8000 \
    VECTOR_EMBEDDER_API_KEY=abc123 \
    MAX_TEXTS_PER_BATCH=100 \
    MAX_IMAGES_PER_BATCH=20 \
    MAX_BATCH_SIZE=8

EXPOSE 8000

CMD exec gunicorn \
    --bind :$PORT \
    --workers 2 \
    --threads 1 \
    --timeout 120 \
    --log-level info \
    --pythonpath /usr/local/bin \
    main:app