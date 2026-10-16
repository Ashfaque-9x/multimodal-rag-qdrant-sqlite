# Multimodal RAG Streamlit application
# Based on the official Streamlit Docker deployment guidance.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    RAG_STORAGE_DIR=/app/rag_storage \
    SQLITE_DB_PATH=/app/rag_storage/rag_app.db

WORKDIR /app

# System packages used by PDF extraction, OCR, image processing, and health checks.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies separately to improve Docker layer caching.
COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt

# Copy only application source/configuration files (see .dockerignore).
COPY app.py multimodel_rag.py ./

# Keep persistent PDF, image, SQLite, and local-Qdrant data outside the image.
RUN mkdir -p /app/rag_storage /app/rag_storage/pdfs /app/rag_storage/images \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail --silent http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
