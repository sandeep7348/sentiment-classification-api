# Multi-stage Docker build for production deployment
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps — cached as separate layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data at build time (avoids runtime delay)
RUN python -c "import nltk; \
    nltk.download('stopwords', quiet=True); \
    nltk.download('wordnet',   quiet=True); \
    nltk.download('omw-1.4',   quiet=True)"

# App source
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
