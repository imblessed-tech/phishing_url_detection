# ─────────────────────────────────────────────
#  Phishing URL Detector — Dockerfile
#  Multi-stage build for a lean production image
# ─────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for lxml and tldextract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/         ./app/
COPY data/        ./data/
COPY train.py     .

# Train the model during the build process
RUN python train.py

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
