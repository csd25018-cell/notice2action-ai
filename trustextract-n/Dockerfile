# Dockerfile for TrustExtract-N
FROM python:3.11-slim

# Install system dependencies & Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-hin \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Setup calibration & model defaults
RUN python scripts/download_models.py

EXPOSE 8501

ENV APP_HOST=0.0.0.0
ENV APP_PORT=8501
ENV DEVICE=cpu

CMD ["python", "run_app.py"]
