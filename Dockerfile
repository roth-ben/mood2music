# Base model
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv && \
    rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3 -m venv /opt/venv

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
RUN mkdir -p /models

# Security and environment setup
RUN groupadd -r appuser && \
    useradd -r -g appuser appuser && \
    chown appuser:appuser /app

# Create a home and cache directory and give ownership
RUN mkdir -p /home/appuser \
    && chown -R appuser:appuser /home/appuser

# Switch to non-root user
USER appuser
ENV HOME=/home/appuser

# Service configuration
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8080/health || exit 1

CMD ["uvicorn", "api.recommend:router", "--host", "0.0.0.0", "--port", "8080"]