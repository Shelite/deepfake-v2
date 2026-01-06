# Spatial Service Dockerfile
# Uses TensorFlow 2.20 + tf-keras for Keras 3.x compatibility
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY spatial_detector.py .
COPY main.py .

# Create directories
RUN mkdir -p weights temp_uploads

# Expose port
EXPOSE 8004

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8004/health')"

# Run service
CMD ["python", "main.py"]
