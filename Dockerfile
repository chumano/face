FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/*

# Install libquadmath0 for certain numerical computations
RUN apt install libquadmath0

WORKDIR /app

ARG TARGET=cpu

# Copy requirements and install Python dependencies
COPY requirements.cpu.txt requirements.cpu.txt
COPY requirements.txt requirements.txt
RUN if [ "$TARGET" = "gpu" ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements.cpu.txt; \
    fi

# Copy application code
COPY src/app.py .
COPY src/config.py .

# Create directories for model files and images
RUN mkdir -p /five /app/f4r/images

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Copy gunicorn configuration
COPY src/gunicorn_config.py .

# Run the application with Gunicorn
CMD ["gunicorn", "--config", "gunicorn_config.py", "app:app"]
# gunicorn --config gunicorn_config.py app:app
