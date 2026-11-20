FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt requirements-dev.txt ./

# Install PyTorch CPU (small footprint)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy NLP model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create runtime directories
RUN mkdir -p uploads static

# Expose Flask/Gunicorn port
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Production entry: gunicorn → wsgi.py
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "wsgi:application"]
