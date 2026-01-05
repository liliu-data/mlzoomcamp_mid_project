FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY ["predict.py", "model.bin", "./"]

# Expose port (Cloud Run will set PORT env var)
EXPOSE 8080

# Use environment variable for port (Cloud Run compatibility)
CMD exec uvicorn predict:app --host 0.0.0.0 --port ${PORT:-8080}
