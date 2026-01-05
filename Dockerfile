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
COPY ["predict.py", "model.bin", "start.sh", "./"]

# Make startup script executable
RUN chmod +x start.sh

# Expose port (Cloud Run will set PORT env var)
EXPOSE 8080

# Use startup script to handle PORT variable
CMD ["./start.sh"]
