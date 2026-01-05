#!/bin/bash
# Startup script for Cloud Run that properly handles PORT environment variable

# Get PORT from environment variable, default to 8080 if not set
PORT=${PORT:-8080}

# Start uvicorn with the correct port
exec uvicorn predict:app --host 0.0.0.0 --port $PORT

