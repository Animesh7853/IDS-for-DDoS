FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create required directories
RUN mkdir -p logs templates static/css static/js

# Create a basic dashboard.html template if it doesn't exist
RUN if [ ! -f templates/dashboard.html ]; then \
    echo '<!DOCTYPE html><html><head><title>Network Attack Monitoring Dashboard</title></head><body><h1>Network Attack Monitoring Dashboard</h1><div id="stats"></div><script src="/static/js/dashboard.js"></script></body></html>' > templates/dashboard.html; \
    fi

# Create a basic dashboard.js file if it doesn't exist
RUN if [ ! -f static/js/dashboard.js ]; then \
    echo 'document.addEventListener("DOMContentLoaded", function() { document.getElementById("stats").innerHTML = "Dashboard is running. Connect to WebSocket for real-time updates."; });' > static/js/dashboard.js; \
    fi

# Environment variables
ENV PORT=8080
ENV MODEL_PATH="model.h5"
ENV SCALER_PATH="scaler.json"
ENV LOG_DIR="logs"

# Create empty model files if they don't exist (to prevent app crash)
RUN touch model.h5
RUN echo '{"mean": [0], "std": [1]}' > scaler.json

# Expose the port that Cloud Run expects
EXPOSE 8080

# Command to run using gunicorn (more reliable for production)
CMD exec gunicorn --bind :$PORT --workers 1 --worker-class uvicorn.workers.UvicornWorker --threads 8 --timeout 0 app:app