# Use slim base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install only required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only necessary files
COPY requirements.txt .

# Install minimal dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Run app
CMD ["uvicorn", "automl.app:app", "--host", "0.0.0.0", "--port", "8000"]
