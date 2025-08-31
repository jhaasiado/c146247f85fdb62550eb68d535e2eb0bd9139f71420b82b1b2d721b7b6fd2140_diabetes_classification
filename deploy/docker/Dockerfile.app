# Placeholder Dockerfile for the application container
# This image runs the training/evaluation pipeline with UV

FROM python:3.12-slim

WORKDIR /app

# System deps you might need (uncomment as necessary)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential curl git && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Use UV to manage the venv and install deps
RUN pip install --no-cache-dir uv && \
    uv sync

# Default command: run the pipeline
CMD ["uv", "run", "python", "main.py"]
