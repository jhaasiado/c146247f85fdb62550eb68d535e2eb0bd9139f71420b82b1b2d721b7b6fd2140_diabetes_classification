# Multi-stage Dockerfile for the diabetes ML pipeline
# Stage 1: builder installs dependencies into a local venv using uv
FROM python:3.12-slim AS builder

WORKDIR /app

# Speed up uv by avoiding hardlinks in some environments
ENV UV_LINK_MODE=copy

# Install uv and resolve dependencies from lock if present
COPY pyproject.toml uv.lock* ./
RUN pip install --no-cache-dir uv \
 && uv sync --frozen || uv sync

# Copy project code and finalize install (installs the project into the venv)
COPY . /app
RUN uv sync --frozen || uv sync

# Stage 2: runtime image copies the venv and minimal project code
FROM python:3.12-slim AS runtime

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}"

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy only what is required at runtime
COPY src ./src
COPY main.py ./main.py

# Create mount points for data and outputs (optional)
RUN mkdir -p /app/data /app/models /app/results

# Default command: run the pipeline
CMD ["python", "-m", "src.run_pipeline"]
