# syntax=docker/dockerfile:1

# Use NVIDIA CUDA runtime as the base to ensure full GPU support for ONNX/TensorRT
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    DEBIAN_FRONTEND=noninteractive

# Install Python 3.13 and minimal runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common curl git && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
    python3.13 \
    python3.13-venv \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Link python3 to python3.13
RUN ln -sf /usr/bin/python3.13 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.13 /usr/bin/python

# Set working directory
WORKDIR /app

# Get uv for lightning-fast system-wide installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install dependencies system-wide to avoid venv/shebang issues in a container
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/requirements.txt,target=backend/requirements.txt \
    uv pip install --system -r backend/requirements.txt

# Create a non-privileged user for security
ARG UID=10001
RUN useradd --create-home --shell /bin/bash --uid "${UID}" appuser

# Prepare cache directories for HuggingFace
RUN mkdir -p /home/appuser/.cache/huggingface && chown -R appuser:appuser /home/appuser

# Copy application source code (filtered by .dockerignore)
COPY . .

# Ensure appuser owns the application files
RUN chown -R appuser:appuser /app

# Switch to the non-privileged user
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Run the backend using 'python3 -m uvicorn' to ensure the correct interpreter is used
# This avoids the "required file not found" shebang issue.
WORKDIR /app/backend
CMD ["python3", "-m", "uvicorn", "app.backend:app", "--host", "0.0.0.0", "--port", "8000"]
