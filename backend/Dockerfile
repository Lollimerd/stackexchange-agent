# syntax=docker/dockerfile:1

# Use official Python 3.13 slim image (Debian Bookworm based)
FROM python:3.13-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app"

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

# Create cache directories for HuggingFace and models
# Ensure appuser owns them so it can write models/logs
RUN mkdir -p /home/appuser/.cache/huggingface /app/backend \
    && chown -R appuser:appuser /home/appuser /app

# Copy application source code
# --chown ensures files are owned by appuser immediately
COPY --chown=appuser:appuser . .

# Switch to non-privileged user
USER appuser

EXPOSE 8000

# Set working directory to your backend folder
WORKDIR /app/backend

# Run the FastAPI app
CMD ["python3", "-m", "uvicorn", "app.backend:app", "--host", "0.0.0.0", "--port", "8000"]