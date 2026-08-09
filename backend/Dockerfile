# syntax=docker/dockerfile:1

# --- Builder Stage ---
# Use the official UV image for fast dependency management
FROM ghcr.io/astral-sh/uv:0.5.11-python3.13-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation and set link mode to copy for virtualenv portability
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Install dependencies into a virtualenv using a cache mount for speed
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/requirements.txt,target=requirements.txt \
    uv init --no-workspace && \
    uv add -r requirements.txt

# --- Final Stage ---
# Use a slim Python image for the lightweight backend
FROM python:3.13-slim-bookworm

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app" \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create a non-privileged user for security
ARG UID=10001
RUN useradd --create-home --shell /bin/bash --uid "${UID}" appuser

# Create cache directories for HuggingFace and models, ensure appuser owns them so it can write models/logs
RUN mkdir -p /home/appuser/.cache/huggingface /app/backend \
    && chown -R appuser:appuser /home/appuser /app

# Copy the virtualenv from the builder stage
COPY --from=builder /opt/venv /opt/venv

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