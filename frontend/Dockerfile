# syntax=docker/dockerfile:1

# --- Builder Stage ---
# Use the official UV image for fast dependency management
FROM ghcr.io/astral-sh/uv:0.5.11-python3.13-bookworm-slim AS builder

WORKDIR /app

# Enable bytecode compilation and set link mode to copy for virtualenv portability
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies into a virtualenv using a cache mount for speed
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install -r requirements.txt

# --- Final Stage ---
# Use a slim Python image for the lightweight frontend
FROM python:3.13-slim-bookworm

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Create a non-privileged user for security
ARG UID=10001
RUN useradd --create-home --shell /bin/bash --uid "${UID}" appuser

# Copy the virtualenv from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Prepare Streamlit cache and config directories
RUN mkdir -p /home/appuser/.cache /home/appuser/.streamlit && \
    chown -R appuser:appuser /home/appuser

# Copy application source code (from the frontend/ context)
COPY . .

# Ensure appuser owns the application files
RUN chown -R appuser:appuser /app

# Switch to the non-privileged user
USER appuser

# Expose Streamlit's default port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "web.py", "--server.address=0.0.0.0"]
