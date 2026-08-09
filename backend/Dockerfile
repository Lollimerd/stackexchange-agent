# syntax=docker/dockerfile:1

# ===========================================================================================================================================================
# Stage 1: Builder — install Python + deps via uv (manages its own Python)
# ===========================================================================================================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS builder

# Pull uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /build

# Let uv install + manage Python 3.13 itself (no apt python needed)
# Install all deps from lockfile into an isolated .venv
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --python 3.13

# ===========================================================================================================================================================
# Stage 2: Runtime — lean CUDA image, copies venv with embedded Python from builder
# ===========================================================================================================================================================
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS runtime

# Prevents Python from writing pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- GPU / ONNX Runtime configuration ---
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Cache compiled TensorRT engines so they aren't recompiled on each restart
ENV ORT_TENSORRT_CACHE_PATH=/tmp/trt_cache

# HuggingFace model cache (mounted as volume from host)
ENV HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

# Create non-privileged user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/home/appuser" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser \
    && mkdir -p /home/appuser/.cache /tmp/trt_cache \
    && chown -R appuser:appuser /home/appuser /tmp/trt_cache

# Copy entire venv from builder (includes uv-managed Python 3.13 + all packages)
COPY --from=builder /build/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

USER appuser

# Copy only the backend application source (context is project root)
COPY backend/ .

EXPOSE 8000

# 1 worker (model is shared in-process); uvloop + httptools for fast async I/O
CMD ["/app/.venv/bin/uvicorn", "app.backend:app", \
    "--host=0.0.0.0", \
    "--port=8000", \
    "--workers=1", \
    "--loop=uvloop", \
    "--http=httptools"]
