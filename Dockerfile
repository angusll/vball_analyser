# syntax=docker/dockerfile:1

ARG GPU=false

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxcb1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml uv.lock README.md ./

ARG GPU=false
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$GPU" = "true" ]; then \
        uv sync --frozen --no-dev --no-install-project --extra tensorrt; \
    else \
        uv sync --frozen --no-dev --no-install-project; \
    fi

# Copy application source and install project package
COPY src/ ./src/
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$GPU" = "true" ]; then \
        uv sync --frozen --no-dev --extra tensorrt; \
    else \
        uv sync --frozen --no-dev; \
    fi

# Create non-root runtime user and writable mount points
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser \
    && mkdir -p /app/data /app/models /videos \
    && chown -R appuser:appuser /app/data /app/models /videos

FROM base AS runtime-false
ENV VBALL_GPU_BUILD=false

FROM base AS runtime-true
ENV VBALL_GPU_BUILD=true \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

FROM runtime-${GPU} AS final

ENV VBALL_HOST=0.0.0.0 \
    VBALL_PORT=8081 \
    VBALL_DATA_DIR=/app/data \
    VBALL_MODELS_DIR=/app/models \
    VBALL_SECRET_FILE=/app/data/.vball_secret \
    VBALL_ALLOWED_LOCAL_DIRS=/app/data:/videos

USER appuser

EXPOSE 8081

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"VBALL_PORT\", \"8081\")}/')" || exit 1

CMD ["vball-analyser"]
