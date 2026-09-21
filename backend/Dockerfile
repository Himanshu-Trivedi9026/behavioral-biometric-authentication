# syntax=docker/dockerfile:1
#
# Phase 15A — FastAPI backend image (CPU PyTorch inference).
#
# Build context: repository ROOT (see compose.yaml `build.context: .`), so the
# top-level packages are importable exactly as in the venv workflow:
#
#     backend.app.main   (ASGI entrypoint `backend.app.main:app`)
#     ml.*               (inference pipeline, lazy-loaded by the API)
#     dataset.*          (schema/loaders)
#     baseline.*         (baseline evaluators)
#
# Model artifacts are intentionally NOT baked into the image. They are
# gitignored and runtime-mounted read-only (./models:/app/models:ro), so the
# existing local artifacts are used without being committed.
#
# .env is never copied into the image. All configuration (PostgreSQL URL, JWT
# secret, CORS, artifact paths) arrives via BBA_* environment variables, which
# fail closed when required secrets are missing (see backend/app/config.py).

FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# libgomp1 is required by the PyTorch CPU wheel at import time (OpenMP).
# Install it before Python deps so per-file caching layers stay stable.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch, pinned to the same CPU build the project was developed
# against. The plain PyPI `torch` wheel bundles CUDA runtimes that this stack
# does not need; the project's own venv instructions use this CPU index.
# Installed BEFORE requirements.txt so the `torch>=2.14.0` requirement is
# satisfied (2.14.0+cpu) and pip never resolves the CUDA bundle from PyPI.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0+cpu"

# Remaining runtime + dev-harness dependencies from the project requirements
# (FastAPI, uvicorn, psycopg[binary], PyJWT, argon2-cffi, no Markdown :).
# Copied before the source trees so dependency layers cache independently.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Top-level packages the ASGI app imports at run time.
COPY backend ./backend
COPY ml ./ml
COPY dataset ./dataset
COPY baseline ./baseline

# Run as an unprivileged user. The models directory is a read-only mount owned
# by the host, so no write permission inside /app/models is required.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Existing ASGI entrypoint; bind to 0.0.0.0 inside the container. Health is
# provided by the Compose healthcheck against GET /health (no auth needed).
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]