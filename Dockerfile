# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# OSIA shared application image
#
# One image, many services. The orchestrator, gateways, and workers are all
# the same codebase invoked with a different `command:` in docker-compose.yml.
# Build once, reuse everywhere:
#
#   docker compose build osia-app
#
# The phone/ADB stack (persona-daemon, phone-bridge, adb-server) deliberately
# stays on the host for now — it needs the physical Moto g06 over USB.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local

# System libraries required by heavy deps:
#   weasyprint -> pango, cairo, gdk-pixbuf, fonts
#   pymupdf/pdf -> already wheels, but need libgl for some render paths
#   ffmpeg -> pydub / video re-encode paths
#   curl -> container healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libffi-dev \
        libjpeg62-turbo \
        libgl1 \
        shared-mime-info \
        fonts-liberation \
        ffmpeg \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, lockfile-faithful installs
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# --- Dependency layer (cached unless pyproject/uv.lock change) ---
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# --- Application code ---
COPY . .

# Install the project itself (no dev extras) now that source is present.
RUN uv sync --frozen --no-dev

# Non-root runtime user; owns /app so log/tmp writes under the repo work.
RUN useradd --create-home --uid 1000 osia && chown -R osia:osia /app
USER osia

# Default command is the orchestrator; every other service overrides `command:`
# in docker-compose.yml.
CMD ["python", "main.py"]
