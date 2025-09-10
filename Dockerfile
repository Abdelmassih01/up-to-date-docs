# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

# --------------------
# Builder stage
# --------------------
FROM base AS builder

# Install system deps for building wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
 && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Copy dependency files first for caching
COPY pyproject.toml poetry.lock* /app/

# 👇 Install with CPU-only torch source
RUN poetry install --no-root --no-dev --source torch-cpu \
 && rm -rf ~/.cache/pypoetry ~/.cache/pip

# --------------------
# Runtime stage
# --------------------
FROM base AS runtime

# Install only what's needed at runtime (no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Copy installed site-packages from builder
COPY --from=builder /usr/local /usr/local

# Copy application code
COPY app /app/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
