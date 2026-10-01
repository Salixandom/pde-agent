FROM python:3.12-slim

# uv for exact locked dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy source
COPY . .

ENV PATH="/app/.venv/bin:$PATH"

# SQLite DB lives in a mounted volume so it survives container restarts
ENV PDE_AGENTS_DB=/app/data/runs.db
RUN mkdir -p /app/data

VOLUME ["/app/data"]

CMD ["python", "main.py"]
