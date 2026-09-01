FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project definition
COPY multi_agent_builder/pyproject.toml /app/

# Copy dependencies
COPY multi_agent_builder/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY multi_agent_builder /app/multi_agent_builder

# Optional: copy example environment file
COPY .env.example /app/.env.example

# Install application
RUN pip install --no-cache-dir -e .

# Non-root user
RUN mkdir -p /app/workspace && \
    addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=10s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "multi_agent_builder.api:app", "--host", "0.0.0.0", "--port", "8000"]