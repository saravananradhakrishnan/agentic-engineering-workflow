# Multistage Dockerfile for multi-agent-builder web service
FROM python:3.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# Install system dependencies including curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project definition files
COPY pyproject.toml /app/
COPY multi_agent_builder/requirements.txt /app/multi_agent_builder/

# Install python dependencies from requirements
RUN pip install --no-cache-dir -r multi_agent_builder/requirements.txt

# Copy application source code
COPY multi_agent_builder /app/multi_agent_builder
COPY .env.example /app/.env.example

# Install python package itself
RUN pip install --no-cache-dir -e .

# Create workspace directory and non-root app user
RUN mkdir -p /app/workspace && \
    addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app

USER appuser

# Expose port
EXPOSE 8000

# Health check using curl against /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command to run FastAPI app using uvicorn
CMD ["uvicorn", "multi_agent_builder.api:app", "--host", "0.0.0.0", "--port", "8000"]
