# Dockerfile for MLFlow Tracking Server
# ======================================

FROM python:3.11-slim

LABEL maintainer="arielgiamportone"
LABEL description="MLFlow Tracking Server for DL_Bayesian project"
LABEL version="1.0.0"

# Environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_HOME=/mlflow

# Set working directory
WORKDIR ${MLFLOW_HOME}

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MLFlow and dependencies
RUN pip install --no-cache-dir \
    mlflow==2.9.0 \
    boto3>=1.33.0 \
    psycopg2-binary>=2.9.0

# Create directories for data persistence
RUN mkdir -p ${MLFLOW_HOME}/mlruns \
             ${MLFLOW_HOME}/artifacts \
             ${MLFLOW_HOME}/db

# Create non-root user
RUN useradd -m -u 1000 mlflow && \
    chown -R mlflow:mlflow ${MLFLOW_HOME}

USER mlflow

# Expose MLFlow port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Default command - SQLite backend for simplicity
# For production, use PostgreSQL backend
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow/db/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--serve-artifacts"]
