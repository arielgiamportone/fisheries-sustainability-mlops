"""Health check endpoint."""

from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, status

from src.api.services.model_service import get_model_service
from src.api.services.mlflow_service import get_mlflow_service

router = APIRouter()


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check the health status of the API and its dependencies"
)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns status of:
    - API service
    - MLFlow connection
    - Model loading status
    """
    # Check MLFlow connection
    mlflow_service = get_mlflow_service()
    mlflow_connected = mlflow_service.check_connection()

    # Check model service
    model_service = get_model_service()
    model_loaded = model_service.is_model_loaded()

    # Overall health
    is_healthy = True  # API is always healthy if it can respond

    return {
        "status": "healthy" if is_healthy else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "checks": {
            "api": True,
            "mlflow_connected": mlflow_connected,
            "model_loaded": model_loaded
        },
        "cache_info": model_service.get_cache_info()
    }


@router.get(
    "/health/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="Simple liveness check for container orchestration"
)
async def liveness() -> Dict[str, str]:
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@router.get(
    "/health/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="Readiness check - confirms the service can handle requests"
)
async def readiness() -> Dict[str, Any]:
    """Kubernetes readiness probe."""
    model_service = get_model_service()
    is_ready = model_service.is_model_loaded()

    return {
        "status": "ready" if is_ready else "not_ready",
        "model_loaded": is_ready
    }
