"""Model registry endpoints."""

from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, status

from src.api.schemas.training import RegisteredModel, ModelStageTransition
from src.api.services.mlflow_service import get_mlflow_service
from src.api.services.model_service import get_model_service

router = APIRouter()


@router.get(
    "/models",
    response_model=List[RegisteredModel],
    summary="List registered models",
    description="List all models in the MLFlow Model Registry"
)
async def list_models() -> List[RegisteredModel]:
    """List all registered models."""
    mlflow_service = get_mlflow_service()

    try:
        models = mlflow_service.list_registered_models()
        return [
            RegisteredModel(
                name=model["name"],
                creation_timestamp=model.get("creation_timestamp"),
                last_updated_timestamp=model.get("last_updated_timestamp"),
                description=model.get("description"),
                latest_versions=model.get("latest_versions", [])
            )
            for model in models
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLFlow service unavailable: {str(e)}"
        )


@router.get(
    "/models/{model_name}",
    summary="Get model details",
    description="Get details of a registered model"
)
async def get_model(model_name: str) -> Dict[str, Any]:
    """Get details of a registered model."""
    mlflow_service = get_mlflow_service()

    try:
        models = mlflow_service.list_registered_models()
        for model in models:
            if model["name"] == model_name:
                return model

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_name}"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLFlow service unavailable: {str(e)}"
        )


@router.get(
    "/models/{model_name}/versions",
    summary="List model versions",
    description="List all versions of a registered model"
)
async def list_model_versions(model_name: str) -> List[Dict[str, Any]]:
    """List all versions of a model."""
    mlflow_service = get_mlflow_service()

    try:
        versions = mlflow_service.get_model_versions(model_name)
        if not versions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_name}"
            )
        return versions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLFlow service unavailable: {str(e)}"
        )


@router.get(
    "/models/{model_name}/production",
    summary="Get production model",
    description="Get the production version of a model"
)
async def get_production_model(model_name: str) -> Dict[str, Any]:
    """Get the production version of a model."""
    mlflow_service = get_mlflow_service()

    try:
        version = mlflow_service.get_production_model_info(model_name)
        if version is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No production version found for model: {model_name}"
            )
        return version
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLFlow service unavailable: {str(e)}"
        )


@router.post(
    "/models/{model_name}/versions/{version}/stage",
    summary="Transition model stage",
    description="Transition a model version to a new stage (Staging, Production, Archived)"
)
async def transition_model_stage(
    model_name: str,
    version: str,
    transition: ModelStageTransition
) -> Dict[str, str]:
    """Transition a model version to a new stage."""
    mlflow_service = get_mlflow_service()

    try:
        mlflow_service.transition_model_stage(
            model_name=model_name,
            version=version,
            stage=transition.stage,
            archive_existing=transition.archive_existing
        )

        # Clear model cache to force reload
        model_service = get_model_service()
        model_service.clear_cache()

        return {
            "message": f"Model {model_name} version {version} transitioned to {transition.stage}",
            "model_name": model_name,
            "version": version,
            "stage": transition.stage
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to transition model: {str(e)}"
        )


@router.get(
    "/models/{model_name}/metrics",
    summary="Get model metrics",
    description="Get metrics for a model (production version by default)"
)
async def get_model_metrics(
    model_name: str,
    version: str = None
) -> Dict[str, float]:
    """Get metrics for a model."""
    mlflow_service = get_mlflow_service()

    try:
        metrics = mlflow_service.get_model_metrics(model_name, version)
        return metrics
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Could not get metrics for model: {model_name}"
        )


@router.post(
    "/models/cache/clear",
    summary="Clear model cache",
    description="Clear all cached models to force reload"
)
async def clear_model_cache() -> Dict[str, str]:
    """Clear the model cache."""
    model_service = get_model_service()
    model_service.clear_cache()
    return {"message": "Model cache cleared"}


@router.get(
    "/models/cache/info",
    summary="Get cache info",
    description="Get information about cached models"
)
async def get_cache_info() -> Dict[str, Any]:
    """Get model cache information."""
    model_service = get_model_service()
    return model_service.get_cache_info()
