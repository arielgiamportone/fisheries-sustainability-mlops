"""Experiment management endpoints."""

from typing import List, Dict, Any

from fastapi import APIRouter, HTTPException, status, Query

from src.api.schemas.training import ExperimentSummary, RunSummary
from src.api.services.mlflow_service import get_mlflow_service

router = APIRouter()


@router.get(
    "/experiments",
    response_model=List[ExperimentSummary],
    summary="List experiments",
    description="List all MLFlow experiments"
)
async def list_experiments() -> List[ExperimentSummary]:
    """List all MLFlow experiments."""
    mlflow_service = get_mlflow_service()

    try:
        experiments = mlflow_service.list_experiments()
        return [
            ExperimentSummary(
                experiment_id=exp["experiment_id"],
                name=exp["name"],
                artifact_location=exp.get("artifact_location"),
                lifecycle_stage=exp.get("lifecycle_stage", "active"),
                creation_time=exp.get("creation_time"),
                last_update_time=exp.get("last_update_time"),
                runs_count=exp.get("runs_count", 0)
            )
            for exp in experiments
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLFlow service unavailable: {str(e)}"
        )


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentSummary,
    summary="Get experiment",
    description="Get details of a specific experiment"
)
async def get_experiment(experiment_id: str) -> ExperimentSummary:
    """Get a specific experiment by ID."""
    mlflow_service = get_mlflow_service()

    try:
        exp = mlflow_service.get_experiment(experiment_id)
        if exp is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Experiment not found: {experiment_id}"
            )

        return ExperimentSummary(
            experiment_id=exp["experiment_id"],
            name=exp["name"],
            artifact_location=exp.get("artifact_location"),
            lifecycle_stage=exp.get("lifecycle_stage", "active"),
            creation_time=exp.get("creation_time"),
            last_update_time=exp.get("last_update_time"),
            runs_count=exp.get("runs_count", 0)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLFlow service unavailable: {str(e)}"
        )


@router.get(
    "/experiments/{experiment_id}/runs",
    response_model=List[RunSummary],
    summary="List runs",
    description="List runs for an experiment"
)
async def list_runs(
    experiment_id: str,
    max_results: int = Query(default=100, ge=1, le=1000),
    filter_string: str = Query(default="")
) -> List[RunSummary]:
    """List runs for an experiment."""
    mlflow_service = get_mlflow_service()

    try:
        runs = mlflow_service.list_runs(
            experiment_id=experiment_id,
            filter_string=filter_string,
            max_results=max_results
        )

        return [
            RunSummary(
                run_id=run["run_id"],
                run_name=run.get("run_name"),
                status=run.get("status", "UNKNOWN"),
                start_time=run.get("start_time"),
                end_time=run.get("end_time"),
                metrics=run.get("metrics", {}),
                params=run.get("params", {}),
                model_uri=f"runs:/{run['run_id']}/model" if run.get("run_id") else None
            )
            for run in runs
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"MLFlow service unavailable: {str(e)}"
        )


@router.get(
    "/runs/{run_id}",
    summary="Get run details",
    description="Get detailed information about a specific run"
)
async def get_run(run_id: str) -> Dict[str, Any]:
    """Get details of a specific run."""
    mlflow_service = get_mlflow_service()

    try:
        run = mlflow_service.get_run(run_id)
        return run
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}"
        )


@router.get(
    "/runs/{run_id}/metrics",
    summary="Get run metrics",
    description="Get metrics for a specific run"
)
async def get_run_metrics(run_id: str) -> Dict[str, float]:
    """Get metrics for a specific run."""
    mlflow_service = get_mlflow_service()

    try:
        run = mlflow_service.get_run(run_id)
        return run.get("metrics", {})
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run not found: {run_id}"
        )
