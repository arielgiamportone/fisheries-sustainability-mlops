"""Training endpoints."""

import uuid
import asyncio
from typing import Dict, Any
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, status

from src.api.schemas.training import (
    TrainRequest,
    TrainResponse,
    TrainingStatus,
    TrainingJobStatus
)

router = APIRouter()

# In-memory job tracking (use Redis in production)
_training_jobs: Dict[str, Dict[str, Any]] = {}


async def run_training_job(job_id: str, request: TrainRequest):
    """
    Background task to run model training.

    Args:
        job_id: Unique job identifier
        request: Training request parameters
    """
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    import numpy as np
    import torch
    import mlflow

    from data.loaders import generate_synthetic_fisheries_data
    from src.deep_learning.models import create_model
    from src.deep_learning.training import (
        Trainer,
        TrainingConfig,
        prepare_data_loaders,
        prepare_sustainability_data,
        evaluate_model
    )
    from src.mlops.mlflow_tracking import MLFlowTracker
    from src.mlops.optuna_tuning import OptunaHyperparameterTuner, TuningConfig

    try:
        # Update job status
        _training_jobs[job_id]["status"] = TrainingStatus.RUNNING
        _training_jobs[job_id]["started_at"] = datetime.utcnow().isoformat()

        # Set seed
        np.random.seed(42)
        torch.manual_seed(42)

        # Generate data
        df = generate_synthetic_fisheries_data(n_samples=request.n_samples)
        X, y, feature_names = prepare_sustainability_data(df, target='Sustainable')

        # Initialize tracker
        tracker = MLFlowTracker(experiment_name=request.experiment_name)

        if request.run_optuna:
            # Run hyperparameter tuning
            config = TuningConfig(
                n_trials=request.optuna_trials,
                epochs_per_trial=min(50, request.epochs),
                early_stopping_patience=5
            )

            tuner = OptunaHyperparameterTuner(
                X, y,
                model_type=request.model_type.value,
                config=config
            )

            results = tuner.run_optimization(show_progress=False)
            final = tuner.train_with_best_params(epochs=request.epochs)

            model = final["model"]
            history = final["history"]
            best_params = results["best_params"]

        else:
            # Standard training
            model = create_model(
                request.model_type.value,
                X.shape[1],
                request.hidden_dims,
                dropout_rate=request.dropout_rate
            )

            training_config = TrainingConfig(
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                early_stopping_patience=request.early_stopping_patience,
                device='cpu'
            )

            train_loader, val_loader, _ = prepare_data_loaders(
                X, y,
                batch_size=request.batch_size,
                val_split=0.2
            )

            trainer = Trainer(model, training_config)
            history = trainer.fit(train_loader, val_loader)
            best_params = None

        # Evaluate
        metrics = evaluate_model(model, X, y, device='cpu')

        # Log to MLFlow
        run_name = request.run_name or f"{request.model_type.value}_{job_id[:8]}"

        with tracker.start_run(run_name=run_name):
            # Log params
            tracker.log_params({
                "model_type": request.model_type.value,
                "hidden_dims": request.hidden_dims,
                "learning_rate": request.learning_rate,
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "n_samples": request.n_samples,
                "optuna_used": request.run_optuna
            })

            if best_params:
                tracker.log_params({"best_" + k: v for k, v in best_params.items()
                                   if not k.startswith("hidden_dim_")})

            # Log metrics
            tracker.log_training_history(history)
            tracker.log_metrics({
                "final_accuracy": metrics['accuracy'],
                "final_precision": metrics['precision'],
                "final_recall": metrics['recall'],
                "final_f1": metrics['f1'],
                "final_auc_roc": metrics['auc_roc']
            })

            # Log model
            input_example = X[:5].astype(np.float32)
            model_uri = tracker.log_model(
                model,
                artifact_path="model",
                input_example=input_example
            )

            run_id = tracker.get_active_run_id()

            # Register if requested
            if request.register_model:
                model_name = f"sustainability_{request.model_type.value}"
                tracker.register_model(
                    model_uri=model_uri,
                    name=model_name,
                    description=f"Trained via API - Job {job_id}"
                )

        # Update job with results
        _training_jobs[job_id].update({
            "status": TrainingStatus.COMPLETED,
            "completed_at": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "model_uri": model_uri,
            "metrics": {
                "accuracy": metrics['accuracy'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "f1": metrics['f1'],
                "auc_roc": metrics['auc_roc']
            },
            "best_params": best_params
        })

    except Exception as e:
        _training_jobs[job_id].update({
            "status": TrainingStatus.FAILED,
            "error_message": str(e),
            "completed_at": datetime.utcnow().isoformat()
        })


@router.post(
    "/train",
    response_model=TrainResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Start model training",
    description="Start a background training job"
)
async def start_training(
    request: TrainRequest,
    background_tasks: BackgroundTasks
) -> TrainResponse:
    """
    Start a model training job.

    Training runs in the background. Use the job_id to check status.
    """
    # Generate job ID
    job_id = f"train_{uuid.uuid4().hex[:12]}"

    # Initialize job tracking
    _training_jobs[job_id] = {
        "status": TrainingStatus.PENDING,
        "created_at": datetime.utcnow().isoformat(),
        "request": request.model_dump()
    }

    # Start background task
    background_tasks.add_task(run_training_job, job_id, request)

    return TrainResponse(
        status=TrainingStatus.PENDING,
        job_id=job_id,
        message="Training job started. Use /train/status/{job_id} to check progress."
    )


@router.get(
    "/train/status/{job_id}",
    response_model=TrainingJobStatus,
    summary="Get training job status",
    description="Check the status of a training job"
)
async def get_training_status(job_id: str) -> TrainingJobStatus:
    """Get the status of a training job."""
    if job_id not in _training_jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job not found: {job_id}"
        )

    job = _training_jobs[job_id]

    return TrainingJobStatus(
        job_id=job_id,
        status=job["status"],
        progress=100.0 if job["status"] == TrainingStatus.COMPLETED else 0.0,
        current_metrics=job.get("metrics"),
        error_message=job.get("error_message"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at")
    )


@router.get(
    "/train/jobs",
    summary="List training jobs",
    description="List all training jobs"
)
async def list_training_jobs() -> Dict[str, Any]:
    """List all training jobs."""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "created_at": job.get("created_at"),
                "completed_at": job.get("completed_at")
            }
            for job_id, job in _training_jobs.items()
        ],
        "total": len(_training_jobs)
    }
