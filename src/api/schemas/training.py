"""Schemas for training endpoints."""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Available model types."""
    MLP = "mlp"
    BNN = "bnn"
    CAUSAL_VAE = "causal_vae"


class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainRequest(BaseModel):
    """Request schema for training a model."""

    model_type: ModelType = Field(
        default=ModelType.BNN,
        description="Type of model to train"
    )
    hidden_dims: List[int] = Field(
        default=[64, 32],
        description="Hidden layer dimensions"
    )
    learning_rate: float = Field(
        default=0.001,
        ge=1e-6,
        le=1.0,
        description="Learning rate"
    )
    epochs: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Number of training epochs"
    )
    batch_size: int = Field(
        default=32,
        ge=8,
        le=512,
        description="Batch size"
    )
    dropout_rate: float = Field(
        default=0.2,
        ge=0.0,
        le=0.9,
        description="Dropout rate"
    )
    early_stopping_patience: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Early stopping patience"
    )

    # Optuna integration
    run_optuna: bool = Field(
        default=False,
        description="Run hyperparameter tuning with Optuna"
    )
    optuna_trials: int = Field(
        default=50,
        ge=5,
        le=500,
        description="Number of Optuna trials (if run_optuna=True)"
    )

    # MLFlow settings
    experiment_name: str = Field(
        default="sustainability_models",
        description="MLFlow experiment name"
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Optional MLFlow run name"
    )
    register_model: bool = Field(
        default=False,
        description="Register model in MLFlow Model Registry"
    )

    # Data settings
    n_samples: int = Field(
        default=1000,
        ge=100,
        le=100000,
        description="Number of synthetic samples to generate"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "bnn",
                "hidden_dims": [64, 32],
                "learning_rate": 0.001,
                "epochs": 100,
                "batch_size": 32,
                "dropout_rate": 0.2,
                "early_stopping_patience": 10,
                "run_optuna": False,
                "optuna_trials": 50,
                "experiment_name": "sustainability_models",
                "run_name": "bnn_training_v1",
                "register_model": True,
                "n_samples": 1000
            }
        }


class TrainResponse(BaseModel):
    """Response schema for training request."""

    status: TrainingStatus = Field(
        ...,
        description="Current status of training job"
    )
    job_id: str = Field(
        ...,
        description="Unique identifier for the training job"
    )
    run_id: Optional[str] = Field(
        default=None,
        description="MLFlow run ID (available when training starts)"
    )
    experiment_id: Optional[str] = Field(
        default=None,
        description="MLFlow experiment ID"
    )
    message: str = Field(
        ...,
        description="Status message"
    )
    metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Final metrics (available when completed)"
    )
    model_uri: Optional[str] = Field(
        default=None,
        description="MLFlow model URI (available when completed)"
    )
    best_params: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Best hyperparameters (if Optuna was used)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "completed",
                "job_id": "train_abc123",
                "run_id": "1234567890abcdef",
                "experiment_id": "1",
                "message": "Training completed successfully",
                "metrics": {
                    "accuracy": 0.85,
                    "f1": 0.83,
                    "auc_roc": 0.91
                },
                "model_uri": "runs:/1234567890abcdef/model",
                "best_params": None
            }
        }


class TrainingJobStatus(BaseModel):
    """Status of a training job."""

    job_id: str
    status: TrainingStatus
    progress: float = Field(
        ge=0,
        le=100,
        description="Progress percentage"
    )
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    current_metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class ExperimentSummary(BaseModel):
    """Summary of an MLFlow experiment."""

    experiment_id: str
    name: str
    artifact_location: Optional[str]
    lifecycle_stage: str
    creation_time: Optional[int]
    last_update_time: Optional[int]
    runs_count: int = 0


class RunSummary(BaseModel):
    """Summary of an MLFlow run."""

    run_id: str
    run_name: Optional[str]
    status: str
    start_time: Optional[int]
    end_time: Optional[int]
    metrics: Dict[str, float]
    params: Dict[str, str]
    model_uri: Optional[str] = None


class RegisteredModel(BaseModel):
    """Information about a registered model."""

    name: str
    creation_timestamp: Optional[int]
    last_updated_timestamp: Optional[int]
    description: Optional[str]
    latest_versions: List[Dict[str, Any]]


class ModelStageTransition(BaseModel):
    """Request to transition model stage."""

    stage: str = Field(
        ...,
        pattern="^(Staging|Production|Archived)$",
        description="Target stage"
    )
    archive_existing: bool = Field(
        default=True,
        description="Archive existing models in target stage"
    )
