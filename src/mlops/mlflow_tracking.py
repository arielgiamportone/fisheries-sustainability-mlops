"""
MLFlow tracking integration for DL_Bayesian project.

Provides a wrapper around MLFlow for experiment tracking, model logging,
and model registry operations.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
import torch
import torch.nn as nn
import numpy as np


class MLFlowTracker:
    """
    Manages MLFlow experiment tracking and model registry.

    Provides methods for:
    - Starting and managing runs
    - Logging parameters, metrics, and artifacts
    - Logging PyTorch models
    - Model registry operations (register, stage transitions)

    Example:
        >>> tracker = MLFlowTracker(experiment_name="sustainability_models")
        >>> with tracker.start_run(run_name="bnn_training"):
        ...     tracker.log_params({"learning_rate": 0.001})
        ...     tracker.log_metrics({"accuracy": 0.95})
        ...     tracker.log_model(model, "model")
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "dl_bayesian_sustainability",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLFlow tracker.

        Args:
            tracking_uri: MLFlow tracking server URI
            experiment_name: Name of the experiment
            artifact_location: Optional custom artifact storage location
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.artifact_location = artifact_location

        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

        # Setup experiment
        self._setup_experiment()

    def _setup_experiment(self):
        """Create or retrieve the experiment."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=self.artifact_location
            )
        else:
            self.experiment_id = experiment.experiment_id
        mlflow.set_experiment(self.experiment_name)

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ):
        """
        Start a new MLFlow run.

        Args:
            run_name: Optional name for the run
            tags: Optional tags dictionary
            description: Optional run description

        Returns:
            MLFlow run context manager
        """
        run_tags = tags or {}
        if description:
            run_tags["mlflow.note.content"] = description

        return mlflow.start_run(
            run_name=run_name,
            tags=run_tags
        )

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to the current run.

        Handles conversion of complex types (lists, dicts) to strings.

        Args:
            params: Dictionary of parameters
        """
        for key, value in params.items():
            if isinstance(value, (list, dict)):
                mlflow.log_param(key, json.dumps(value))
            elif value is not None:
                mlflow.log_param(key, value)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """
        Log metrics to the current run.

        Args:
            metrics: Dictionary of metric name to value
            step: Optional step number (for time series metrics)
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not np.isnan(value):
                mlflow.log_metric(key, value, step=step)

    def log_training_history(self, history):
        """
        Log complete training history from a TrainingHistory object.

        Args:
            history: TrainingHistory object from training.py
        """
        # Log loss per epoch
        for epoch, train_loss in enumerate(history.train_loss):
            mlflow.log_metric("train_loss", train_loss, step=epoch)

        for epoch, val_loss in enumerate(history.val_loss):
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Log metrics per epoch
        if history.train_metrics:
            for epoch, metrics in enumerate(history.train_metrics):
                for key, value in metrics.items():
                    mlflow.log_metric(f"train_{key}", value, step=epoch)

        if history.val_metrics:
            for epoch, metrics in enumerate(history.val_metrics):
                for key, value in metrics.items():
                    mlflow.log_metric(f"val_{key}", value, step=epoch)

        # Log best results
        mlflow.log_metric("best_val_loss", history.best_val_loss)
        mlflow.log_metric("best_epoch", history.best_epoch)

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str = "model",
        input_example: Optional[np.ndarray] = None,
        registered_model_name: Optional[str] = None
    ) -> str:
        """
        Log a PyTorch model to MLFlow.

        Args:
            model: PyTorch model to log
            artifact_path: Path within artifacts to store model
            input_example: Optional example input for signature inference
            registered_model_name: If provided, register model with this name

        Returns:
            Model URI
        """
        # Infer signature if example provided
        signature = None
        if input_example is not None:
            model.eval()
            with torch.no_grad():
                example_tensor = torch.FloatTensor(input_example)
                output = model(example_tensor)
                if isinstance(output, tuple):
                    output = output[0]  # Handle BNN output (logits, kl)
                output_example = output.numpy()
            signature = infer_signature(input_example, output_example)

        # Log model
        mlflow.pytorch.log_model(
            model,
            artifact_path,
            signature=signature,
            input_example=input_example,
            registered_model_name=registered_model_name
        )

        run_id = mlflow.active_run().info.run_id
        return f"runs:/{run_id}/{artifact_path}"

    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None
    ):
        """
        Log a local file or directory as an artifact.

        Args:
            local_path: Path to local file/directory
            artifact_path: Optional destination path in artifacts
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_figure(self, figure, artifact_name: str):
        """
        Log a matplotlib figure as an artifact.

        Args:
            figure: Matplotlib figure object
            artifact_name: Name for the artifact file
        """
        mlflow.log_figure(figure, artifact_name)

    def log_dict(self, dictionary: Dict, artifact_name: str):
        """
        Log a dictionary as a JSON artifact.

        Args:
            dictionary: Dictionary to log
            artifact_name: Name for the JSON file
        """
        mlflow.log_dict(dictionary, artifact_name)

    def set_tag(self, key: str, value: str):
        """Set a tag on the current run."""
        mlflow.set_tag(key, value)

    def set_tags(self, tags: Dict[str, str]):
        """Set multiple tags on the current run."""
        mlflow.set_tags(tags)

    # ==========================================
    # Model Registry Operations
    # ==========================================

    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ):
        """
        Register a model in the Model Registry.

        Args:
            model_uri: URI of the model (e.g., "runs:/run_id/model")
            name: Name for the registered model
            tags: Optional tags for the model version
            description: Optional description

        Returns:
            ModelVersion object
        """
        result = mlflow.register_model(model_uri, name)

        # Add tags if provided
        if tags:
            for key, value in tags.items():
                self.client.set_model_version_tag(
                    name, result.version, key, value
                )

        # Add description if provided
        if description:
            self.client.update_model_version(
                name=name,
                version=result.version,
                description=description
            )

        return result

    def transition_model_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ):
        """
        Transition a model version to a new stage.

        Args:
            name: Registered model name
            version: Model version
            stage: Target stage ("Staging", "Production", "Archived")
            archive_existing: Whether to archive existing models in target stage
        """
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )

    def get_latest_version(
        self,
        name: str,
        stages: Optional[List[str]] = None
    ) -> Optional[str]:
        """
        Get the latest version of a registered model.

        Args:
            name: Registered model name
            stages: Optional list of stages to filter by

        Returns:
            Latest version number or None
        """
        try:
            versions = self.client.get_latest_versions(name, stages=stages)
            if versions:
                return versions[0].version
        except Exception:
            pass
        return None

    def load_model(self, model_uri: str) -> nn.Module:
        """
        Load a PyTorch model from MLFlow.

        Args:
            model_uri: URI of the model

        Returns:
            Loaded PyTorch model
        """
        return mlflow.pytorch.load_model(model_uri)

    def get_production_model(self, model_name: str) -> Optional[nn.Module]:
        """
        Load the production version of a registered model.

        Args:
            model_name: Name of the registered model

        Returns:
            Model in Production stage, or None if not found
        """
        try:
            model_uri = f"models:/{model_name}/Production"
            return self.load_model(model_uri)
        except Exception:
            return None

    def get_staging_model(self, model_name: str) -> Optional[nn.Module]:
        """
        Load the staging version of a registered model.

        Args:
            model_name: Name of the registered model

        Returns:
            Model in Staging stage, or None if not found
        """
        try:
            model_uri = f"models:/{model_name}/Staging"
            return self.load_model(model_uri)
        except Exception:
            return None

    # ==========================================
    # Query Operations
    # ==========================================

    def list_experiments(self) -> List[Dict]:
        """
        List all experiments.

        Returns:
            List of experiment info dictionaries
        """
        experiments = self.client.search_experiments()
        return [
            {
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "creation_time": exp.creation_time,
                "last_update_time": exp.last_update_time
            }
            for exp in experiments
        ]

    def list_runs(
        self,
        experiment_id: Optional[str] = None,
        filter_string: str = "",
        max_results: int = 100
    ) -> List[Dict]:
        """
        List runs for an experiment.

        Args:
            experiment_id: Experiment ID (uses current if not provided)
            filter_string: MLFlow filter string
            max_results: Maximum number of results

        Returns:
            List of run info dictionaries
        """
        exp_id = experiment_id or self.experiment_id
        runs = self.client.search_runs(
            experiment_ids=[exp_id],
            filter_string=filter_string,
            max_results=max_results
        )

        return [
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            }
            for run in runs
        ]

    def list_registered_models(self) -> List[Dict]:
        """
        List all registered models.

        Returns:
            List of registered model info dictionaries
        """
        models = self.client.search_registered_models()
        return [
            {
                "name": model.name,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "description": model.description,
                "latest_versions": [
                    {
                        "version": v.version,
                        "current_stage": v.current_stage,
                        "creation_timestamp": v.creation_timestamp,
                        "source": v.source,
                        "run_id": v.run_id,
                        "status": v.status
                    }
                    for v in model.latest_versions
                ]
            }
            for model in models
        ]

    def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        """
        Get metrics for a specific run.

        Args:
            run_id: Run ID

        Returns:
            Dictionary of metrics
        """
        run = self.client.get_run(run_id)
        return run.data.metrics

    def get_run_params(self, run_id: str) -> Dict[str, str]:
        """
        Get parameters for a specific run.

        Args:
            run_id: Run ID

        Returns:
            Dictionary of parameters
        """
        run = self.client.get_run(run_id)
        return run.data.params

    @staticmethod
    def get_active_run_id() -> Optional[str]:
        """Get the ID of the currently active run."""
        run = mlflow.active_run()
        return run.info.run_id if run else None

    @staticmethod
    def end_run():
        """End the current run."""
        mlflow.end_run()


def create_tracker(
    tracking_uri: str = None,
    experiment_name: str = "dl_bayesian_sustainability"
) -> MLFlowTracker:
    """
    Factory function to create an MLFlow tracker.

    Uses environment variables if tracking_uri not provided:
    - MLFLOW_TRACKING_URI: Tracking server URI (default: http://localhost:5000)

    Args:
        tracking_uri: Optional tracking URI override
        experiment_name: Name of the experiment

    Returns:
        Configured MLFlowTracker instance
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    return MLFlowTracker(tracking_uri=uri, experiment_name=experiment_name)
