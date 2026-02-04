"""
Optuna hyperparameter tuning integration for DL_Bayesian project.

Provides automated hyperparameter optimization with MLFlow logging.
"""

import os
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass

import numpy as np
import optuna
from optuna.integration.mlflow import MLflowCallback
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn

from src.deep_learning.models import create_model, ModelConfig
from src.deep_learning.training import (
    Trainer,
    TrainingConfig,
    prepare_data_loaders
)


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    n_trials: int = 50
    timeout: Optional[int] = None  # seconds
    study_name: str = "sustainability_tuning"
    direction: str = "minimize"  # minimize val_loss
    metric_name: str = "val_loss"

    # Search space bounds
    n_layers_range: Tuple[int, int] = (1, 4)
    hidden_dim_range: Tuple[int, int] = (16, 128)
    dropout_range: Tuple[float, float] = (0.1, 0.5)
    lr_range: Tuple[float, float] = (1e-5, 1e-2)
    batch_sizes: List[int] = None
    weight_decay_range: Tuple[float, float] = (1e-6, 1e-3)

    # Training settings for each trial
    epochs_per_trial: int = 50
    early_stopping_patience: int = 5

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [16, 32, 64, 128]


class OptunaHyperparameterTuner:
    """
    Hyperparameter tuning using Optuna with MLFlow integration.

    Optimizes model architecture and training hyperparameters,
    logging all trials to MLFlow for analysis.

    Example:
        >>> tuner = OptunaHyperparameterTuner(X, y, model_type="bnn")
        >>> results = tuner.run_optimization()
        >>> best_model = tuner.train_with_best_params()
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "bnn",
        config: Optional[TuningConfig] = None,
        mlflow_tracking_uri: str = "http://localhost:5000"
    ):
        """
        Initialize the hyperparameter tuner.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
            model_type: Type of model ("mlp", "bnn", "causal_vae")
            config: Tuning configuration
            mlflow_tracking_uri: MLFlow tracking server URI
        """
        self.X = X
        self.y = y
        self.model_type = model_type
        self.config = config or TuningConfig()
        self.mlflow_tracking_uri = mlflow_tracking_uri

        self.study: Optional[optuna.Study] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_value: Optional[float] = None

    def _suggest_hidden_dims(self, trial: optuna.Trial) -> List[int]:
        """Suggest hidden layer dimensions."""
        n_layers = trial.suggest_int(
            "n_layers",
            self.config.n_layers_range[0],
            self.config.n_layers_range[1]
        )

        hidden_dims = []
        prev_dim = self.config.hidden_dim_range[1]  # Start from max

        for i in range(n_layers):
            # Each layer can be smaller or equal to previous
            max_dim = min(prev_dim, self.config.hidden_dim_range[1])
            min_dim = self.config.hidden_dim_range[0]

            dim = trial.suggest_int(
                f"hidden_dim_{i}",
                min_dim,
                max_dim,
                step=16
            )
            hidden_dims.append(dim)
            prev_dim = dim

        return hidden_dims

    def _create_objective(self) -> Callable:
        """Create the objective function for Optuna."""

        def objective(trial: optuna.Trial) -> float:
            """Objective function to minimize validation loss."""
            # Suggest hyperparameters
            hidden_dims = self._suggest_hidden_dims(trial)

            params = {
                "hidden_dims": hidden_dims,
                "dropout_rate": trial.suggest_float(
                    "dropout_rate",
                    self.config.dropout_range[0],
                    self.config.dropout_range[1]
                ),
                "learning_rate": trial.suggest_float(
                    "learning_rate",
                    self.config.lr_range[0],
                    self.config.lr_range[1],
                    log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size",
                    self.config.batch_sizes
                ),
                "weight_decay": trial.suggest_float(
                    "weight_decay",
                    self.config.weight_decay_range[0],
                    self.config.weight_decay_range[1],
                    log=True
                )
            }

            # Create model
            model = create_model(
                self.model_type,
                self.X.shape[1],
                params["hidden_dims"],
                dropout_rate=params["dropout_rate"]
            )

            # Training configuration
            training_config = TrainingConfig(
                epochs=self.config.epochs_per_trial,
                batch_size=params["batch_size"],
                learning_rate=params["learning_rate"],
                weight_decay=params["weight_decay"],
                early_stopping_patience=self.config.early_stopping_patience,
                device='cpu',
                verbose=False
            )

            # Prepare data
            train_loader, val_loader, _ = prepare_data_loaders(
                self.X, self.y,
                batch_size=params["batch_size"],
                val_split=0.2
            )

            # Train
            trainer = Trainer(model, training_config)

            # Use pruning callback
            for epoch in range(training_config.epochs):
                # Train one epoch
                train_loss, _ = trainer._train_epoch(train_loader)

                if val_loader is not None:
                    val_loss, _ = trainer._validate_epoch(val_loader)

                    # Report to Optuna for pruning
                    trial.report(val_loss, epoch)

                    # Check if trial should be pruned
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    # Check early stopping
                    if trainer.early_stopping(val_loss, model):
                        break

                    trainer.history.train_loss.append(train_loss)
                    trainer.history.val_loss.append(val_loss)

            # Return best validation loss
            return trainer.history.best_val_loss

        return objective

    def run_optimization(
        self,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            show_progress: Whether to show progress bar

        Returns:
            Dictionary with optimization results
        """
        # Create study
        self.study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_warmup_steps=10)
        )

        # MLFlow callback for logging
        mlflow_callback = MLflowCallback(
            tracking_uri=self.mlflow_tracking_uri,
            metric_name=self.config.metric_name,
            create_experiment=True,
            mlflow_kwargs={
                "experiment_name": f"optuna_{self.config.study_name}"
            }
        )

        # Run optimization
        self.study.optimize(
            self._create_objective(),
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            callbacks=[mlflow_callback],
            show_progress_bar=show_progress,
            gc_after_trial=True
        )

        # Store results
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        # Reconstruct hidden_dims from best params
        n_layers = self.best_params.get("n_layers", 2)
        self.best_params["hidden_dims"] = [
            self.best_params[f"hidden_dim_{i}"]
            for i in range(n_layers)
        ]

        return {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.study.trials),
            "n_completed": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "study": self.study
        }

    def train_with_best_params(
        self,
        epochs: int = 100,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train a model using the best hyperparameters found.

        Args:
            epochs: Number of epochs for final training
            verbose: Whether to print progress

        Returns:
            Dictionary with trained model and history
        """
        if self.best_params is None:
            raise ValueError("Run run_optimization() first")

        # Create model with best params
        model = create_model(
            self.model_type,
            self.X.shape[1],
            self.best_params["hidden_dims"],
            dropout_rate=self.best_params["dropout_rate"]
        )

        # Training config with best params
        config = TrainingConfig(
            epochs=epochs,
            batch_size=self.best_params["batch_size"],
            learning_rate=self.best_params["learning_rate"],
            weight_decay=self.best_params["weight_decay"],
            early_stopping_patience=15,
            device='cpu',
            verbose=verbose
        )

        # Prepare data
        train_loader, val_loader, _ = prepare_data_loaders(
            self.X, self.y,
            batch_size=self.best_params["batch_size"],
            val_split=0.2
        )

        # Train
        trainer = Trainer(model, config)
        history = trainer.fit(train_loader, val_loader)

        return {
            "model": model,
            "history": history,
            "params": self.best_params,
            "trainer": trainer
        }

    def get_optimization_history(self) -> List[Dict]:
        """
        Get the optimization history as a list of trial results.

        Returns:
            List of dictionaries with trial information
        """
        if self.study is None:
            return []

        return [
            {
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "datetime_start": trial.datetime_start,
                "datetime_complete": trial.datetime_complete
            }
            for trial in self.study.trials
        ]

    def get_param_importances(self) -> Dict[str, float]:
        """
        Calculate parameter importances using fANOVA.

        Returns:
            Dictionary mapping parameter names to importance scores
        """
        if self.study is None:
            return {}

        try:
            importances = optuna.importance.get_param_importances(self.study)
            return dict(importances)
        except Exception:
            return {}

    def plot_optimization_history(self):
        """
        Create optimization history plot.

        Returns:
            Plotly figure
        """
        if self.study is None:
            return None

        return optuna.visualization.plot_optimization_history(self.study)

    def plot_param_importances(self):
        """
        Create parameter importance plot.

        Returns:
            Plotly figure
        """
        if self.study is None:
            return None

        try:
            return optuna.visualization.plot_param_importances(self.study)
        except Exception:
            return None

    def plot_parallel_coordinate(self):
        """
        Create parallel coordinate plot.

        Returns:
            Plotly figure
        """
        if self.study is None:
            return None

        return optuna.visualization.plot_parallel_coordinate(self.study)


def quick_tune(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "bnn",
    n_trials: int = 20,
    mlflow_tracking_uri: str = "http://localhost:5000"
) -> Dict[str, Any]:
    """
    Quick hyperparameter tuning with sensible defaults.

    Args:
        X: Feature matrix
        y: Target vector
        model_type: Model type ("mlp", "bnn")
        n_trials: Number of trials
        mlflow_tracking_uri: MLFlow URI

    Returns:
        Best parameters and trained model
    """
    config = TuningConfig(
        n_trials=n_trials,
        epochs_per_trial=30,
        early_stopping_patience=5
    )

    tuner = OptunaHyperparameterTuner(
        X, y,
        model_type=model_type,
        config=config,
        mlflow_tracking_uri=mlflow_tracking_uri
    )

    # Run optimization
    results = tuner.run_optimization()

    # Train final model
    final = tuner.train_with_best_params(epochs=100)

    return {
        **results,
        "model": final["model"],
        "history": final["history"]
    }
