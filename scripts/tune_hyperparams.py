#!/usr/bin/env python
"""
Hyperparameter tuning script using Optuna with MLFlow logging.

Usage:
    python scripts/tune_hyperparams.py --model-type bnn --n-trials 50
    python scripts/tune_hyperparams.py --model-type mlp --timeout 3600
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import mlflow

from data.loaders import generate_synthetic_fisheries_data
from src.deep_learning.training import prepare_sustainability_data, evaluate_model
from src.mlops.optuna_tuning import OptunaHyperparameterTuner, TuningConfig
from src.mlops.mlflow_tracking import MLFlowTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with Optuna"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="bnn",
        choices=["mlp", "bnn"],
        help="Type of model to tune"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Timeout in seconds (optional)"
    )
    parser.add_argument(
        "--epochs-per-trial",
        type=int,
        default=50,
        help="Epochs per trial"
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=100,
        help="Epochs for final model training"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name"
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLFlow tracking URI"
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register final model in MLFlow"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("DL_Bayesian - Hyperparameter Tuning with Optuna")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"Trials: {args.n_trials}")
    print(f"Timeout: {args.timeout}s" if args.timeout else "Timeout: None")
    print(f"Epochs per trial: {args.epochs_per_trial}")
    print(f"MLFlow URI: {args.tracking_uri}")
    print("=" * 60)

    # Generate data
    print("\nGenerating synthetic data...")
    df = generate_synthetic_fisheries_data(n_samples=args.n_samples)
    X, y, feature_names = prepare_sustainability_data(df, target='Sustainable')
    print(f"Data shape: X={X.shape}, y={y.shape}")

    # Configure tuning
    study_name = args.study_name or f"{args.model_type}_tuning"
    config = TuningConfig(
        n_trials=args.n_trials,
        timeout=args.timeout,
        study_name=study_name,
        epochs_per_trial=args.epochs_per_trial,
        early_stopping_patience=5
    )

    # Create tuner
    tuner = OptunaHyperparameterTuner(
        X, y,
        model_type=args.model_type,
        config=config,
        mlflow_tracking_uri=args.tracking_uri
    )

    # Run optimization
    print("\nStarting hyperparameter optimization...")
    print("This may take a while...\n")

    results = tuner.run_optimization(show_progress=True)

    # Print results
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Total trials: {results['n_trials']}")
    print(f"Completed trials: {results['n_completed']}")
    print(f"Pruned trials: {results['n_pruned']}")
    print(f"Best validation loss: {results['best_value']:.4f}")
    print("\nBest hyperparameters:")
    for key, value in results['best_params'].items():
        if not key.startswith("hidden_dim_"):
            print(f"  {key}: {value}")
    print(f"  hidden_dims: {results['best_params']['hidden_dims']}")

    # Parameter importances
    importances = tuner.get_param_importances()
    if importances:
        print("\nParameter Importances:")
        for param, importance in sorted(importances.items(), key=lambda x: -x[1]):
            print(f"  {param}: {importance:.4f}")

    # Train final model with best params
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)

    final_results = tuner.train_with_best_params(
        epochs=args.final_epochs,
        verbose=True
    )

    model = final_results["model"]
    history = final_results["history"]

    # Evaluate final model
    metrics = evaluate_model(model, X, y, device='cpu')

    print("\nFinal Model Metrics:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    print(f"  AUC-ROC: {metrics['auc_roc']:.4f}")

    # Register model if requested
    if args.register:
        print("\nRegistering model in MLFlow...")
        tracker = MLFlowTracker(
            tracking_uri=args.tracking_uri,
            experiment_name=f"final_{study_name}"
        )

        with tracker.start_run(run_name=f"best_{args.model_type}"):
            # Log best params
            tracker.log_params(results['best_params'])

            # Log final metrics
            tracker.log_metrics({
                "accuracy": metrics['accuracy'],
                "precision": metrics['precision'],
                "recall": metrics['recall'],
                "f1": metrics['f1'],
                "auc_roc": metrics['auc_roc'],
                "best_val_loss": results['best_value']
            })

            # Log training history
            tracker.log_training_history(history)

            # Log model
            input_example = X[:5].astype(np.float32)
            model_uri = tracker.log_model(
                model,
                artifact_path="model",
                input_example=input_example
            )

            # Register
            model_name = f"sustainability_{args.model_type}_tuned"
            tracker.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={
                    "model_type": args.model_type,
                    "tuned": "true",
                    "n_trials": str(args.n_trials)
                },
                description=f"Best {args.model_type.upper()} model from Optuna tuning ({args.n_trials} trials)"
            )

            print(f"Model registered as: {model_name}")

    print("\n" + "=" * 60)
    print("TUNING COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"\nView results at: {args.tracking_uri}")


if __name__ == "__main__":
    main()
