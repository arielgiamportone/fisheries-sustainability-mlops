#!/usr/bin/env python
"""
Training script with MLFlow integration.

This script demonstrates how to train models with full MLFlow tracking,
including parameter logging, metric tracking, and model registration.

Usage:
    python scripts/train_with_mlflow.py --model-type bnn --epochs 100
    python scripts/train_with_mlflow.py --model-type mlp --register
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
from src.deep_learning.models import create_model, ModelConfig
from src.deep_learning.training import (
    Trainer,
    TrainingConfig,
    prepare_data_loaders,
    prepare_sustainability_data,
    evaluate_model
)
from src.mlops.mlflow_tracking import MLFlowTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train sustainability model with MLFlow tracking"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="bnn",
        choices=["mlp", "bnn", "causal_vae"],
        help="Type of model to train"
    )
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="64,32",
        help="Hidden layer dimensions (comma-separated)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="sustainability_models",
        help="MLFlow experiment name"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run name"
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register model in MLFlow Model Registry"
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLFlow tracking URI"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def train_with_mlflow(args):
    """
    Main training function with MLFlow integration.

    Args:
        args: Parsed command line arguments

    Returns:
        Training history and final metrics
    """
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Parse hidden dimensions
    hidden_dims = [int(x) for x in args.hidden_dims.split(",")]

    # Initialize MLFlow tracker
    print(f"Connecting to MLFlow at {args.tracking_uri}...")
    tracker = MLFlowTracker(
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name
    )

    # Generate data
    print(f"Generating {args.n_samples} synthetic samples...")
    df = generate_synthetic_fisheries_data(n_samples=args.n_samples)

    # Prepare data for deep learning
    X, y, feature_names = prepare_sustainability_data(df, target='Sustainable')
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {feature_names}")

    # Model configuration
    model_config = ModelConfig(
        input_dim=X.shape[1],
        hidden_dims=hidden_dims,
        output_dim=1,
        dropout_rate=args.dropout,
        activation='relu',
        use_batch_norm=True
    )

    # Training configuration
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=1e-5,
        early_stopping_patience=10,
        val_split=0.2,
        device='cpu',  # CPU only
        verbose=True
    )

    # Create model
    print(f"Creating {args.model_type.upper()} model...")
    model = create_model(
        args.model_type,
        model_config.input_dim,
        hidden_dims,
        dropout_rate=args.dropout
    )

    # Prepare data loaders
    train_loader, val_loader, _ = prepare_data_loaders(
        X, y,
        batch_size=training_config.batch_size,
        val_split=training_config.val_split,
        random_state=args.seed
    )

    # Generate run name if not provided
    run_name = args.run_name or f"{args.model_type}_{args.epochs}ep"

    print(f"\nStarting MLFlow run: {run_name}")
    print("=" * 50)

    # Start MLFlow run
    with tracker.start_run(run_name=run_name):
        # Log all parameters
        tracker.log_params({
            "model_type": args.model_type,
            "input_dim": model_config.input_dim,
            "hidden_dims": hidden_dims,
            "output_dim": model_config.output_dim,
            "dropout_rate": args.dropout,
            "activation": model_config.activation,
            "use_batch_norm": model_config.use_batch_norm,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "early_stopping_patience": training_config.early_stopping_patience,
            "val_split": training_config.val_split,
            "n_samples": args.n_samples,
            "n_features": len(feature_names),
            "seed": args.seed
        })

        # Set tags
        tracker.set_tags({
            "model_type": args.model_type,
            "stage": "training",
            "data_source": "synthetic"
        })

        # Log feature names
        tracker.log_dict({"features": feature_names}, "features.json")

        # Train model
        print("\nTraining model...")
        trainer = Trainer(model, training_config)
        history = trainer.fit(train_loader, val_loader)

        # Log training history
        tracker.log_training_history(history)

        # Evaluate on validation set
        print("\nEvaluating model...")
        metrics = evaluate_model(model, X, y, device='cpu')

        # Log final metrics
        final_metrics = {
            "final_accuracy": metrics['accuracy'],
            "final_precision": metrics['precision'],
            "final_recall": metrics['recall'],
            "final_f1": metrics['f1'],
            "final_auc_roc": metrics['auc_roc'],
            "total_epochs": len(history.train_loss),
            "best_epoch": history.best_epoch,
            "best_val_loss": history.best_val_loss
        }
        tracker.log_metrics(final_metrics)

        # Log confusion matrix as artifact
        cm = metrics['confusion_matrix']
        tracker.log_dict(
            {"confusion_matrix": cm.tolist()},
            "confusion_matrix.json"
        )

        # Log model
        print("\nLogging model to MLFlow...")
        input_example = X[:5].astype(np.float32)
        model_uri = tracker.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        # Register model if requested
        if args.register:
            print("\nRegistering model in Model Registry...")
            model_name = f"sustainability_{args.model_type}"
            tracker.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={"model_type": args.model_type},
                description=f"Sustainability prediction model ({args.model_type.upper()})"
            )
            print(f"Model registered as: {model_name}")

        # Print results
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED")
        print("=" * 50)
        print(f"Run ID: {tracker.get_active_run_id()}")
        print(f"Model URI: {model_uri}")
        print(f"\nFinal Metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

        return history, metrics


def main():
    """Main entry point."""
    args = parse_args()

    print("DL_Bayesian - Training with MLFlow")
    print("=" * 50)
    print(f"Model Type: {args.model_type}")
    print(f"Hidden Dims: {args.hidden_dims}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Register Model: {args.register}")
    print("=" * 50)

    try:
        history, metrics = train_with_mlflow(args)
        print("\nTraining completed successfully!")
        print(f"\nView results at: {args.tracking_uri}")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise


if __name__ == "__main__":
    main()
