#!/usr/bin/env python
"""
Train a model specifically for the API with 10 input features.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import mlflow
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.deep_learning.models import create_model
from src.deep_learning.training import (
    Trainer, TrainingConfig, prepare_data_loaders
)


def generate_api_training_data(n_samples=1000):
    """Generate training data with only the 10 API features."""
    np.random.seed(42)

    # Generate features that match API input
    data = {
        'SST_C': np.random.uniform(15, 30, n_samples),
        'Salinity_ppt': np.random.uniform(30, 40, n_samples),
        'Chlorophyll_mg_m3': np.random.uniform(0.1, 10, n_samples),
        'pH': np.random.uniform(7.5, 8.5, n_samples),
        'Fleet_Size': np.random.randint(10, 500, n_samples),
        'Fishing_Effort_hours': np.random.uniform(100, 5000, n_samples),
        'Fuel_Consumption_L': np.random.uniform(500, 20000, n_samples),
        'Fish_Price_USD_ton': np.random.uniform(500, 5000, n_samples),
        'Fuel_Price_USD_L': np.random.uniform(0.5, 2.5, n_samples),
        'Operating_Cost_USD': np.random.uniform(5000, 100000, n_samples),
    }

    df = pd.DataFrame(data)

    # Calculate CPUE (catch per unit effort)
    cpue = df['Chlorophyll_mg_m3'] * 100 / (df['Fishing_Effort_hours'] + 1)

    # Calculate sustainability score based on multiple factors
    sustainability_score = (
        0.3 * (1 - df['Fishing_Effort_hours'] / 5000) +
        0.2 * (cpue / cpue.max()) +
        0.2 * (1 - df['Fleet_Size'] / 500) +
        0.15 * (df['Fish_Price_USD_ton'] / 5000) +
        0.15 * (1 - df['Fuel_Consumption_L'] / 20000)
    )

    # Add noise
    sustainability_score += np.random.normal(0, 0.1, n_samples)

    # Binary target
    df['Sustainable'] = (sustainability_score > 0.4).astype(int)

    return df


def prepare_data(df):
    """Prepare data for training."""
    feature_cols = [
        'SST_C', 'Salinity_ppt', 'Chlorophyll_mg_m3', 'pH',
        'Fleet_Size', 'Fishing_Effort_hours', 'Fuel_Consumption_L',
        'Fish_Price_USD_ton', 'Fuel_Price_USD_L', 'Operating_Cost_USD'
    ]

    X = df[feature_cols].values.astype(np.float32)
    y = df['Sustainable'].values

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, feature_cols, scaler


def main():
    print("=" * 50)
    print("Training API-compatible model (10 features)")
    print("=" * 50)

    # Setup MLFlow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("api_models")

    # Generate data
    print("\nGenerating training data...")
    df = generate_api_training_data(n_samples=1000)
    X, y, feature_names, scaler = prepare_data(df)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Features: {feature_names}")
    print(f"Class balance: {np.mean(y):.2%} sustainable")

    # Create model with 10 input features
    print("\nCreating BNN model...")
    model = create_model("bnn", input_dim=10, hidden_dims=[64, 32])

    # Training config
    config = TrainingConfig(
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        early_stopping_patience=15,
        device='cpu',
        verbose=True
    )

    # Prepare loaders
    train_loader, val_loader, _ = prepare_data_loaders(X, y, batch_size=32, val_split=0.2)

    # Start MLFlow run
    with mlflow.start_run(run_name="bnn_api_model"):
        # Log params
        mlflow.log_params({
            "model_type": "bnn",
            "input_dim": 10,
            "hidden_dims": "[64, 32]",
            "n_samples": len(df),
            "features": str(feature_names)
        })

        # Train
        print("\nTraining...")
        trainer = Trainer(model, config)
        history = trainer.fit(train_loader, val_loader)

        # Log metrics
        for epoch, (tl, vl) in enumerate(zip(history.train_loss, history.val_loss)):
            mlflow.log_metrics({"train_loss": tl, "val_loss": vl}, step=epoch)

        # Final evaluation
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            output = model(X_tensor)
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            probs = torch.sigmoid(logits).numpy().flatten()
            preds = (probs > 0.5).astype(int)

        accuracy = np.mean(preds == y)
        print(f"\nFinal Accuracy: {accuracy:.4f}")

        mlflow.log_metrics({
            "final_accuracy": accuracy,
            "best_val_loss": history.best_val_loss,
            "best_epoch": history.best_epoch
        })

        # Save model
        print("\nSaving model...")
        mlflow.pytorch.log_model(model, "model")

        # Register model
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/model"

        # Delete existing model if exists, then register new one
        client = mlflow.tracking.MlflowClient()
        try:
            client.delete_registered_model("sustainability_bnn_api")
        except:
            pass

        mlflow.register_model(model_uri, "sustainability_bnn_api")
        print("Model registered as: sustainability_bnn_api")

        # Promote to Production
        client.transition_model_version_stage(
            name="sustainability_bnn_api",
            version="1",
            stage="Production"
        )
        print("Model promoted to Production")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print("=" * 50)
    print(f"Accuracy: {accuracy:.4f}")
    print("Model: sustainability_bnn_api (Production)")


if __name__ == "__main__":
    main()
