"""
Tests for MLOps components (MLFlow, Optuna).

Run with: pytest tests/test_mlops.py -v
"""

import pytest
import numpy as np


class TestMLFlowTracker:
    """Tests for MLFlow tracking wrapper."""

    def test_import_mlflow_tracking(self):
        """Test that mlflow_tracking module can be imported."""
        from src.mlops.mlflow_tracking import MLFlowTracker, create_tracker
        assert MLFlowTracker is not None
        assert create_tracker is not None

    def test_tracker_initialization(self):
        """Test MLFlowTracker can be instantiated."""
        from src.mlops.mlflow_tracking import MLFlowTracker

        # Should not raise even if MLFlow server is not running
        try:
            tracker = MLFlowTracker(
                tracking_uri="http://localhost:5000",
                experiment_name="test_experiment"
            )
            assert tracker is not None
            assert tracker.experiment_name == "test_experiment"
        except Exception:
            # MLFlow server might not be running, which is OK for unit tests
            pytest.skip("MLFlow server not available")


class TestOptunaHyperparameterTuner:
    """Tests for Optuna hyperparameter tuning."""

    def test_import_optuna_tuning(self):
        """Test that optuna_tuning module can be imported."""
        from src.mlops.optuna_tuning import OptunaHyperparameterTuner, TuningConfig
        assert OptunaHyperparameterTuner is not None
        assert TuningConfig is not None

    def test_tuning_config_defaults(self):
        """Test TuningConfig default values."""
        from src.mlops.optuna_tuning import TuningConfig

        config = TuningConfig()
        assert config.n_trials == 50
        assert config.direction == "minimize"
        assert config.epochs_per_trial == 50
        assert config.batch_sizes == [16, 32, 64, 128]

    def test_tuning_config_custom(self):
        """Test TuningConfig with custom values."""
        from src.mlops.optuna_tuning import TuningConfig

        config = TuningConfig(
            n_trials=20,
            epochs_per_trial=30,
            study_name="custom_study"
        )
        assert config.n_trials == 20
        assert config.epochs_per_trial == 30
        assert config.study_name == "custom_study"

    def test_tuner_initialization(self):
        """Test OptunaHyperparameterTuner initialization."""
        from src.mlops.optuna_tuning import OptunaHyperparameterTuner, TuningConfig

        # Create dummy data
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)

        config = TuningConfig(n_trials=2, epochs_per_trial=2)

        tuner = OptunaHyperparameterTuner(
            X=X,
            y=y,
            model_type="mlp",
            config=config
        )

        assert tuner is not None
        assert tuner.model_type == "mlp"
        assert tuner.config.n_trials == 2


class TestMLOpsIntegration:
    """Integration tests for MLOps components."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
        return X, y

    def test_model_training_basic(self, sample_data):
        """Test basic model training works."""
        from src.deep_learning.models import create_model
        from src.deep_learning.training import (
            Trainer, TrainingConfig, prepare_data_loaders
        )

        X, y = sample_data

        # Create simple model
        model = create_model("mlp", input_dim=10, hidden_dims=[16, 8])

        # Create config
        config = TrainingConfig(
            epochs=5,
            batch_size=32,
            learning_rate=0.01,
            device='cpu',
            verbose=False
        )

        # Prepare data
        train_loader, val_loader, _ = prepare_data_loaders(X, y, batch_size=32)

        # Train
        trainer = Trainer(model, config)
        history = trainer.fit(train_loader, val_loader)

        assert len(history.train_loss) > 0
        assert len(history.val_loss) > 0

    def test_bnn_prediction_with_uncertainty(self, sample_data):
        """Test BNN model provides uncertainty estimates."""
        import torch
        from src.deep_learning.models import create_model

        X, y = sample_data

        # Create BNN
        bnn = create_model("bnn", input_dim=10, hidden_dims=[16, 8])

        # Make prediction with uncertainty
        X_tensor = torch.FloatTensor(X[:5])

        mean, std, samples = bnn.predict_with_uncertainty(X_tensor, n_samples=10)

        assert mean.shape == (5, 1)
        assert std.shape == (5, 1)
        assert samples.shape[0] == 10  # n_samples
        assert torch.all(std >= 0)  # Uncertainty should be non-negative


class TestDataLoaders:
    """Tests for data loading utilities."""

    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        from data.loaders import generate_synthetic_fisheries_data

        df = generate_synthetic_fisheries_data(n_samples=100)

        assert len(df) == 100
        assert 'Sustainable' in df.columns
        assert 'SST_C' in df.columns

    def test_prepare_sustainability_data(self):
        """Test data preparation for deep learning."""
        from data.loaders import generate_synthetic_fisheries_data
        from src.deep_learning.training import prepare_sustainability_data

        df = generate_synthetic_fisheries_data(n_samples=100)
        X, y, feature_names = prepare_sustainability_data(df, target='Sustainable')

        assert X.shape[0] == 100
        assert len(y) == 100
        assert len(feature_names) > 0
        assert np.unique(y).tolist() in [[0, 1], [0], [1]]
