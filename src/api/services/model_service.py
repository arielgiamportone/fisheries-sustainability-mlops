"""Service for model loading, caching, and inference."""

import os
import time
from typing import Dict, Any, Optional, Tuple, List
from threading import Lock

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.mlops.mlflow_tracking import MLFlowTracker


class ModelService:
    """
    Manages model loading, caching, and inference operations.

    Provides thread-safe model loading with caching and supports
    both production and staging models.

    Example:
        >>> service = ModelService()
        >>> await service.initialize()
        >>> result = service.predict(features, model_name="sustainability_bnn")
    """

    def __init__(
        self,
        mlflow_tracking_uri: str = None,
        default_model_name: str = "sustainability_bnn_api",
        cache_models: bool = True
    ):
        """
        Initialize the model service.

        Args:
            mlflow_tracking_uri: MLFlow tracking server URI
            default_model_name: Default model name to use
            cache_models: Whether to cache loaded models
        """
        self.tracking_uri = mlflow_tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self.default_model_name = default_model_name
        self.cache_models = cache_models

        # Model cache: {(model_name, version): (model, scaler, timestamp)}
        self._model_cache: Dict[Tuple[str, str], Tuple[nn.Module, Any, float]] = {}
        self._cache_lock = Lock()

        # Tracker
        self._tracker: Optional[MLFlowTracker] = None

        # Feature configuration (must match training)
        self.feature_names = [
            'SST_C', 'Salinity_ppt', 'Chlorophyll_mg_m3', 'pH',
            'Fleet_Size', 'Fishing_Effort_hours', 'Fuel_Consumption_L',
            'Fish_Price_USD_ton', 'Fuel_Price_USD_L', 'Operating_Cost_USD'
        ]

        self._initialized = False

    async def initialize(self):
        """Initialize the service and load default model."""
        if self._initialized:
            return

        try:
            self._tracker = MLFlowTracker(
                tracking_uri=self.tracking_uri,
                experiment_name="api_inference"
            )
            self._initialized = True

            # Try to load production model
            await self._preload_production_model()
        except Exception as e:
            print(f"Warning: Could not initialize MLFlow connection: {e}")
            self._initialized = True  # Mark as initialized anyway

    async def _preload_production_model(self):
        """Preload the production model into cache."""
        try:
            model = self._tracker.get_production_model(self.default_model_name)
            if model is not None:
                cache_key = (self.default_model_name, "Production")
                with self._cache_lock:
                    self._model_cache[cache_key] = (model, None, time.time())
                print(f"Preloaded production model: {self.default_model_name}")
        except Exception as e:
            print(f"Could not preload production model: {e}")

    def _get_model(
        self,
        model_name: str,
        model_version: str = "Production"
    ) -> Optional[nn.Module]:
        """
        Get a model from cache or load from MLFlow.

        Args:
            model_name: Name of the registered model
            model_version: Version or stage

        Returns:
            Loaded model or None
        """
        cache_key = (model_name, model_version)

        # Check cache first
        if self.cache_models:
            with self._cache_lock:
                if cache_key in self._model_cache:
                    model, scaler, _ = self._model_cache[cache_key]
                    return model

        # Load from MLFlow
        try:
            if model_version in ("Production", "Staging", "Archived"):
                model_uri = f"models:/{model_name}/{model_version}"
            else:
                model_uri = f"models:/{model_name}/{model_version}"

            model = self._tracker.load_model(model_uri)

            # Cache if enabled
            if self.cache_models:
                with self._cache_lock:
                    self._model_cache[cache_key] = (model, None, time.time())

            return model

        except Exception as e:
            print(f"Error loading model {model_name}/{model_version}: {e}")
            return None

    def _preprocess_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Preprocess input features for model inference.

        Args:
            features: Dictionary of feature values

        Returns:
            Preprocessed feature array
        """
        # Extract features in correct order
        feature_vector = []
        for name in self.feature_names:
            # Map API field names to internal names
            api_name = name.lower()
            if api_name in features:
                feature_vector.append(features[api_name])
            elif name in features:
                feature_vector.append(features[name])
            else:
                raise ValueError(f"Missing feature: {name}")

        # Convert to array and normalize (simple z-score)
        X = np.array(feature_vector, dtype=np.float32).reshape(1, -1)

        # Apply standard scaling (approximate, should match training)
        # In production, you'd save/load the scaler with the model
        X = (X - X.mean()) / (X.std() + 1e-8)

        return X

    def predict(
        self,
        features: Dict[str, float],
        model_name: Optional[str] = None,
        model_version: str = "Production",
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Make a prediction using the specified model.

        Args:
            features: Dictionary of input features
            model_name: Model name (uses default if None)
            model_version: Model version or stage
            n_samples: Number of samples for BNN uncertainty estimation

        Returns:
            Prediction results dictionary
        """
        start_time = time.time()

        model_name = model_name or self.default_model_name

        # Get model
        model = self._get_model(model_name, model_version)
        if model is None:
            raise ValueError(f"Model not found: {model_name}/{model_version}")

        # Preprocess
        X = self._preprocess_features(features)
        X_tensor = torch.FloatTensor(X)

        # Inference
        model.eval()
        with torch.no_grad():
            output = model(X_tensor)

            # Handle different model types
            if isinstance(output, tuple):
                # BNN returns (logits, kl)
                logits = output[0]
            else:
                logits = output

            probability = torch.sigmoid(logits).item()
            prediction = 1 if probability >= 0.5 else 0

        # Calculate uncertainty for BNN
        uncertainty = None
        confidence_interval = None

        if hasattr(model, 'predict_with_uncertainty'):
            try:
                mean_prob, std_prob, _ = model.predict_with_uncertainty(
                    X_tensor, n_samples=n_samples
                )
                uncertainty = std_prob.item()
                # 95% confidence interval
                ci_low = max(0, mean_prob.item() - 1.96 * std_prob.item())
                ci_high = min(1, mean_prob.item() + 1.96 * std_prob.item())
                confidence_interval = (ci_low, ci_high)
                probability = mean_prob.item()
                prediction = 1 if probability >= 0.5 else 0
            except Exception:
                pass

        inference_time = (time.time() - start_time) * 1000

        return {
            "prediction": prediction,
            "probability": probability,
            "uncertainty": uncertainty,
            "confidence_interval": confidence_interval,
            "model_used": model_name,
            "model_version": model_version,
            "inference_time_ms": inference_time
        }

    def predict_batch(
        self,
        samples: List[Dict[str, float]],
        model_name: Optional[str] = None,
        model_version: str = "Production"
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple samples.

        Args:
            samples: List of feature dictionaries
            model_name: Model name
            model_version: Model version

        Returns:
            List of prediction results
        """
        results = []
        for sample in samples:
            result = self.predict(sample, model_name, model_version)
            results.append(result)
        return results

    def is_model_loaded(self, model_name: str = None, version: str = "Production") -> bool:
        """Check if a model is loaded in cache."""
        model_name = model_name or self.default_model_name
        cache_key = (model_name, version)
        with self._cache_lock:
            return cache_key in self._model_cache

    def clear_cache(self):
        """Clear all cached models."""
        with self._cache_lock:
            self._model_cache.clear()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models."""
        with self._cache_lock:
            return {
                "cached_models": list(self._model_cache.keys()),
                "cache_size": len(self._model_cache)
            }


# Global instance
_model_service: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get or create the global model service instance."""
    global _model_service
    if _model_service is None:
        _model_service = ModelService()
    return _model_service
