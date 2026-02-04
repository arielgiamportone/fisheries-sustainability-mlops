"""Service for MLFlow operations."""

import os
from typing import Dict, Any, Optional, List

from src.mlops.mlflow_tracking import MLFlowTracker


class MLFlowService:
    """
    Service for MLFlow-related operations.

    Provides methods for querying experiments, runs, and models
    from the MLFlow tracking server.
    """

    def __init__(self, tracking_uri: str = None):
        """
        Initialize the MLFlow service.

        Args:
            tracking_uri: MLFlow tracking server URI
        """
        self.tracking_uri = tracking_uri or os.getenv(
            "MLFLOW_TRACKING_URI", "http://localhost:5000"
        )
        self._tracker: Optional[MLFlowTracker] = None

    def _get_tracker(self) -> MLFlowTracker:
        """Get or create MLFlow tracker."""
        if self._tracker is None:
            self._tracker = MLFlowTracker(
                tracking_uri=self.tracking_uri,
                experiment_name="api_queries"
            )
        return self._tracker

    def check_connection(self) -> bool:
        """Check if MLFlow server is reachable."""
        try:
            tracker = self._get_tracker()
            tracker.list_experiments()
            return True
        except Exception:
            return False

    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        List all experiments.

        Returns:
            List of experiment dictionaries
        """
        tracker = self._get_tracker()
        experiments = tracker.list_experiments()

        # Add run count
        for exp in experiments:
            try:
                runs = tracker.list_runs(
                    experiment_id=exp["experiment_id"],
                    max_results=1000
                )
                exp["runs_count"] = len(runs)
            except Exception:
                exp["runs_count"] = 0

        return experiments

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment dictionary or None
        """
        experiments = self.list_experiments()
        for exp in experiments:
            if exp["experiment_id"] == experiment_id:
                return exp
        return None

    def list_runs(
        self,
        experiment_id: str,
        filter_string: str = "",
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        List runs for an experiment.

        Args:
            experiment_id: Experiment ID
            filter_string: Optional MLFlow filter
            max_results: Maximum results

        Returns:
            List of run dictionaries
        """
        tracker = self._get_tracker()
        return tracker.list_runs(
            experiment_id=experiment_id,
            filter_string=filter_string,
            max_results=max_results
        )

    def get_run(self, run_id: str) -> Dict[str, Any]:
        """
        Get a specific run.

        Args:
            run_id: Run ID

        Returns:
            Run dictionary
        """
        tracker = self._get_tracker()
        return {
            "run_id": run_id,
            "metrics": tracker.get_run_metrics(run_id),
            "params": tracker.get_run_params(run_id)
        }

    def list_registered_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.

        Returns:
            List of registered model dictionaries
        """
        tracker = self._get_tracker()
        return tracker.list_registered_models()

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        Get all versions of a registered model.

        Args:
            model_name: Model name

        Returns:
            List of version dictionaries
        """
        models = self.list_registered_models()
        for model in models:
            if model["name"] == model_name:
                return model.get("latest_versions", [])
        return []

    def get_production_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get info about the production version of a model.

        Args:
            model_name: Model name

        Returns:
            Version info or None
        """
        versions = self.get_model_versions(model_name)
        for version in versions:
            if version.get("current_stage") == "Production":
                return version
        return None

    def transition_model_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ):
        """
        Transition a model version to a new stage.

        Args:
            model_name: Model name
            version: Version number
            stage: Target stage
            archive_existing: Archive existing models in target stage
        """
        tracker = self._get_tracker()
        tracker.transition_model_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing=archive_existing
        )

    def get_model_metrics(self, model_name: str, version: str = None) -> Dict[str, float]:
        """
        Get metrics for a model version.

        Args:
            model_name: Model name
            version: Version (uses production if None)

        Returns:
            Metrics dictionary
        """
        versions = self.get_model_versions(model_name)

        target_version = None
        if version is None:
            # Get production version
            for v in versions:
                if v.get("current_stage") == "Production":
                    target_version = v
                    break
        else:
            for v in versions:
                if v.get("version") == version:
                    target_version = v
                    break

        if target_version is None:
            return {}

        run_id = target_version.get("run_id")
        if run_id:
            tracker = self._get_tracker()
            return tracker.get_run_metrics(run_id)

        return {}


# Global instance
_mlflow_service: Optional[MLFlowService] = None


def get_mlflow_service() -> MLFlowService:
    """Get or create the global MLFlow service instance."""
    global _mlflow_service
    if _mlflow_service is None:
        _mlflow_service = MLFlowService()
    return _mlflow_service
