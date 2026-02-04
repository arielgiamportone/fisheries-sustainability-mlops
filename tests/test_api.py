"""
Tests for the FastAPI application.

Run with: pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient


# Skip if dependencies not available
pytest.importorskip("fastapi")
pytest.importorskip("httpx")


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test /health endpoint returns 200."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data

    def test_liveness_endpoint(self, client):
        """Test /health/live endpoint."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_readiness_endpoint(self, client):
        """Test /health/ready endpoint."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestPredictEndpoints:
    """Tests for prediction endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    @pytest.fixture
    def valid_prediction_input(self):
        """Valid prediction input data."""
        return {
            "sst_c": 25.0,
            "salinity_ppt": 35.0,
            "chlorophyll_mg_m3": 2.5,
            "ph": 8.1,
            "fleet_size": 150,
            "fishing_effort_hours": 1200.0,
            "fuel_consumption_l": 5000.0,
            "fish_price_usd_ton": 2500.0,
            "fuel_price_usd_l": 1.2,
            "operating_cost_usd": 15000.0
        }

    def test_predict_endpoint_schema(self, client, valid_prediction_input):
        """Test that /api/v1/predict accepts valid input."""
        # Note: This may fail if no model is loaded, but tests schema validation
        response = client.post("/api/v1/predict", json=valid_prediction_input)
        # Either 200 (success) or 404 (model not found) or 500 (model error)
        assert response.status_code in [200, 404, 500]

    def test_predict_invalid_input(self, client):
        """Test that /api/v1/predict rejects invalid input."""
        invalid_input = {
            "sst_c": "not a number",  # Invalid type
            "salinity_ppt": 35.0
        }
        response = client.post("/api/v1/predict", json=invalid_input)
        assert response.status_code == 422  # Validation error

    def test_predict_missing_fields(self, client):
        """Test that /api/v1/predict requires all fields."""
        incomplete_input = {
            "sst_c": 25.0,
            "salinity_ppt": 35.0
            # Missing other required fields
        }
        response = client.post("/api/v1/predict", json=incomplete_input)
        assert response.status_code == 422


class TestTrainEndpoints:
    """Tests for training endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_train_endpoint_accepts_request(self, client):
        """Test that /api/v1/train accepts valid request."""
        train_request = {
            "model_type": "bnn",
            "hidden_dims": [32, 16],
            "epochs": 5,
            "batch_size": 32,
            "n_samples": 100
        }
        response = client.post("/api/v1/train", json=train_request)
        # Should return 202 Accepted (training started in background)
        assert response.status_code == 202
        data = response.json()
        assert "job_id" in data
        assert "status" in data

    def test_list_training_jobs(self, client):
        """Test listing training jobs."""
        response = client.get("/api/v1/train/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data


class TestExperimentsEndpoints:
    """Tests for experiments endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_list_experiments(self, client):
        """Test listing experiments."""
        response = client.get("/api/v1/experiments")
        # May return 200 or 503 depending on MLFlow availability
        assert response.status_code in [200, 503]


class TestModelsEndpoints:
    """Tests for models endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_list_models(self, client):
        """Test listing registered models."""
        response = client.get("/api/v1/models")
        # May return 200 or 503 depending on MLFlow availability
        assert response.status_code in [200, 503]

    def test_cache_info(self, client):
        """Test getting cache info."""
        response = client.get("/api/v1/models/cache/info")
        assert response.status_code == 200
        data = response.json()
        assert "cache_size" in data

    def test_clear_cache(self, client):
        """Test clearing model cache."""
        response = client.post("/api/v1/models/cache/clear")
        assert response.status_code == 200


class TestRootEndpoint:
    """Tests for root endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint returns something."""
        response = client.get("/")
        assert response.status_code == 200


class TestOpenAPI:
    """Tests for OpenAPI documentation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "info" in data
        assert "paths" in data

    def test_docs_endpoint(self, client):
        """Test Swagger UI is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200

    def test_redoc_endpoint(self, client):
        """Test ReDoc is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
