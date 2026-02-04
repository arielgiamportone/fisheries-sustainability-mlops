"""Schemas for prediction endpoints."""

from typing import Optional, List, Tuple
from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Input schema for sustainability prediction."""

    # Environmental variables
    sst_c: float = Field(
        ...,
        ge=-5,
        le=40,
        description="Sea Surface Temperature in Celsius"
    )
    salinity_ppt: float = Field(
        ...,
        ge=20,
        le=45,
        description="Salinity in parts per thousand"
    )
    chlorophyll_mg_m3: float = Field(
        ...,
        ge=0,
        le=100,
        description="Chlorophyll concentration in mg/m³"
    )
    ph: float = Field(
        ...,
        ge=6,
        le=9,
        description="pH level"
    )

    # Operational variables
    fleet_size: int = Field(
        ...,
        ge=1,
        le=50000,
        description="Number of vessels in fleet"
    )
    fishing_effort_hours: float = Field(
        ...,
        ge=0,
        description="Total fishing effort in hours"
    )
    fuel_consumption_l: float = Field(
        ...,
        ge=0,
        description="Fuel consumption in liters"
    )

    # Economic variables
    fish_price_usd_ton: float = Field(
        ...,
        ge=0,
        description="Fish price in USD per ton"
    )
    fuel_price_usd_l: float = Field(
        ...,
        ge=0,
        description="Fuel price in USD per liter"
    )
    operating_cost_usd: float = Field(
        ...,
        ge=0,
        description="Operating cost in USD"
    )

    # Model selection (optional)
    model_name: Optional[str] = Field(
        default="sustainability_bnn_api",
        description="Name of the registered model to use"
    )
    model_version: Optional[str] = Field(
        default="Production",
        description="Model version or stage (Production, Staging, or version number)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sst_c": 25.5,
                "salinity_ppt": 35.0,
                "chlorophyll_mg_m3": 2.5,
                "ph": 8.1,
                "fleet_size": 150,
                "fishing_effort_hours": 1200.0,
                "fuel_consumption_l": 5000.0,
                "fish_price_usd_ton": 2500.0,
                "fuel_price_usd_l": 1.2,
                "operating_cost_usd": 15000.0,
                "model_name": "sustainability_bnn",
                "model_version": "Production"
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for sustainability prediction."""

    prediction: int = Field(
        ...,
        description="Binary prediction (0: Not Sustainable, 1: Sustainable)"
    )
    probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of sustainability (0-1)"
    )
    confidence_interval: Optional[Tuple[float, float]] = Field(
        default=None,
        description="95% confidence interval (for BNN models)"
    )
    uncertainty: Optional[float] = Field(
        default=None,
        description="Epistemic uncertainty estimate (for BNN models)"
    )
    model_used: str = Field(
        ...,
        description="Name of the model used for prediction"
    )
    model_version: str = Field(
        ...,
        description="Version of the model used"
    )
    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "probability": 0.78,
                "confidence_interval": [0.72, 0.84],
                "uncertainty": 0.06,
                "model_used": "sustainability_bnn",
                "model_version": "1",
                "inference_time_ms": 12.5
            }
        }


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""

    samples: List[PredictionInput] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="List of samples to predict"
    )
    model_name: Optional[str] = Field(
        default="sustainability_bnn",
        description="Model to use for all predictions"
    )
    model_version: Optional[str] = Field(
        default="Production",
        description="Model version or stage"
    )


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""

    predictions: List[PredictionOutput] = Field(
        ...,
        description="List of prediction results"
    )
    total_samples: int = Field(
        ...,
        description="Total number of samples processed"
    )
    total_time_ms: float = Field(
        ...,
        description="Total processing time in milliseconds"
    )
    model_used: str
    model_version: str
