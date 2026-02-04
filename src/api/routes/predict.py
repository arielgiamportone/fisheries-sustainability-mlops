"""Prediction endpoints."""

import time
from typing import List

from fastapi import APIRouter, HTTPException, status

from src.api.schemas.prediction import (
    PredictionInput,
    PredictionOutput,
    BatchPredictionInput,
    BatchPredictionOutput
)
from src.api.services.model_service import get_model_service

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Make a prediction",
    description="Predict sustainability using the specified model"
)
async def predict(input_data: PredictionInput) -> PredictionOutput:
    """
    Make a sustainability prediction.

    Uses the specified model (or default production model) to predict
    whether a fishery operation is sustainable.

    For BNN models, also returns uncertainty estimates and confidence intervals.
    """
    model_service = get_model_service()

    # Convert input to feature dict
    features = {
        "sst_c": input_data.sst_c,
        "salinity_ppt": input_data.salinity_ppt,
        "chlorophyll_mg_m3": input_data.chlorophyll_mg_m3,
        "ph": input_data.ph,
        "fleet_size": input_data.fleet_size,
        "fishing_effort_hours": input_data.fishing_effort_hours,
        "fuel_consumption_l": input_data.fuel_consumption_l,
        "fish_price_usd_ton": input_data.fish_price_usd_ton,
        "fuel_price_usd_l": input_data.fuel_price_usd_l,
        "operating_cost_usd": input_data.operating_cost_usd
    }

    try:
        result = model_service.predict(
            features=features,
            model_name=input_data.model_name,
            model_version=input_data.model_version
        )

        return PredictionOutput(
            prediction=result["prediction"],
            probability=result["probability"],
            confidence_interval=result.get("confidence_interval"),
            uncertainty=result.get("uncertainty"),
            model_used=result["model_used"],
            model_version=result["model_version"],
            inference_time_ms=result["inference_time_ms"]
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Batch predictions",
    description="Make predictions for multiple samples"
)
async def predict_batch(input_data: BatchPredictionInput) -> BatchPredictionOutput:
    """
    Make predictions for multiple samples.

    Processes up to 1000 samples in a single request.
    """
    model_service = get_model_service()
    start_time = time.time()

    predictions: List[PredictionOutput] = []

    for sample in input_data.samples:
        features = {
            "sst_c": sample.sst_c,
            "salinity_ppt": sample.salinity_ppt,
            "chlorophyll_mg_m3": sample.chlorophyll_mg_m3,
            "ph": sample.ph,
            "fleet_size": sample.fleet_size,
            "fishing_effort_hours": sample.fishing_effort_hours,
            "fuel_consumption_l": sample.fuel_consumption_l,
            "fish_price_usd_ton": sample.fish_price_usd_ton,
            "fuel_price_usd_l": sample.fuel_price_usd_l,
            "operating_cost_usd": sample.operating_cost_usd
        }

        try:
            result = model_service.predict(
                features=features,
                model_name=input_data.model_name,
                model_version=input_data.model_version
            )

            predictions.append(PredictionOutput(
                prediction=result["prediction"],
                probability=result["probability"],
                confidence_interval=result.get("confidence_interval"),
                uncertainty=result.get("uncertainty"),
                model_used=result["model_used"],
                model_version=result["model_version"],
                inference_time_ms=result["inference_time_ms"]
            ))

        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch prediction failed: {str(e)}"
            )

    total_time = (time.time() - start_time) * 1000

    return BatchPredictionOutput(
        predictions=predictions,
        total_samples=len(predictions),
        total_time_ms=total_time,
        model_used=input_data.model_name or "sustainability_bnn",
        model_version=input_data.model_version or "Production"
    )
