"""
FastAPI application for DL_Bayesian project.

Provides REST API for:
- Sustainability predictions using trained models
- Model training with MLFlow tracking
- Experiment management
- Model registry operations
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.api.routes import health, predict, train, experiments, models
from src.api.services.model_service import get_model_service


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager.

    Handles startup and shutdown events.
    """
    # Startup
    print("Starting DL_Bayesian API...")

    # Initialize model service
    model_service = get_model_service()
    await model_service.initialize()

    print("API startup complete.")

    yield

    # Shutdown
    print("Shutting down DL_Bayesian API...")
    model_service.clear_cache()
    print("API shutdown complete.")


# Create FastAPI app
app = FastAPI(
    title="DL_Bayesian Sustainability API",
    description="""
    API for fisheries sustainability prediction using Deep Learning and Bayesian Networks.

    ## Features

    - **Predictions**: Make sustainability predictions using BNN/MLP models
    - **Training**: Train new models with MLFlow tracking
    - **Experiments**: View and manage MLFlow experiments
    - **Models**: Manage models in the MLFlow Model Registry

    ## Model Types

    - **BNN (Bayesian Neural Network)**: Provides uncertainty estimates
    - **MLP (Multi-Layer Perceptron)**: Standard feedforward network
    - **Causal VAE**: Variational autoencoder with causal structure

    ## Quick Start

    1. Check API health: `GET /health`
    2. Make a prediction: `POST /api/v1/predict`
    3. Start training: `POST /api/v1/train`
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(train.router, prefix="/api/v1", tags=["Training"])
app.include_router(experiments.router, prefix="/api/v1", tags=["Experiments"])
app.include_router(models.router, prefix="/api/v1", tags=["Models"])

# Static files and templates (if frontend directory exists)
frontend_path = Path(__file__).parent.parent.parent / "frontend"
if frontend_path.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(frontend_path / "static")),
        name="static"
    )
    templates = Jinja2Templates(directory=str(frontend_path / "templates"))

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def home(request: Request):
        """Serve the home page."""
        return templates.TemplateResponse("index.html", {"request": request})

    @app.get("/predict", response_class=HTMLResponse, include_in_schema=False)
    async def predict_page(request: Request):
        """Serve the prediction page."""
        return templates.TemplateResponse("predict.html", {"request": request})

    @app.get("/models", response_class=HTMLResponse, include_in_schema=False)
    async def models_page(request: Request):
        """Serve the models page."""
        return templates.TemplateResponse("models.html", {"request": request})

    @app.get("/experiments", response_class=HTMLResponse, include_in_schema=False)
    async def experiments_page(request: Request):
        """Serve the experiments page."""
        return templates.TemplateResponse("experiments.html", {"request": request})
else:
    @app.get("/", include_in_schema=False)
    async def home():
        """Root endpoint."""
        return {
            "message": "DL_Bayesian Sustainability API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    return {
        "error": "Internal server error",
        "detail": str(exc),
        "path": request.url.path
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
