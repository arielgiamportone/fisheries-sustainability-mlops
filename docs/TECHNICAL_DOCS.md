# Documentación Técnica — Fisheries Sustainability MLOps

## Sistema MLOps para Sostenibilidad Pesquera con Deep Learning y Redes Bayesianas

**Versión:** 1.0.0  
**Última actualización:** Febrero 2026  
**Autor:** Ing. Ariel Luján Giamportone  
**ORCID:** [0009-0000-1607-9743](https://orcid.org/0009-0000-1607-9743)  
**Repositorio:** [github.com/arielgiamportone/fisheries-sustainability-mlops](https://github.com/arielgiamportone/fisheries-sustainability-mlops)

---

## Tabla de Contenidos

1. [Arquitectura del Sistema](#1-arquitectura-del-sistema)
2. [Stack Tecnológico](#2-stack-tecnológico)
3. [Modelos de Deep Learning](#3-modelos-de-deep-learning)
4. [Redes Bayesianas](#4-redes-bayesianas)
5. [Pipeline de Datos](#5-pipeline-de-datos)
6. [API REST](#6-api-rest)
7. [MLFlow Integration](#7-mlflow-integration)
8. [Optuna Hyperparameter Tuning](#8-optuna-hyperparameter-tuning)
9. [Containerización](#9-containerización)
10. [Infraestructura AWS](#10-infraestructura-aws)
11. [CI/CD Pipeline](#11-cicd-pipeline)
12. [Seguridad](#12-seguridad)
13. [Monitoreo y Logging](#13-monitoreo-y-logging)
14. [Testing](#14-testing)

---

## 1. Arquitectura del Sistema

### 1.1 Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CAPA DE PRESENTACIÓN                          │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐  │
│  │  Web Dashboard  │  │   REST API      │  │    MLFlow UI            │  │
│  │  (HTML/CSS/JS)  │  │   Consumers     │  │    (Experimentos)       │  │
│  │  Puerto: 8000   │  │   (Externos)    │  │    Puerto: 5000         │  │
│  └────────┬────────┘  └────────┬────────┘  └────────────┬────────────┘  │
└───────────┼─────────────────────┼───────────────────────┼───────────────┘
            │                     │                       │
            ▼                     ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CAPA DE APLICACIÓN                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        FastAPI Application                        │  │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐  │  │
│  │  │   Routes    │ │   Schemas   │ │  Services   │ │ Middleware  │  │  │
│  │  │  /health    │ │  Pydantic   │ │ModelService │ │   CORS      │  │  │
│  │  │  /predict   │ │  Validation │ │MLFlowService│ │   Logging   │  │  │
│  │  │  /train     │ │             │ │             │ │             │  │  │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
            │                                             │
            ▼                                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CAPA DE ML/AI                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐  │
│  │   Deep Learning     │  │  Bayesian Networks  │  │    MLOps        │  │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │  │  ┌───────────┐  │  │
│  │  │      BNN      │  │  │  │    pgmpy      │  │  │  │  MLFlow   │  │  │
│  │  │  (PyTorch)    │  │  │  │   DAG Model   │  │  │  │  Tracking │  │  │
│  │  └───────────────┘  │  │  └───────────────┘  │  │  └───────────┘  │  │
│  │  ┌───────────────┐  │  │  ┌───────────────┐  │  │  ┌───────────┐  │  │
│  │  │      MLP      │  │  │  │   Inference   │  │  │  │  Optuna   │  │  │
│  │  │  (PyTorch)    │  │  │  │   Engine      │  │  │  │  Tuning   │  │  │
│  │  └───────────────┘  │  │  └───────────────┘  │  │  └───────────┘  │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
            │                                             │
            ▼                                             ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CAPA DE DATOS                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────┐  │
│  │   Model Registry    │  │  Artifact Storage   │  │  Training Data  │  │
│  │   (MLFlow)          │  │   (mlruns/)         │  │   (data/)       │  │
│  └─────────────────────┘  └─────────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Flujo de Datos

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Datos de   │───▶│  Preproceso  │───▶│ Entrenamiento│───▶│   Modelo     │
│   Entrada    │    │  (Scaling)   │    │  (PyTorch)   │    │  Registrado  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Respuesta  │◀───│  Post-proc   │◀───│  Inferencia  │◀───│   Modelo     │
│   con Uncert.│    │  (Sigmoid)   │    │  (BNN)       │    │  Producción  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### 1.3 Componentes Principales

| Componente | Tecnología | Propósito |
|------------|------------|-----------|
| API Backend | FastAPI | REST endpoints, validación |
| Frontend | HTML/CSS/JS | Dashboard web interactivo |
| Deep Learning | PyTorch | BNN y MLP para predicciones |
| Bayesian Networks | pgmpy | Modelado causal |
| Experiment Tracking | MLFlow | Logging, registry, UI |
| HP Tuning | Optuna | Optimización automática |
| Containerización | Docker | Portabilidad |
| Orquestación | Docker Compose | Multi-container local |
| Cloud | AWS ECS/ECR | Deployment producción |
| CI/CD | GitHub Actions | Automatización |

---

## 2. Stack Tecnológico

### 2.1 Dependencias Core

```python
# requirements.txt - Versiones mínimas
python >= 3.9

# Deep Learning
torch >= 2.0.0
numpy >= 1.24.0
pandas >= 2.0.0
scikit-learn >= 1.3.0

# Bayesian
pgmpy >= 0.1.23

# API
fastapi >= 0.104.0
uvicorn[standard] >= 0.24.0
pydantic >= 2.5.0
pydantic-settings >= 2.1.0
jinja2 >= 3.1.2
aiofiles >= 23.2.1
python-multipart >= 0.0.6

# MLOps
mlflow >= 2.9.0
optuna >= 3.4.0
optuna-integration[mlflow] >= 3.6.0

# Utilities
python-dotenv >= 1.0.0
httpx >= 0.25.0
pyyaml >= 6.0.1
```

### 2.2 Versiones de Python Soportadas

| Python | Estado | Notas |
|--------|--------|-------|
| 3.9 | Soportado | Mínimo requerido |
| 3.10 | Soportado | Recomendado |
| 3.11 | Soportado | Mejor rendimiento |
| 3.12 | Experimental | No todas las deps compatibles |

### 2.3 Compatibilidad de Plataformas

| Plataforma | Estado | Notas |
|------------|--------|-------|
| Windows 10/11 | Completo | Desarrollo principal |
| Linux (Ubuntu 20.04+) | Completo | Producción |
| macOS (Intel/ARM) | Completo | - |
| Docker | Completo | Recomendado para deploy |

---

## 3. Modelos de Deep Learning

### 3.1 Bayesian Neural Network (BNN)

#### Arquitectura

```python
class BayesianNeuralNetwork(nn.Module):
    """
    Red neuronal con dropout como aproximación variacional bayesiana.
    Permite cuantificar incertidumbre mediante Monte Carlo Dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 1,
        dropout_rate: float = 0.2
    ):
        # Capas lineales con dropout
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)  # MC Dropout
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
```

#### Predicción con Incertidumbre

```python
def predict_with_uncertainty(
    self,
    x: torch.Tensor,
    n_samples: int = 100
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Monte Carlo Dropout para cuantificación de incertidumbre.

    Args:
        x: Tensor de entrada [batch_size, input_dim]
        n_samples: Número de pasadas forward con dropout activo

    Returns:
        mean: Media de predicciones [batch_size, output_dim]
        std: Desviación estándar (incertidumbre) [batch_size, output_dim]
        samples: Todas las muestras [n_samples, batch_size, output_dim]
    """
    self.train()  # Mantiene dropout activo

    samples = []
    for _ in range(n_samples):
        with torch.no_grad():
            output = self.forward(x)
            samples.append(torch.sigmoid(output))

    samples = torch.stack(samples)
    mean = samples.mean(dim=0)
    std = samples.std(dim=0)

    return mean, std, samples
```

#### Interpretación de Incertidumbre

| Nivel de Incertidumbre (std) | Interpretación |
|------------------------------|----------------|
| < 0.1 | Alta confianza en la predicción |
| 0.1 - 0.2 | Confianza moderada |
| 0.2 - 0.3 | Incertidumbre significativa |
| > 0.3 | Baja confianza, revisar datos |

### 3.2 Multilayer Perceptron (MLP)

#### Arquitectura

```python
class MultilayerPerceptron(nn.Module):
    """
    MLP estándar para clasificación binaria.
    Más rápido que BNN pero sin cuantificación de incertidumbre.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        output_dim: int = 1,
        activation: str = 'relu'
    ):
        layers = []
        prev_dim = input_dim

        activation_fn = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU()
        }[activation]

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                activation_fn,
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
```

### 3.3 Factory Pattern

```python
def create_model(
    model_type: str,
    input_dim: int,
    hidden_dims: List[int] = [64, 32],
    **kwargs
) -> nn.Module:
    """
    Factory para crear modelos de forma consistente.

    Args:
        model_type: 'bnn' o 'mlp'
        input_dim: Dimensión de entrada
        hidden_dims: Lista de dimensiones ocultas
        **kwargs: Parámetros adicionales del modelo

    Returns:
        Modelo PyTorch instanciado
    """
    models = {
        'bnn': BayesianNeuralNetwork,
        'mlp': MultilayerPerceptron
    }

    if model_type not in models:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")

    return models[model_type](
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        **kwargs
    )
```

### 3.4 Training Pipeline

```python
@dataclass
class TrainingConfig:
    """Configuración para entrenamiento."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    device: str = 'cpu'
    verbose: bool = True

class Trainer:
    """Entrenador genérico para modelos PyTorch."""

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> TrainingHistory:
        """
        Entrena el modelo con early stopping.

        Returns:
            TrainingHistory con métricas por época
        """
        history = TrainingHistory()
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config.epochs):
            # Training step
            train_loss = self._train_epoch(train_loader)

            # Validation step
            val_loss = self._validate_epoch(val_loader)

            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                patience_counter = 0
                # Save best model weights
                self._save_checkpoint()
            else:
                patience_counter += 1

            if patience_counter >= self.config.early_stopping_patience:
                if self.config.verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        # Restore best weights
        self._load_checkpoint()
        return history
```

---

## 4. Redes Bayesianas

### 4.1 Estructura del DAG

```python
# Grafo Acíclico Dirigido para Sostenibilidad Pesquera
dag_structure = [
    # Variables ambientales → Biomasa
    ('SST_C', 'Fish_Biomass'),
    ('Salinity_ppt', 'Fish_Biomass'),
    ('Chlorophyll_mg_m3', 'Fish_Biomass'),
    ('pH', 'Fish_Biomass'),

    # Biomasa → Captura
    ('Fish_Biomass', 'Catch_tons'),

    # Esfuerzo pesquero → Captura
    ('Fleet_Size', 'Catch_tons'),
    ('Fishing_Effort_hours', 'Catch_tons'),

    # Factores económicos
    ('Catch_tons', 'Revenue_USD'),
    ('Fish_Price_USD_ton', 'Revenue_USD'),

    # Costos operativos
    ('Fuel_Consumption_L', 'Operating_Cost_USD'),
    ('Fuel_Price_USD_L', 'Operating_Cost_USD'),
    ('Fleet_Size', 'Operating_Cost_USD'),

    # Sostenibilidad
    ('Catch_tons', 'Sustainable'),
    ('Fish_Biomass', 'Sustainable'),
    ('Operating_Cost_USD', 'Sustainable'),
    ('Revenue_USD', 'Sustainable')
]
```

### 4.2 Implementación con pgmpy

```python
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

class FisheriesBayesianNetwork:
    """Red Bayesiana para análisis causal de sostenibilidad."""

    def __init__(self, structure: List[Tuple[str, str]]):
        self.model = BayesianNetwork(structure)
        self.inference_engine = None

    def fit(self, data: pd.DataFrame):
        """
        Aprende CPDs desde datos usando MLE.

        Args:
            data: DataFrame con variables discretizadas
        """
        # Discretizar variables continuas
        discretized_data = self._discretize(data)

        # Ajustar CPDs
        self.model.fit(
            discretized_data,
            estimator=MaximumLikelihoodEstimator
        )

        # Preparar motor de inferencia
        self.inference_engine = VariableElimination(self.model)

    def query(
        self,
        target: str,
        evidence: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Realiza inferencia probabilística.

        Args:
            target: Variable a consultar
            evidence: Diccionario de evidencias observadas

        Returns:
            Distribución de probabilidad del target
        """
        result = self.inference_engine.query(
            variables=[target],
            evidence=evidence
        )
        return result

    def _discretize(self, data: pd.DataFrame) -> pd.DataFrame:
        """Discretiza variables continuas en bins."""
        discretized = data.copy()

        continuous_vars = [
            'SST_C', 'Salinity_ppt', 'Chlorophyll_mg_m3',
            'pH', 'Catch_tons', 'Revenue_USD'
        ]

        for var in continuous_vars:
            if var in discretized.columns:
                discretized[var] = pd.qcut(
                    discretized[var],
                    q=3,
                    labels=['Low', 'Medium', 'High']
                )

        return discretized
```

### 4.3 Análisis Causal

```python
def causal_analysis(
    bn: FisheriesBayesianNetwork,
    intervention: Dict[str, str],
    outcome: str
) -> Dict[str, float]:
    """
    Análisis do-calculus para intervenciones.

    Args:
        bn: Red Bayesiana entrenada
        intervention: Variables a intervenir (do-operator)
        outcome: Variable resultado

    Returns:
        Distribución del outcome bajo intervención
    """
    # Simular intervención (do-calculus)
    # P(Sustainable | do(Fishing_Effort = Low))

    modified_bn = bn.model.copy()

    # Eliminar aristas entrantes a variable intervenida
    for var in intervention:
        parents = list(modified_bn.get_parents(var))
        for parent in parents:
            modified_bn.remove_edge(parent, var)

    # Realizar inferencia
    inference = VariableElimination(modified_bn)
    result = inference.query(
        variables=[outcome],
        evidence=intervention
    )

    return result.values
```

---

## 5. Pipeline de Datos

### 5.1 Variables del Sistema

#### Variables Ambientales

| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| `SST_C` | float | 15-30 | Temperatura superficial del mar (°C) |
| `Salinity_ppt` | float | 30-40 | Salinidad (partes por mil) |
| `Chlorophyll_mg_m3` | float | 0.1-10 | Clorofila-a (mg/m³) |
| `pH` | float | 7.5-8.5 | pH del agua |

#### Variables Operativas

| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| `Fleet_Size` | int | 10-500 | Número de embarcaciones |
| `Fishing_Effort_hours` | float | 100-5000 | Horas de pesca |
| `Fuel_Consumption_L` | float | 500-20000 | Consumo de combustible (L) |

#### Variables Económicas

| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| `Fish_Price_USD_ton` | float | 500-5000 | Precio del pescado (USD/ton) |
| `Fuel_Price_USD_L` | float | 0.5-2.5 | Precio del combustible (USD/L) |
| `Operating_Cost_USD` | float | 5000-100000 | Costo operativo (USD) |

#### Variable Objetivo

| Variable | Tipo | Valores | Descripción |
|----------|------|---------|-------------|
| `Sustainable` | int | 0, 1 | Clasificación binaria de sostenibilidad |

### 5.2 Preprocesamiento

```python
def prepare_sustainability_data(
    df: pd.DataFrame,
    target: str = 'Sustainable',
    scaler: StandardScaler = None
) -> Tuple[np.ndarray, np.ndarray, List[str], StandardScaler]:
    """
    Prepara datos para entrenamiento de modelos.

    Pipeline:
    1. Selección de features
    2. Manejo de valores faltantes
    3. Estandarización (z-score)
    4. Separación X/y

    Args:
        df: DataFrame con datos crudos
        target: Nombre de columna objetivo
        scaler: Scaler pre-ajustado (opcional)

    Returns:
        X: Features escaladas [n_samples, n_features]
        y: Labels [n_samples]
        feature_names: Lista de nombres de features
        scaler: Scaler ajustado
    """
    feature_cols = [
        'SST_C', 'Salinity_ppt', 'Chlorophyll_mg_m3', 'pH',
        'Fleet_Size', 'Fishing_Effort_hours', 'Fuel_Consumption_L',
        'Fish_Price_USD_ton', 'Fuel_Price_USD_L', 'Operating_Cost_USD'
    ]

    # Filtrar columnas disponibles
    available_features = [c for c in feature_cols if c in df.columns]

    X = df[available_features].values.astype(np.float32)
    y = df[target].values if target in df.columns else None

    # Estandarización
    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y, available_features, scaler
```

### 5.3 Data Loaders

```python
def prepare_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    test_split: float = 0.1,
    random_state: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders para train/val/test.

    Args:
        X: Features [n_samples, n_features]
        y: Labels [n_samples]
        batch_size: Tamaño de batch
        val_split: Proporción para validación
        test_split: Proporción para test
        random_state: Semilla para reproducibilidad

    Returns:
        train_loader, val_loader, test_loader
    """
    # Split train/temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_split + test_split,
        random_state=random_state
    )

    # Split val/test
    relative_test = test_split / (val_split + test_split)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test,
        random_state=random_state
    )

    # Create datasets
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test).unsqueeze(1)
    )

    # Create loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader
```

---

## 6. API REST

### 6.1 Estructura de la API

```
src/api/
├── __init__.py
├── main.py              # Aplicación FastAPI
├── routes/
│   ├── __init__.py
│   ├── health.py        # Endpoints de salud
│   ├── predict.py       # Predicciones
│   ├── train.py         # Entrenamiento
│   ├── experiments.py   # MLFlow experiments
│   └── models.py        # Model registry
├── schemas/
│   ├── __init__.py
│   ├── prediction.py    # Schemas de predicción
│   └── training.py      # Schemas de entrenamiento
└── services/
    ├── __init__.py
    ├── model_service.py   # Servicio de modelos
    └── mlflow_service.py  # Servicio MLFlow
```

### 6.2 Endpoints

#### Health Endpoints

```python
# GET /health
{
    "status": "healthy",
    "timestamp": "2025-02-04T12:00:00Z",
    "version": "1.0.0",
    "components": {
        "api": "healthy",
        "mlflow": "healthy",
        "model": "loaded"
    }
}

# GET /health/live
{"status": "alive"}

# GET /health/ready
{"status": "ready", "model_loaded": true}
```

#### Prediction Endpoint

```python
# POST /api/v1/predict
# Request
{
    "sst_c": 25.0,
    "salinity_ppt": 35.0,
    "chlorophyll_mg_m3": 2.5,
    "ph": 8.1,
    "fleet_size": 150,
    "fishing_effort_hours": 1200.0,
    "fuel_consumption_l": 5000.0,
    "fish_price_usd_ton": 2500.0,
    "fuel_price_usd_l": 1.2,
    "operating_cost_usd": 15000.0,
    "model_name": "sustainability_bnn_api",
    "model_stage": "Production"
}

# Response
{
    "prediction": 1,
    "probability": 0.847,
    "uncertainty": 0.089,
    "is_sustainable": true,
    "confidence": "high",
    "model_info": {
        "name": "sustainability_bnn_api",
        "version": "1",
        "stage": "Production"
    },
    "timestamp": "2025-02-04T12:00:00Z"
}
```

#### Training Endpoints

```python
# POST /api/v1/train
# Request
{
    "model_type": "bnn",
    "hidden_dims": [64, 32],
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "n_samples": 1000
}

# Response (202 Accepted)
{
    "job_id": "train_abc123",
    "status": "started",
    "message": "Training job started",
    "created_at": "2025-02-04T12:00:00Z"
}

# GET /api/v1/train/jobs
{
    "jobs": [
        {
            "job_id": "train_abc123",
            "status": "completed",
            "model_type": "bnn",
            "accuracy": 0.856,
            "created_at": "2025-02-04T12:00:00Z",
            "completed_at": "2025-02-04T12:05:00Z"
        }
    ],
    "total": 1
}
```

### 6.3 Schemas Pydantic

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum

class ModelStage(str, Enum):
    NONE = "None"
    STAGING = "Staging"
    PRODUCTION = "Production"
    ARCHIVED = "Archived"

class PredictionInput(BaseModel):
    """Input schema para predicción."""

    sst_c: float = Field(
        ..., ge=0, le=40,
        description="Temperatura superficial del mar (°C)"
    )
    salinity_ppt: float = Field(
        ..., ge=0, le=50,
        description="Salinidad (partes por mil)"
    )
    chlorophyll_mg_m3: float = Field(
        ..., ge=0, le=50,
        description="Concentración de clorofila (mg/m³)"
    )
    ph: float = Field(
        ..., ge=6.0, le=9.0,
        description="pH del agua"
    )
    fleet_size: int = Field(
        ..., ge=1, le=1000,
        description="Número de embarcaciones"
    )
    fishing_effort_hours: float = Field(
        ..., ge=0,
        description="Horas de esfuerzo pesquero"
    )
    fuel_consumption_l: float = Field(
        ..., ge=0,
        description="Consumo de combustible (litros)"
    )
    fish_price_usd_ton: float = Field(
        ..., ge=0,
        description="Precio del pescado (USD/tonelada)"
    )
    fuel_price_usd_l: float = Field(
        ..., ge=0,
        description="Precio del combustible (USD/litro)"
    )
    operating_cost_usd: float = Field(
        ..., ge=0,
        description="Costo operativo (USD)"
    )
    model_name: Optional[str] = Field(
        default="sustainability_bnn_api",
        description="Nombre del modelo registrado"
    )
    model_stage: Optional[ModelStage] = Field(
        default=ModelStage.PRODUCTION,
        description="Stage del modelo"
    )

class PredictionOutput(BaseModel):
    """Output schema para predicción."""

    prediction: int = Field(
        description="Predicción binaria (0=No sostenible, 1=Sostenible)"
    )
    probability: float = Field(
        ge=0, le=1,
        description="Probabilidad de sostenibilidad"
    )
    uncertainty: Optional[float] = Field(
        default=None,
        description="Incertidumbre de la predicción (solo BNN)"
    )
    is_sustainable: bool = Field(
        description="Interpretación booleana"
    )
    confidence: str = Field(
        description="Nivel de confianza: high, medium, low"
    )
    model_info: dict = Field(
        description="Información del modelo usado"
    )
    timestamp: str = Field(
        description="Timestamp de la predicción"
    )
```

### 6.4 Servicios

#### ModelService

```python
class ModelService:
    """Servicio para carga y predicción de modelos."""

    def __init__(
        self,
        mlflow_tracking_uri: str = None,
        default_model_name: str = "sustainability_bnn_api",
        cache_models: bool = True
    ):
        self.tracking_uri = mlflow_tracking_uri or "http://127.0.0.1:5000"
        self.default_model_name = default_model_name
        self.cache_models = cache_models
        self._model_cache: Dict[str, nn.Module] = {}
        self._scaler = StandardScaler()

        mlflow.set_tracking_uri(self.tracking_uri)

    def load_model(
        self,
        model_name: str = None,
        stage: str = "Production"
    ) -> nn.Module:
        """
        Carga modelo desde MLFlow registry.

        Args:
            model_name: Nombre del modelo registrado
            stage: Stage del modelo (Production, Staging, etc.)

        Returns:
            Modelo PyTorch cargado
        """
        name = model_name or self.default_model_name
        cache_key = f"{name}_{stage}"

        if self.cache_models and cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model_uri = f"models:/{name}/{stage}"
        model = mlflow.pytorch.load_model(model_uri)

        if self.cache_models:
            self._model_cache[cache_key] = model

        return model

    def predict(
        self,
        features: np.ndarray,
        model_name: str = None,
        stage: str = "Production"
    ) -> Dict:
        """
        Realiza predicción con modelo cargado.

        Args:
            features: Array de features [1, n_features]
            model_name: Nombre del modelo
            stage: Stage del modelo

        Returns:
            Dict con predicción, probabilidad e incertidumbre
        """
        model = self.load_model(model_name, stage)

        # Escalar features
        scaled_features = self._scaler.transform(features)
        x_tensor = torch.FloatTensor(scaled_features)

        # Predicción
        model.eval()

        if hasattr(model, 'predict_with_uncertainty'):
            # BNN con incertidumbre
            mean, std, _ = model.predict_with_uncertainty(
                x_tensor, n_samples=100
            )
            probability = mean.item()
            uncertainty = std.item()
        else:
            # MLP sin incertidumbre
            with torch.no_grad():
                logits = model(x_tensor)
                probability = torch.sigmoid(logits).item()
                uncertainty = None

        prediction = 1 if probability > 0.5 else 0

        return {
            "prediction": prediction,
            "probability": probability,
            "uncertainty": uncertainty,
            "is_sustainable": prediction == 1
        }
```

---

## 7. MLFlow Integration

### 7.1 MLFlowTracker

```python
class MLFlowTracker:
    """Wrapper para tracking de experimentos con MLFlow."""

    def __init__(
        self,
        tracking_uri: str = "http://127.0.0.1:5000",
        experiment_name: str = "fisheries_sustainability"
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str = None, nested: bool = False):
        """
        Context manager para runs de MLFlow.

        Usage:
            with tracker.start_run("my_experiment"):
                tracker.log_params({...})
                # train model
                tracker.log_metrics({...})
        """
        with mlflow.start_run(run_name=run_name, nested=nested) as run:
            self.active_run = run
            yield run

    def log_params(self, params: Dict[str, Any]):
        """Log parámetros del experimento."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log métricas del experimento."""
        mlflow.log_metrics(metrics, step=step)

    def log_model(
        self,
        model: nn.Module,
        artifact_path: str = "model",
        registered_model_name: str = None
    ):
        """
        Log modelo a MLFlow.

        Args:
            model: Modelo PyTorch
            artifact_path: Path dentro del artifact store
            registered_model_name: Nombre para registro (opcional)
        """
        mlflow.pytorch.log_model(
            model,
            artifact_path=artifact_path,
            registered_model_name=registered_model_name
        )

    def register_model(
        self,
        model_uri: str,
        name: str
    ) -> str:
        """
        Registra modelo en Model Registry.

        Args:
            model_uri: URI del modelo (runs:/run_id/artifact_path)
            name: Nombre para el modelo registrado

        Returns:
            Versión del modelo registrado
        """
        result = mlflow.register_model(model_uri, name)
        return result.version

    def transition_model_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing: bool = True
    ):
        """
        Transiciona modelo entre stages.

        Args:
            name: Nombre del modelo registrado
            version: Versión del modelo
            stage: Nuevo stage (Staging, Production, Archived)
            archive_existing: Archivar modelos existentes en ese stage
        """
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=name,
            version=version,
            stage=stage,
            archive_existing_versions=archive_existing
        )
```

### 7.2 Model Registry Workflow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    None     │────▶│   Staging   │────▶│ Production  │────▶│  Archived   │
│  (Initial)  │     │  (Testing)  │     │   (Live)    │     │ (Retired)   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │                   │
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
  Modelo nuevo        Validación         Sirviendo            Histórico
  registrado          A/B testing        predicciones         referencia
```

### 7.3 Artifact Storage

```
mlruns/
├── 0/                          # Default experiment
├── 1/                          # fisheries_sustainability experiment
│   ├── meta.yaml
│   └── abc123def456/           # Run ID
│       ├── meta.yaml
│       ├── params/
│       │   ├── epochs
│       │   ├── learning_rate
│       │   └── model_type
│       ├── metrics/
│       │   ├── train_loss
│       │   ├── val_loss
│       │   └── accuracy
│       ├── artifacts/
│       │   └── model/
│       │       ├── MLmodel
│       │       ├── data/
│       │       │   ├── model.pth
│       │       │   └── pickle_module_info.txt
│       │       └── requirements.txt
│       └── tags/
└── models/                     # Model Registry
    └── sustainability_bnn_api/
        ├── version-1/
        └── version-2/
```

---

## 8. Optuna Hyperparameter Tuning

### 8.1 TuningConfig

```python
@dataclass
class TuningConfig:
    """Configuración para tuning de hiperparámetros."""

    n_trials: int = 50
    timeout: int = None  # Segundos, None = sin límite
    direction: str = "minimize"  # minimize val_loss
    study_name: str = "hp_optimization"

    # Search space
    n_layers_range: Tuple[int, int] = (1, 4)
    hidden_dim_range: Tuple[int, int] = (16, 128)
    dropout_range: Tuple[float, float] = (0.1, 0.5)
    lr_range: Tuple[float, float] = (1e-5, 1e-2)
    batch_sizes: List[int] = field(
        default_factory=lambda: [16, 32, 64, 128]
    )

    # Training per trial
    epochs_per_trial: int = 50
    early_stopping_patience: int = 5
```

### 8.2 OptunaHyperparameterTuner

```python
class OptunaHyperparameterTuner:
    """Optimizador de hiperparámetros con Optuna."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "bnn",
        config: TuningConfig = None
    ):
        self.X = X
        self.y = y
        self.model_type = model_type
        self.config = config or TuningConfig()

    def _objective(self, trial: optuna.Trial) -> float:
        """
        Función objetivo para Optuna.

        Args:
            trial: Trial de Optuna

        Returns:
            Validation loss (a minimizar)
        """
        # Sample hyperparameters
        n_layers = trial.suggest_int(
            "n_layers",
            *self.config.n_layers_range
        )

        hidden_dims = [
            trial.suggest_int(
                f"hidden_dim_{i}",
                *self.config.hidden_dim_range
            )
            for i in range(n_layers)
        ]

        dropout = trial.suggest_float(
            "dropout",
            *self.config.dropout_range
        )

        lr = trial.suggest_float(
            "learning_rate",
            *self.config.lr_range,
            log=True  # Log scale
        )

        batch_size = trial.suggest_categorical(
            "batch_size",
            self.config.batch_sizes
        )

        # Create model
        model = create_model(
            self.model_type,
            input_dim=self.X.shape[1],
            hidden_dims=hidden_dims,
            dropout_rate=dropout
        )

        # Training config
        train_config = TrainingConfig(
            epochs=self.config.epochs_per_trial,
            batch_size=batch_size,
            learning_rate=lr,
            early_stopping_patience=self.config.early_stopping_patience,
            device='cpu',
            verbose=False
        )

        # Prepare data
        train_loader, val_loader, _ = prepare_data_loaders(
            self.X, self.y, batch_size=batch_size
        )

        # Train
        trainer = Trainer(model, train_config)
        history = trainer.fit(train_loader, val_loader)

        return history.best_val_loss

    def tune(self) -> Tuple[Dict, optuna.Study]:
        """
        Ejecuta optimización de hiperparámetros.

        Returns:
            best_params: Mejores hiperparámetros encontrados
            study: Objeto Study de Optuna
        """
        # Create study with MLFlow callback
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction
        )

        # MLFlow callback para logging automático
        mlflow_callback = MLflowCallback(
            tracking_uri="http://127.0.0.1:5000",
            metric_name="val_loss"
        )

        # Run optimization
        study.optimize(
            self._objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            callbacks=[mlflow_callback]
        )

        return study.best_params, study
```

### 8.3 Visualización de Resultados

```python
# Optuna provee visualizaciones built-in
import optuna.visualization as viz

# Historia de optimización
fig = viz.plot_optimization_history(study)

# Importancia de hiperparámetros
fig = viz.plot_param_importances(study)

# Parallel coordinate plot
fig = viz.plot_parallel_coordinate(study)

# Slice plot
fig = viz.plot_slice(study)
```

---

## 9. Containerización

### 9.1 Dockerfile API

```dockerfile
# docker/api/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY data/ ./data/
COPY config/ ./config/

# Environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 9.2 Dockerfile MLFlow

```dockerfile
# docker/mlflow/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install MLFlow
RUN pip install --no-cache-dir mlflow==2.9.0

# Create directories for data persistence
RUN mkdir -p /mlflow/mlruns /mlflow/artifacts

# Environment variables
ENV MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlflow/artifacts

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run MLFlow server
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:///mlflow/mlflow.db", \
     "--default-artifact-root", "/mlflow/artifacts"]
```

### 9.3 Docker Compose

```yaml
# docker/docker-compose.yml
version: '3.8'

services:
  mlflow:
    build:
      context: ..
      dockerfile: docker/mlflow/Dockerfile
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  api:
    build:
      context: ..
      dockerfile: docker/api/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - PYTHONUNBUFFERED=1
    depends_on:
      mlflow:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

volumes:
  mlflow-data:
    driver: local
```

### 9.4 Comandos Docker

```bash
# Build images
docker-compose -f docker/docker-compose.yml build

# Start services
docker-compose -f docker/docker-compose.yml up -d

# View logs
docker-compose -f docker/docker-compose.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.yml down

# Stop and remove volumes
docker-compose -f docker/docker-compose.yml down -v
```

---

## 10. Infraestructura AWS

### 10.1 ECR Setup

```bash
#!/bin/bash
# infrastructure/ecr/setup-ecr.sh

AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create repositories
for repo in dl-bayesian-api dl-bayesian-mlflow; do
    aws ecr create-repository \
        --repository-name $repo \
        --region $AWS_REGION \
        --image-scanning-configuration scanOnPush=true \
        --encryption-configuration encryptionType=AES256
done

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag and push images
docker tag dl-bayesian-api:latest \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/dl-bayesian-api:latest

docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/dl-bayesian-api:latest
```

### 10.2 ECS Task Definition

```json
{
  "family": "dl-bayesian-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "ACCOUNT.dkr.ecr.REGION.amazonaws.com/dl-bayesian-api:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "MLFLOW_TRACKING_URI",
          "value": "http://mlflow.internal:5000"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/dl-bayesian-api",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

### 10.3 Arquitectura AWS

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              AWS Cloud                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐        ┌─────────────────────────────────────┐    │
│  │   Route 53      │───────▶│        Application Load Balancer    │    │
│  │   (DNS)         │        │        (HTTPS termination)          │    │
│  └─────────────────┘        └──────────────┬──────────────────────┘    │
│                                            │                            │
│                              ┌─────────────┴─────────────┐              │
│                              │                           │              │
│                              ▼                           ▼              │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                         VPC                                        │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │                    Private Subnet                            │  │  │
│  │  │  ┌─────────────────┐        ┌─────────────────────────────┐  │  │  │
│  │  │  │  ECS Fargate    │        │      ECS Fargate            │  │  │  │
│  │  │  │  ┌───────────┐  │        │      ┌───────────────────┐  │  │  │  │
│  │  │  │  │    API    │  │◀──────▶│      │     MLFlow        │  │  │  │  │
│  │  │  │  │  :8000    │  │        │      │     :5000         │  │  │  │  │
│  │  │  │  └───────────┘  │        │      └───────────────────┘  │  │  │  │
│  │  │  └─────────────────┘        └─────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  │                                    │                               │  │
│  │                                    ▼                               │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │                   EFS (Elastic File System)                  │  │  │
│  │  │                   - MLFlow artifacts                         │  │  │
│  │  │                   - Model storage                            │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌─────────────────┐        ┌─────────────────┐                        │
│  │      ECR        │        │   CloudWatch    │                        │
│  │  (Images)       │        │   (Logs)        │                        │
│  └─────────────────┘        └─────────────────┘                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 11. CI/CD Pipeline

### 11.1 CI Pipeline (ci.yml)

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install linters
        run: pip install ruff black isort

      - name: Run ruff
        run: ruff check src/ tests/

      - name: Check black formatting
        run: black --check src/ tests/

      - name: Check import sorting
        run: isort --check-only src/ tests/

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov pytest-asyncio

      - name: Run tests
        run: |
          pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          files: ./coverage.xml

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build API image
        uses: docker/build-push-action@v5
        with:
          context: .
          file: docker/api/Dockerfile
          push: false
          tags: dl-bayesian-api:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### 11.2 CD Pipeline (cd.yml)

```yaml
name: CD

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

env:
  AWS_REGION: us-east-1
  ECR_REPOSITORY: dl-bayesian-api
  ECS_SERVICE: dl-bayesian-api
  ECS_CLUSTER: dl-bayesian
  CONTAINER_NAME: api

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build and push image
        id: build-image
        env:
          ECR_REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          IMAGE_TAG: ${{ github.sha }}
        run: |
          docker build -t $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG \
            -f docker/api/Dockerfile .
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG
          echo "image=$ECR_REGISTRY/$ECR_REPOSITORY:$IMAGE_TAG" >> $GITHUB_OUTPUT

      - name: Deploy to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v1
        with:
          task-definition: infrastructure/ecs/task-definition-api.json
          service: ${{ env.ECS_SERVICE }}
          cluster: ${{ env.ECS_CLUSTER }}
          wait-for-service-stability: true
```

---

## 12. Seguridad

### 12.1 Autenticación y Autorización

```python
# Ejemplo de middleware de autenticación (no implementado por defecto)
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key
```

### 12.2 Validación de Entrada

```python
# Pydantic maneja validación automáticamente
class PredictionInput(BaseModel):
    sst_c: float = Field(..., ge=0, le=40)  # Rangos válidos
    # ... otros campos con validación

    @validator('sst_c')
    def validate_sst(cls, v):
        if v < 0 or v > 40:
            raise ValueError('SST debe estar entre 0 y 40°C')
        return v
```

### 12.3 Headers de Seguridad

```python
# Middleware de seguridad
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware

# Solo hosts confiables
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "*.example.com"]
)

# HTTPS en producción
if settings.ENVIRONMENT == "production":
    app.add_middleware(HTTPSRedirectMiddleware)
```

### 12.4 Variables de Entorno

```bash
# .env.example
# NUNCA commitear .env con valores reales

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secret-api-key

# MLFlow
MLFLOW_TRACKING_URI=http://127.0.0.1:5000

# AWS (solo producción)
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION=us-east-1

# Database (si aplica)
DATABASE_URL=
```

---

## 13. Monitoreo y Logging

### 13.1 Configuración de Logging

```python
import logging
import sys
from datetime import datetime

def setup_logging(level: str = "INFO"):
    """Configura logging para la aplicación."""

    # Format
    log_format = (
        "%(asctime)s | %(levelname)8s | %(name)s | "
        "%(filename)s:%(lineno)d | %(message)s"
    )

    # Handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format))

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))
    root_logger.addHandler(handler)

    # Reduce noise from libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
```

### 13.2 Métricas de la API

```python
from prometheus_client import Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

# Métricas
REQUEST_COUNT = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'Request latency in seconds',
    ['method', 'endpoint']
)

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()

        response = await call_next(request)

        latency = time.time() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(latency)

        return response
```

### 13.3 Health Checks

```python
@router.get("/health")
async def health_check():
    """Health check completo."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {
            "api": check_api_health(),
            "mlflow": check_mlflow_health(),
            "model": check_model_health()
        }
    }

def check_mlflow_health() -> str:
    try:
        client = mlflow.tracking.MlflowClient()
        client.search_experiments(max_results=1)
        return "healthy"
    except Exception:
        return "unhealthy"
```

---

## 14. Testing

### 14.1 Estructura de Tests

```
tests/
├── __init__.py
├── conftest.py           # Fixtures compartidos
├── test_api.py           # Tests de API
├── test_mlops.py         # Tests de MLOps
├── test_models.py        # Tests de modelos
└── test_integration.py   # Tests de integración
```

### 14.2 Fixtures

```python
# tests/conftest.py
import pytest
import numpy as np
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    """Cliente de test para la API."""
    from src.api.main import app
    return TestClient(app)

@pytest.fixture
def sample_data():
    """Datos de muestra para tests."""
    np.random.seed(42)
    X = np.random.randn(100, 10).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)
    return X, y

@pytest.fixture
def valid_prediction_input():
    """Input válido para predicción."""
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
```

### 14.3 Tests de API

```python
# tests/test_api.py
class TestHealthEndpoints:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data

class TestPredictEndpoints:
    def test_predict_valid_input(self, client, valid_prediction_input):
        response = client.post(
            "/api/v1/predict",
            json=valid_prediction_input
        )
        # 200 si modelo cargado, 404/500 si no
        assert response.status_code in [200, 404, 500]

    def test_predict_invalid_input(self, client):
        response = client.post(
            "/api/v1/predict",
            json={"sst_c": "not a number"}
        )
        assert response.status_code == 422
```

### 14.4 Tests de Modelos

```python
# tests/test_models.py
class TestBNN:
    def test_bnn_creation(self):
        from src.deep_learning.models import create_model

        model = create_model("bnn", input_dim=10, hidden_dims=[32, 16])
        assert model is not None

    def test_bnn_uncertainty(self, sample_data):
        import torch
        from src.deep_learning.models import create_model

        X, y = sample_data
        model = create_model("bnn", input_dim=10)

        x_tensor = torch.FloatTensor(X[:5])
        mean, std, samples = model.predict_with_uncertainty(x_tensor)

        assert mean.shape == (5, 1)
        assert std.shape == (5, 1)
        assert torch.all(std >= 0)  # Incertidumbre no negativa
```

### 14.5 Ejecutar Tests

```bash
# Todos los tests
pytest tests/ -v

# Solo tests de API
pytest tests/test_api.py -v

# Con coverage
pytest tests/ -v --cov=src --cov-report=html

# Tests rápidos (excluir integración)
pytest tests/ -v -m "not integration"
```

---

## Apéndices

### A. Glosario

| Término | Definición |
|---------|------------|
| BNN | Bayesian Neural Network - Red neuronal con incertidumbre |
| CPD | Conditional Probability Distribution |
| DAG | Directed Acyclic Graph |
| ECR | Elastic Container Registry (AWS) |
| ECS | Elastic Container Service (AWS) |
| MC Dropout | Monte Carlo Dropout - técnica para incertidumbre |
| MLE | Maximum Likelihood Estimation |
| MLP | Multilayer Perceptron |
| MLOps | Machine Learning Operations |

### B. Referencias

1. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation
2. FastAPI Documentation: https://fastapi.tiangolo.com/
3. MLFlow Documentation: https://mlflow.org/docs/latest/
4. Optuna Documentation: https://optuna.readthedocs.io/
5. pgmpy Documentation: https://pgmpy.org/

### C. Changelog

| Versión | Fecha | Cambios |
|---------|-------|---------|
| 1.0.0 | 2025-02 | Versión inicial con MLOps completo |

---

---

**Autor:** Ing. Ariel Luján Giamportone  
**ORCID:** [0009-0000-1607-9743](https://orcid.org/0009-0000-1607-9743) · [LinkedIn](https://www.linkedin.com/in/agiamportone/) · [ResearchGate](https://www.researchgate.net/profile/Ariel-Lujan-Giamportone)

*Documentación técnica del proyecto Fisheries Sustainability MLOps*
