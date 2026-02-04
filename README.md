# DL_Bayesian: Deep Learning y Redes Bayesianas para Sostenibilidad Pesquera

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)](https://fastapi.tiangolo.com)
[![MLFlow](https://img.shields.io/badge/MLFlow-2.9-orange.svg)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistema completo de Machine Learning para predecir la sostenibilidad de operaciones pesqueras utilizando **Redes Bayesianas**, **Deep Learning** y un stack **MLOps** completo.

---

## Tabla de Contenidos

1. [Descripcion del Proyecto](#1-descripcion-del-proyecto)
2. [Para Que Sirve](#2-para-que-sirve)
3. [Arquitectura del Sistema](#3-arquitectura-del-sistema)
4. [Estado de Desarrollo](#4-estado-de-desarrollo)
5. [Instalacion](#5-instalacion)
6. [Guia de Uso Rapido](#6-guia-de-uso-rapido)
7. [Manual de Uso Completo](#7-manual-de-uso-completo)
8. [Documentacion Tecnica](#8-documentacion-tecnica)
9. [Datos y Referencias](#9-datos-y-referencias)
10. [Estructura del Proyecto](#10-estructura-del-proyecto)
11. [API Reference](#11-api-reference)
12. [Contribuir](#12-contribuir)

---

## 1. Descripcion del Proyecto

### Que es DL_Bayesian?

DL_Bayesian es un sistema de inteligencia artificial que predice si una operacion pesquera es **sostenible** o **no sostenible** basandose en:

- **Variables ambientales**: temperatura del agua, salinidad, pH, clorofila
- **Variables operativas**: tamano de flota, esfuerzo pesquero, consumo de combustible
- **Variables economicas**: precio del pescado, costos operativos

### Caracteristicas Principales

| Caracteristica | Descripcion |
|----------------|-------------|
| **Redes Bayesianas** | Modelado probabilistico con estructura causal aprendida automaticamente |
| **Deep Learning** | Red Neuronal Bayesiana (BNN) que cuantifica incertidumbre en predicciones |
| **API REST** | FastAPI con endpoints para prediccion, entrenamiento y gestion de modelos |
| **MLOps** | MLFlow para tracking de experimentos, versionado y registro de modelos |
| **Hyperparameter Tuning** | Optuna para optimizacion automatica de hiperparametros |
| **Interfaz Web** | Dashboard HTML para hacer predicciones sin codigo |

### Proposito Dual

Este proyecto tiene dos objetivos:

1. **Aplicacion Profesional**: Sistema End-to-End con MLOps completo para deploy en produccion (AWS ECS)
2. **Material Educativo**: Recurso de aprendizaje sobre Deep Learning y Redes Bayesianas para la comunidad de IA en pesquerias

---

## 2. Para Que Sirve

### Casos de Uso

#### 1. Evaluacion de Sostenibilidad
```
Entrada: Datos de una operacion pesquera
Salida: Prediccion (Sostenible/No Sostenible) + Probabilidad + Incertidumbre
```

#### 2. Analisis de Escenarios
```
Pregunta: "Si reduzco el esfuerzo pesquero en 20%, como cambia la probabilidad de sostenibilidad?"
Respuesta: Analisis causal con do-calculus
```

#### 3. Toma de Decisiones
```
Uso: Reguladores pesqueros, empresas de pesca, ONGs ambientales
Beneficio: Decisiones basadas en datos con cuantificacion de incertidumbre
```

### Flujo de Trabajo Tipico

```
1. Usuario ingresa datos → 2. Modelo procesa → 3. Prediccion con incertidumbre
                                                         ↓
4. Dashboard muestra resultado ← 5. Decision informada
```

---

## 3. Arquitectura del Sistema

### Diagrama de Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                         USUARIO                                  │
│    (Navegador Web / API Client / Jupyter Notebook)              │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CAPA DE PRESENTACION                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Dashboard  │  │  Formulario │  │  Swagger UI (API Docs)  │  │
│  │  (HTML/JS)  │  │  Prediccion │  │  /docs                  │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CAPA DE API (FastAPI)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────┐  │
│  │ /health  │ │ /predict │ │ /train   │ │ /models /experiments│  │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE SERVICIOS                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │  ModelService   │  │  MLFlowService  │  │ TrainingService │  │
│  │  (Inferencia)   │  │  (Tracking)     │  │ (Entrenamiento) │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE ML/AI                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Bayesian   │  │     BNN      │  │    Causal VAE        │   │
│  │   Network    │  │   (PyTorch)  │  │    (PyTorch)         │   │
│  │   (pgmpy)    │  │              │  │                      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CAPA DE DATOS                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Synthetic  │  │  Processed   │  │    MLFlow Artifacts  │   │
│  │   Generator  │  │    CSV       │  │    (Models, Metrics) │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Stack Tecnologico

| Capa | Tecnologia | Version | Proposito |
|------|------------|---------|-----------|
| **Frontend** | HTML/CSS/JS | - | Interfaz de usuario |
| **API** | FastAPI | 0.104+ | REST API |
| **ML Framework** | PyTorch | 2.0+ | Deep Learning |
| **Bayesian** | pgmpy | 0.1.24+ | Redes Bayesianas |
| **MLOps** | MLFlow | 2.9+ | Experiment tracking |
| **HP Tuning** | Optuna | 3.4+ | Optimizacion |
| **Validacion** | Pydantic | 2.5+ | Schemas |
| **Data** | Pandas/NumPy | 2.0+/1.24+ | Procesamiento |

---

## 4. Estado de Desarrollo

### Fases Completadas

| Fase | Estado | Descripcion |
|------|--------|-------------|
| **A** Fundamentos | ✅ 100% | Estructura base, redes bayesianas, validacion |
| **B** Datos | ✅ 100% | Generador sintetico, loaders, preprocesamiento |
| **C** Modularizacion | ✅ 100% | Organizacion en modulos, tests, config |
| **D** Deep Learning | ✅ 100% | BNN, MLP, CausalVAE, training pipeline |
| **E** MLOps | ✅ 100% | MLFlow, Optuna, FastAPI, Docker, CI/CD |

### Componentes Implementados

```
✅ Redes Bayesianas (pgmpy)
   ├── Aprendizaje de estructura (Hill Climb Search)
   ├── Estimacion de parametros (MLE, Bayesian)
   ├── Inferencia (Variable Elimination)
   └── Validacion cruzada

✅ Deep Learning (PyTorch)
   ├── MLP (Multi-Layer Perceptron)
   ├── BNN (Bayesian Neural Network)
   ├── CausalVAE (Variational Autoencoder)
   └── Training con Early Stopping

✅ MLOps
   ├── MLFlow Tracking & Registry
   ├── Optuna Hyperparameter Tuning
   ├── FastAPI REST API
   ├── Docker (Dockerfile, docker-compose)
   └── GitHub Actions (CI/CD)

✅ Frontend
   ├── Dashboard principal
   ├── Formulario de prediccion
   ├── Gestion de modelos
   └── Visualizacion de experimentos
```

### Lo que falta (Roadmap)

| Componente | Estado | Prioridad |
|------------|--------|-----------|
| Deploy AWS (ECR/ECS) | Pendiente | Media |
| Datos reales OWID | Pendiente | Alta |
| Notebooks educativos | Pendiente | Alta |
| Autenticacion API | Pendiente | Baja |
| Monitoring (Prometheus) | Pendiente | Baja |

---

## 5. Instalacion

### Requisitos Previos

- **Python**: 3.10, 3.11 o 3.12 (NO usar 3.13)
- **pip**: Version actualizada
- **Git**: Para clonar el repositorio
- **Docker** (opcional): Para despliegue containerizado

### Paso 1: Clonar Repositorio

```bash
git clone https://github.com/arielgiamportone/DL_Bayesian.git
cd DL_Bayesian
```

### Paso 2: Crear Entorno Virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install optuna-integration[mlflow]  # Dependencia adicional
```

### Paso 4: Verificar Instalacion

```bash
python test_installation.py
```

Resultado esperado:
```
✓✓✓ Todas las dependencias estan instaladas correctamente! ✓✓✓
```

---

## 6. Guia de Uso Rapido

### Opcion A: Usar la API Web (Recomendado)

**1. Iniciar MLFlow (tracking de experimentos):**
```bash
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlruns/mlflow.db --default-artifact-root ./mlruns/artifacts
```

**2. Entrenar un modelo:**
```bash
python scripts/train_api_model.py
```

**3. Iniciar la API:**
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```

**4. Abrir en navegador:**
- Dashboard: http://127.0.0.1:8000
- Prediccion: http://127.0.0.1:8000/predict
- API Docs: http://127.0.0.1:8000/docs
- MLFlow: http://127.0.0.1:5000

### Opcion B: Usar desde Python

```python
# Prediccion simple con Red Bayesiana
from data.loaders import generate_synthetic_fisheries_data, prepare_bayesian_dataset
from src.bayesian.networks import BayesianSustainabilityModel

# Generar datos
df = generate_synthetic_fisheries_data(n_samples=1000)
df_bayesian = prepare_bayesian_dataset(df, target='Sustainable')

# Entrenar modelo
model = BayesianSustainabilityModel(target='Sustainable')
model.fit(df_bayesian)

# Hacer prediccion
result = model.query(
    variables=['Sustainable'],
    evidence={'CPUE_disc': 'Alto', 'Fishing_Effort_hours_disc': 'Bajo'}
)
print(f"P(Sostenible) = {result['values'][1]:.2%}")
```

### Opcion C: Usar Notebooks Jupyter

```bash
jupyter notebook
# Abrir: BayesianNetworks_SostenibilidadPesquera.ipynb
```

---

## 7. Manual de Uso Completo

### 7.1 Usando la Interfaz Web

#### Dashboard (http://127.0.0.1:8000)

El dashboard muestra:
- **Estado del sistema**: Verde = funcionando
- **Conexion MLFlow**: Verde = conectado
- **Modelo cargado**: Verde = listo para predicciones
- **Prediccion rapida**: Formulario con campos esenciales

#### Formulario de Prediccion (http://127.0.0.1:8000/predict)

**Campos a completar:**

| Campo | Descripcion | Rango Tipico |
|-------|-------------|--------------|
| Temperatura (°C) | Temperatura superficial del mar | 15-30 |
| Salinidad (ppt) | Salinidad en partes por mil | 30-40 |
| Clorofila (mg/m³) | Concentracion de clorofila | 0.1-10 |
| pH | Nivel de acidez del agua | 7.5-8.5 |
| Tamano de Flota | Numero de embarcaciones | 10-500 |
| Esfuerzo (horas) | Horas totales de pesca | 100-5000 |
| Combustible (L) | Consumo total | 500-20000 |
| Precio Pescado (USD/ton) | Precio de venta | 500-5000 |
| Precio Combustible (USD/L) | Costo del combustible | 0.5-2.5 |
| Costo Operativo (USD) | Costos totales | 5000-100000 |

**Interpretacion del resultado:**

```
Prediccion: SOSTENIBLE / NO SOSTENIBLE
Probabilidad: 0-100%
Intervalo de Confianza: [min%, max%]  <- Rango donde esta el valor real
Incertidumbre: 0-100%  <- Que tan seguro esta el modelo
```

### 7.2 Usando la API Programaticamente

#### Hacer una Prediccion

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sst_c": 25.0,
    "salinity_ppt": 35.0,
    "chlorophyll_mg_m3": 2.5,
    "ph": 8.1,
    "fleet_size": 150,
    "fishing_effort_hours": 1200,
    "fuel_consumption_l": 5000,
    "fish_price_usd_ton": 2500,
    "fuel_price_usd_l": 1.2,
    "operating_cost_usd": 15000
  }'
```

**Respuesta:**
```json
{
  "prediction": 1,
  "probability": 0.656,
  "confidence_interval": [0.48, 0.83],
  "uncertainty": 0.088,
  "model_used": "sustainability_bnn_api",
  "model_version": "Production",
  "inference_time_ms": 295
}
```

#### Iniciar Entrenamiento

```bash
curl -X POST http://127.0.0.1:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "bnn",
    "epochs": 100,
    "batch_size": 32,
    "register_model": true
  }'
```

#### Listar Modelos Registrados

```bash
curl http://127.0.0.1:8000/api/v1/models
```

### 7.3 Entrenamiento de Modelos

#### Entrenamiento Basico con MLFlow

```bash
python scripts/train_with_mlflow.py \
  --model-type bnn \
  --epochs 100 \
  --hidden-dims 64,32 \
  --register
```

**Parametros disponibles:**
- `--model-type`: mlp, bnn (default: bnn)
- `--epochs`: Numero de epocas (default: 100)
- `--hidden-dims`: Capas ocultas (default: 64,32)
- `--learning-rate`: Tasa de aprendizaje (default: 0.001)
- `--batch-size`: Tamano del batch (default: 32)
- `--register`: Registrar modelo en MLFlow

#### Hyperparameter Tuning con Optuna

```bash
python scripts/tune_hyperparams.py \
  --model-type bnn \
  --n-trials 50 \
  --register
```

### 7.4 Explorando MLFlow

**Abrir MLFlow UI:** http://127.0.0.1:5000

**Que puedes ver:**
1. **Experiments**: Grupos de entrenamientos
2. **Runs**: Cada ejecucion individual
3. **Metrics**: Graficos de loss, accuracy, etc.
4. **Parameters**: Hiperparametros usados
5. **Artifacts**: Modelos guardados
6. **Models**: Registro de modelos con versiones

**Promover modelo a produccion:**
1. Ir a "Models" en MLFlow
2. Seleccionar modelo
3. Click en version
4. "Stage" → "Production"

---

## 8. Documentacion Tecnica

### 8.1 Modelos de Machine Learning

#### Bayesian Neural Network (BNN)

```python
class BayesianNeuralNetwork(nn.Module):
    """
    Red neuronal con pesos probabilisticos.

    En lugar de pesos fijos, cada peso tiene una distribucion:
    w ~ N(mu, softplus(rho))

    Ventajas:
    - Cuantifica incertidumbre en predicciones
    - Mas robusto a overfitting
    - Ideal para decisiones criticas
    """
```

**Arquitectura:**
```
Input (10) → BayesianLinear(64) → ReLU → BayesianLinear(32) → ReLU → BayesianLinear(1) → Sigmoid
```

**Funcion de perdida (ELBO):**
```
ELBO = -E[log p(y|x,w)] + KL(q(w)||p(w))
     = Reconstruction Loss + KL Divergence
```

#### Red Bayesiana (pgmpy)

```python
class BayesianSustainabilityModel:
    """
    Modelo grafico probabilistico que representa relaciones causales.

    Estructura aprendida automaticamente con Hill Climb Search.
    Parametros estimados con Maximum Likelihood.
    """
```

**Grafo Causal Aprendido:**
```
SST_C → Fishing_Effort
Chlorophyll → CPUE
CPUE → Sustainable
Fishing_Effort → Sustainable
Operating_Cost → Sustainable
```

### 8.2 Pipeline de Datos

```python
# Generacion de datos sinteticos
def generate_synthetic_fisheries_data(n_samples=1000):
    """
    Genera datos realistas simulando operaciones pesqueras.

    Variables generadas:
    - Ambientales: SST, salinidad, pH, clorofila
    - Operativas: flota, esfuerzo, combustible
    - Economicas: precios, costos
    - Derivadas: CPUE, score de sostenibilidad

    La variable target 'Sustainable' se calcula como:
    Sustainable = f(CPUE, esfuerzo, acuicultura, costos) > threshold
    """
```

### 8.3 API Endpoints

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/health` | GET | Estado del sistema |
| `/health/live` | GET | Liveness probe (K8s) |
| `/health/ready` | GET | Readiness probe (K8s) |
| `/api/v1/predict` | POST | Hacer prediccion |
| `/api/v1/predict/batch` | POST | Predicciones en lote |
| `/api/v1/train` | POST | Iniciar entrenamiento |
| `/api/v1/train/status/{job_id}` | GET | Estado del entrenamiento |
| `/api/v1/experiments` | GET | Listar experimentos |
| `/api/v1/experiments/{id}/runs` | GET | Runs de un experimento |
| `/api/v1/models` | GET | Listar modelos registrados |
| `/api/v1/models/{name}/versions/{v}/stage` | POST | Cambiar stage |
| `/api/v1/models/cache/clear` | POST | Limpiar cache |

### 8.4 Configuracion

**config/config.yaml:**
```yaml
project:
  name: "DL_Bayesian"
  version: "1.0.0"

data:
  synthetic_samples: 1000
  test_size: 0.2
  random_state: 42

bayesian:
  target_variable: "Sustainable"
  scoring_method: "bic"
  max_indegree: 5

deep_learning:
  model_type: "bnn"
  hidden_dims: [64, 32]
  learning_rate: 0.001
  epochs: 100
  batch_size: 32
```

---

## 9. Datos y Referencias

### 9.1 Variables del Modelo

| Variable | Tipo | Unidad | Descripcion |
|----------|------|--------|-------------|
| SST_C | Ambiental | °C | Temperatura superficial del mar |
| Salinity_ppt | Ambiental | ppt | Salinidad |
| Chlorophyll_mg_m3 | Ambiental | mg/m³ | Clorofila (indicador de productividad) |
| pH | Ambiental | - | Acidez del agua |
| Fleet_Size | Operativa | # | Numero de embarcaciones |
| Fishing_Effort_hours | Operativa | horas | Esfuerzo pesquero total |
| Fuel_Consumption_L | Operativa | litros | Consumo de combustible |
| Fish_Price_USD_ton | Economica | USD/ton | Precio de venta |
| Fuel_Price_USD_L | Economica | USD/L | Precio del combustible |
| Operating_Cost_USD | Economica | USD | Costos operativos |
| **Sustainable** | **Target** | **0/1** | **Variable a predecir** |

### 9.2 Fuentes de Datos

**Actualmente:** Datos sinteticos generados algoritmicamente

**Fuentes reales disponibles (para implementar):**
- [Our World in Data - Fish & Overfishing](https://ourworldindata.org/fish-and-overfishing)
- [FAO FishStatJ](https://www.fao.org/fishery/statistics/software/fishstatj)
- [Sea Around Us](https://www.seaaroundus.org/)

### 9.3 Referencias Academicas

1. **Redes Bayesianas:**
   - Koller, D. & Friedman, N. (2009). Probabilistic Graphical Models

2. **Bayesian Deep Learning:**
   - Blundell et al. (2015). Weight Uncertainty in Neural Networks

3. **Sostenibilidad Pesquera:**
   - FAO. (2022). The State of World Fisheries and Aquaculture

---

## 10. Estructura del Proyecto

```
DL_Bayesian/
│
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Pipeline de CI (tests, lint)
│       └── cd.yml                 # Pipeline de CD (deploy AWS)
│
├── config/
│   ├── config.yaml                # Configuracion general
│   └── optuna_config.yaml         # Configuracion Optuna
│
├── data/
│   ├── __init__.py
│   ├── loaders.py                 # Generador de datos, loaders
│   └── processed/
│       ├── fisheries_data.csv     # Datos procesados
│       └── fisheries_bayesian.csv # Datos discretizados
│
├── docker/
│   ├── api/
│   │   └── Dockerfile             # Imagen API
│   ├── mlflow/
│   │   └── Dockerfile             # Imagen MLFlow
│   └── docker-compose.yml         # Orquestacion local
│
├── frontend/
│   ├── static/
│   │   ├── css/style.css          # Estilos
│   │   └── js/app.js              # JavaScript
│   └── templates/
│       ├── index.html             # Dashboard
│       ├── predict.html           # Formulario prediccion
│       ├── models.html            # Gestion modelos
│       └── experiments.html       # Experimentos
│
├── infrastructure/
│   ├── ecr/
│   │   └── setup-ecr.sh           # Script crear repos ECR
│   ├── ecs/
│   │   ├── task-definition-api.json
│   │   └── task-definition-mlflow.json
│   └── README.md                  # Guia de deploy AWS
│
├── scripts/
│   ├── train_with_mlflow.py       # Entrenamiento con tracking
│   ├── train_api_model.py         # Entrenar modelo para API
│   └── tune_hyperparams.py        # Optimizacion con Optuna
│
├── src/
│   ├── __init__.py
│   │
│   ├── api/                       # FastAPI Application
│   │   ├── __init__.py
│   │   ├── main.py                # App principal
│   │   ├── routes/
│   │   │   ├── health.py          # /health
│   │   │   ├── predict.py         # /predict
│   │   │   ├── train.py           # /train
│   │   │   ├── experiments.py     # /experiments
│   │   │   └── models.py          # /models
│   │   ├── schemas/
│   │   │   ├── prediction.py      # Schemas de prediccion
│   │   │   └── training.py        # Schemas de entrenamiento
│   │   └── services/
│   │       ├── model_service.py   # Carga e inferencia
│   │       └── mlflow_service.py  # Queries MLFlow
│   │
│   ├── bayesian/                  # Redes Bayesianas
│   │   ├── __init__.py
│   │   ├── networks.py            # BayesianSustainabilityModel
│   │   ├── inference.py           # BayesianInference
│   │   └── validation.py          # BayesianValidator
│   │
│   ├── causal/                    # Analisis Causal
│   │   ├── __init__.py
│   │   ├── dag.py                 # CausalDAG, SustainabilityDAG
│   │   └── interventions.py       # CausalInterventions
│   │
│   ├── deep_learning/             # Deep Learning
│   │   ├── __init__.py
│   │   ├── models.py              # MLP, BNN, CausalVAE
│   │   └── training.py            # Trainer, TrainingConfig
│   │
│   ├── mlops/                     # MLOps Components
│   │   ├── __init__.py
│   │   ├── mlflow_tracking.py     # MLFlowTracker
│   │   └── optuna_tuning.py       # OptunaHyperparameterTuner
│   │
│   └── visualization/             # Visualizacion
│       ├── __init__.py
│       └── plots.py               # Funciones de graficos
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Fixtures pytest
│   ├── test_bayesian.py           # Tests redes bayesianas
│   ├── test_causal.py             # Tests analisis causal
│   ├── test_api.py                # Tests API
│   └── test_mlops.py              # Tests MLOps
│
├── outputs/
│   ├── figures/                   # Graficos generados
│   ├── models/                    # Modelos guardados
│   └── reports/                   # Reportes
│
├── mlruns/                        # Directorio MLFlow (local)
│
├── .dockerignore
├── .env.example                   # Template variables entorno
├── .gitignore
├── requirements.txt               # Dependencias Python
├── README.md                      # Este archivo
├── PLAN_MEJORA.md                 # Plan de desarrollo
└── test_installation.py           # Verificar instalacion
```

---

## 11. API Reference

### Schemas de Entrada

#### PredictionInput
```json
{
  "sst_c": 25.0,           // float, -5 a 40
  "salinity_ppt": 35.0,    // float, 20 a 45
  "chlorophyll_mg_m3": 2.5, // float, 0 a 100
  "ph": 8.1,               // float, 6 a 9
  "fleet_size": 150,       // int, 1 a 50000
  "fishing_effort_hours": 1200, // float, >= 0
  "fuel_consumption_l": 5000,   // float, >= 0
  "fish_price_usd_ton": 2500,   // float, >= 0
  "fuel_price_usd_l": 1.2,      // float, >= 0
  "operating_cost_usd": 15000,  // float, >= 0
  "model_name": "sustainability_bnn_api",  // opcional
  "model_version": "Production"            // opcional
}
```

#### TrainRequest
```json
{
  "model_type": "bnn",     // "mlp", "bnn", "causal_vae"
  "hidden_dims": [64, 32], // lista de enteros
  "learning_rate": 0.001,  // float
  "epochs": 100,           // int
  "batch_size": 32,        // int
  "run_optuna": false,     // bool
  "register_model": true   // bool
}
```

### Schemas de Salida

#### PredictionOutput
```json
{
  "prediction": 1,         // 0 o 1
  "probability": 0.656,    // 0 a 1
  "confidence_interval": [0.48, 0.83], // opcional
  "uncertainty": 0.088,    // opcional
  "model_used": "sustainability_bnn_api",
  "model_version": "Production",
  "inference_time_ms": 295.0
}
```

---

## 12. Contribuir

### Repositorios

- **Desarrollo personal:** https://github.com/arielgiamportone
- **Comunidad educativa:** https://github.com/PesquerosEnIA

### Como Contribuir

1. Fork del repositorio
2. Crear rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m "Agregar nueva funcionalidad"`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Crear Pull Request

### Ejecutar Tests

```bash
pytest tests/ -v --cov=src
```

### Estilo de Codigo

```bash
# Formatear
black src/ tests/

# Lint
ruff check src/ tests/
```

---

## Licencia

Este proyecto esta bajo licencia MIT.

---

## Contacto

- **Autor:** Ariel Giamportone
- **GitHub:** [@arielgiamportone](https://github.com/arielgiamportone)
- **Comunidad:** [PesquerosEnIA](https://github.com/PesquerosEnIA)

---

*Ultima actualizacion: Febrero 2026*
