# AGENTS.md - Documentación del Proyecto

## Deep Learning y Redes Bayesianas para Sostenibilidad Pesquera

Este proyecto implementa un sistema completo de análisis causal y predicción de sostenibilidad pesquera utilizando Redes Bayesianas y Deep Learning.

---

## Tabla de Contenidos

1. [Descripción General](#descripción-general)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Instalación](#instalación)
4. [Módulos](#módulos)
   - [Data Loaders](#data-loaders)
   - [Bayesian Networks](#bayesian-networks)
   - [Causal Analysis](#causal-analysis)
   - [Deep Learning](#deep-learning)
   - [Visualization](#visualization)
5. [Notebooks](#notebooks)
6. [Configuración](#configuración)
7. [Testing](#testing)
8. [API Reference](#api-reference)
9. [Ejemplos de Uso](#ejemplos-de-uso)

---

## Descripción General

Este proyecto combina técnicas de:

- **Redes Bayesianas**: Modelado probabilístico de relaciones entre variables pesqueras
- **Análisis Causal**: Inferencia causal con do-calculus y contrafactuales
- **Deep Learning**: Redes neuronales bayesianas (BNN) y VAE causales

### Objetivo Principal

Predecir y analizar la sostenibilidad de operaciones pesqueras considerando:
- Variables ambientales (temperatura, salinidad, clorofila)
- Variables operativas (esfuerzo pesquero, CPUE, flota)
- Variables económicas (costos, precios, márgenes)

---

## Estructura del Proyecto

```
Deep_Learning_Causalidad_RedesBayesianas/
├── AGENTS.md                    # Esta documentación
├── PLAN_MEJORA.md              # Plan de mejora del proyecto
├── README.md                    # README principal
├── requirements.txt             # Dependencias Python
├── setup.py                     # Configuración de instalación
│
├── config/
│   └── config.yaml             # Configuración centralizada
│
├── data/
│   ├── __init__.py
│   ├── loaders.py              # Funciones de carga y generación de datos
│   ├── raw/                    # Datos crudos
│   └── processed/              # Datos procesados
│       ├── fisheries_data.csv
│       └── fisheries_bayesian.csv
│
├── src/
│   ├── __init__.py
│   ├── bayesian/               # Módulo de Redes Bayesianas
│   │   ├── __init__.py
│   │   ├── networks.py         # BayesianSustainabilityModel
│   │   ├── inference.py        # Motor de inferencia
│   │   └── validation.py       # Validación y métricas
│   │
│   ├── causal/                 # Módulo de Análisis Causal
│   │   ├── __init__.py
│   │   ├── dag.py              # CausalDAG, SustainabilityDAG
│   │   └── interventions.py    # CausalInterventions
│   │
│   ├── deep_learning/          # Módulo de Deep Learning
│   │   ├── __init__.py
│   │   ├── models.py           # MLP, BNN, CausalVAE
│   │   └── training.py         # Trainer, EarlyStopping
│   │
│   └── visualization/          # Módulo de Visualización
│       ├── __init__.py
│       └── plots.py            # Funciones de gráficos
│
├── notebooks/
│   ├── 01_EDA_Datos_Pesqueros.ipynb
│   ├── 02_Bayesian_Networks.ipynb (legacy)
│   ├── 03_Causal_Analysis.ipynb (legacy)
│   ├── 04_Deep_Learning_Sustainability.ipynb
│   ├── BayesianNetworks_SostenibilidadPesquera.ipynb
│   └── CausalNetwork_SostenibilidadRAS.ipynb
│
├── tests/
│   ├── test_bayesian.py        # Tests de redes bayesianas
│   └── test_causal.py          # Tests de análisis causal
│
└── outputs/
    ├── models/                 # Modelos guardados
    ├── figures/                # Figuras generadas
    └── reports/                # Reportes
```

---

## Instalación

### Requisitos

- Python 3.10+
- pip

### Instalación de dependencias

```bash
pip install -r requirements.txt
```

### Dependencias principales

```
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
pgmpy>=0.1.24
matplotlib>=3.7.0
seaborn>=0.12.0
networkx>=3.0
scikit-learn>=1.2.0
torch>=2.0.0
PyYAML>=6.0
tqdm>=4.65.0
jupyter>=1.0.0
```

---

## Módulos

### Data Loaders

**Ubicación**: `data/loaders.py`

Funciones para generación y procesamiento de datos de pesquerías.

#### Funciones principales

```python
from data.loaders import (
    generate_synthetic_fisheries_data,
    discretize_for_bayesian,
    prepare_bayesian_dataset
)
```

##### `generate_synthetic_fisheries_data()`

Genera datos sintéticos realistas basados en distribuciones FAO.

```python
df = generate_synthetic_fisheries_data(
    n_samples=1000,
    years=(2010, 2023),
    countries=['Argentina', 'Chile', 'Peru'],
    random_state=42
)
```

**Parámetros:**
- `n_samples`: Número de muestras a generar
- `years`: Tupla con rango de años
- `countries`: Lista de países (opcional)
- `random_state`: Semilla para reproducibilidad

**Retorna:** DataFrame con variables ambientales, operativas y económicas.

##### `discretize_for_bayesian()`

Discretiza variables continuas para uso en redes bayesianas.

```python
df_disc = discretize_for_bayesian(
    df,
    columns=['SST_C', 'CPUE', 'Production_tons'],
    n_bins=3,
    labels=['Bajo', 'Medio', 'Alto']
)
```

##### `prepare_bayesian_dataset()`

Prepara un dataset completo para modelado bayesiano.

```python
df_bayesian = prepare_bayesian_dataset(
    df,
    target='Sustainable',
    features=['SST_C', 'CPUE', 'Fishing_Effort_hours'],
    discretize=True
)
```

---

### Bayesian Networks

**Ubicación**: `src/bayesian/`

#### BayesianSustainabilityModel

Modelo principal de Red Bayesiana para sostenibilidad.

```python
from src.bayesian.networks import BayesianSustainabilityModel

model = BayesianSustainabilityModel(
    target='Sustainable',
    scoring_method='bic',  # 'bic', 'k2', 'bdeu'
    max_indegree=5,
    random_state=42
)

# Ajustar modelo
model.fit(df_discretized)

# Predicción
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Consulta de inferencia
result = model.query(
    variables=['Sustainable'],
    evidence={'CPUE_disc': 'Alto', 'SST_C_disc': 'Medio'}
)

# Comparar métodos de scoring
comparison = model.compare_scoring_methods(df_discretized)
```

**Métodos principales:**

| Método | Descripción |
|--------|-------------|
| `fit(data, predefined_edges, estimator)` | Aprende estructura y parámetros |
| `predict(X)` | Predicción de clase (0/1) |
| `predict_proba(X)` | Probabilidad de clase positiva |
| `query(variables, evidence)` | Inferencia probabilística |
| `get_edges()` | Obtiene aristas del DAG |
| `get_nodes()` | Obtiene nodos del modelo |
| `get_cpd(variable)` | Obtiene CPD de una variable |
| `compare_scoring_methods(data)` | Compara BIC, K2, BDeu |

#### BayesianValidator

Validación y evaluación de modelos bayesianos.

```python
from src.bayesian.validation import BayesianValidator

validator = BayesianValidator(
    target='Sustainable',
    scoring_method='bic',
    random_state=42
)

# Validación train/test
results = validator.train_test_validation(data, test_size=0.2)

# Validación cruzada k-fold
cv_results = validator.cross_validate(data, n_splits=5)

# Análisis de estabilidad estructural (bootstrap)
stability = validator.bootstrap_structure_stability(data, n_iterations=50)

# Comparar métodos de scoring
comparison = validator.compare_scoring_methods(data)
```

**Métricas disponibles:**
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- Matriz de confusión

---

### Causal Analysis

**Ubicación**: `src/causal/`

#### CausalDAG

Definición y manipulación de grafos acíclicos dirigidos causales.

```python
from src.causal.dag import CausalDAG, SustainabilityDAG

# Crear DAG desde aristas
dag = CausalDAG([
    ('Temperature', 'FeedingRate'),
    ('FeedingRate', 'Growth'),
    ('Growth', 'Sustainable')
])

# O añadir aristas individualmente
dag = CausalDAG()
dag.add_edge('A', 'B', strength=0.8, mechanism='Lineal positivo')
dag.add_edges_from([('B', 'C'), ('C', 'D')])

# Validar estructura
validation = dag.validate()
# {'is_valid_dag': True, 'n_nodes': 4, 'n_edges': 3, ...}

# Obtener relaciones
parents = dag.get_parents('Growth')
children = dag.get_children('Temperature')
ancestors = dag.get_ancestors('Sustainable')
descendants = dag.get_descendants('Temperature')
markov_blanket = dag.get_markov_blanket('Growth')

# D-separación
is_sep = dag.is_d_separated('A', 'C', {'B'})

# Comparar con estructura aprendida
comparison = dag.compare_with_learned(learned_edges)
```

#### SustainabilityDAG

DAGs predefinidos para dominios de sostenibilidad.

```python
from src.causal.dag import SustainabilityDAG

# DAG para pesquerías marinas
fisheries_dag = SustainabilityDAG.create('fisheries')

# DAG para sistemas RAS (Recirculating Aquaculture Systems)
ras_dag = SustainabilityDAG.create('ras')

# Obtener pares tratamiento-resultado
pairs = fisheries_dag.get_treatment_outcome_pairs()
```

#### CausalInterventions

Análisis de intervenciones causales (do-calculus).

```python
from src.causal.interventions import CausalInterventions

ci = CausalInterventions(bayesian_model, target='Sustainable')

# Intervención do(X=x)
result = ci.do_intervention(
    treatment='CPUE_disc',
    treatment_value='Alto'
)
print(result.summary())

# Estimar efecto causal promedio (ATE)
ate = ci.estimate_ate(
    treatment='CPUE_disc',
    treatment_value_1='Alto',
    treatment_value_0='Bajo',
    outcome='Sustainable'
)

# Comparar todas las intervenciones posibles
all_interventions = ci.compare_interventions('CPUE_disc')

# Consulta contrafactual
counterfactual = ci.counterfactual_query(
    observed={'CPUE_disc': 'Bajo', 'Sustainable': 0},
    intervention={'CPUE_disc': 'Alto'},
    query_var='Sustainable'
)
```

**InterventionResult:**

```python
@dataclass
class InterventionResult:
    treatment: str
    treatment_value: str
    outcome: str
    baseline_prob: Dict[str, float]
    intervention_prob: Dict[str, float]
    ate: float  # Average Treatment Effect
    relative_effect: float
```

---

### Deep Learning

**Ubicación**: `src/deep_learning/`

#### Modelos disponibles

##### SustainabilityMLP

Red neuronal feedforward para clasificación de sostenibilidad.

```python
from src.deep_learning.models import SustainabilityMLP, ModelConfig

config = ModelConfig(
    input_dim=10,
    hidden_dims=[64, 32, 16],
    output_dim=1,
    dropout=0.3,
    activation='relu',
    use_batch_norm=True
)

model = SustainabilityMLP(config)
```

##### BayesianNeuralNetwork

Red neuronal bayesiana con cuantificación de incertidumbre.

```python
from src.deep_learning.models import BayesianNeuralNetwork

bnn = BayesianNeuralNetwork(
    input_dim=10,
    hidden_dims=[64, 32],
    output_dim=1
)

# Forward pass retorna (logits, kl_divergence)
logits, kl = bnn(x)

# Predicción con incertidumbre
mean, std, samples = bnn.predict_with_uncertainty(x, n_samples=100)

# Pérdida ELBO
loss = bnn.elbo_loss(x, y, n_samples=5, kl_weight=0.01)
```

##### CausalVAE

Variational Autoencoder con estructura causal en el espacio latente.

```python
from src.deep_learning.models import CausalVAE

vae = CausalVAE(
    input_dim=10,
    latent_dim=5,
    hidden_dims=[64, 32],
    causal_order=['var1', 'var2', 'var3', 'var4', 'var5']
)

# Forward pass
x_recon, mu, logvar = vae(x)

# Pérdida VAE
recon_loss, kl_loss = vae.loss_function(x_recon, x, mu, logvar)

# Generar contrafactual
counterfactual = vae.generate_counterfactual(
    x,
    intervention={'latent_idx': 0, 'value': 2.0}
)
```

#### Training Infrastructure

##### Trainer

Clase para entrenamiento con callbacks y logging.

```python
from src.deep_learning.training import (
    Trainer, TrainingConfig, TrainingHistory,
    prepare_data_loaders, train_model, evaluate_model
)

config = TrainingConfig(
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    weight_decay=1e-5,
    early_stopping_patience=10,
    val_split=0.2,
    device='auto',  # 'cuda', 'cpu', o 'auto'
    save_best=True,
    verbose=True
)

trainer = Trainer(model, config)
history = trainer.fit(train_loader, val_loader)

# Predicciones
predictions, probabilities = trainer.predict(test_loader)

# Guardar/cargar modelo
trainer.save_model('outputs/models/model.pt')
trainer.load_model('outputs/models/model.pt')
```

##### Funciones de conveniencia

```python
# Preparar DataLoaders
train_loader, val_loader, test_loader = prepare_data_loaders(
    X, y,
    batch_size=32,
    val_split=0.2,
    test_split=0.1,
    random_state=42
)

# Preparar datos de sostenibilidad
X_scaled, y, feature_names = prepare_sustainability_data(
    df,
    target='Sustainable',
    exclude_cols=['Year', 'Country']
)

# Entrenar modelo (función de conveniencia)
model, history = train_model(model, X, y, config)

# Evaluar modelo
metrics = evaluate_model(model, X_test, y_test, device='cpu')
```

---

### Visualization

**Ubicación**: `src/visualization/plots.py`

```python
from src.visualization.plots import (
    plot_bayesian_network,
    plot_confusion_matrix,
    plot_training_history,
    plot_feature_importance,
    plot_cpd_heatmap,
    plot_intervention_effects
)

# Visualizar red bayesiana
plot_bayesian_network(model, figsize=(12, 8), save_path='network.png')

# Matriz de confusión
plot_confusion_matrix(y_true, y_pred, labels=['No Sostenible', 'Sostenible'])

# Historial de entrenamiento
plot_training_history(history, metrics=['loss', 'accuracy'])

# Importancia de features
plot_feature_importance(model, feature_names)

# CPD como heatmap
plot_cpd_heatmap(model, variable='Sustainable')

# Efectos de intervención
plot_intervention_effects(intervention_results)
```

---

## Notebooks

### 01_EDA_Datos_Pesqueros.ipynb

Análisis exploratorio de datos pesqueros:
- Estadísticas descriptivas
- Distribuciones de variables
- Análisis de correlaciones
- Análisis temporal y geográfico
- Detección de outliers

### 04_Deep_Learning_Sustainability.ipynb

Demostración de modelos de Deep Learning:
- Comparación MLP vs BNN
- Análisis de incertidumbre
- Generación de contrafactuales con VAE
- Visualización de resultados

### BayesianNetworks_SostenibilidadPesquera.ipynb

Red Bayesiana para pesquerías marinas:
- Generación de datos sintéticos
- Aprendizaje de estructura
- Inferencia probabilística
- Validación con métricas

### CausalNetwork_SostenibilidadRAS.ipynb

Análisis causal para sistemas RAS:
- Definición de DAG teórico
- Aprendizaje de estructura
- Análisis de intervenciones
- Comparación con estructura teórica

---

## Configuración

**Ubicación**: `config/config.yaml`

```yaml
# Configuración del proyecto
project:
  name: "Deep Learning y Redes Bayesianas - Sostenibilidad Pesquera"
  version: "1.0.0"
  random_seed: 42

# Configuración de datos
data:
  n_samples: 1000
  test_size: 0.2
  val_size: 0.2
  years: [2010, 2023]

# Configuración de Redes Bayesianas
bayesian:
  scoring_method: "bic"  # bic, k2, bdeu
  max_indegree: 5
  estimator: "mle"  # mle, bayesian
  bootstrap_iterations: 50
  cv_folds: 5

# Configuración de Deep Learning
deep_learning:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.00001
  early_stopping_patience: 10
  hidden_dims: [64, 32, 16]
  dropout: 0.3
  device: "auto"

# Configuración de visualización
visualization:
  figure_size: [10, 8]
  dpi: 100
  style: "seaborn-v0_8-whitegrid"
```

### Uso de configuración

```python
import yaml
from pathlib import Path

config_path = Path('config/config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f)

# Acceder a configuración
seed = config['project']['random_seed']
epochs = config['deep_learning']['epochs']
```

---

## Testing

### Ejecutar tests

```bash
# Todos los tests
pytest tests/ -v

# Solo tests de bayesian
pytest tests/test_bayesian.py -v

# Solo tests de causal
pytest tests/test_causal.py -v

# Con cobertura
pytest tests/ --cov=src --cov-report=html
```

### Estructura de tests

#### test_bayesian.py

```python
class TestBayesianSustainabilityModel:
    def test_model_initialization()
    def test_model_fit()
    def test_model_predict()
    def test_model_predict_proba()
    def test_model_query()
    def test_model_get_edges()
    def test_model_not_fitted_error()
    def test_different_scoring_methods()
    def test_compare_scoring_methods()
    def test_create_sustainability_model()

class TestBayesianValidator:
    def test_validator_initialization()
    def test_train_test_validation()
    def test_cross_validate()
    def test_bootstrap_stability()
    def test_compare_scoring_methods()

class TestValidationResults:
    def test_validation_results_to_dict()
```

#### test_causal.py

```python
class TestCausalRelation:
    def test_relation_creation()
    def test_relation_as_tuple()

class TestCausalDAG:
    def test_dag_initialization()
    def test_dag_with_initial_edges()
    def test_add_edge()
    def test_add_edges_from()
    def test_validate_dag()
    def test_validate_cyclic_graph()
    def test_get_parents()
    def test_get_children()
    def test_get_ancestors()
    def test_get_descendants()
    def test_get_markov_blanket()
    def test_compare_with_learned()
    def test_d_separation()
    def test_to_dot()

class TestSustainabilityDAG:
    def test_fisheries_dag()
    def test_ras_dag()
    def test_invalid_domain()
    def test_get_treatment_outcome_pairs()

class TestInterventionResult:
    def test_intervention_result_creation()
    def test_intervention_result_summary()

class TestCausalInterventionsIntegration:
    def test_do_intervention()
    def test_compare_interventions()
    def test_sensitivity_to_intervention()
```

---

## API Reference

### Imports rápidos

```python
# Data
from data.loaders import (
    generate_synthetic_fisheries_data,
    discretize_for_bayesian,
    prepare_bayesian_dataset
)

# Bayesian
from src.bayesian.networks import BayesianSustainabilityModel
from src.bayesian.validation import BayesianValidator, ValidationResults

# Causal
from src.causal.dag import CausalDAG, SustainabilityDAG
from src.causal.interventions import CausalInterventions, InterventionResult

# Deep Learning
from src.deep_learning.models import (
    SustainabilityMLP,
    BayesianNeuralNetwork,
    CausalVAE,
    ModelConfig
)
from src.deep_learning.training import (
    Trainer,
    TrainingConfig,
    TrainingHistory,
    EarlyStopping,
    prepare_data_loaders,
    prepare_sustainability_data,
    train_model,
    evaluate_model
)

# Visualization
from src.visualization.plots import (
    plot_bayesian_network,
    plot_confusion_matrix,
    plot_training_history
)
```

---

## Ejemplos de Uso

### Ejemplo 1: Pipeline completo de Red Bayesiana

```python
from data.loaders import generate_synthetic_fisheries_data, prepare_bayesian_dataset
from src.bayesian.networks import BayesianSustainabilityModel
from src.bayesian.validation import BayesianValidator

# 1. Generar datos
df = generate_synthetic_fisheries_data(n_samples=1000, random_state=42)
df_bayesian = prepare_bayesian_dataset(df, target='Sustainable')

# 2. Crear y entrenar modelo
model = BayesianSustainabilityModel(target='Sustainable', scoring_method='bic')
model.fit(df_bayesian)

# 3. Validar modelo
validator = BayesianValidator(target='Sustainable')
cv_results = validator.cross_validate(df_bayesian, n_splits=5)

print(f"Accuracy: {cv_results['summary']['accuracy']['mean']:.4f}")
print(f"F1-Score: {cv_results['summary']['f1']['mean']:.4f}")

# 4. Hacer inferencia
result = model.query(
    variables=['Sustainable'],
    evidence={'CPUE_disc': 'Alto', 'Fishing_Effort_hours_disc': 'Bajo'}
)
print(f"P(Sustainable=1 | evidencia) = {result['values'][1]:.4f}")
```

### Ejemplo 2: Análisis de intervención causal

```python
from src.causal.interventions import CausalInterventions

# Crear analizador de intervenciones
ci = CausalInterventions(model.model, target='Sustainable')

# Evaluar efecto de aumentar CPUE
result = ci.do_intervention('CPUE_disc', 'Alto')
print(result.summary())

# Comparar intervenciones
for var in ['CPUE_disc', 'Fishing_Effort_hours_disc']:
    results = ci.compare_interventions(var)
    print(f"\nIntervenciones en {var}:")
    for value, res in results.items():
        print(f"  {value}: ATE = {res.ate:+.4f}")
```

### Ejemplo 3: Deep Learning con incertidumbre

```python
import torch
from src.deep_learning.models import BayesianNeuralNetwork
from src.deep_learning.training import (
    TrainingConfig, Trainer, prepare_sustainability_data, prepare_data_loaders
)

# 1. Preparar datos
X, y, features = prepare_sustainability_data(df, target='Sustainable')
train_loader, val_loader, _ = prepare_data_loaders(X, y, batch_size=32)

# 2. Crear modelo BNN
bnn = BayesianNeuralNetwork(
    input_dim=X.shape[1],
    hidden_dims=[64, 32],
    output_dim=1
)

# 3. Entrenar
config = TrainingConfig(epochs=50, learning_rate=0.001)
trainer = Trainer(bnn, config)
history = trainer.fit(train_loader, val_loader)

# 4. Predicción con incertidumbre
X_test = torch.FloatTensor(X[:10])
mean, std, samples = bnn.predict_with_uncertainty(X_test, n_samples=100)

for i in range(len(mean)):
    print(f"Muestra {i}: P(Sust=1) = {mean[i]:.3f} +/- {std[i]:.3f}")
```

### Ejemplo 4: Contrafactuales con VAE

```python
from src.deep_learning.models import CausalVAE

# Crear VAE causal
vae = CausalVAE(
    input_dim=10,
    latent_dim=5,
    hidden_dims=[64, 32]
)

# Entrenar VAE (similar a ejemplo anterior)
# ...

# Generar contrafactual
# "¿Qué pasaría si el CPUE hubiera sido alto?"
original = torch.FloatTensor(X[0:1])
counterfactual = vae.generate_counterfactual(
    original,
    intervention={'latent_idx': 0, 'value': 2.0}  # Aumentar primera variable latente
)

print("Original:", original)
print("Contrafactual:", counterfactual)
```

---

## Notas Técnicas

### Compatibilidad de versiones

El proyecto es compatible con:
- **pgmpy 1.0.0+**: Usa `DiscreteBayesianNetwork` y nuevos nombres de scoring (`BIC`, `K2`, `BDeu`)
- **networkx 3.0+**: Usa `nx.is_d_separator()` para d-separación
- **PyTorch 2.0+**: Para modelos de Deep Learning

### Consideraciones de rendimiento

- Para datasets grandes (>10,000 muestras), considerar usar `max_indegree` bajo
- El bootstrap de estructura puede ser lento; reducir `n_iterations` si es necesario
- Para Deep Learning, usar GPU si está disponible (`device='cuda'`)

### Reproducibilidad

Todos los componentes aceptan `random_state` o `random_seed` para garantizar reproducibilidad:

```python
# Data
df = generate_synthetic_fisheries_data(random_state=42)

# Bayesian
model = BayesianSustainabilityModel(random_state=42)

# Deep Learning
torch.manual_seed(42)
config = TrainingConfig(...)
```

---

## Contribuciones

Para contribuir al proyecto:

1. Fork del repositorio
2. Crear rama de feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -m 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

### Guías de estilo

- Código en español para docstrings y comentarios
- Type hints en todas las funciones
- Tests para toda nueva funcionalidad
- Docstrings en formato Google

---

## Licencia

Este proyecto está bajo licencia MIT.

---

## Contacto

Para preguntas o sugerencias, abrir un issue en el repositorio.

---

*Documentación generada para el proyecto Deep Learning y Redes Bayesianas para Sostenibilidad Pesquera.*
*Última actualización: 2026-01-28*
