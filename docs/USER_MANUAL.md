# Manual de Usuario - DL_Bayesian

## Sistema de Prediccion de Sostenibilidad Pesquera

**Version:** 1.0.0
**Fecha:** Febrero 2025

---

## Tabla de Contenidos

1. [Introduccion](#1-introduccion)
2. [Requisitos del Sistema](#2-requisitos-del-sistema)
3. [Instalacion](#3-instalacion)
4. [Inicio Rapido](#4-inicio-rapido)
5. [Interfaz Web](#5-interfaz-web)
6. [Hacer Predicciones](#6-hacer-predicciones)
7. [Entrenar Modelos](#7-entrenar-modelos)
8. [Gestion de Modelos](#8-gestion-de-modelos)
9. [MLFlow UI](#9-mlflow-ui)
10. [API REST](#10-api-rest)
11. [Solucion de Problemas](#11-solucion-de-problemas)
12. [Preguntas Frecuentes](#12-preguntas-frecuentes)

---

## 1. Introduccion

### Que es DL_Bayesian?

DL_Bayesian es un sistema de Machine Learning que predice la sostenibilidad de operaciones pesqueras. Utiliza:

- **Redes Neuronales Bayesianas (BNN)**: Proporcionan predicciones con nivel de confianza
- **Redes Bayesianas**: Para analisis causal de factores
- **MLFlow**: Para gestionar experimentos y modelos
- **FastAPI**: Como interfaz web y API

### Para que sirve?

El sistema permite:

1. **Predecir** si una operacion pesquera es sostenible
2. **Cuantificar la incertidumbre** de cada prediccion
3. **Entrenar modelos** con nuevos datos
4. **Gestionar versiones** de modelos
5. **Comparar experimentos** de entrenamiento

### Quien puede usarlo?

- Gestores de pesquerias
- Investigadores en sostenibilidad marina
- Cientificos de datos del sector pesquero
- Estudiantes de Machine Learning

---

## 2. Requisitos del Sistema

### Hardware Minimo

| Componente | Requisito |
|------------|-----------|
| CPU | 2 cores |
| RAM | 4 GB |
| Disco | 2 GB libres |
| Red | Conexion a internet |

### Hardware Recomendado

| Componente | Requisito |
|------------|-----------|
| CPU | 4+ cores |
| RAM | 8+ GB |
| Disco | 10 GB libres |
| Red | Conexion estable |

### Software Requerido

- **Python 3.9 o superior**
- **pip** (gestor de paquetes Python)
- **Git** (opcional, para clonar repositorio)
- **Navegador web moderno** (Chrome, Firefox, Edge)

### Sistemas Operativos Soportados

- Windows 10/11
- Ubuntu 20.04+
- macOS 10.15+

---

## 3. Instalacion

### Paso 1: Descargar el Proyecto

**Opcion A: Con Git**
```bash
git clone https://github.com/arielgiamportone/DL_Bayesian.git
cd DL_Bayesian
```

**Opcion B: Descarga directa**
1. Ir a https://github.com/arielgiamportone/DL_Bayesian
2. Click en "Code" > "Download ZIP"
3. Extraer el archivo
4. Abrir terminal en la carpeta extraida

### Paso 2: Crear Entorno Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 4: Verificar Instalacion

```bash
python -c "import torch; import mlflow; import fastapi; print('OK')"
```

Si ve "OK", la instalacion fue exitosa.

---

## 4. Inicio Rapido

### Iniciar el Sistema (3 pasos)

**Terminal 1 - Iniciar MLFlow:**
```bash
mlflow server --host 127.0.0.1 --port 5000
```

**Terminal 2 - Iniciar API:**
```bash
cd DL_Bayesian
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 3 - Entrenar Modelo (primera vez):**
```bash
python scripts/train_api_model.py
```

### Acceder a las Interfaces

| Interfaz | URL | Descripcion |
|----------|-----|-------------|
| Dashboard | http://127.0.0.1:8000 | Interfaz principal |
| API Docs | http://127.0.0.1:8000/docs | Documentacion interactiva |
| MLFlow | http://127.0.0.1:5000 | Gestion de experimentos |

---

## 5. Interfaz Web

### 5.1 Dashboard Principal

Al acceder a http://127.0.0.1:8000 vera el dashboard principal:

```
+------------------------------------------------------------------+
|  DL_Bayesian - Sostenibilidad Pesquera                           |
+------------------------------------------------------------------+
|                                                                   |
|  [Dashboard]  [Prediccion]  [Modelos]  [Experimentos]            |
|                                                                   |
|  +------------------+  +------------------+  +------------------+ |
|  | Estado Sistema   |  | Modelo Activo    |  | Ultima Prediccion| |
|  | OK               |  | BNN v1           |  | Sostenible       | |
|  | API: Activa      |  | Production       |  | Prob: 84.7%      | |
|  | MLFlow: Activo   |  | Accuracy: 85.6%  |  | Conf: Alta       | |
|  +------------------+  +------------------+  +------------------+ |
|                                                                   |
+------------------------------------------------------------------+
```

### 5.2 Navegacion

| Seccion | Funcion |
|---------|---------|
| Dashboard | Vista general del sistema |
| Prediccion | Formulario para nuevas predicciones |
| Modelos | Lista de modelos registrados |
| Experimentos | Historial de entrenamientos |

---

## 6. Hacer Predicciones

### 6.1 Via Interfaz Web

1. Click en "Prediccion" en el menu
2. Complete el formulario con los datos:

**Variables Ambientales:**
| Campo | Descripcion | Rango Tipico |
|-------|-------------|--------------|
| Temperatura (SST) | Temperatura del mar | 15-30 C |
| Salinidad | Concentracion de sal | 30-40 ppt |
| Clorofila | Indicador de productividad | 0.1-10 mg/m3 |
| pH | Acidez del agua | 7.5-8.5 |

**Variables Operativas:**
| Campo | Descripcion | Rango Tipico |
|-------|-------------|--------------|
| Tamano Flota | Numero de barcos | 10-500 |
| Esfuerzo Pesquero | Horas de pesca | 100-5000 h |
| Consumo Combustible | Litros usados | 500-20000 L |

**Variables Economicas:**
| Campo | Descripcion | Rango Tipico |
|-------|-------------|--------------|
| Precio Pescado | USD por tonelada | 500-5000 |
| Precio Combustible | USD por litro | 0.5-2.5 |
| Costo Operativo | USD total | 5000-100000 |

3. Click en "Predecir"

### 6.2 Interpretar Resultados

```
+------------------------------------------------------------------+
|  RESULTADO DE PREDICCION                                          |
+------------------------------------------------------------------+
|                                                                   |
|  Prediccion: SOSTENIBLE                                          |
|                                                                   |
|  Probabilidad: 84.7%                                             |
|  [========================================----]                   |
|                                                                   |
|  Incertidumbre: 8.9% (Confianza ALTA)                           |
|                                                                   |
|  Interpretacion:                                                  |
|  El modelo predice que esta operacion pesquera es sostenible     |
|  con alta confianza. Los factores ambientales y el nivel de      |
|  esfuerzo pesquero estan dentro de rangos sostenibles.           |
|                                                                   |
+------------------------------------------------------------------+
```

**Niveles de Confianza:**

| Incertidumbre | Nivel | Interpretacion |
|---------------|-------|----------------|
| < 10% | Alta | Prediccion muy confiable |
| 10-20% | Media | Prediccion razonablemente confiable |
| 20-30% | Baja | Considerar factores adicionales |
| > 30% | Muy Baja | Revisar datos de entrada |

### 6.3 Via API (curl)

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### 6.4 Via Python

```python
import requests

url = "http://127.0.0.1:8000/api/v1/predict"

data = {
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

response = requests.post(url, json=data)
result = response.json()

print(f"Sostenible: {result['is_sustainable']}")
print(f"Probabilidad: {result['probability']:.1%}")
print(f"Incertidumbre: {result['uncertainty']:.1%}")
```

---

## 7. Entrenar Modelos

### 7.1 Entrenamiento Rapido (Interfaz Web)

1. Ir a "Dashboard" o "Modelos"
2. Click en "Entrenar Nuevo Modelo"
3. Configurar parametros:

| Parametro | Descripcion | Valor por Defecto |
|-----------|-------------|-------------------|
| Tipo de Modelo | BNN o MLP | BNN |
| Capas Ocultas | Arquitectura | [64, 32] |
| Epocas | Iteraciones de entrenamiento | 100 |
| Batch Size | Muestras por iteracion | 32 |
| Learning Rate | Tasa de aprendizaje | 0.001 |
| Muestras | Datos de entrenamiento | 1000 |

4. Click en "Iniciar Entrenamiento"

### 7.2 Entrenamiento via Script

```bash
# Entrenamiento basico
python scripts/train_api_model.py

# Entrenamiento con MLFlow tracking
python scripts/train_with_mlflow.py
```

### 7.3 Entrenamiento via API

```bash
curl -X POST http://127.0.0.1:8000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "bnn",
    "hidden_dims": [64, 32],
    "epochs": 100,
    "batch_size": 32,
    "n_samples": 1000
  }'
```

Respuesta:
```json
{
  "job_id": "train_abc123",
  "status": "started",
  "message": "Training job started"
}
```

### 7.4 Monitorear Entrenamiento

```bash
# Ver estado del job
curl http://127.0.0.1:8000/api/v1/train/jobs/train_abc123
```

O en MLFlow UI: http://127.0.0.1:5000

---

## 8. Gestion de Modelos

### 8.1 Ver Modelos Registrados

**Via Web:**
1. Ir a "Modelos" en el menu
2. Ver lista de modelos con versiones

**Via API:**
```bash
curl http://127.0.0.1:8000/api/v1/models
```

### 8.2 Ciclo de Vida del Modelo

```
   NONE          STAGING        PRODUCTION      ARCHIVED
     |               |               |               |
  [Nuevo]  --->  [Testing]  --->  [En Uso]  --->  [Retirado]
     |               |               |               |
  Recien         Validando      Sirviendo       Historico
  registrado     rendimiento    predicciones    para ref.
```

### 8.3 Promover Modelo a Produccion

**Via Web:**
1. Ir a "Modelos"
2. Encontrar el modelo deseado
3. Click en "Promover a Produccion"
4. Confirmar

**Via API:**
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/models/sustainability_bnn_api/stage" \
  -H "Content-Type: application/json" \
  -d '{"version": "2", "stage": "Production"}'
```

### 8.4 Limpiar Cache de Modelos

Si actualizo un modelo y no ve cambios:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/models/cache/clear
```

---

## 9. MLFlow UI

### 9.1 Acceder a MLFlow

Abrir http://127.0.0.1:5000 en el navegador.

### 9.2 Navegacion de MLFlow

```
+------------------------------------------------------------------+
|  MLFlow                                                           |
+------------------------------------------------------------------+
|  [Experiments]  [Models]  [Artifacts]                            |
|                                                                   |
|  Experiments:                                                     |
|  +-- api_models                                                   |
|  |   +-- Run: bnn_api_model (2025-02-04)                        |
|  |   |   Accuracy: 0.856                                         |
|  |   |   Best Val Loss: 0.423                                    |
|  |   +-- Run: bnn_api_model (2025-02-03)                        |
|  +-- fisheries_sustainability                                     |
|      +-- Run: experiment_001                                      |
|                                                                   |
+------------------------------------------------------------------+
```

### 9.3 Comparar Experimentos

1. Seleccionar multiples runs (checkbox)
2. Click en "Compare"
3. Ver graficos comparativos de metricas

### 9.4 Ver Artefactos del Modelo

1. Click en un run
2. Ir a "Artifacts"
3. Explorar:
   - `model/` - Modelo PyTorch guardado
   - `requirements.txt` - Dependencias
   - `MLmodel` - Metadatos

### 9.5 Registrar Modelo desde MLFlow

1. Ir al run deseado
2. En "Artifacts", click en la carpeta "model"
3. Click en "Register Model"
4. Dar nombre o seleccionar existente

---

## 10. API REST

### 10.1 Documentacion Interactiva

Acceder a http://127.0.0.1:8000/docs para Swagger UI.

Aqui puede:
- Ver todos los endpoints disponibles
- Probar endpoints directamente
- Ver schemas de request/response

### 10.2 Endpoints Principales

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | /health | Estado del sistema |
| GET | /health/ready | Listo para recibir requests |
| POST | /api/v1/predict | Hacer prediccion |
| POST | /api/v1/train | Iniciar entrenamiento |
| GET | /api/v1/train/jobs | Listar trabajos |
| GET | /api/v1/models | Listar modelos |
| GET | /api/v1/experiments | Listar experimentos |

### 10.3 Codigos de Estado

| Codigo | Significado |
|--------|-------------|
| 200 | Exito |
| 202 | Aceptado (trabajo en segundo plano) |
| 400 | Error en los datos enviados |
| 404 | Recurso no encontrado |
| 422 | Error de validacion |
| 500 | Error interno del servidor |
| 503 | Servicio no disponible |

### 10.4 Ejemplos de Uso

**Python con requests:**
```python
import requests

# Health check
r = requests.get("http://127.0.0.1:8000/health")
print(r.json())

# Prediccion
data = {
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
r = requests.post("http://127.0.0.1:8000/api/v1/predict", json=data)
print(r.json())
```

**JavaScript (fetch):**
```javascript
// Prediccion
const data = {
    sst_c: 25.0,
    salinity_ppt: 35.0,
    chlorophyll_mg_m3: 2.5,
    ph: 8.1,
    fleet_size: 150,
    fishing_effort_hours: 1200.0,
    fuel_consumption_l: 5000.0,
    fish_price_usd_ton: 2500.0,
    fuel_price_usd_l: 1.2,
    operating_cost_usd: 15000.0
};

fetch('http://127.0.0.1:8000/api/v1/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(data)
})
.then(response => response.json())
.then(result => console.log(result));
```

---

## 11. Solucion de Problemas

### 11.1 El servidor no inicia

**Problema:** `uvicorn` no encuentra el modulo

**Solucion:**
```bash
# Asegurar que esta en el directorio correcto
cd DL_Bayesian

# Verificar entorno virtual activo
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# Reinstalar dependencias
pip install -r requirements.txt
```

### 11.2 Error "Model not found"

**Problema:** La API retorna 404 al predecir

**Solucion:**
```bash
# 1. Verificar MLFlow esta corriendo
curl http://127.0.0.1:5000/health

# 2. Entrenar modelo si es primera vez
python scripts/train_api_model.py

# 3. Verificar modelo registrado
curl http://127.0.0.1:8000/api/v1/models
```

### 11.3 MLFlow no conecta

**Problema:** API reporta "MLFlow unavailable"

**Solucion:**
```bash
# 1. Verificar MLFlow esta corriendo en puerto 5000
# Terminal 1:
mlflow server --host 127.0.0.1 --port 5000

# 2. Reiniciar la API
# Terminal 2:
# Ctrl+C para detener
python -m uvicorn src.api.main:app --reload
```

### 11.4 Predicciones incorrectas

**Problema:** Modelo predice siempre lo mismo

**Posibles causas:**
1. Datos fuera de rango
2. Modelo no entrenado correctamente
3. Cache desactualizado

**Solucion:**
```bash
# 1. Verificar rangos de datos
# Los valores deben estar en rangos realistas

# 2. Limpiar cache
curl -X POST http://127.0.0.1:8000/api/v1/models/cache/clear

# 3. Re-entrenar modelo
python scripts/train_api_model.py
```

### 11.5 Error de dimension del modelo

**Problema:** "mat1 and mat2 shapes cannot be multiplied"

**Causa:** El modelo fue entrenado con diferente numero de features

**Solucion:**
```bash
# Re-entrenar modelo con las 10 features correctas
python scripts/train_api_model.py
```

### 11.6 Puerto ya en uso

**Problema:** "Address already in use"

**Solucion (Windows):**
```bash
# Encontrar proceso
netstat -ano | findstr :8000

# Matar proceso (usar el PID encontrado)
taskkill /PID <PID> /F
```

**Solucion (Linux/macOS):**
```bash
# Encontrar y matar proceso
lsof -i :8000
kill -9 <PID>
```

---

## 12. Preguntas Frecuentes

### General

**P: Necesito GPU para usar este sistema?**
R: No, el sistema esta optimizado para funcionar solo con CPU.

**P: Puedo usar mis propios datos?**
R: Si, puede modificar `data/loaders.py` para cargar sus datos en formato CSV.

**P: Que precision tiene el modelo?**
R: Con datos sinteticos, aproximadamente 85%. Con datos reales, dependera de la calidad de los datos.

### Predicciones

**P: Que significa la incertidumbre?**
R: La incertidumbre indica que tan confiable es la prediccion. Valores bajos (<10%) indican alta confianza.

**P: Puedo hacer predicciones por lotes?**
R: Actualmente la API procesa una prediccion a la vez. Para lotes, puede hacer multiples llamadas.

**P: Que modelo debo usar, BNN o MLP?**
R: Use BNN si necesita cuantificar incertidumbre. Use MLP si solo necesita la prediccion binaria (mas rapido).

### Entrenamiento

**P: Cuanto tiempo toma entrenar un modelo?**
R: Con 1000 muestras y 100 epocas, aproximadamente 1-2 minutos en CPU.

**P: Puedo detener un entrenamiento en progreso?**
R: Si usa la API, puede reiniciar el servidor. El modelo parcialmente entrenado se perdera.

**P: Como mejoro la precision del modelo?**
R:
1. Use mas datos de entrenamiento
2. Ajuste hiperparametros con Optuna
3. Asegure que los datos sean representativos

### MLFlow

**P: Donde se guardan los modelos?**
R: En la carpeta `mlruns/` del directorio del proyecto.

**P: Como exporto un modelo?**
R: Puede descargar los artefactos desde MLFlow UI o usar `mlflow.pytorch.load_model()`.

**P: Puedo usar una base de datos en lugar de archivos?**
R: Si, MLFlow soporta PostgreSQL, MySQL, etc. Configure `--backend-store-uri`.

### Deployment

**P: Como despliego en produccion?**
R: Use Docker con `docker-compose` o despliegue en AWS ECS siguiendo la documentacion tecnica.

**P: Es seguro exponer la API a internet?**
R: Para produccion, agregue autenticacion (API keys), HTTPS, y use un balanceador de carga.

---

## Contacto y Soporte

- **Repositorio:** https://github.com/arielgiamportone/DL_Bayesian
- **Issues:** https://github.com/arielgiamportone/DL_Bayesian/issues
- **Comunidad:** https://github.com/PesquerosEnIA

---

## Apendice: Referencia Rapida

### Comandos Esenciales

```bash
# Iniciar MLFlow
mlflow server --host 127.0.0.1 --port 5000

# Iniciar API
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# Entrenar modelo
python scripts/train_api_model.py

# Ejecutar tests
pytest tests/ -v

# Health check
curl http://127.0.0.1:8000/health
```

### URLs Importantes

| Servicio | URL |
|----------|-----|
| Dashboard | http://127.0.0.1:8000 |
| API Docs | http://127.0.0.1:8000/docs |
| MLFlow UI | http://127.0.0.1:5000 |
| Health Check | http://127.0.0.1:8000/health |

### Variables de Entrada (API)

```json
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
  "operating_cost_usd": 15000.0
}
```

---

*Manual de Usuario - DL_Bayesian v1.0.0*
