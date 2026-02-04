"""
Módulo de Redes Bayesianas para Sostenibilidad Pesquera.

Proporciona clases y funciones para:
- Aprendizaje de estructura de redes bayesianas
- Estimación de parámetros
- Inferencia probabilística
- Validación de modelos
"""

from .networks import BayesianSustainabilityModel
from .inference import BayesianInference
from .validation import BayesianValidator

__all__ = [
    'BayesianSustainabilityModel',
    'BayesianInference',
    'BayesianValidator'
]
