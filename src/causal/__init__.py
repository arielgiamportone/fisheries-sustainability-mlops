"""
Módulo de Análisis Causal para Sostenibilidad Pesquera.

Proporciona clases y funciones para:
- Definición y validación de DAGs causales
- Análisis de intervenciones (do-calculus)
- Estimación de efectos causales
- Análisis contrafactual
"""

from .dag import CausalDAG, SustainabilityDAG
from .interventions import CausalInterventions

__all__ = [
    'CausalDAG',
    'SustainabilityDAG',
    'CausalInterventions'
]
