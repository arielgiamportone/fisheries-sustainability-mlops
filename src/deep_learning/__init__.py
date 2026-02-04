"""
Módulo de Deep Learning para Sostenibilidad Pesquera.

Proporciona modelos avanzados:
- Redes Neuronales Bayesianas (BNN) para cuantificación de incertidumbre
- Variational Autoencoders Causales para generación contrafactual
- Modelos híbridos que combinan estructura causal con redes neuronales

Requiere: torch, pyro-ppl (opcional)
"""

from .models import (
    SustainabilityMLP,
    BayesianNeuralNetwork,
    CausalEncoder
)
from .training import (
    Trainer,
    EarlyStopping,
    train_model,
    evaluate_model
)

__all__ = [
    'SustainabilityMLP',
    'BayesianNeuralNetwork',
    'CausalEncoder',
    'Trainer',
    'EarlyStopping',
    'train_model',
    'evaluate_model'
]
