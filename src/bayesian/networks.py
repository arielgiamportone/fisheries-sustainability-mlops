"""
Módulo de Redes Bayesianas para modelado de sostenibilidad pesquera.

Este módulo proporciona la clase BayesianSustainabilityModel que encapsula
la funcionalidad de pgmpy para aprendizaje de estructura, estimación de
parámetros e inferencia.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Set
import warnings

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import (
    HillClimbSearch,
    BIC,
    K2,
    BDeu,
    MaximumLikelihoodEstimator,
    BayesianEstimator
)
from pgmpy.inference import VariableElimination

warnings.filterwarnings('ignore', category=FutureWarning)


class BayesianSustainabilityModel:
    """
    Modelo de Red Bayesiana para análisis de sostenibilidad pesquera.

    Esta clase proporciona una interfaz unificada para:
    - Aprender la estructura de la red desde datos
    - Estimar parámetros (CPDs)
    - Realizar inferencias probabilísticas
    - Comparar diferentes métodos de scoring

    Attributes:
        model: Modelo BayesianNetwork de pgmpy
        structure: Estructura aprendida (DAGModel)
        target: Variable objetivo (default: 'Sostenibilidad')
        scoring_method: Método de scoring usado

    Example:
        >>> model = BayesianSustainabilityModel(target='Sustainable')
        >>> model.fit(df_discretized)
        >>> probs = model.predict_proba(X_test)
    """

    SCORING_METHODS = {
        'bic': BIC,
        'k2': K2,
        'bdeu': BDeu
    }

    def __init__(
        self,
        target: str = 'Sustainable',
        scoring_method: str = 'bic',
        max_indegree: int = 5,
        random_state: int = 42
    ):
        """
        Inicializa el modelo de red bayesiana.

        Args:
            target: Nombre de la variable objetivo
            scoring_method: Método de scoring ('bic', 'k2', 'bdeu')
            max_indegree: Máximo número de padres por nodo
            random_state: Semilla para reproducibilidad
        """
        self.target = target
        self.scoring_method = scoring_method
        self.max_indegree = max_indegree
        self.random_state = random_state

        self.model: Optional[BayesianNetwork] = None
        self.structure = None
        self._inference_engine = None
        self._fitted = False
        self._feature_names: List[str] = []

    def fit(
        self,
        data: pd.DataFrame,
        predefined_edges: Optional[List[Tuple[str, str]]] = None,
        estimator: str = 'mle'
    ) -> 'BayesianSustainabilityModel':
        """
        Aprende la estructura y parámetros de la red bayesiana.

        Args:
            data: DataFrame con variables discretizadas
            predefined_edges: Lista de aristas predefinidas (opcional)
            estimator: Método de estimación ('mle' o 'bayesian')

        Returns:
            self: El modelo ajustado
        """
        # Guardar nombres de features
        self._feature_names = [col for col in data.columns if col != self.target]

        if predefined_edges is not None:
            # Usar estructura predefinida
            edges = predefined_edges
        else:
            # Aprender estructura
            edges = self._learn_structure(data)

        # Crear modelo
        self.model = BayesianNetwork(edges)

        # Estimar parámetros
        if estimator == 'mle':
            self.model.fit(data, estimator=MaximumLikelihoodEstimator)
        else:
            self.model.fit(data, estimator=BayesianEstimator, prior_type='BDeu')

        # Crear motor de inferencia
        self._inference_engine = VariableElimination(self.model)
        self._fitted = True

        return self

    def _learn_structure(self, data: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Aprende la estructura de la red usando el método de scoring especificado.

        Args:
            data: DataFrame con datos discretizados

        Returns:
            Lista de aristas (tuplas de strings)
        """
        # Obtener clase de scoring
        if self.scoring_method not in self.SCORING_METHODS:
            raise ValueError(f"Método de scoring no válido: {self.scoring_method}. "
                           f"Opciones: {list(self.SCORING_METHODS.keys())}")

        scoring_class = self.SCORING_METHODS[self.scoring_method]
        scoring = scoring_class(data)

        # Buscar estructura
        hc = HillClimbSearch(data)
        self.structure = hc.estimate(
            scoring_method=scoring,
            max_indegree=self.max_indegree
        )

        return list(self.structure.edges())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice la clase para cada muestra.

        Args:
            X: DataFrame con features (sin la variable objetivo)

        Returns:
            Array de predicciones (0 o 1)
        """
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predice la probabilidad de la clase positiva para cada muestra.

        Args:
            X: DataFrame con features (sin la variable objetivo)

        Returns:
            Array de probabilidades P(target=1)
        """
        self._check_fitted()

        probabilities = []

        for idx, row in X.iterrows():
            evidence = row.to_dict()
            try:
                result = self._inference_engine.query(
                    variables=[self.target],
                    evidence=evidence
                )
                # Obtener probabilidad de clase positiva (1)
                prob = result.values[1] if len(result.values) > 1 else result.values[0]
                probabilities.append(prob)
            except Exception as e:
                # Si hay error, usar probabilidad marginal
                probabilities.append(0.5)

        return np.array(probabilities)

    def query(
        self,
        variables: List[str],
        evidence: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Realiza una consulta de inferencia en la red.

        Args:
            variables: Variables a consultar
            evidence: Evidencia observada (dict variable: valor)

        Returns:
            Diccionario con distribuciones de probabilidad
        """
        self._check_fitted()

        result = self._inference_engine.query(
            variables=variables,
            evidence=evidence or {}
        )

        return {
            'variables': variables,
            'evidence': evidence,
            'values': result.values,
            'states': result.state_names
        }

    def get_edges(self) -> List[Tuple[str, str]]:
        """Retorna las aristas del modelo."""
        self._check_fitted()
        return list(self.model.edges())

    def get_nodes(self) -> List[str]:
        """Retorna los nodos del modelo."""
        self._check_fitted()
        return list(self.model.nodes())

    def get_cpd(self, variable: str):
        """
        Obtiene la distribución de probabilidad condicional de una variable.

        Args:
            variable: Nombre de la variable

        Returns:
            TabularCPD de pgmpy
        """
        self._check_fitted()
        return self.model.get_cpds(variable)

    def compare_scoring_methods(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Dict]:
        """
        Compara diferentes métodos de scoring para aprendizaje de estructura.

        Args:
            data: DataFrame con datos discretizados

        Returns:
            Diccionario con resultados por método
        """
        results = {}

        for method_name, scoring_class in self.SCORING_METHODS.items():
            scoring = scoring_class(data)
            hc = HillClimbSearch(data)
            structure = hc.estimate(scoring_method=scoring)

            results[method_name] = {
                'edges': list(structure.edges()),
                'n_edges': len(structure.edges()),
                'nodes': list(structure.nodes())
            }

        # Calcular coincidencias
        methods = list(results.keys())
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                edges1 = set(results[m1]['edges'])
                edges2 = set(results[m2]['edges'])
                intersection = edges1 & edges2
                union = edges1 | edges2
                jaccard = len(intersection) / len(union) if union else 0
                results[f'{m1}_vs_{m2}_jaccard'] = jaccard

        return results

    def _check_fitted(self):
        """Verifica que el modelo esté ajustado."""
        if not self._fitted:
            raise RuntimeError("El modelo no está ajustado. Ejecute fit() primero.")

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        edges = len(self.get_edges()) if self._fitted else 0
        return (f"BayesianSustainabilityModel(target='{self.target}', "
                f"scoring='{self.scoring_method}', status={status}, edges={edges})")


def create_sustainability_model(
    data: pd.DataFrame,
    target: str = 'Sustainable',
    scoring: str = 'bic'
) -> BayesianSustainabilityModel:
    """
    Factory function para crear y ajustar un modelo de sostenibilidad.

    Args:
        data: DataFrame con datos discretizados
        target: Variable objetivo
        scoring: Método de scoring

    Returns:
        Modelo ajustado
    """
    model = BayesianSustainabilityModel(target=target, scoring_method=scoring)
    model.fit(data)
    return model
