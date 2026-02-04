"""
Módulo de inferencia para redes bayesianas.

Proporciona funcionalidades avanzadas de inferencia incluyendo:
- Consultas probabilísticas
- Inferencia con evidencia parcial
- Análisis de sensibilidad
- Cálculo de probabilidades marginales
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.models import BayesianNetwork


class BayesianInference:
    """
    Clase para realizar inferencias avanzadas en redes bayesianas.

    Proporciona métodos para consultas, análisis de sensibilidad
    y cálculo de probabilidades condicionales.

    Attributes:
        model: BayesianNetwork de pgmpy
        inference_engine: Motor de inferencia (VariableElimination)

    Example:
        >>> inference = BayesianInference(model)
        >>> prob = inference.query_probability('Sustainable', {'SST_C_disc': 'Alto'})
    """

    def __init__(
        self,
        model: BayesianNetwork,
        method: str = 'variable_elimination'
    ):
        """
        Inicializa el motor de inferencia.

        Args:
            model: Modelo BayesianNetwork ajustado
            method: Método de inferencia ('variable_elimination' o 'belief_propagation')
        """
        self.model = model

        if method == 'variable_elimination':
            self.inference_engine = VariableElimination(model)
        elif method == 'belief_propagation':
            self.inference_engine = BeliefPropagation(model)
        else:
            raise ValueError(f"Método no válido: {method}")

    def query(
        self,
        variables: List[str],
        evidence: Optional[Dict[str, str]] = None
    ) -> Dict:
        """
        Realiza una consulta de inferencia.

        Args:
            variables: Variables a consultar
            evidence: Evidencia observada

        Returns:
            Diccionario con resultados de la consulta
        """
        result = self.inference_engine.query(
            variables=variables,
            evidence=evidence or {}
        )

        return {
            'variables': variables,
            'evidence': evidence,
            'distribution': result,
            'values': result.values,
            'states': result.state_names
        }

    def query_probability(
        self,
        variable: str,
        evidence: Optional[Dict[str, str]] = None,
        state: Optional[str] = None
    ) -> Union[float, Dict[str, float]]:
        """
        Consulta la probabilidad de una variable dado evidencia.

        Args:
            variable: Variable a consultar
            evidence: Evidencia observada
            state: Estado específico (si None, retorna todas las probabilidades)

        Returns:
            Probabilidad o diccionario de probabilidades
        """
        result = self.query([variable], evidence)

        probs = {}
        for i, s in enumerate(result['states'][variable]):
            probs[s] = float(result['values'][i])

        if state is not None:
            return probs.get(state, 0.0)

        return probs

    def marginal_probability(self, variable: str) -> Dict[str, float]:
        """
        Calcula la probabilidad marginal de una variable (sin evidencia).

        Args:
            variable: Variable a consultar

        Returns:
            Diccionario de probabilidades por estado
        """
        return self.query_probability(variable, evidence=None)

    def conditional_probability(
        self,
        target: str,
        given: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calcula P(target | given).

        Args:
            target: Variable objetivo
            given: Variables condicionantes

        Returns:
            Distribución condicional
        """
        return self.query_probability(target, evidence=given)

    def most_probable_explanation(
        self,
        evidence: Dict[str, str],
        variables: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Encuentra la explicación más probable dado evidencia (MPE).

        Args:
            evidence: Evidencia observada
            variables: Variables a inferir (si None, todas las no observadas)

        Returns:
            Diccionario con los estados más probables
        """
        if variables is None:
            variables = [n for n in self.model.nodes() if n not in evidence]

        mpe = {}
        for var in variables:
            probs = self.query_probability(var, evidence)
            mpe[var] = max(probs, key=probs.get)

        return mpe

    def sensitivity_analysis(
        self,
        target: str,
        target_state: str,
        variable: str,
        base_evidence: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Analiza cómo cambia P(target=state) al variar una variable.

        Args:
            target: Variable objetivo
            target_state: Estado del objetivo a analizar
            variable: Variable a variar
            base_evidence: Evidencia base (opcional)

        Returns:
            Diccionario con probabilidades por cada estado de la variable
        """
        base_evidence = base_evidence or {}
        results = {}

        # Obtener estados posibles de la variable
        cpd = self.model.get_cpds(variable)
        if cpd is None:
            raise ValueError(f"Variable '{variable}' no encontrada en el modelo")

        states = cpd.state_names[variable]

        for state in states:
            evidence = {**base_evidence, variable: state}
            prob = self.query_probability(target, evidence, target_state)
            results[state] = prob

        return results

    def batch_predict(
        self,
        X: pd.DataFrame,
        target: str,
        positive_state: str = '1'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Realiza predicciones en lote.

        Args:
            X: DataFrame con evidencia por fila
            target: Variable objetivo
            positive_state: Estado considerado positivo

        Returns:
            Tupla (predicciones, probabilidades)
        """
        predictions = []
        probabilities = []

        for idx, row in X.iterrows():
            evidence = row.to_dict()

            try:
                probs = self.query_probability(target, evidence)
                prob_positive = probs.get(positive_state, probs.get(1, 0.5))
                predictions.append(1 if prob_positive >= 0.5 else 0)
                probabilities.append(prob_positive)
            except Exception:
                predictions.append(0)
                probabilities.append(0.5)

        return np.array(predictions), np.array(probabilities)

    def intervention_effect(
        self,
        treatment: str,
        treatment_value: str,
        outcome: str,
        outcome_state: str,
        confounders: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """
        Estima el efecto de una intervención (análisis do-calculus simplificado).

        Nota: Esta es una aproximación usando evidencia condicional.
        Para análisis causal riguroso, usar el módulo causal.

        Args:
            treatment: Variable de tratamiento
            treatment_value: Valor del tratamiento
            outcome: Variable de resultado
            outcome_state: Estado del resultado a medir
            confounders: Variables confusoras a controlar

        Returns:
            Diccionario con probabilidades pre y post intervención
        """
        # Probabilidad base (sin intervención)
        base_prob = self.query_probability(outcome, confounders, outcome_state)

        # Probabilidad con intervención
        evidence = {**(confounders or {}), treatment: treatment_value}
        intervention_prob = self.query_probability(outcome, evidence, outcome_state)

        return {
            'baseline': base_prob,
            'intervention': intervention_prob,
            'effect': intervention_prob - base_prob,
            'relative_effect': (intervention_prob - base_prob) / base_prob if base_prob > 0 else 0
        }


def predict_with_bayesian_network(
    model: BayesianNetwork,
    X: pd.DataFrame,
    target: str = 'Sustainable'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Función de conveniencia para predicción con red bayesiana.

    Args:
        model: Modelo BayesianNetwork ajustado
        X: DataFrame con features
        target: Variable objetivo

    Returns:
        Tupla (predicciones, probabilidades)
    """
    inference = BayesianInference(model)
    return inference.batch_predict(X, target)
