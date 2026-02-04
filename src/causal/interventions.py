"""
Módulo para análisis de intervenciones causales.

Proporciona funcionalidades para:
- Simulación de intervenciones (do-calculus)
- Estimación de efectos causales
- Análisis contrafactual
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.inference import VariableElimination, CausalInference


@dataclass
class InterventionResult:
    """Resultado de una intervención causal."""
    treatment: str
    treatment_value: str
    outcome: str
    baseline_prob: Dict[str, float]
    intervention_prob: Dict[str, float]
    ate: float  # Average Treatment Effect
    relative_effect: float

    def summary(self) -> str:
        return (f"Intervención: do({self.treatment}={self.treatment_value})\n"
                f"Outcome: {self.outcome}\n"
                f"P(outcome) baseline: {self.baseline_prob}\n"
                f"P(outcome) intervención: {self.intervention_prob}\n"
                f"ATE: {self.ate:+.4f}\n"
                f"Efecto relativo: {self.relative_effect:+.2%}")


class CausalInterventions:
    """
    Clase para análisis de intervenciones causales.

    Permite simular intervenciones (do-calculus), estimar efectos
    causales y realizar análisis contrafactual.

    Example:
        >>> ci = CausalInterventions(model, target='Sustainable')
        >>> effect = ci.do_intervention('CPUE_disc', 'Alto')
        >>> print(effect.summary())
    """

    def __init__(
        self,
        model: BayesianNetwork,
        target: str = 'Sustainable'
    ):
        """
        Inicializa el analizador de intervenciones.

        Args:
            model: Modelo BayesianNetwork ajustado
            target: Variable objetivo
        """
        self.model = model
        self.target = target
        self.inference = VariableElimination(model)

    def _query_distribution(
        self,
        variable: str,
        evidence: Optional[Dict[str, str]] = None
    ) -> Dict[str, float]:
        """Consulta la distribución de una variable."""
        result = self.inference.query(
            variables=[variable],
            evidence=evidence or {}
        )

        probs = {}
        for i, state in enumerate(result.state_names[variable]):
            probs[state] = float(result.values[i])

        return probs

    def do_intervention(
        self,
        treatment: str,
        treatment_value: str,
        outcome: Optional[str] = None,
        confounders: Optional[Dict[str, str]] = None
    ) -> InterventionResult:
        """
        Simula una intervención do(treatment=value).

        Esta es una aproximación usando inferencia condicional.
        Para análisis causal riguroso con ajuste de confusores,
        se requiere especificar los confusores.

        Args:
            treatment: Variable de tratamiento
            treatment_value: Valor del tratamiento
            outcome: Variable de resultado (default: self.target)
            confounders: Variables confusoras a controlar

        Returns:
            InterventionResult con efectos estimados
        """
        outcome = outcome or self.target

        # Distribución baseline (marginal o condicional a confusores)
        baseline_prob = self._query_distribution(outcome, confounders)

        # Distribución post-intervención
        intervention_evidence = {**(confounders or {}), treatment: treatment_value}
        intervention_prob = self._query_distribution(outcome, intervention_evidence)

        # Calcular efectos (asumiendo estado positivo es '1' o 'Sostenible')
        positive_states = ['1', 1, 'Sostenible', 'Alto', 'Si']
        positive_state = None
        for state in positive_states:
            if state in baseline_prob:
                positive_state = state
                break

        if positive_state is None:
            positive_state = list(baseline_prob.keys())[-1]

        baseline_positive = baseline_prob.get(positive_state, 0)
        intervention_positive = intervention_prob.get(positive_state, 0)

        ate = intervention_positive - baseline_positive
        relative_effect = ate / baseline_positive if baseline_positive > 0 else 0

        return InterventionResult(
            treatment=treatment,
            treatment_value=treatment_value,
            outcome=outcome,
            baseline_prob=baseline_prob,
            intervention_prob=intervention_prob,
            ate=ate,
            relative_effect=relative_effect
        )

    def compare_interventions(
        self,
        treatment: str,
        outcome: Optional[str] = None
    ) -> Dict[str, InterventionResult]:
        """
        Compara el efecto de diferentes valores del tratamiento.

        Args:
            treatment: Variable de tratamiento
            outcome: Variable de resultado

        Returns:
            Diccionario con resultados por valor de tratamiento
        """
        outcome = outcome or self.target

        # Obtener valores posibles del tratamiento
        cpd = self.model.get_cpds(treatment)
        if cpd is None:
            raise ValueError(f"Variable '{treatment}' no encontrada")

        treatment_values = cpd.state_names[treatment]

        results = {}
        for value in treatment_values:
            results[value] = self.do_intervention(treatment, value, outcome)

        return results

    def estimate_ate(
        self,
        treatment: str,
        treatment_value_1: str,
        treatment_value_0: str,
        outcome: Optional[str] = None
    ) -> Dict:
        """
        Estima el Average Treatment Effect (ATE).

        ATE = E[Y | do(T=1)] - E[Y | do(T=0)]

        Args:
            treatment: Variable de tratamiento
            treatment_value_1: Valor de tratamiento (grupo tratado)
            treatment_value_0: Valor de control (grupo control)
            outcome: Variable de resultado

        Returns:
            Diccionario con ATE y componentes
        """
        outcome = outcome or self.target

        # P(Y | do(T=1))
        result_1 = self.do_intervention(treatment, treatment_value_1, outcome)

        # P(Y | do(T=0))
        result_0 = self.do_intervention(treatment, treatment_value_0, outcome)

        # Encontrar estado positivo
        states = list(result_1.intervention_prob.keys())
        positive_states = ['1', 1, 'Sostenible', 'Alto']
        positive_state = None
        for ps in positive_states:
            if ps in states:
                positive_state = ps
                break
        if positive_state is None:
            positive_state = states[-1]

        y1 = result_1.intervention_prob.get(positive_state, 0)
        y0 = result_0.intervention_prob.get(positive_state, 0)

        ate = y1 - y0

        return {
            'ate': ate,
            'E[Y|do(T=1)]': y1,
            'E[Y|do(T=0)]': y0,
            'treatment': treatment,
            'outcome': outcome,
            'treatment_value_1': treatment_value_1,
            'treatment_value_0': treatment_value_0,
            'interpretation': self._interpret_ate(ate)
        }

    def _interpret_ate(self, ate: float) -> str:
        """Interpreta el valor del ATE."""
        if abs(ate) < 0.05:
            return "Efecto mínimo o nulo"
        elif ate > 0.2:
            return "Efecto positivo fuerte"
        elif ate > 0.1:
            return "Efecto positivo moderado"
        elif ate > 0:
            return "Efecto positivo débil"
        elif ate < -0.2:
            return "Efecto negativo fuerte"
        elif ate < -0.1:
            return "Efecto negativo moderado"
        else:
            return "Efecto negativo débil"

    def sensitivity_to_intervention(
        self,
        treatment: str,
        outcome: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Analiza la sensibilidad del outcome a diferentes valores del tratamiento.

        Args:
            treatment: Variable de tratamiento
            outcome: Variable de resultado

        Returns:
            DataFrame con probabilidades por valor de tratamiento
        """
        results = self.compare_interventions(treatment, outcome)

        data = []
        for value, result in results.items():
            row = {
                'treatment_value': value,
                'ate': result.ate,
                'relative_effect': result.relative_effect
            }
            for state, prob in result.intervention_prob.items():
                row[f'P({outcome}={state})'] = prob
            data.append(row)

        return pd.DataFrame(data)

    def counterfactual_query(
        self,
        observed: Dict[str, str],
        intervention: Dict[str, str],
        query_var: str
    ) -> Dict:
        """
        Realiza una consulta contrafactual simplificada.

        "¿Qué habría pasado con query_var si hubiéramos intervenido?"

        Args:
            observed: Valores observados
            intervention: Intervención hipotética
            query_var: Variable a consultar

        Returns:
            Diccionario con resultados contrafactuales
        """
        # Distribución factual (lo que observamos)
        factual = self._query_distribution(query_var, observed)

        # Distribución contrafactual (si hubiéramos intervenido)
        counterfactual_evidence = {**observed, **intervention}
        # Remover el tratamiento de la evidencia observada si estaba
        for key in intervention:
            if key in observed:
                del counterfactual_evidence[key]
        counterfactual_evidence.update(intervention)

        counterfactual = self._query_distribution(query_var, counterfactual_evidence)

        return {
            'query_variable': query_var,
            'observed': observed,
            'intervention': intervention,
            'factual_distribution': factual,
            'counterfactual_distribution': counterfactual,
            'difference': {
                k: counterfactual.get(k, 0) - factual.get(k, 0)
                for k in set(factual.keys()) | set(counterfactual.keys())
            }
        }


def quick_intervention_analysis(
    model: BayesianNetwork,
    treatment: str,
    target: str = 'Sustainable'
) -> pd.DataFrame:
    """
    Función de conveniencia para análisis rápido de intervenciones.

    Args:
        model: Modelo BayesianNetwork ajustado
        treatment: Variable de tratamiento
        target: Variable objetivo

    Returns:
        DataFrame con resultados
    """
    ci = CausalInterventions(model, target)
    return ci.sensitivity_to_intervention(treatment, target)
