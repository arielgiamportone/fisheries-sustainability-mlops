"""
Módulo de validación para redes bayesianas.

Proporciona funcionalidades para:
- Validación cruzada
- Cálculo de métricas
- Análisis de estabilidad estructural (bootstrap)
- Comparación de modelos
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from collections import Counter
from dataclasses import dataclass

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BIC, K2, BDeu, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


@dataclass
class ValidationResults:
    """Contenedor para resultados de validación."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    confusion_matrix: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray

    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'auc_roc': self.auc_roc
        }

    def __repr__(self) -> str:
        return (f"ValidationResults(accuracy={self.accuracy:.4f}, "
                f"precision={self.precision:.4f}, recall={self.recall:.4f}, "
                f"f1={self.f1:.4f}, auc_roc={self.auc_roc:.4f})")


class BayesianValidator:
    """
    Clase para validación de redes bayesianas.

    Proporciona métodos para validación cruzada, análisis de estabilidad
    estructural y comparación de modelos.

    Example:
        >>> validator = BayesianValidator(target='Sustainable')
        >>> results = validator.cross_validate(data, n_splits=5)
        >>> stability = validator.bootstrap_structure_stability(data, n_iterations=50)
    """

    def __init__(
        self,
        target: str = 'Sustainable',
        scoring_method: str = 'bic',
        random_state: int = 42
    ):
        """
        Inicializa el validador.

        Args:
            target: Variable objetivo
            scoring_method: Método de scoring para aprendizaje de estructura
            random_state: Semilla para reproducibilidad
        """
        self.target = target
        self.scoring_method = scoring_method
        self.random_state = random_state

    def _get_scoring_class(self, method: str):
        """Obtiene la clase de scoring apropiada."""
        scoring_map = {
            'bic': BIC,
            'k2': K2,
            'bdeu': BDeu
        }
        return scoring_map.get(method, BIC)

    def _predict_with_model(
        self,
        model: BayesianNetwork,
        X: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Realiza predicciones con un modelo bayesiano."""
        inference = VariableElimination(model)
        predictions = []
        probabilities = []

        for idx, row in X.iterrows():
            evidence = row.to_dict()
            try:
                result = inference.query(variables=[self.target], evidence=evidence)
                prob_1 = result.values[1] if len(result.values) > 1 else 0.5
                predictions.append(1 if prob_1 >= 0.5 else 0)
                probabilities.append(prob_1)
            except Exception:
                predictions.append(0)
                probabilities.append(0.5)

        return np.array(predictions), np.array(probabilities)

    def train_test_validation(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2
    ) -> ValidationResults:
        """
        Realiza validación con split train/test.

        Args:
            data: DataFrame con datos discretizados
            test_size: Proporción para test

        Returns:
            ValidationResults con métricas
        """
        X = data.drop(self.target, axis=1)
        y = data[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size,
            random_state=self.random_state, stratify=y
        )

        # Entrenar modelo
        df_train = pd.concat([X_train, y_train], axis=1)
        scoring = self._get_scoring_class(self.scoring_method)(df_train)
        hc = HillClimbSearch(df_train)
        structure = hc.estimate(scoring_method=scoring)

        model = BayesianNetwork(structure.edges())
        model.fit(df_train, estimator=MaximumLikelihoodEstimator)

        # Predecir
        y_pred, y_prob = self._predict_with_model(model, X_test)

        # Calcular métricas
        return self._calculate_metrics(y_test.values, y_pred, y_prob)

    def cross_validate(
        self,
        data: pd.DataFrame,
        n_splits: int = 5
    ) -> Dict:
        """
        Realiza validación cruzada k-fold.

        Args:
            data: DataFrame con datos discretizados
            n_splits: Número de folds

        Returns:
            Diccionario con métricas por fold y promedios
        """
        X = data.drop(self.target, axis=1)
        y = data[self.target]

        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=self.random_state)

        fold_results = []
        all_metrics = {
            'accuracy': [], 'precision': [], 'recall': [],
            'f1': [], 'auc_roc': []
        }

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Entrenar
            df_train = pd.concat([X_train, y_train], axis=1)
            scoring = self._get_scoring_class(self.scoring_method)(df_train)
            hc = HillClimbSearch(df_train)
            structure = hc.estimate(scoring_method=scoring)

            model = BayesianNetwork(structure.edges())
            model.fit(df_train, estimator=MaximumLikelihoodEstimator)

            # Predecir
            y_pred, y_prob = self._predict_with_model(model, X_test)

            # Métricas
            results = self._calculate_metrics(y_test.values, y_pred, y_prob)
            fold_results.append(results)

            for metric, value in results.to_dict().items():
                all_metrics[metric].append(value)

        # Calcular estadísticas
        summary = {}
        for metric, values in all_metrics.items():
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }

        return {
            'fold_results': fold_results,
            'summary': summary,
            'n_splits': n_splits
        }

    def bootstrap_structure_stability(
        self,
        data: pd.DataFrame,
        n_iterations: int = 50
    ) -> Dict:
        """
        Analiza la estabilidad de la estructura mediante bootstrap.

        Args:
            data: DataFrame con datos discretizados
            n_iterations: Número de iteraciones de bootstrap

        Returns:
            Diccionario con frecuencia de aristas y estadísticas
        """
        np.random.seed(self.random_state)
        n_samples = len(data)
        edge_counts = Counter()
        all_structures = []

        for i in range(n_iterations):
            # Muestreo con reemplazo
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            data_bootstrap = data.iloc[indices].reset_index(drop=True)

            # Aprender estructura
            scoring = self._get_scoring_class(self.scoring_method)(data_bootstrap)
            hc = HillClimbSearch(data_bootstrap)
            structure = hc.estimate(scoring_method=scoring)

            edges = list(structure.edges())
            all_structures.append(set(edges))

            for edge in edges:
                edge_counts[edge] += 1

        # Calcular estabilidad
        edge_stability = {}
        for edge, count in edge_counts.items():
            percentage = (count / n_iterations) * 100
            stability = 'alta' if percentage >= 80 else 'media' if percentage >= 50 else 'baja'
            edge_stability[edge] = {
                'count': count,
                'percentage': percentage,
                'stability': stability
            }

        # Ordenar por frecuencia
        sorted_edges = sorted(edge_stability.items(),
                            key=lambda x: x[1]['count'], reverse=True)

        return {
            'edge_stability': dict(sorted_edges),
            'n_iterations': n_iterations,
            'total_unique_edges': len(edge_counts),
            'highly_stable_edges': [e for e, s in edge_stability.items() if s['stability'] == 'alta'],
            'unstable_edges': [e for e, s in edge_stability.items() if s['stability'] == 'baja']
        }

    def compare_scoring_methods(
        self,
        data: pd.DataFrame
    ) -> Dict:
        """
        Compara diferentes métodos de scoring.

        Args:
            data: DataFrame con datos discretizados

        Returns:
            Diccionario con resultados por método y comparaciones
        """
        methods = ['bic', 'k2', 'bdeu']
        results = {}

        for method in methods:
            scoring = self._get_scoring_class(method)(data)
            hc = HillClimbSearch(data)
            structure = hc.estimate(scoring_method=scoring)

            results[method] = {
                'edges': set(structure.edges()),
                'n_edges': len(structure.edges())
            }

        # Calcular similitudes
        comparisons = {}
        for i, m1 in enumerate(methods):
            for m2 in methods[i+1:]:
                edges1 = results[m1]['edges']
                edges2 = results[m2]['edges']
                intersection = edges1 & edges2
                union = edges1 | edges2
                jaccard = len(intersection) / len(union) if union else 0

                comparisons[f'{m1}_vs_{m2}'] = {
                    'common_edges': len(intersection),
                    'total_unique_edges': len(union),
                    'jaccard_similarity': jaccard
                }

        return {
            'methods': results,
            'comparisons': comparisons
        }

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> ValidationResults:
        """Calcula métricas de clasificación."""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_roc = 0.5

        cm = confusion_matrix(y_true, y_pred)

        return ValidationResults(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc_roc,
            confusion_matrix=cm,
            predictions=y_pred,
            probabilities=y_prob
        )


def quick_validate(
    data: pd.DataFrame,
    target: str = 'Sustainable',
    n_splits: int = 5
) -> Dict:
    """
    Función de conveniencia para validación rápida.

    Args:
        data: DataFrame con datos discretizados
        target: Variable objetivo
        n_splits: Número de folds para CV

    Returns:
        Resumen de métricas
    """
    validator = BayesianValidator(target=target)
    cv_results = validator.cross_validate(data, n_splits=n_splits)

    summary = cv_results['summary']
    print("=" * 50)
    print("RESUMEN DE VALIDACIÓN CRUZADA")
    print("=" * 50)
    for metric, stats in summary.items():
        print(f"{metric:12}: {stats['mean']:.4f} (+/- {stats['std']:.4f})")
    print("=" * 50)

    return cv_results
