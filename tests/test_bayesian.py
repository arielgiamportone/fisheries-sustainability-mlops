"""
Tests unitarios para el módulo bayesian.

Ejecutar con: pytest tests/test_bayesian.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.bayesian.networks import BayesianSustainabilityModel, create_sustainability_model
from src.bayesian.validation import BayesianValidator, ValidationResults
from data.loaders import generate_synthetic_fisheries_data, prepare_bayesian_dataset


@pytest.fixture
def sample_data():
    """Genera datos de prueba."""
    df = generate_synthetic_fisheries_data(n_samples=200, random_state=42)
    df_bayesian = prepare_bayesian_dataset(df, target='Sustainable')
    return df_bayesian


@pytest.fixture
def small_data():
    """Genera un dataset pequeño con dependencias para tests."""
    np.random.seed(42)
    n = 200

    # Generar datos con dependencias reales
    A = np.random.choice(['Bajo', 'Medio', 'Alto'], n, p=[0.3, 0.4, 0.3])
    B = np.random.choice(['Bajo', 'Medio', 'Alto'], n, p=[0.3, 0.4, 0.3])

    # C depende de A
    C = []
    for a in A:
        if a == 'Alto':
            C.append(np.random.choice(['Bajo', 'Medio', 'Alto'], p=[0.1, 0.3, 0.6]))
        elif a == 'Medio':
            C.append(np.random.choice(['Bajo', 'Medio', 'Alto'], p=[0.3, 0.4, 0.3]))
        else:
            C.append(np.random.choice(['Bajo', 'Medio', 'Alto'], p=[0.6, 0.3, 0.1]))

    # Sustainable depende de B y C
    Sustainable = []
    for b, c in zip(B, C):
        prob = 0.3
        if b == 'Alto':
            prob += 0.25
        if c == 'Alto':
            prob += 0.25
        Sustainable.append(1 if np.random.random() < prob else 0)

    data = pd.DataFrame({
        'A_disc': A,
        'B_disc': B,
        'C_disc': C,
        'Sustainable': Sustainable
    })
    return data


class TestBayesianSustainabilityModel:
    """Tests para BayesianSustainabilityModel."""

    def test_model_initialization(self):
        """Test inicialización del modelo."""
        model = BayesianSustainabilityModel(target='Sustainable')
        assert model.target == 'Sustainable'
        assert model.scoring_method == 'bic'
        assert model._fitted == False

    def test_model_fit(self, small_data):
        """Test ajuste del modelo."""
        model = BayesianSustainabilityModel(target='Sustainable')
        model.fit(small_data)

        assert model._fitted == True
        assert model.model is not None
        assert len(model.get_nodes()) > 0

    def test_model_predict(self, small_data):
        """Test predicción."""
        model = BayesianSustainabilityModel(target='Sustainable')
        model.fit(small_data)

        X = small_data.drop('Sustainable', axis=1)
        predictions = model.predict(X.head(10))

        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_model_predict_proba(self, small_data):
        """Test probabilidades de predicción."""
        model = BayesianSustainabilityModel(target='Sustainable')
        model.fit(small_data)

        X = small_data.drop('Sustainable', axis=1)
        probs = model.predict_proba(X.head(10))

        assert len(probs) == 10
        assert all(0 <= p <= 1 for p in probs)

    def test_model_query(self, small_data):
        """Test consulta de inferencia."""
        model = BayesianSustainabilityModel(target='Sustainable')
        model.fit(small_data)

        result = model.query(
            variables=['Sustainable'],
            evidence={'A_disc': 'Alto'}
        )

        assert 'variables' in result
        assert 'values' in result
        assert len(result['values']) > 0

    def test_model_get_edges(self, small_data):
        """Test obtención de aristas."""
        model = BayesianSustainabilityModel(target='Sustainable')
        model.fit(small_data)

        edges = model.get_edges()
        assert isinstance(edges, list)
        assert all(isinstance(e, tuple) and len(e) == 2 for e in edges)

    def test_model_not_fitted_error(self):
        """Test error si modelo no está ajustado."""
        model = BayesianSustainabilityModel(target='Sustainable')

        with pytest.raises(RuntimeError):
            model.get_edges()

    def test_different_scoring_methods(self, small_data):
        """Test diferentes métodos de scoring."""
        for method in ['bic', 'k2', 'bdeu']:
            model = BayesianSustainabilityModel(
                target='Sustainable',
                scoring_method=method
            )
            model.fit(small_data)
            assert model._fitted == True

    def test_compare_scoring_methods(self, small_data):
        """Test comparación de métodos de scoring."""
        model = BayesianSustainabilityModel(target='Sustainable')
        model.fit(small_data)

        comparison = model.compare_scoring_methods(small_data)

        assert 'bic' in comparison
        assert 'k2' in comparison
        assert 'bdeu' in comparison

    def test_create_sustainability_model(self, small_data):
        """Test factory function."""
        model = create_sustainability_model(small_data, target='Sustainable')

        assert model._fitted == True
        assert model.target == 'Sustainable'


class TestBayesianValidator:
    """Tests para BayesianValidator."""

    def test_validator_initialization(self):
        """Test inicialización del validador."""
        validator = BayesianValidator(target='Sustainable')
        assert validator.target == 'Sustainable'
        assert validator.scoring_method == 'bic'

    def test_train_test_validation(self, small_data):
        """Test validación train/test."""
        validator = BayesianValidator(target='Sustainable')
        results = validator.train_test_validation(small_data, test_size=0.3)

        assert isinstance(results, ValidationResults)
        assert 0 <= results.accuracy <= 1
        assert 0 <= results.precision <= 1
        assert 0 <= results.recall <= 1
        assert 0 <= results.f1 <= 1

    def test_cross_validate(self, small_data):
        """Test validación cruzada."""
        validator = BayesianValidator(target='Sustainable')
        results = validator.cross_validate(small_data, n_splits=3)

        assert 'summary' in results
        assert 'fold_results' in results
        assert len(results['fold_results']) == 3

        summary = results['summary']
        assert 'accuracy' in summary
        assert 'mean' in summary['accuracy']
        assert 'std' in summary['accuracy']

    def test_bootstrap_stability(self, small_data):
        """Test análisis de estabilidad bootstrap."""
        validator = BayesianValidator(target='Sustainable')
        results = validator.bootstrap_structure_stability(small_data, n_iterations=10)

        assert 'edge_stability' in results
        assert 'n_iterations' in results
        assert results['n_iterations'] == 10

    def test_compare_scoring_methods(self, small_data):
        """Test comparación de métodos de scoring."""
        validator = BayesianValidator(target='Sustainable')
        results = validator.compare_scoring_methods(small_data)

        assert 'methods' in results
        assert 'comparisons' in results
        assert 'bic' in results['methods']


class TestValidationResults:
    """Tests para ValidationResults."""

    def test_validation_results_to_dict(self):
        """Test conversión a diccionario."""
        results = ValidationResults(
            accuracy=0.85,
            precision=0.80,
            recall=0.75,
            f1=0.77,
            auc_roc=0.82,
            confusion_matrix=np.array([[10, 2], [3, 15]]),
            predictions=np.array([0, 1, 1, 0]),
            probabilities=np.array([0.3, 0.7, 0.8, 0.2])
        )

        d = results.to_dict()

        assert d['accuracy'] == 0.85
        assert d['precision'] == 0.80
        assert d['recall'] == 0.75
        assert d['f1'] == 0.77
        assert d['auc_roc'] == 0.82


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
