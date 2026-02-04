"""
Tests unitarios para el módulo causal.

Ejecutar con: pytest tests/test_causal.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.causal.dag import CausalDAG, SustainabilityDAG, CausalRelation
from src.causal.interventions import CausalInterventions, InterventionResult


class TestCausalRelation:
    """Tests para CausalRelation."""

    def test_relation_creation(self):
        """Test creación de relación causal."""
        relation = CausalRelation(
            cause='Temperature',
            effect='Sustainability',
            strength=0.7,
            mechanism='Higher temperature reduces fish stocks'
        )

        assert relation.cause == 'Temperature'
        assert relation.effect == 'Sustainability'
        assert relation.strength == 0.7

    def test_relation_as_tuple(self):
        """Test conversión a tupla."""
        relation = CausalRelation(cause='A', effect='B')
        assert relation.as_tuple() == ('A', 'B')


class TestCausalDAG:
    """Tests para CausalDAG."""

    def test_dag_initialization(self):
        """Test inicialización del DAG."""
        dag = CausalDAG()
        assert len(dag.nodes) == 0
        assert len(dag.edges) == 0

    def test_dag_with_initial_edges(self):
        """Test inicialización con aristas."""
        edges = [('A', 'B'), ('B', 'C'), ('A', 'C')]
        dag = CausalDAG(edges=edges)

        assert len(dag.nodes) == 3
        assert len(dag.edges) == 3

    def test_add_edge(self):
        """Test añadir arista."""
        dag = CausalDAG()
        dag.add_edge('X', 'Y')
        dag.add_edge('Y', 'Z')

        assert ('X', 'Y') in dag.edges
        assert ('Y', 'Z') in dag.edges
        assert len(dag.nodes) == 3

    def test_add_edges_from(self):
        """Test añadir múltiples aristas."""
        dag = CausalDAG()
        dag.add_edges_from([('A', 'B'), ('B', 'C')])

        assert len(dag.edges) == 2

    def test_validate_dag(self):
        """Test validación de DAG."""
        dag = CausalDAG([('A', 'B'), ('B', 'C')])
        result = dag.validate()

        assert result['is_valid_dag'] == True
        assert result['n_nodes'] == 3
        assert result['n_edges'] == 2
        assert 'topological_order' in result

    def test_validate_cyclic_graph(self):
        """Test detección de ciclos."""
        dag = CausalDAG()
        dag.add_edge('A', 'B')
        dag.add_edge('B', 'C')
        dag.add_edge('C', 'A')  # Crea ciclo

        result = dag.validate()
        assert result['is_valid_dag'] == False
        assert len(result['cycles']) > 0

    def test_get_parents(self):
        """Test obtener padres."""
        dag = CausalDAG([('A', 'C'), ('B', 'C')])
        parents = dag.get_parents('C')

        assert set(parents) == {'A', 'B'}

    def test_get_children(self):
        """Test obtener hijos."""
        dag = CausalDAG([('A', 'B'), ('A', 'C')])
        children = dag.get_children('A')

        assert set(children) == {'B', 'C'}

    def test_get_ancestors(self):
        """Test obtener ancestros."""
        dag = CausalDAG([('A', 'B'), ('B', 'C'), ('C', 'D')])
        ancestors = dag.get_ancestors('D')

        assert ancestors == {'A', 'B', 'C'}

    def test_get_descendants(self):
        """Test obtener descendientes."""
        dag = CausalDAG([('A', 'B'), ('B', 'C'), ('B', 'D')])
        descendants = dag.get_descendants('A')

        assert descendants == {'B', 'C', 'D'}

    def test_get_markov_blanket(self):
        """Test obtener Markov blanket."""
        dag = CausalDAG([
            ('A', 'B'),
            ('C', 'B'),
            ('B', 'D'),
            ('E', 'D')
        ])

        blanket = dag.get_markov_blanket('B')
        # Markov blanket de B: padres (A, C), hijos (D), padres de hijos (E)
        assert blanket == {'A', 'C', 'D', 'E'}

    def test_compare_with_learned(self):
        """Test comparación con estructura aprendida."""
        theoretical = CausalDAG([('A', 'B'), ('B', 'C'), ('A', 'C')])
        learned_edges = [('A', 'B'), ('B', 'C'), ('D', 'C')]

        comparison = theoretical.compare_with_learned(learned_edges)

        assert len(comparison['common_edges']) == 2
        assert len(comparison['only_in_theoretical']) == 1
        assert len(comparison['only_in_learned']) == 1
        assert 0 <= comparison['jaccard'] <= 1

    def test_d_separation(self):
        """Test d-separación."""
        dag = CausalDAG([('A', 'C'), ('B', 'C')])

        # En estructura de colisionador A → C ← B:
        # - Sin condicionar en C: A y B ESTÁN d-separados (independientes)
        # - Condicionando en C: A y B NO están d-separados (dependientes)
        assert dag.is_d_separated('A', 'B', set())  # d-separados sin C
        assert not dag.is_d_separated('A', 'B', {'C'})  # no d-separados dado C

    def test_to_dot(self):
        """Test generación de formato DOT."""
        dag = CausalDAG([('A', 'B')])
        dot = dag.to_dot()

        assert 'digraph' in dot
        assert '"A" -> "B"' in dot


class TestSustainabilityDAG:
    """Tests para SustainabilityDAG."""

    def test_fisheries_dag(self):
        """Test DAG de pesquerías."""
        dag = SustainabilityDAG(domain='fisheries')

        assert dag.domain == 'fisheries'
        assert len(dag.edges) > 0
        validation = dag.validate()
        assert validation['is_valid_dag'] == True

    def test_ras_dag(self):
        """Test DAG de sistemas RAS."""
        dag = SustainabilityDAG(domain='ras')

        assert dag.domain == 'ras'
        assert len(dag.edges) > 0
        validation = dag.validate()
        assert validation['is_valid_dag'] == True

    def test_invalid_domain(self):
        """Test dominio inválido."""
        with pytest.raises(ValueError):
            SustainabilityDAG(domain='invalid')

    def test_get_treatment_outcome_pairs(self):
        """Test obtener pares tratamiento-resultado."""
        dag = SustainabilityDAG(domain='fisheries')
        pairs = dag.get_treatment_outcome_pairs()

        assert len(pairs) > 0
        assert 'treatment' in pairs[0]
        assert 'outcome' in pairs[0]


class TestInterventionResult:
    """Tests para InterventionResult."""

    def test_intervention_result_creation(self):
        """Test creación de resultado de intervención."""
        result = InterventionResult(
            treatment='X',
            treatment_value='Alto',
            outcome='Y',
            baseline_prob={'0': 0.6, '1': 0.4},
            intervention_prob={'0': 0.4, '1': 0.6},
            ate=0.2,
            relative_effect=0.5
        )

        assert result.treatment == 'X'
        assert result.ate == 0.2

    def test_intervention_result_summary(self):
        """Test resumen de resultado."""
        result = InterventionResult(
            treatment='X',
            treatment_value='Alto',
            outcome='Y',
            baseline_prob={'0': 0.6, '1': 0.4},
            intervention_prob={'0': 0.4, '1': 0.6},
            ate=0.2,
            relative_effect=0.5
        )

        summary = result.summary()
        assert 'Intervención' in summary
        assert 'ATE' in summary


# Tests de integración con modelo real requieren datos,
# se ejecutan condicionalmente
class TestCausalInterventionsIntegration:
    """Tests de integración para CausalInterventions."""

    @pytest.fixture
    def fitted_model(self):
        """Crea un modelo ajustado para tests."""
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
        from pgmpy.estimators import MaximumLikelihoodEstimator

        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'A': np.random.choice(['Bajo', 'Alto'], n),
            'B': np.random.choice(['Bajo', 'Alto'], n),
            'Sustainable': np.random.choice([0, 1], n)
        })

        model = BayesianNetwork([('A', 'Sustainable'), ('B', 'Sustainable')])
        model.fit(data, estimator=MaximumLikelihoodEstimator)

        return model, data

    def test_do_intervention(self, fitted_model):
        """Test intervención do()."""
        model, data = fitted_model
        ci = CausalInterventions(model, target='Sustainable')

        result = ci.do_intervention('A', 'Alto')

        assert isinstance(result, InterventionResult)
        assert result.treatment == 'A'
        assert result.treatment_value == 'Alto'

    def test_compare_interventions(self, fitted_model):
        """Test comparación de intervenciones."""
        model, data = fitted_model
        ci = CausalInterventions(model, target='Sustainable')

        results = ci.compare_interventions('A')

        assert 'Bajo' in results or 'Alto' in results
        assert all(isinstance(r, InterventionResult) for r in results.values())

    def test_sensitivity_to_intervention(self, fitted_model):
        """Test sensibilidad a intervenciones."""
        model, data = fitted_model
        ci = CausalInterventions(model, target='Sustainable')

        df = ci.sensitivity_to_intervention('A')

        assert isinstance(df, pd.DataFrame)
        assert 'treatment_value' in df.columns
        assert 'ate' in df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
