"""
Módulo para definición y manipulación de DAGs causales.

Proporciona clases para definir estructuras causales teóricas,
validarlas contra datos y compararlas con estructuras aprendidas.
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field


@dataclass
class CausalRelation:
    """Representa una relación causal entre dos variables."""
    cause: str
    effect: str
    strength: Optional[float] = None
    confidence: Optional[float] = None
    mechanism: Optional[str] = None

    def as_tuple(self) -> Tuple[str, str]:
        return (self.cause, self.effect)

    def __repr__(self) -> str:
        return f"{self.cause} -> {self.effect}"


class CausalDAG:
    """
    Clase para definir y manipular DAGs causales.

    Permite definir relaciones causales teóricas, validarlas,
    y compararlas con estructuras aprendidas de datos.

    Attributes:
        edges: Lista de aristas causales
        nodes: Conjunto de nodos
        graph: Grafo dirigido de NetworkX

    Example:
        >>> dag = CausalDAG()
        >>> dag.add_edge('Temperature', 'FeedingRate')
        >>> dag.add_edge('FeedingRate', 'Sustainability')
        >>> dag.validate()
    """

    def __init__(self, edges: Optional[List[Tuple[str, str]]] = None):
        """
        Inicializa el DAG causal.

        Args:
            edges: Lista opcional de aristas iniciales
        """
        self.graph = nx.DiGraph()
        self.relations: Dict[Tuple[str, str], CausalRelation] = {}

        if edges:
            for edge in edges:
                self.add_edge(edge[0], edge[1])

    def add_edge(
        self,
        cause: str,
        effect: str,
        strength: Optional[float] = None,
        mechanism: Optional[str] = None
    ) -> 'CausalDAG':
        """
        Añade una relación causal.

        Args:
            cause: Variable causa
            effect: Variable efecto
            strength: Fuerza de la relación (opcional)
            mechanism: Descripción del mecanismo (opcional)

        Returns:
            self para encadenamiento
        """
        self.graph.add_edge(cause, effect)
        relation = CausalRelation(
            cause=cause,
            effect=effect,
            strength=strength,
            mechanism=mechanism
        )
        self.relations[(cause, effect)] = relation
        return self

    def add_edges_from(self, edges: List[Tuple[str, str]]) -> 'CausalDAG':
        """Añade múltiples aristas."""
        for edge in edges:
            self.add_edge(edge[0], edge[1])
        return self

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """Retorna lista de aristas."""
        return list(self.graph.edges())

    @property
    def nodes(self) -> List[str]:
        """Retorna lista de nodos."""
        return list(self.graph.nodes())

    def validate(self) -> Dict:
        """
        Valida que el grafo sea un DAG válido.

        Returns:
            Diccionario con resultados de validación
        """
        is_dag = nx.is_directed_acyclic_graph(self.graph)

        results = {
            'is_valid_dag': is_dag,
            'n_nodes': self.graph.number_of_nodes(),
            'n_edges': self.graph.number_of_edges(),
            'cycles': [] if is_dag else list(nx.simple_cycles(self.graph))
        }

        if is_dag:
            results['topological_order'] = list(nx.topological_sort(self.graph))
            results['root_nodes'] = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
            results['leaf_nodes'] = [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]

        return results

    def get_parents(self, node: str) -> List[str]:
        """Obtiene los padres directos de un nodo."""
        return list(self.graph.predecessors(node))

    def get_children(self, node: str) -> List[str]:
        """Obtiene los hijos directos de un nodo."""
        return list(self.graph.successors(node))

    def get_ancestors(self, node: str) -> Set[str]:
        """Obtiene todos los ancestros de un nodo."""
        return nx.ancestors(self.graph, node)

    def get_descendants(self, node: str) -> Set[str]:
        """Obtiene todos los descendientes de un nodo."""
        return nx.descendants(self.graph, node)

    def get_markov_blanket(self, node: str) -> Set[str]:
        """
        Obtiene el Markov blanket de un nodo.
        (padres, hijos, y padres de los hijos)
        """
        parents = set(self.get_parents(node))
        children = set(self.get_children(node))
        parents_of_children = set()
        for child in children:
            parents_of_children.update(self.get_parents(child))

        blanket = parents | children | parents_of_children
        blanket.discard(node)
        return blanket

    def compare_with_learned(
        self,
        learned_edges: List[Tuple[str, str]]
    ) -> Dict:
        """
        Compara el DAG teórico con una estructura aprendida.

        Args:
            learned_edges: Lista de aristas de la estructura aprendida

        Returns:
            Diccionario con métricas de comparación
        """
        theoretical = set(self.edges)
        learned = set(learned_edges)

        # Conjuntos
        common = theoretical & learned
        only_theoretical = theoretical - learned
        only_learned = learned - theoretical

        # Métricas
        precision = len(common) / len(learned) if learned else 0
        recall = len(common) / len(theoretical) if theoretical else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        jaccard = len(common) / len(theoretical | learned) if (theoretical | learned) else 0

        return {
            'common_edges': list(common),
            'only_in_theoretical': list(only_theoretical),
            'only_in_learned': list(only_learned),
            'n_common': len(common),
            'n_only_theoretical': len(only_theoretical),
            'n_only_learned': len(only_learned),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'jaccard': jaccard
        }

    def is_d_separated(
        self,
        x: str,
        y: str,
        z: Optional[Set[str]] = None
    ) -> bool:
        """
        Verifica si X e Y están d-separados dado Z.

        Args:
            x: Primera variable
            y: Segunda variable
            z: Conjunto de variables condicionantes

        Returns:
            True si X e Y están d-separados dado Z
        """
        z = z or set()
        return nx.is_d_separator(self.graph, {x}, {y}, z)

    def to_dot(self) -> str:
        """Genera representación DOT del grafo."""
        lines = ["digraph {"]
        for edge in self.edges:
            lines.append(f'    "{edge[0]}" -> "{edge[1]}";')
        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CausalDAG(nodes={len(self.nodes)}, edges={len(self.edges)})"


class SustainabilityDAG(CausalDAG):
    """
    DAG causal predefinido para análisis de sostenibilidad pesquera.

    Define la estructura causal teórica basada en conocimiento del dominio.
    """

    # Estructura causal teórica para pesquerías
    FISHERIES_DAG = [
        # Variables ambientales -> Operativas
        ('SST_C_disc', 'Fishing_Effort_hours_disc'),
        ('Chlorophyll_mg_m3_disc', 'CPUE_disc'),

        # Variables operativas -> Económicas
        ('Fishing_Effort_hours_disc', 'Fuel_Consumption_L_disc'),
        ('Fleet_Size_disc', 'Operating_Cost_USD_disc'),

        # Variables -> Sostenibilidad
        ('CPUE_disc', 'Sustainable'),
        ('Fishing_Effort_hours_disc', 'Sustainable'),
        ('Operating_Cost_USD_disc', 'Sustainable'),
        ('Fish_Price_USD_ton_disc', 'Sustainable'),
    ]

    # Estructura causal para sistemas RAS
    RAS_DAG = [
        # Ambientales -> Tasa de Alimentación
        ('Temperatura_disc', 'Tasa_Alimentacion_disc'),
        ('Salinidad_disc', 'Tasa_Alimentacion_disc'),
        ('pH_disc', 'Tasa_Alimentacion_disc'),

        # Operativas -> Sostenibilidad
        ('Tasa_Alimentacion_disc', 'Sostenibilidad'),
        ('Flota_disc', 'Sostenibilidad'),
        ('Precio_disc', 'Sostenibilidad'),
        ('Mantenimiento_disc', 'Sostenibilidad'),
    ]

    def __init__(self, domain: str = 'fisheries'):
        """
        Inicializa el DAG de sostenibilidad.

        Args:
            domain: Dominio ('fisheries' o 'ras')
        """
        if domain == 'fisheries':
            edges = self.FISHERIES_DAG
        elif domain == 'ras':
            edges = self.RAS_DAG
        else:
            raise ValueError(f"Dominio no válido: {domain}. Use 'fisheries' o 'ras'")

        super().__init__(edges)
        self.domain = domain

    def get_treatment_outcome_pairs(self) -> List[Dict]:
        """
        Retorna pares tratamiento-resultado para análisis causal.

        Returns:
            Lista de diccionarios con pares y confusores
        """
        if self.domain == 'fisheries':
            return [
                {
                    'treatment': 'Fishing_Effort_hours_disc',
                    'outcome': 'Sustainable',
                    'confounders': ['SST_C_disc']
                },
                {
                    'treatment': 'CPUE_disc',
                    'outcome': 'Sustainable',
                    'confounders': ['Chlorophyll_mg_m3_disc']
                }
            ]
        else:  # RAS
            return [
                {
                    'treatment': 'Tasa_Alimentacion_disc',
                    'outcome': 'Sostenibilidad',
                    'confounders': ['Temperatura_disc', 'pH_disc']
                }
            ]

    def __repr__(self) -> str:
        return f"SustainabilityDAG(domain='{self.domain}', nodes={len(self.nodes)}, edges={len(self.edges)})"
