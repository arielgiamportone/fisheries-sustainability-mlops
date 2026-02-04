"""
Funciones de visualización para análisis de sostenibilidad pesquera.

Proporciona gráficos para redes bayesianas, métricas de validación,
análisis causal y EDA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
from pgmpy.models import BayesianNetwork


def plot_bayesian_network(
    model: BayesianNetwork,
    title: str = "Red Bayesiana",
    figsize: Tuple[int, int] = (12, 8),
    node_color: str = 'lightblue',
    edge_color: str = 'gray',
    target_node: Optional[str] = None,
    target_color: str = 'lightgreen',
    layout: str = 'spring',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza una red bayesiana.

    Args:
        model: Modelo BayesianNetwork de pgmpy
        title: Título del gráfico
        figsize: Tamaño de la figura
        node_color: Color de los nodos
        edge_color: Color de las aristas
        target_node: Nodo objetivo a destacar
        target_color: Color del nodo objetivo
        layout: Tipo de layout ('spring', 'circular', 'kamada_kawai')
        save_path: Ruta para guardar la figura

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Crear grafo de NetworkX
    G = nx.DiGraph()
    G.add_edges_from(model.edges())

    # Seleccionar layout
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=42, k=2)
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42)

    # Colores de nodos
    colors = []
    for node in G.nodes():
        if target_node and node == target_node:
            colors.append(target_color)
        else:
            colors.append(node_color)

    # Dibujar
    nx.draw(G, pos, ax=ax,
            with_labels=True,
            node_color=colors,
            edge_color=edge_color,
            node_size=2500,
            font_size=9,
            font_weight='bold',
            arrows=True,
            arrowsize=20,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.1')

    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = None,
    title: str = "Matriz de Confusión",
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza una matriz de confusión.

    Args:
        cm: Matriz de confusión (numpy array)
        labels: Etiquetas de las clases
        title: Título del gráfico
        figsize: Tamaño de la figura
        cmap: Mapa de colores
        save_path: Ruta para guardar

    Returns:
        Figura de matplotlib
    """
    if labels is None:
        labels = ['Insostenible', 'Sostenible']

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=labels, yticklabels=labels,
                ax=ax, cbar_kws={'shrink': 0.8})

    ax.set_xlabel('Predicción', fontsize=12)
    ax.set_ylabel('Valor Real', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    # Añadir métricas
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    metrics_text = f'Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}'
    ax.text(0.5, -0.15, metrics_text, transform=ax.transAxes,
            ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_metrics_comparison(
    metrics: Dict[str, Dict],
    title: str = "Comparación de Métricas",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza comparación de métricas de validación cruzada.

    Args:
        metrics: Diccionario con métricas (estructura de cross_validate)
        title: Título del gráfico
        figsize: Tamaño de la figura
        save_path: Ruta para guardar

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    metric_names = list(metrics.keys())
    means = [metrics[m]['mean'] for m in metric_names]
    stds = [metrics[m]['std'] for m in metric_names]

    # Gráfico de barras con error
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    bars = axes[0].bar(metric_names, means, yerr=stds, capsize=5,
                       color=colors[:len(metric_names)], alpha=0.8)

    axes[0].set_ylabel('Score')
    axes[0].set_title('Media ± Desv. Est.')
    axes[0].set_ylim(0, 1)

    # Añadir valores
    for bar, mean, std in zip(bars, means, stds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                     f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

    # Boxplot de valores por fold
    data_for_box = [metrics[m]['values'] for m in metric_names]
    bp = axes[1].boxplot(data_for_box, labels=metric_names, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors[:len(metric_names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1].set_ylabel('Score')
    axes[1].set_title('Distribución por Fold')
    axes[1].set_ylim(0, 1)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_edge_stability(
    edge_stability: Dict,
    top_n: int = 15,
    title: str = "Estabilidad de Aristas (Bootstrap)",
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza la estabilidad de aristas del análisis bootstrap.

    Args:
        edge_stability: Diccionario de estabilidad de aristas
        top_n: Número de aristas a mostrar
        title: Título del gráfico
        figsize: Tamaño de la figura
        save_path: Ruta para guardar

    Returns:
        Figura de matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ordenar por frecuencia
    sorted_edges = sorted(edge_stability.items(),
                         key=lambda x: x[1]['percentage'], reverse=True)[:top_n]

    labels = [f"{e[0][:12]}→{e[1][:12]}" for e, _ in sorted_edges]
    percentages = [s['percentage'] for _, s in sorted_edges]

    # Colores según estabilidad
    colors = []
    for pct in percentages:
        if pct >= 80:
            colors.append('#2ecc71')  # Verde - alta
        elif pct >= 50:
            colors.append('#f39c12')  # Naranja - media
        else:
            colors.append('#e74c3c')  # Rojo - baja

    bars = ax.barh(labels, percentages, color=colors, alpha=0.8)

    # Líneas de referencia
    ax.axvline(x=80, color='green', linestyle='--', linewidth=1.5,
               label='Alta estabilidad (80%)')
    ax.axvline(x=50, color='orange', linestyle='--', linewidth=1.5,
               label='Media estabilidad (50%)')

    ax.set_xlabel('Frecuencia en Bootstrap (%)')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0, 105)

    # Añadir valores
    for bar, pct in zip(bars, percentages):
        ax.text(pct + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_intervention_effects(
    results: pd.DataFrame,
    treatment: str,
    outcome: str,
    title: str = None,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualiza efectos de intervenciones causales.

    Args:
        results: DataFrame con resultados de intervenciones
        treatment: Nombre de la variable de tratamiento
        outcome: Nombre de la variable de resultado
        title: Título del gráfico
        figsize: Tamaño de la figura
        save_path: Ruta para guardar

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if title is None:
        title = f"Efecto de {treatment} en {outcome}"

    # Gráfico 1: Probabilidades por valor de tratamiento
    prob_cols = [c for c in results.columns if c.startswith('P(')]
    if prob_cols:
        results.plot(x='treatment_value', y=prob_cols, kind='bar',
                    ax=axes[0], alpha=0.8)
        axes[0].set_xlabel(treatment)
        axes[0].set_ylabel('Probabilidad')
        axes[0].set_title('Distribución del Outcome')
        axes[0].legend(title='Estado')
        axes[0].tick_params(axis='x', rotation=45)

    # Gráfico 2: ATE por valor de tratamiento
    colors = ['green' if x >= 0 else 'red' for x in results['ate']]
    bars = axes[1].bar(results['treatment_value'], results['ate'],
                       color=colors, alpha=0.7)

    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_xlabel(treatment)
    axes[1].set_ylabel('Average Treatment Effect')
    axes[1].set_title('Efecto Causal Estimado')
    axes[1].tick_params(axis='x', rotation=45)

    # Añadir valores
    for bar, ate in zip(bars, results['ate']):
        ypos = bar.get_height() + 0.01 if ate >= 0 else bar.get_height() - 0.03
        axes[1].text(bar.get_x() + bar.get_width()/2, ypos,
                     f'{ate:+.3f}', ha='center', va='bottom' if ate >= 0 else 'top',
                     fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_dag_comparison(
    theoretical_edges: Set[Tuple[str, str]],
    learned_edges: Set[Tuple[str, str]],
    title: str = "Comparación: DAG Teórico vs Aprendido",
    figsize: Tuple[int, int] = (16, 7),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compara visualmente un DAG teórico con una estructura aprendida.

    Args:
        theoretical_edges: Aristas del DAG teórico
        learned_edges: Aristas aprendidas
        title: Título del gráfico
        figsize: Tamaño de la figura
        save_path: Ruta para guardar

    Returns:
        Figura de matplotlib
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Crear grafos
    G_theoretical = nx.DiGraph()
    G_theoretical.add_edges_from(theoretical_edges)

    G_learned = nx.DiGraph()
    G_learned.add_edges_from(learned_edges)

    # Todos los nodos
    all_nodes = set(G_theoretical.nodes()) | set(G_learned.nodes())

    # DAG Teórico
    pos_t = nx.spring_layout(G_theoretical, seed=42, k=2)
    nx.draw(G_theoretical, pos_t, ax=axes[0],
            with_labels=True,
            node_color='lightgreen',
            edge_color='darkgreen',
            node_size=2000,
            font_size=8,
            arrows=True,
            arrowsize=15)
    axes[0].set_title('DAG Teórico', fontsize=12, fontweight='bold')

    # DAG Aprendido
    pos_l = nx.spring_layout(G_learned, seed=42, k=2)
    nx.draw(G_learned, pos_l, ax=axes[1],
            with_labels=True,
            node_color='lightblue',
            edge_color='darkblue',
            node_size=2000,
            font_size=8,
            arrows=True,
            arrowsize=15)
    axes[1].set_title('Estructura Aprendida', fontsize=12, fontweight='bold')

    # Calcular métricas
    common = theoretical_edges & learned_edges
    only_theoretical = theoretical_edges - learned_edges
    only_learned = learned_edges - theoretical_edges
    jaccard = len(common) / len(theoretical_edges | learned_edges) if (theoretical_edges | learned_edges) else 0

    metrics_text = (f"Aristas comunes: {len(common)} | "
                   f"Solo teóricas: {len(only_theoretical)} | "
                   f"Solo aprendidas: {len(only_learned)} | "
                   f"Jaccard: {jaccard:.3f}")

    fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, style='italic')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
