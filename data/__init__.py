"""
Módulo de datos para el proyecto de Sostenibilidad Pesquera.

Este paquete proporciona funciones para:
- Cargar datos de fuentes públicas (FAO, OWID)
- Generar datos sintéticos realistas
- Preparar datos para redes bayesianas
- Discretizar variables continuas
"""

from .loaders import (
    download_owid_data,
    generate_synthetic_fisheries_data,
    discretize_for_bayesian,
    prepare_bayesian_dataset,
    load_or_generate_data,
    get_data_summary,
    print_data_info,
    RAW_DIR,
    PROCESSED_DIR,
    DATA_DIR
)

__all__ = [
    'download_owid_data',
    'generate_synthetic_fisheries_data',
    'discretize_for_bayesian',
    'prepare_bayesian_dataset',
    'load_or_generate_data',
    'get_data_summary',
    'print_data_info',
    'RAW_DIR',
    'PROCESSED_DIR',
    'DATA_DIR'
]
