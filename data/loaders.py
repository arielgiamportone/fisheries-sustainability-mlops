"""
Módulo de carga y procesamiento de datos para análisis de sostenibilidad pesquera.

Este módulo proporciona funciones para:
- Descargar datos de fuentes públicas (Our World in Data, FAO)
- Cargar y procesar datasets locales
- Generar datos sintéticos para testing
- Preparar datos para redes bayesianas

Fuentes de datos:
- Our World in Data: https://ourworldindata.org/fish-and-overfishing
- FAO FishStatJ: https://www.fao.org/fishery/statistics/software/fishstatj/en
- Global Fishing Watch: https://globalfishingwatch.org/datasets-and-code/
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
import ssl
from typing import Optional, Dict, List, Tuple
import warnings

# Directorio base del proyecto
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# URLs de datos de Our World in Data (acceso libre)
OWID_URLS = {
    "fish_seafood_production": "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Fish%20and%20seafood%20production%20-%20FAO%20(2020)/Fish%20and%20seafood%20production%20-%20FAO%20(2020).csv",
    "aquaculture_production": "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Aquaculture%20production%20(FAO%2C%202020)/Aquaculture%20production%20(FAO%2C%202020).csv",
    "capture_fishery": "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Capture%20fishery%20production%20(FAO%2C%202020)/Capture%20fishery%20production%20(FAO%2C%202020).csv",
}

# URLs alternativas directas de OWID grapher
OWID_GRAPHER_URLS = {
    "fish_seafood_production": "https://ourworldindata.org/grapher/fish-seafood-production",
    "aquaculture_production": "https://ourworldindata.org/grapher/aquaculture-farmed-fish-production",
    "capture_vs_aquaculture": "https://ourworldindata.org/grapher/capture-and-aquaculture-production",
    "fish_consumption_capita": "https://ourworldindata.org/grapher/fish-and-seafood-consumption-per-capita",
}


def ensure_directories():
    """Crea los directorios necesarios si no existen."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, filename: str, force: bool = False) -> Path:
    """
    Descarga un archivo desde una URL.

    Args:
        url: URL del archivo a descargar
        filename: Nombre del archivo destino
        force: Si True, descarga aunque el archivo exista

    Returns:
        Path al archivo descargado
    """
    ensure_directories()
    filepath = RAW_DIR / filename

    if filepath.exists() and not force:
        print(f"Archivo ya existe: {filepath}")
        return filepath

    print(f"Descargando {url}...")

    # Crear contexto SSL que ignore verificación (para algunos servidores)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"Descargado exitosamente: {filepath}")
    except Exception as e:
        print(f"Error descargando {url}: {e}")
        # Intentar con contexto SSL relajado
        try:
            with urllib.request.urlopen(url, context=ssl_context) as response:
                with open(filepath, 'wb') as f:
                    f.write(response.read())
            print(f"Descargado exitosamente (SSL relajado): {filepath}")
        except Exception as e2:
            print(f"Error en segundo intento: {e2}")
            return None

    return filepath


def download_owid_data(dataset_name: str = "fish_seafood_production", force: bool = False) -> Optional[pd.DataFrame]:
    """
    Descarga datos de Our World in Data.

    Args:
        dataset_name: Nombre del dataset (fish_seafood_production, aquaculture_production, etc.)
        force: Si True, descarga aunque el archivo exista

    Returns:
        DataFrame con los datos o None si falla
    """
    if dataset_name not in OWID_URLS:
        available = list(OWID_URLS.keys())
        raise ValueError(f"Dataset '{dataset_name}' no disponible. Opciones: {available}")

    url = OWID_URLS[dataset_name]
    filename = f"owid_{dataset_name}.csv"

    filepath = download_file(url, filename, force)

    if filepath and filepath.exists():
        try:
            df = pd.read_csv(filepath)
            print(f"Datos cargados: {len(df)} registros")
            return df
        except Exception as e:
            print(f"Error leyendo CSV: {e}")
            return None

    return None


def generate_synthetic_fisheries_data(
    n_samples: int = 1000,
    years: Tuple[int, int] = (2010, 2023),
    countries: List[str] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Genera datos sintéticos realistas de pesquerías basados en distribuciones
    similares a datos reales de FAO.

    Args:
        n_samples: Número de muestras a generar
        years: Rango de años (inicio, fin)
        countries: Lista de países (si None, usa países principales)
        random_state: Semilla para reproducibilidad

    Returns:
        DataFrame con datos sintéticos
    """
    np.random.seed(random_state)

    if countries is None:
        # Principales países pesqueros según FAO
        countries = [
            'China', 'Indonesia', 'India', 'Vietnam', 'Peru',
            'Russia', 'USA', 'Japan', 'Chile', 'Philippines',
            'Norway', 'Thailand', 'South Korea', 'Malaysia', 'Mexico',
            'Argentina', 'Spain', 'Iceland', 'Morocco', 'Ecuador'
        ]

    # Generar años
    year_range = list(range(years[0], years[1] + 1))

    data = []

    for _ in range(n_samples):
        country = np.random.choice(countries)
        year = np.random.choice(year_range)

        # Producción base según país (China lidera)
        base_production = {
            'China': 80000000, 'Indonesia': 23000000, 'India': 14000000,
            'Vietnam': 8000000, 'Peru': 7000000, 'Russia': 5000000,
            'USA': 5000000, 'Japan': 4000000, 'Chile': 4000000
        }
        base = base_production.get(country, np.random.uniform(500000, 3000000))

        # Variables ambientales
        sst = np.random.normal(22, 5)  # Sea Surface Temperature (°C)
        salinity = np.random.normal(35, 2)  # ppt
        chlorophyll = np.random.lognormal(0, 1)  # mg/m³
        ph = np.random.normal(8.1, 0.2)

        # Variables operativas
        fleet_size = int(np.random.lognormal(8, 1.5))  # número de embarcaciones
        fishing_effort = np.random.lognormal(10, 1)  # horas de pesca
        fuel_consumption = fishing_effort * np.random.uniform(10, 50)  # litros

        # Variables económicas
        fish_price = np.random.lognormal(7, 0.5)  # USD/ton
        fuel_price = np.random.normal(1.2, 0.3)  # USD/litro
        operating_cost = fuel_consumption * fuel_price + np.random.normal(50000, 20000)

        # Producción (depende de esfuerzo y condiciones)
        efficiency = 1.0
        if sst < 15 or sst > 28:
            efficiency *= 0.7
        if chlorophyll < 0.5:
            efficiency *= 0.8

        capture_production = base * (year - 2000) / 20 * efficiency * np.random.uniform(0.8, 1.2) / 1000000
        aquaculture_production = base * (year - 2000) / 15 * np.random.uniform(0.5, 1.5) / 1000000

        # CPUE (Catch Per Unit Effort) - indicador de sostenibilidad
        cpue = capture_production * 1000 / (fishing_effort + 1)

        # Índice de sostenibilidad (calculado basado en múltiples factores)
        # Factores positivos: CPUE alto, bajo consumo combustible relativo
        # Factores negativos: sobreexplotación, alto esfuerzo
        sustainability_score = (
            0.3 * min(cpue / 10, 1) +  # CPUE normalizado
            0.2 * (1 - min(fishing_effort / 100000, 1)) +  # Esfuerzo bajo es mejor
            0.2 * min(aquaculture_production / capture_production if capture_production > 0 else 0, 1) +  # Más acuicultura es mejor
            0.15 * (1 if 18 < sst < 26 else 0.5) +  # Temperatura óptima
            0.15 * (1 if chlorophyll > 0.5 else 0.5)  # Productividad
        )

        # Clasificación binaria de sostenibilidad
        sustainable = 1 if sustainability_score > 0.5 else 0

        data.append({
            'Country': country,
            'Year': year,
            'SST_C': round(sst, 2),
            'Salinity_ppt': round(salinity, 2),
            'Chlorophyll_mg_m3': round(chlorophyll, 3),
            'pH': round(ph, 2),
            'Fleet_Size': fleet_size,
            'Fishing_Effort_hours': round(fishing_effort, 0),
            'Fuel_Consumption_L': round(fuel_consumption, 0),
            'Fish_Price_USD_ton': round(fish_price, 2),
            'Fuel_Price_USD_L': round(fuel_price, 2),
            'Operating_Cost_USD': round(operating_cost, 0),
            'Capture_Production_Mt': round(capture_production, 3),
            'Aquaculture_Production_Mt': round(aquaculture_production, 3),
            'Total_Production_Mt': round(capture_production + aquaculture_production, 3),
            'CPUE': round(cpue, 4),
            'Sustainability_Score': round(sustainability_score, 3),
            'Sustainable': sustainable
        })

    df = pd.DataFrame(data)
    return df


def discretize_for_bayesian(
    df: pd.DataFrame,
    columns: List[str] = None,
    n_bins: int = 3,
    labels: List[str] = None
) -> pd.DataFrame:
    """
    Discretiza variables continuas para uso en redes bayesianas.

    Args:
        df: DataFrame con datos
        columns: Columnas a discretizar (si None, todas las numéricas)
        n_bins: Número de categorías
        labels: Etiquetas para las categorías

    Returns:
        DataFrame con columnas discretizadas añadidas
    """
    df_disc = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    if labels is None:
        if n_bins == 3:
            labels = ['Bajo', 'Medio', 'Alto']
        elif n_bins == 2:
            labels = ['Bajo', 'Alto']
        else:
            labels = [f'Cat_{i}' for i in range(n_bins)]

    for col in columns:
        if col in df.columns and df[col].dtype in [np.float64, np.int64, float, int]:
            try:
                df_disc[f'{col}_disc'] = pd.qcut(
                    df[col],
                    q=n_bins,
                    labels=labels,
                    duplicates='drop'
                )
            except ValueError:
                # Si hay muchos valores repetidos, usar cut en lugar de qcut
                df_disc[f'{col}_disc'] = pd.cut(
                    df[col],
                    bins=n_bins,
                    labels=labels[:n_bins]
                )

    return df_disc


def prepare_bayesian_dataset(
    df: pd.DataFrame,
    target: str = 'Sustainable',
    features: List[str] = None,
    discretize: bool = True
) -> pd.DataFrame:
    """
    Prepara dataset para análisis con redes bayesianas.

    Args:
        df: DataFrame original
        target: Variable objetivo
        features: Lista de features a incluir
        discretize: Si True, discretiza variables continuas

    Returns:
        DataFrame preparado para pgmpy
    """
    if features is None:
        features = [
            'SST_C', 'Salinity_ppt', 'Chlorophyll_mg_m3', 'pH',
            'Fleet_Size', 'Fishing_Effort_hours', 'Fuel_Consumption_L',
            'Fish_Price_USD_ton', 'Operating_Cost_USD', 'CPUE'
        ]

    # Filtrar columnas existentes
    available_features = [f for f in features if f in df.columns]

    if target in df.columns:
        cols = available_features + [target]
    else:
        cols = available_features

    df_prep = df[cols].copy()

    if discretize:
        df_prep = discretize_for_bayesian(df_prep, columns=available_features)
        # Mantener solo columnas discretizadas y target
        disc_cols = [c for c in df_prep.columns if c.endswith('_disc')] + [target]
        disc_cols = [c for c in disc_cols if c in df_prep.columns]
        df_prep = df_prep[disc_cols]

    return df_prep


def load_or_generate_data(
    use_synthetic: bool = True,
    n_samples: int = 1000,
    save: bool = True
) -> pd.DataFrame:
    """
    Carga datos existentes o genera nuevos datos sintéticos.

    Args:
        use_synthetic: Si True, genera datos sintéticos
        n_samples: Número de muestras si genera sintéticos
        save: Si True, guarda los datos generados

    Returns:
        DataFrame con datos
    """
    ensure_directories()

    processed_file = PROCESSED_DIR / "fisheries_data.csv"

    # Intentar cargar datos procesados existentes
    if processed_file.exists() and not use_synthetic:
        print(f"Cargando datos existentes: {processed_file}")
        return pd.read_csv(processed_file)

    if use_synthetic:
        print(f"Generando {n_samples} muestras sintéticas...")
        df = generate_synthetic_fisheries_data(n_samples=n_samples)

        if save:
            df.to_csv(processed_file, index=False)
            print(f"Datos guardados en: {processed_file}")

        return df

    # Intentar descargar datos reales
    print("Intentando descargar datos de Our World in Data...")
    df = download_owid_data("fish_seafood_production")

    if df is not None and save:
        df.to_csv(processed_file, index=False)
        print(f"Datos guardados en: {processed_file}")

    return df


# === FUNCIONES AUXILIARES ===

def get_data_summary(df: pd.DataFrame) -> Dict:
    """Retorna un resumen del dataset."""
    summary = {
        'n_samples': len(df),
        'n_features': len(df.columns),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_stats': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
    }
    return summary


def print_data_info(df: pd.DataFrame):
    """Imprime información del dataset."""
    print("=" * 60)
    print("INFORMACIÓN DEL DATASET")
    print("=" * 60)
    print(f"Registros: {len(df)}")
    print(f"Columnas: {len(df.columns)}")
    print(f"\nColumnas disponibles:")
    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        print(f"  - {col}: {dtype} (nulls: {null_count})")

    if 'Sustainable' in df.columns:
        print(f"\nDistribución de Sostenibilidad:")
        print(df['Sustainable'].value_counts())

    print("=" * 60)


if __name__ == "__main__":
    # Test del módulo
    print("Testing data loaders...")

    # Generar datos sintéticos
    df = generate_synthetic_fisheries_data(n_samples=500)
    print_data_info(df)

    # Preparar para bayesian
    df_bayesian = prepare_bayesian_dataset(df)
    print(f"\nDataset para Bayesian Network:")
    print(df_bayesian.head())
