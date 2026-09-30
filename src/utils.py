"""Funções utilitárias para carregamento e processamento de dados"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Carrega dados da base Kaggle Bank Marketing
    
    Args:
        filepath: caminho para arquivo CSV
        
    Returns:
        DataFrame com dados brutos
    """
    logger.info(f"Carregando dados de {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def clean_and_prepare_data(df: pd.DataFrame, drop_duration: bool = True) -> pd.DataFrame:
    """
    Limpa e prepara dados para modelagem
    
    Args:
        df: DataFrame bruto
        drop_duration: remover coluna 'duration' (vazamento temporal)
        
    Returns:
        DataFrame limpo e preparado
    """
    df = df.copy()
    
    # Remover coluna duration (vazamento temporal)
    if drop_duration and 'duration' in df.columns:
        logger.info("Removendo coluna 'duration' (vazamento temporal)")
        df = df.drop(columns=['duration'])
    
    # Remover duplicatas
    initial_shape = df.shape[0]
    df = df.drop_duplicates()
    if df.shape[0] < initial_shape:
        logger.info(f"Removidas {initial_shape - df.shape[0]} linhas duplicadas")
    
    # Remover valores faltantes
    missing = df.isnull().sum()
    if missing.any():
        logger.info(f"Removendo {df[df.isnull().any(axis=1)].shape[0]} linhas com valores faltantes")
        df = df.dropna()
    
    return df


def encode_features(df: pd.DataFrame, target_col: str = 'y') -> tuple:
    """
    Codifica features categóricas e retorna arrays X e y
    
    Args:
        df: DataFrame com dados limpos
        target_col: nome da coluna target
        
    Returns:
        Tupla (X_encoded, y_encoded, feature_names, encoders)
    """
    df = df.copy()
    
    # Separar features e target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Codificar target (sim -> 1, não -> 0)
    y_encoded = (y == 'yes').astype(int)
    
    # Identificar colunas categóricas e numéricas
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    logger.info(f"Colunas categóricas: {categorical_cols}")
    logger.info(f"Colunas numéricas: {numeric_cols}")
    
    # Codificar categóricas
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        logger.info(f"Codificada coluna '{col}': {len(le.classes_)} classes")
    
    feature_names = X.columns.tolist()
    X_encoded = X.values
    y_encoded = y_encoded.values
    
    return X_encoded, y_encoded, feature_names, encoders


def scale_features(X_train: np.ndarray, X_test: np.ndarray = None) -> tuple:
    """
    Normaliza features usando StandardScaler
    
    Args:
        X_train: dados de treinamento
        X_test: dados de teste (opcional)
        
    Returns:
        Tupla (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, None, scaler


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
               random_state: int = 42) -> tuple:
    """
    Divide dados em treino e teste
    
    Args:
        X: features
        y: target
        test_size: proporção para teste
        random_state: seed para reprodutibilidade
        
    Returns:
        Tupla (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Dados divididos: treino {X_train.shape[0]}, teste {X_test.shape[0]}")
    logger.info(f"Taxa de conversão (treino): {y_train.mean():.2%}")
    logger.info(f"Taxa de conversão (teste): {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test


def create_golden_set(X_test: np.ndarray, y_test: np.ndarray, n_samples: int = 5) -> tuple:
    """
    Cria golden set (casos de teste) variados
    
    Args:
        X_test: dados de teste
        y_test: targets de teste
        n_samples: número de exemplos
        
    Returns:
        Tupla (X_golden, y_golden, indices)
    """
    # Selecionar exemplos balanceados
    indices_yes = np.where(y_test == 1)[0]
    indices_no = np.where(y_test == 0)[0]
    
    # Garantir mix de conversões positivas e negativas
    n_yes = max(1, n_samples // 2)
    n_no = n_samples - n_yes
    
    selected_indices = np.concatenate([
        np.random.choice(indices_yes, size=min(n_yes, len(indices_yes)), replace=False),
        np.random.choice(indices_no, size=min(n_no, len(indices_no)), replace=False)
    ])
    
    X_golden = X_test[selected_indices]
    y_golden = y_test[selected_indices]
    
    return X_golden, y_golden, selected_indices


def print_data_summary(df: pd.DataFrame):
    """Imprime sumário dos dados"""
    logger.info("=" * 80)
    logger.info("SUMÁRIO DOS DADOS")
    logger.info("=" * 80)
    logger.info(f"Shape: {df.shape}")
    logger.info(f"\nTipos de dados:\n{df.dtypes}")
    logger.info(f"\nValores faltantes:\n{df.isnull().sum()}")
    logger.info(f"\nEstatísticas numéricas:\n{df.describe()}")
    logger.info("=" * 80)
