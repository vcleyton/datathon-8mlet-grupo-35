"""Configurações do projeto Datathon"""

import os
from pathlib import Path

# Diretórios
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

# Criar diretórios se não existirem
for dir_path in [DATA_DIR, NOTEBOOKS_DIR, MODELS_DIR, MLRUNS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Base Kaggle
KAGGLE_DATASET = "bank-marketing"
DATA_FILE = DATA_DIR / "bank-marketing.csv"

# Configurações MLflow
MLFLOW_TRACKING_URI = MLRUNS_DIR.resolve().as_uri()
MLFLOW_EXPERIMENT_NAME = "datathon-grupo-35"

# Thompson Sampling
THOMPSON_ALPHA_PRIOR = 1.0
THOMPSON_BETA_PRIOR = 1.0
THOMPSON_NUM_ARMS = 3  # Número de ofertas/braços

# Seed para reprodutibilidade
RANDOM_SEED = 42

# Configurações de modelo
TEST_SIZE = 0.2
RANDOM_STATE = RANDOM_SEED
SCALER_TYPE = "standard"  # StandardScaler

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
