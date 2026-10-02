"""
Script de Treinamento do Modelo (Etapas 1-4, 7)
Análise Exploratória + Preparação + Baseline + Thompson Sampling + Avaliação + MLflow

Execução:
    python train_model.py

Este script:
1. Carrega dados da base Kaggle Bank Marketing
2. Realiza EDA (Análise Exploratória)
3. Limpa e prepara dados
4. Treina modelo Baseline
5. Treina modelo Thompson Sampling
6. Compara performances
7. Registra tudo no MLflow
8. Salva modelos para API
"""

import sys
import logging
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import mlflow
import mlflow.sklearn

# Adicionar src ao path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import *
from src.utils import (
    load_data, clean_and_prepare_data, encode_features, 
    scale_features, split_data, create_golden_set, print_data_summary
)
from src.models import (
    BaselinePolicy, ThompsonSampling, evaluate_strategy, simulate_bandit
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def plot_eda(df: pd.DataFrame, output_dir: Path):
    """Gera visualizações de EDA"""
    logger.info("Gerando visualizações de EDA...")
    
    # Criar diretório de saída
    output_dir.mkdir(exist_ok=True)
    
    # 1. Distribuição do target
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    df['y'].value_counts().plot(kind='bar', ax=axes[0], color=['#FF6B6B', '#4ECDC4'])
    axes[0].set_title('Distribuição do Target (Conversão)')
    axes[0].set_ylabel('Contagem')
    axes[0].set_xticklabels(['Não', 'Sim'], rotation=0)
    
    df['y'].value_counts(normalize=True).plot(kind='pie', ax=axes[1], autopct='%1.1f%%',
                                               colors=['#FF6B6B', '#4ECDC4'])
    axes[1].set_title('Proporção de Conversões')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'target_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()
    logger.info("✓ Salvo: target_distribution.png")
    
    # 2. Distribuição de idade
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    df['age'].hist(bins=30, ax=axes[0], color='#95E1D3', edgecolor='black')
    axes[0].set_title('Distribuição de Idade')
    axes[0].set_xlabel('Idade')
    axes[0].set_ylabel('Frequência')
    
    df.boxplot(column='age', by='y', ax=axes[1])
    axes[1].set_title('Idade por Status de Conversão')
    axes[1].set_xlabel('Conversão')
    axes[1].set_ylabel('Idade')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'age_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()
    logger.info("✓ Salvo: age_distribution.png")
    
    # 3. Top 10 categorias de job
    fig, ax = plt.subplots(figsize=(10, 6))
    df['job'].value_counts().head(10).plot(kind='barh', ax=ax, color='#F8B500')
    ax.set_title('Top 10 Tipos de Emprego')
    ax.set_xlabel('Contagem')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'job_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()
    logger.info("✓ Salvo: job_distribution.png")
    
    # 4. Taxa de conversão por job
    conversion_by_job = df.groupby('job')['y'].apply(lambda x: (x == 'yes').sum() / len(x))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    conversion_by_job.head(15).plot(kind='barh', ax=ax, color='#A8E6CF')
    ax.set_title('Taxa de Conversão por Tipo de Emprego')
    ax.set_xlabel('Taxa de Conversão')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'conversion_by_job.png', dpi=100, bbox_inches='tight')
    plt.close()
    logger.info("✓ Salvo: conversion_by_job.png")
    
    # 5. Distribuição de contatos na campanha
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    df['campaign'].hist(bins=50, ax=axes[0], color='#FFD3B6', edgecolor='black')
    axes[0].set_title('Contatos na Campanha')
    axes[0].set_xlabel('Número de contatos')
    axes[0].set_ylabel('Frequência')
    
    df.boxplot(column='campaign', by='y', ax=axes[1])
    axes[1].set_title('Contatos por Status de Conversão')
    axes[1].set_xlabel('Conversão')
    axes[1].set_ylabel('Número de contatos')
    plt.suptitle('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'campaign_distribution.png', dpi=100, bbox_inches='tight')
    plt.close()
    logger.info("✓ Salvo: campaign_distribution.png")
    
    logger.info(f"✓ Visualizações salvas em {output_dir}")


def create_golden_set_examples(X_test: np.ndarray, y_test: np.ndarray, 
                                feature_names: List[str], 
                                baseline_pred, thompson_pred,
                                n_examples: int = 5) -> pd.DataFrame:
    """Cria golden set com exemplos de teste (Etapa 4)"""
    
    logger.info(f"Criando Golden Set com {n_examples} exemplos...")
    
    # Selecionar exemplos
    X_golden, y_golden, indices = create_golden_set(X_test, y_test, n_examples)
    
    baseline_golden = baseline_pred[indices]
    thompson_golden = thompson_pred[indices]
    
    # Criar DataFrame com exemplos
    golden_data = []
    
    for i, idx in enumerate(indices):
        # Resumo das features (usar média dos numéricos)
        feature_summary = X_test[idx]
        
        golden_data.append({
            'Cliente_ID': f'C{idx:04d}',
            'Conversão_Real': 'Sim' if y_golden[i] == 1 else 'Não',
            'Oferta_Baseline': int(baseline_golden[i]),
            'Oferta_Thompson': int(thompson_golden[i]),
            'Acerto_Baseline': 'Sim' if baseline_golden[i] == 0 else 'Não',  # Baseline sempre oferece 0
            'Feature_1_Idade': f'{feature_summary[0]:.0f}',
            'Feature_2_Emprego_Code': f'{feature_summary[1]:.0f}',
            'Feature_3_Estado_Civil_Code': f'{feature_summary[2]:.0f}',
        })
    
    golden_df = pd.DataFrame(golden_data)
    
    logger.info("Golden Set criado:")
    logger.info(golden_df.to_string())
    
    return golden_df


def main():
    """Executa pipeline completo de treinamento"""
    
    logger.info("=" * 80)
    logger.info("PIPELINE DE TREINAMENTO - DATATHON GRUPO 35")
    logger.info("=" * 80)
    
    # ===== ETAPA 1: Carregar Base Kaggle e EDA =====
    logger.info("\n[ETAPA 1] Carregando Base Kaggle e Análise Exploratória (EDA)")
    logger.info("-" * 80)
    
    if not DATA_FILE.exists():
        logger.error(f"✗ Arquivo não encontrado: {DATA_FILE}")
        logger.error("Instruções: Baixe a base de https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing")
        logger.error(f"Descompacte em: {DATA_DIR}")
        return
    
    # Carregar dados
    df = load_data(str(DATA_FILE))
    logger.info(f"✓ Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    
    # Sumário dos dados
    print_data_summary(df)
    
    # Gerar visualizações
    plot_eda(df, PROJECT_ROOT / 'visualizations')
    
    # ===== ETAPA 2: Preparação da Base =====
    logger.info("\n[ETAPA 2] Preparação e Limpeza de Dados")
    logger.info("-" * 80)
    
    df_clean = clean_and_prepare_data(df, drop_duration=True)
    logger.info(f"✓ Dados limpos: {df_clean.shape[0]} linhas (removidas {df.shape[0] - df_clean.shape[0]})")
    
    # Codificar features
    X_encoded, y_encoded, feature_names, encoders = encode_features(df_clean, target_col='y')
    logger.info(f"✓ Features codificadas: {len(feature_names)} features")
    logger.info(f"  Features: {feature_names[:5]}...")
    
    # Dividir em treino/teste
    X_train, X_test, y_train, y_test = split_data(
        X_encoded, y_encoded, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    # Normalizar features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    logger.info(f"✓ Features normalizadas com StandardScaler")
    
    # ===== ETAPA 3: Baseline e Thompson Sampling =====
    logger.info("\n[ETAPA 3] Baseline e Algoritmo Adaptativo")
    logger.info("-" * 80)
    
    # Simular três ofertas proxy usando tercis de idade no conjunto de treino.
    age_index = feature_names.index('age')
    arm_quantiles = np.arange(1, THOMPSON_NUM_ARMS) / THOMPSON_NUM_ARMS
    arm_thresholds = np.quantile(X_train[:, age_index], arm_quantiles)
    y_train_arms = np.digitize(X_train[:, age_index], arm_thresholds)
    y_test_arms = np.digitize(X_test[:, age_index], arm_thresholds)
    
    logger.info(f"✓ {THOMPSON_NUM_ARMS} ofertas simuladas")
    logger.info(f"  Treino: {np.bincount(y_train_arms)}")
    logger.info(f"  Teste: {np.bincount(y_test_arms)}")
    
    # Baseline: Always offer arm 0
    logger.info("\n• Treinando Baseline (oferta fixa)...")
    baseline = BaselinePolicy(num_arms=THOMPSON_NUM_ARMS)
    baseline.fit(y_train, y_train_arms)
    baseline_pred_test = baseline.predict(X_test_scaled)
    baseline_conv = baseline.get_conversion_rate()
    
    logger.info(f"✓ Baseline treinado:")
    logger.info(f"  Taxa de conversão: {baseline_conv:.2%}")
    
    # Thompson Sampling
    logger.info("\n• Treinando Thompson Sampling...")
    thompson = ThompsonSampling(
        num_arms=THOMPSON_NUM_ARMS,
        alpha_prior=THOMPSON_ALPHA_PRIOR,
        beta_prior=THOMPSON_BETA_PRIOR,
        random_state=RANDOM_SEED
    )
    thompson.fit(y_train, y_train_arms)
    thompson_pred_test = thompson.predict(X_test_scaled)
    thompson_conv = thompson.get_conversion_rate()
    
    logger.info(f"✓ Thompson Sampling treinado:")
    logger.info(f"  Taxa de conversão esperada: {thompson_conv:.2%}")
    logger.info(f"  Parâmetros Beta:")
    for arm in range(THOMPSON_NUM_ARMS):
        logger.info(f"    Oferta {arm}: α={thompson.alpha[arm]:.0f}, β={thompson.beta[arm]:.0f}")
    
    # ===== ETAPA 4: Avaliação =====
    logger.info("\n[ETAPA 4] Avaliação e Casos de Teste")
    logger.info("-" * 80)
    
    # Métricas do Baseline
    baseline_metrics = evaluate_strategy(y_test, baseline_pred_test)
    logger.info(f"\n Baseline:")
    logger.info(f"  Taxa de conversão: {baseline_metrics['conversion_rate']:.2%}")
    logger.info(f"  Recompensa total: {baseline_metrics['total_reward']:.0f}")
    logger.info(f"  Taxa de exploração: {baseline_metrics['exploration_rate']:.2%}")
    
    # Métricas do Thompson Sampling
    thompson_metrics = evaluate_strategy(y_test, thompson_pred_test)
    logger.info(f"\n Thompson Sampling:")
    logger.info(f"  Taxa de conversão: {thompson_metrics['conversion_rate']:.2%}")
    logger.info(f"  Recompensa total: {thompson_metrics['total_reward']:.0f}")
    logger.info(f"  Taxa de exploração: {thompson_metrics['exploration_rate']:.2%}")
    
    # Melhoria
    melhoria = thompson_metrics['conversion_rate'] - baseline_metrics['conversion_rate']
    melhoria_rel = melhoria / baseline_metrics['conversion_rate'] * 100
    logger.info(f"\n Melhoria relativa: {melhoria_rel:.1f}% ({melhoria:+.2%})")
    
    # Golden Set (5 exemplos)
    logger.info("\nGolden Set (5 casos de teste):")
    golden_df = create_golden_set_examples(
        X_test, y_test, feature_names,
        baseline_pred_test, thompson_pred_test, n_examples=5
    )
    
    # ===== ETAPA 7: MLflow =====
    logger.info("\n[ETAPA 7] Registrando Experimento com MLflow")
    logger.info("-" * 80)
    
    # Configurar MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="thompson_sampling_baseline_comparison"):
        
        # Log de parâmetros
        mlflow.log_param("algoritmo_principal", "thompson_sampling")
        mlflow.log_param("algoritmo_baseline", "always_best_arm")
        mlflow.log_param("num_ofertas", THOMPSON_NUM_ARMS)
        mlflow.log_param("alpha_prior", THOMPSON_ALPHA_PRIOR)
        mlflow.log_param("beta_prior", THOMPSON_BETA_PRIOR)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_seed", RANDOM_SEED)
        
        # Log de dados
        mlflow.log_param("n_samples_treino", X_train.shape[0])
        mlflow.log_param("n_samples_teste", X_test.shape[0])
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("kaggle_dataset", KAGGLE_DATASET)
        
        # Log de métricas - Baseline
        mlflow.log_metric("baseline_conversion_rate", baseline_metrics['conversion_rate'])
        mlflow.log_metric("baseline_total_reward", baseline_metrics['total_reward'])
        mlflow.log_metric("baseline_exploration_rate", baseline_metrics['exploration_rate'])
        
        # Log de métricas - Thompson Sampling
        mlflow.log_metric("thompson_conversion_rate", thompson_metrics['conversion_rate'])
        mlflow.log_metric("thompson_total_reward", thompson_metrics['total_reward'])
        mlflow.log_metric("thompson_exploration_rate", thompson_metrics['exploration_rate'])
        
        # Log de métricas - Comparação
        mlflow.log_metric("melhoria_conversao_absoluta", melhoria)
        mlflow.log_metric("melhoria_conversao_relativa_pct", melhoria_rel)
        
        # Log do modelo Thompson
        mlflow.sklearn.log_model(
            thompson,
            "thompson_sampling_model",
            registered_model_name="thompson_sampling"
        )
        
        # Log de artefatos
        golden_df.to_csv("golden_set.csv", index=False)
        mlflow.log_artifact("golden_set.csv")
        
        logger.info("✓ Experimento registrado no MLflow")
        logger.info(f"  Run ID: {mlflow.active_run().info.run_id}")
    
    # ===== Salvar Modelos para API =====
    logger.info("\n[ETAPA 5] Salvando Modelos para API")
    logger.info("-" * 80)
    
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    joblib.dump(thompson, models_dir / "thompson_model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    
    logger.info(f"✓ Modelos salvos em {models_dir}")
    logger.info("  - thompson_model.pkl")
    logger.info("  - scaler.pkl")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ PIPELINE CONCLUÍDO COM SUCESSO!")
    logger.info("=" * 80)
    logger.info("\nPróximos passos:")
    logger.info("1. Visualizar experimentos: mlflow ui")
    logger.info("2. Iniciar API: python api_service.py")
    logger.info("3. Testar recomendações em http://localhost:8000/docs")
    
    return {
        'baseline_metrics': baseline_metrics,
        'thompson_metrics': thompson_metrics,
        'melhoria': melhoria,
        'melhoria_rel': melhoria_rel,
        'golden_set': golden_df
    }


if __name__ == "__main__":
    main()
