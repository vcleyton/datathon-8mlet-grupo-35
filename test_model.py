"""
Script de Teste e Validação (Golden Set)
Demonstra o funcionamento do modelo com 5 exemplos práticos

Execução após train_model.py:
    python test_model.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path(__file__).parent))

from src.config import PROJECT_ROOT
from src.models import ThompsonSampling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_golden_set():
    """Testa o modelo com Golden Set (5 exemplos)"""
    
    logger.info("=" * 80)
    logger.info("TESTE DE VALIDAÇÃO - GOLDEN SET (5 Exemplos)")
    logger.info("=" * 80)
    
    # Carregar modelo
    models_dir = PROJECT_ROOT / "models"
    model_file = models_dir / "thompson_model.pkl"
    
    if not model_file.exists():
        logger.error(f"✗ Modelo não encontrado em {model_file}")
        logger.error("Execute train_model.py primeiro")
        return
    
    thompson = joblib.load(model_file)
    logger.info("✓ Modelo Thompson Sampling carregado")
    
    # Golden Set - 5 exemplos fictícios
    golden_examples = [
        {
            "id": "CLI_001",
            "idade": 32,
            "emprego": "technician",
            "saldo": 1500,
            "descricao": "Cliente jovem, técnico, saldo médio"
        },
        {
            "id": "CLI_002",
            "idade": 58,
            "emprego": "retired",
            "saldo": 5000,
            "descricao": "Cliente sênior, aposentado, alto saldo"
        },
        {
            "id": "CLI_003",
            "idade": 25,
            "emprego": "student",
            "saldo": 100,
            "descricao": "Cliente jovem, estudante, baixo saldo"
        },
        {
            "id": "CLI_004",
            "idade": 45,
            "emprego": "management",
            "saldo": 3000,
            "descricao": "Cliente meio-carreira, gerência, saldo alto"
        },
        {
            "id": "CLI_005",
            "idade": 50,
            "emprego": "blue-collar",
            "saldo": 500,
            "descricao": "Cliente experiente, trabalho manual, saldo baixo"
        }
    ]
    
    logger.info("\n" + "=" * 80)
    logger.info("RECOMENDAÇÕES DO MODELO (Thompson Sampling)")
    logger.info("=" * 80)
    
    # Mapeamento de ofertas
    ofertas = {
        0: "Produto Premium (Alto Rendimento)",
        1: "Produto Standard (Renda Média)",
        2: "Produto Básico (Baixa Renda)"
    }
    
    # Processar cada exemplo
    for i, exemplo in enumerate(golden_examples, 1):
        
        logger.info(f"\n[Cliente {i}/5] {exemplo['id']}")
        logger.info(f"Descrição: {exemplo['descricao']}")
        logger.info(f"  • Idade: {exemplo['idade']} anos")
        logger.info(f"  • Emprego: {exemplo['emprego']}")
        logger.info(f"  • Saldo: €{exemplo['saldo']:,.0f}")
        
        # Criar feature vector fictício (escala 0-1)
        X = np.array([[
            exemplo['idade'] / 100,
            hash(exemplo['emprego']) % 10 / 10,
            exemplo['saldo'] / 10000,
            1.0,  # campaign
            100.0  # pdays
        ]])
        
        # Recomendação
        recomendacao = thompson.predict(X)[0]
        proba = thompson.predict_proba(X)[0]
        
        logger.info(f"  ➜ Oferta Recomendada: {recomendacao}")
        logger.info(f"     {ofertas[recomendacao]}")
        logger.info(f"  ➜ Confiança da Recomendação:")
        for j in range(thompson.num_arms):
            confianca = proba[j]
            barra = "█" * int(confianca * 20) + "░" * (20 - int(confianca * 20))
            logger.info(f"     Oferta {j}: {barra} {confianca:.1%}")
        
        # Raciocínio
        if recomendacao == 0:
            logger.info(f"  💡 Raciocínio: Cliente tem saldo alto, oferecer produto premium")
        elif recomendacao == 1:
            logger.info(f"  💡 Raciocínio: Cliente tem saldo médio, oferecer produto standard")
        else:
            logger.info(f"  💡 Raciocínio: Cliente tem saldo baixo, oferecer produto básico")
    
    logger.info("\n" + "=" * 80)
    logger.info("✓ TESTE CONCLUÍDO")
    logger.info("=" * 80)
    
    # Mostrar estado do modelo
    logger.info("\nEstado Atual do Modelo Thompson Sampling:")
    logger.info(f"  Número de ofertas: {thompson.num_arms}")
    logger.info(f"  Prior Beta (α, β): ({thompson.alpha_prior}, {thompson.beta_prior})")
    logger.info(f"\n  Parâmetros por Oferta:")
    
    for arm in range(thompson.num_arms):
        alpha = thompson.alpha[arm]
        beta = thompson.beta[arm]
        taxa_esperada = alpha / (alpha + beta)
        
        logger.info(f"    Oferta {arm}:")
        logger.info(f"      • α (sucessos+prior): {alpha:.0f}")
        logger.info(f"      • β (falhas+prior): {beta:.0f}")
        logger.info(f"      • Taxa esperada: {taxa_esperada:.2%}")
    
    logger.info("\n" + "-" * 80)
    logger.info("Próximas Etapas:")
    logger.info("1. Iniciar API: python api_service.py")
    logger.info("2. Visualizar MLflow: mlflow ui")
    logger.info("3. Consultar documentação: README.md")


if __name__ == "__main__":
    test_golden_set()
