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
            "campanha": 2,
            "pdays": 999,
            "mes": "may",
            "poutcome": "nonexistent",
            "descricao": "Cliente jovem, técnico, sem contato anterior"
        },
        {
            "id": "CLI_002",
            "idade": 58,
            "emprego": "retired",
            "campanha": 1,
            "pdays": 999,
            "mes": "sep",
            "poutcome": "nonexistent",
            "descricao": "Cliente sênior, aposentado, sem contato anterior"
        },
        {
            "id": "CLI_003",
            "idade": 25,
            "emprego": "student",
            "campanha": 3,
            "pdays": 15,
            "mes": "may",
            "poutcome": "failure",
            "descricao": "Cliente jovem, estudante, contato anterior sem conversão"
        },
        {
            "id": "CLI_004",
            "idade": 45,
            "emprego": "management",
            "campanha": 2,
            "pdays": 15,
            "mes": "oct",
            "poutcome": "success",
            "descricao": "Cliente de meia-idade, gerência, contato anterior convertido"
        },
        {
            "id": "CLI_005",
            "idade": 50,
            "emprego": "blue-collar",
            "campanha": 4,
            "pdays": 999,
            "mes": "jul",
            "poutcome": "nonexistent",
            "descricao": "Cliente experiente, trabalho manual, sem contato anterior"
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
        logger.info(f"  • Campanha: {exemplo['campanha']} contatos")
        logger.info(f"  • Dias desde o contato anterior: {exemplo['pdays']}")
        
        # A política atual nao usa as features do cliente.
        X = np.empty((1, 0))
        
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
        
        logger.info("  Nota: a recomendação usa as taxas globais dos braços, não o perfil do cliente.")
    
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
