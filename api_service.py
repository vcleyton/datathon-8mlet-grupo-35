"""
API FastAPI para servir recomendações (Etapa 5 - Serviço Demonstrável)

Execução:
    python api_service.py

Acesso:
    - API: http://localhost:8000/recomenda-oferta
    - Docs: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI app
app = FastAPI(
    title="Datathon - Plataforma Adaptativa de Ofertas",
    description="API para recomendação de ofertas usando Thompson Sampling",
    version="1.0.0"
)

# Estado global
model_state = {
    'thompson_model': None,
    'scaler': None,
    'feature_names': None,
    'loaded': False
}


# ============= MODELOS PYDANTIC =============

class ClienteRequest(BaseModel):
    """Request com features do cliente"""
    idade: int
    emprego: str
    saldo: float
    campanha: int
    pdays: int
    mes: str
    poutcome: str
    
    class Config:
        example = {
            "idade": 35,
            "emprego": "technician",
            "saldo": 1234.56,
            "campanha": 2,
            "pdays": 15,
            "mes": "may",
            "poutcome": "success"
        }


class RecomendacaoResponse(BaseModel):
    """Response com recomendação"""
    cliente_id: str
    oferta_recomendada: int
    nome_oferta: str
    probabilidade_sucesso: float
    algoritmo: str
    timestamp: str
    confianca: str


class HealthResponse(BaseModel):
    """Response de health check"""
    status: str
    modelo_carregado: bool
    timestamp: str


# ============= ENDPOINTS =============

@app.get("/", tags=["Status"])
@app.get("/health", tags=["Status"], response_model=HealthResponse)
async def health_check():
    """Health check da API"""
    return HealthResponse(
        status="ok" if model_state['loaded'] else "modelo não carregado",
        modelo_carregado=model_state['loaded'],
        timestamp=datetime.now().isoformat()
    )


@app.post("/recomenda-oferta", response_model=RecomendacaoResponse, tags=["Recomendações"])
async def recomenda_oferta(cliente: ClienteRequest):
    """
    Endpoint principal: retorna oferta recomendada para um cliente
    
    O algoritmo Thompson Sampling:
    - Modela cada oferta como uma distribuição Beta
    - Amostra da distribuição para cada oferta
    - Escolhe a oferta com maior amostra
    - Adapta-se conforme recebe feedback
    """
    
    if not model_state['loaded']:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda não foi carregado. Execute 02_modelo.ipynb primeiro."
        )
    
    try:
        # Construir vetor de features
        # Nota: em produção, isso seria feito com transformação apropriada
        features = np.array([[
            cliente.idade,
            hash(cliente.emprego) % 100,  # Codificação simples
            cliente.saldo,
            cliente.campanha,
            cliente.pdays,
            hash(cliente.mes) % 12,  # Codificação simples
            hash(cliente.poutcome) % 10  # Codificação simples
        ]])
        
        # Normalizar
        if model_state['scaler']:
            features = model_state['scaler'].transform(features)
        
        # Predição
        thompson_model = model_state['thompson_model']
        
        # Braço recomendado
        bracos_amostrados = np.random.beta(thompson_model.alpha, thompson_model.beta)
        braço_recomendado = int(np.argmax(bracos_amostrados))
        
        # Probabilidade de sucesso
        proba = thompson_model.alpha[braço_recomendado] / (
            thompson_model.alpha[braço_recomendado] + thompson_model.beta[braço_recomendado]
        )
        
        # Mapa de nomes de ofertas
        nomes_ofertas = {
            0: "Produto Premium",
            1: "Produto Standard",
            2: "Produto Básico"
        }
        
        # Classificar confiança
        if proba >= 0.7:
            confianca = "Alta"
        elif proba >= 0.5:
            confianca = "Média"
        else:
            confianca = "Baixa"
        
        response = RecomendacaoResponse(
            cliente_id=f"CLI_{datetime.now().timestamp():.0f}",
            oferta_recomendada=braço_recomendado,
            nome_oferta=nomes_ofertas.get(braço_recomendado, f"Oferta {braço_recomendado}"),
            probabilidade_sucesso=float(proba),
            algoritmo="Thompson Sampling",
            timestamp=datetime.now().isoformat(),
            confianca=confianca
        )
        
        logger.info(f"Recomendação: Cliente {cliente.idade}a, "
                   f"Oferta {response.oferta_recomendada}, "
                   f"Prob={proba:.2%}")
        
        return response
        
    except Exception as e:
        logger.error(f"Erro ao gerar recomendação: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recomenda-lote", tags=["Recomendações"])
async def recomenda_lote(clientes: List[ClienteRequest]):
    """
    Endpoint para processar múltiplos clientes
    """
    recomendacoes = []
    
    for cliente in clientes:
        rec = await recomenda_oferta(cliente)
        recomendacoes.append(rec)
    
    return {
        "n_recomendacoes": len(recomendacoes),
        "recomendacoes": recomendacoes,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/modelo/info", tags=["Modelo"])
async def modelo_info():
    """Retorna informações sobre o modelo carregado"""
    
    if not model_state['loaded']:
        raise HTTPException(
            status_code=503,
            detail="Modelo ainda não foi carregado"
        )
    
    thompson_model = model_state['thompson_model']
    
    return {
        "algoritmo": "Thompson Sampling",
        "num_ofertas": thompson_model.num_arms,
        "alpha_prior": thompson_model.alpha_prior,
        "beta_prior": thompson_model.beta_prior,
        "ofertas_info": [
            {
                "oferta_id": i,
                "alpha": float(thompson_model.alpha[i]),
                "beta": float(thompson_model.beta[i]),
                "prob_sucesso_esperada": float(
                    thompson_model.alpha[i] / (thompson_model.alpha[i] + thompson_model.beta[i])
                ),
                "conversoes_totais": int(thompson_model.alpha[i] - thompson_model.alpha_prior),
                "rejeicoes_totais": int(thompson_model.beta[i] - thompson_model.beta_prior)
            }
            for i in range(thompson_model.num_arms)
        ],
        "timestamp": datetime.now().isoformat()
    }


@app.post("/modelo/atualizar", tags=["Modelo"])
async def atualizar_modelo(oferta_id: int, conversao: int):
    """
    Atualiza o modelo com novo feedback (em produção, seria via batch)
    
    Args:
        oferta_id: ID da oferta recomendada (0, 1 ou 2)
        conversao: 1 se cliente aceitou, 0 caso contrário
    """
    
    if not model_state['loaded']:
        raise HTTPException(status_code=503, detail="Modelo não carregado")
    
    try:
        thompson_model = model_state['thompson_model']
        thompson_model.update(oferta_id, conversao)
        
        logger.info(f"Modelo atualizado: Oferta {oferta_id}, Conversão {conversao}")
        
        return {
            "status": "sucesso",
            "mensagem": f"Modelo atualizado com feedback da oferta {oferta_id}",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Erro ao atualizar modelo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============= INICIALIZAÇÃO =============

def load_model():
    """Carrega modelo treinado do arquivo"""
    
    try:
        # Caminhos
        models_dir = Path(__file__).parent / "models"
        model_file = models_dir / "thompson_model.pkl"
        scaler_file = models_dir / "scaler.pkl"
        
        if not model_file.exists():
            logger.warning(f"Arquivo de modelo não encontrado: {model_file}")
            logger.warning("Execute o notebook 02_modelo.ipynb para treinar o modelo")
            return False
        
        # Carregar modelo
        thompson_model = joblib.load(model_file)
        scaler = joblib.load(scaler_file) if scaler_file.exists() else None
        
        model_state['thompson_model'] = thompson_model
        model_state['scaler'] = scaler
        model_state['loaded'] = True
        
        logger.info("✓ Modelo Thompson Sampling carregado com sucesso")
        logger.info(f"  Ofertas: {thompson_model.num_arms}")
        logger.info(f"  Alpha prior: {thompson_model.alpha_prior}")
        logger.info(f"  Beta prior: {thompson_model.beta_prior}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Erro ao carregar modelo: {str(e)}")
        return False


@app.on_event("startup")
async def startup_event():
    """Event executado ao iniciar a API"""
    logger.info("=" * 80)
    logger.info("Iniciando API - Plataforma Adaptativa de Ofertas")
    logger.info("=" * 80)
    
    if load_model():
        logger.info("✓ API pronta para recomendações!")
    else:
        logger.warning("⚠ API iniciada mas modelo não está disponível")
    
    logger.info("Acesse http://localhost:8000/docs para documentação interativa")


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Iniciando servidor FastAPI...")
    logger.info("Acesse http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
