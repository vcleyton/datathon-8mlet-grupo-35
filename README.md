# Datathon 8MLet - Grupo 35: Plataforma de Experimentação Adaptativa

## 📋 Visão do Problema (Etapa 0 e Definição)

Uma instituição financeira digital precisa decidir, em diferentes canais, qual oferta, mensagem ou próximo passo apresentar para cada cliente elegível. As abordagens tradicionais (regras fixas e testes A/B longos) apresentam limitações:

- **Desperdício de tráfego**: Testes A/B longos usam muitos clientes para decisões simples
- **Reatividade lenta**: Mudanças de contexto e preferências de clientes não são capturadas em tempo real
- **Personalização limitada**: Regras estáticas não adaptam-se a comportamentos individuais

**Solução proposta**: Uma abordagem **adaptativa baseada em Multi-Armed Bandit** (Thompson Sampling) que:
- ✅ Identifica comportamentos distintos dos clientes
- ✅ Equilibra exploração (testar novas ofertas) com explotação (usar o melhor conhecimento)
- ✅ Aprende continuamente com respostas observadas
- ✅ Personaliza ofertas sem congelar decisões em regras estáticas

---

## 📊 Dados e Base Kaggle (Etapa 1)

**Base selecionada**: [Bank Marketing Dataset](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing) por henriqueyamahata

### Descrição
- **Origem**: Campanha de marketing telefônico de uma instituição financeira portuguesa
- **Tamanho**: 41.188 registros com 20 atributos
- **Target**: `y` - Cliente subscreve depósito a prazo? (sim/não)
- **Período**: Maio 2008 - Novembro 2013

### Atributos principais
| Atributo | Tipo | Descrição |
|----------|------|-----------|
| age | Numérico | Idade do cliente |
| job | Categórico | Tipo de emprego |
| balance | Numérico | Saldo médio anual (€) |
| duration | Numérico | Duração da última chamada (segundos) |
| campaign | Numérico | Número de contatos durante esta campanha |
| pdays | Numérico | Dias desde último contato |
| y | Binário | Subscrição (sim/não) |

---

## 🧹 Preparação da Base (Etapa 2)

### Limpeza de Dados
- ✅ Removida coluna `duration` (vazamento temporal - não disponível antes da decisão)
- ✅ Mantidas features de cliente (age, job, balance, etc.)
- ✅ Codificação numérica de categóricas (`LabelEncoder`)
- ✅ Escalagem de numéricos (StandardScaler)

---

## 🤖 Estratégia Algorítmica (Etapa 3)

### 1. Baseline Determinístico
**Política**: Oferecer sempre o produto com melhor taxa histórica de conversão

```python
# Pseudo-código
melhor_produto = produtos.groupby('nome').agg({'conversao': 'mean'}).idxmax()
oferta_base = melhor_produto  # sempre igual
```

**Métrica de baseline**: Taxa de conversão média histórica (~11%)

### 2. Modelo Adaptativo: Thompson Sampling

**Conceito**: Modelar a incerteza sobre a taxa de sucesso de cada oferta usando distribuições Beta e amostrar a melhor ação.

**Algoritmo**:
```
Para cada cliente:
  1. Para cada oferta disponível:
     - Amostra taxa_sucesso ~ Beta(alpha, beta)  # Alpha = conversões+1, Beta = rejeições+1
  2. Seleciona oferta com maior taxa_sucesso amostrada
  3. Registra resultado (conversão ou não)
  4. Atualiza parâmetros Beta para essa oferta
```

**Vantagens**:
- Bayesiano: incorpora incerteza na decisão
- Exploração automática: novos produtos explorados naturalmente
- Explotação inteligente: produtos bons escolhidos com mais frequência
- Convergência: aprende a melhor oferta conforme acumula dados

**Parâmetros**:
- `alpha_prior = 1` (prior não-informativo)
- `beta_prior = 1` (prior não-informativo)

---

## 📈 Avaliação e Métricas (Etapa 4)

### Métricas Calculadas

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **Taxa de Conversão** | conversões / total | % de clientes que aceitam |
| **Recompensa Acumulada** | Σ conversões | Total de sucessos |
| **Regret** | (baseline_total - adaptativo_total) | Oportunidades perdidas |
| **Exploration Rate** | ação_não_ótima / total | % de vezes que explora |

### Golden Set (5 Exemplos de Teste)
O projeto inclui 5 casos de teste representativos mostrando:
- Cliente jovem / cliente senior
- Alta renda / baixa renda
- Cliente novo / cliente conhecido
- Para cada um: qual oferta o modelo recomenda e raciocínio

---

## 💻 Serviço Demonstrável (Etapa 5)

### API FastAPI

**Arquivo**: `api_service.py`

**Endpoint**: `POST /recomenda-oferta`

**Request**:
```json
{
  "idade": 35,
  "emprego": "technician",
  "saldo": 1234.56,
  "campanha": 2,
  "pdays": 15,
  "mes": "may",
  "poutcome": "success"
}
```

**Response**:
```json
{
  "cliente_id": "C001",
  "oferta_recomendada": "produto_A",
  "probabilidade_sucesso": 0.62,
  "algoritmo": "thompson_sampling",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Como Usar

```bash
# Instalar dependências
pip install -r requirements.txt

# Treinar modelo e registrar com MLflow
python train_model.py

# Iniciar API
python api_service.py

# Testar (em outro terminal)
curl -X POST http://localhost:8000/recomenda-oferta \
  -H "Content-Type: application/json" \
  -d '{"idade": 35, "emprego": "technician", ...}'
```

---

## ☁️ Arquitetura-alvo em Nuvem (Etapa 6)

### Arquitetura Proposta para AWS

#### Componentes

1. **Data Ingestion** (AWS S3 + AWS Glue)
   - Dados brutos de campanhas armazenados em S3
   - Glue ETL jobs para limpeza e transformação
   - Frequência: diária

2. **Model Training** (Amazon SageMaker)
   - Pipeline de treinamento automático (SageMaker Pipelines)
   - Hyperparameter tuning com Bayesian Optimization
   - Versionamento com Model Registry
   - Registro automático com MLflow

3. **Model Serving** (Amazon SageMaker Endpoints + API Gateway)
   - SageMaker Endpoints para inferência em tempo real (latência <100ms)
   - Auto-scaling baseado em volume de requisições
   - Multi-model endpoints para suportar A/B testing

4. **Observabilidade e Monitoramento** (CloudWatch + Amazon Lookout)
   - Logs de recomendações (CloudWatch Logs)
   - Métricas de performance do modelo (data drift, feature importance)
   - Alertas automáticos se conversão cai >5%
   - Amazon Lookout for Metrics: detecção de anomalias

5. **Retroalimentação e Retrainamento** (Lambda + DynamoDB)
   - Lambda: captura resultado real (cliente aceitou oferta?)
   - DynamoDB: armazena histórico de recomendações e outcomes
   - Redrive: mensagens para fila SNS para atualizar parâmetros Bayesianos em tempo real

6. **Governance** (AWS Governance)
   - Compliance com LGPD/GDPR: dados de clientes são anonimizados
   - Auditoria: logs centralizados em CloudTrail
   - Limites: human-in-the-loop para decisões sensíveis via SNS notifications

#### Fluxo End-to-End

```
Cliente (app/web) 
  → API Gateway 
  → Lambda (validação) 
  → SageMaker Endpoint (Thompson Sampling) 
  → DynamoDB (log) 
  → SNS (callback de resultado) 
  → Lambda (atualizar parâmetros Beta) 
  → CloudWatch (métricas)
```

#### Benefícios
- **Escalabilidade**: auto-scaling automático para picos de tráfego
- **Observabilidade**: métricas e logs centralizados
- **Governança**: LGPD-compliant, auditável, com humano no loop
- **Baixa latência**: <100ms por recomendação em produção

---

## 🔄 Ciclo de Vida MLOps (Etapa 7)

### MLflow Integration

Todos os experimentos são rastreados com MLflow:

```python
import mlflow
import mlflow.sklearn

# Log de parâmetros
mlflow.log_param("algorithm", "thompson_sampling")
mlflow.log_param("alpha_prior", 1.0)
mlflow.log_param("beta_prior", 1.0)

# Log de métricas
mlflow.log_metric("baseline_conversion", 0.11)
mlflow.log_metric("thompson_conversion", 0.15)
mlflow.log_metric("regret", 0.04)

# Log do modelo
mlflow.sklearn.log_model(model, "thompson_model")
```

### Artefatos Registrados
- ✅ Parâmetros do modelo (priors, estratégia de exploração)
- ✅ Métricas de desempenho (conversão, regret, exploration rate)
- ✅ Dados de treinamento (versão da base Kaggle)
- ✅ Modelo serializado (pickle)

### Execução

```bash
# Visualizar experimentos
mlflow ui

# Navegar a http://localhost:5000
```

---

## 🎬 Apresentação Final (Etapa 8)

### Vídeo Pitch (até 5 minutos)

**Roteiro completo para gravação:** [ROTEIRO_VIDEO_PITCH.md](ROTEIRO_VIDEO_PITCH.md)

Conteúdo esperado:
1. **Problema de Negócio** (30s)
   - Contexto: instituição financeira digital
   - Limitação: regras fixas e testes A/B longos

2. **Solução Proposta** (60s)
   - Thompson Sampling: exploração + explotação
   - Adaptação contínua ao comportamento do cliente

3. **Dados e Preparação** (30s)
   - Base Kaggle: Bank Marketing
   - Features: idade, emprego, saldo, etc.

4. **Modelo e Resultados** (90s)
   - Baseline: ~11% conversão
   - Thompson Sampling: ~15% conversão
   - Ganho: +36% de melhoria relativa

5. **Demo Prática** (60s)
   - Executar Etapa 5: API em ação
   - Cliente exemplo → Oferta recomendada

6. **Conclusões** (30s)
   - Impacto de negócio: mais ofertas aceitas
   - Próximos passos: produção com MLflow + AWS

---

## 📂 Estrutura do Repositório

```
datathon-8mlet-grupo-35/
├── README.md                          # Este arquivo (documentação consolidada)
├── requirements.txt                   # Dependências Python
├── .gitignore                         # Arquivos a ignorar
├── train_model.py                     # Pipeline de EDA, treino e MLflow
├── test_model.py                      # Teste do Golden Set
├── setup_check.py                     # Validação do ambiente
├── api_service.py                     # API FastAPI demonstrável
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Etapa 1: Análise Exploratória
│   └── 02_modelo.ipynb               # Etapas 2-4 e 7: Modelo e MLflow
│
├── src/
│   ├── __init__.py
│   ├── models.py                     # Baseline e Thompson Sampling
│   ├── utils.py                      # Funções auxiliares
│   └── config.py                     # Configurações
│
├── data/
│   └── bank-marketing.csv            # Base Kaggle (baixar)
│
├── models/                            # Criado após o treinamento
│   ├── thompson_model.pkl
│   └── scaler.pkl
├── visualizations/                    # Gráficos gerados pela EDA
├── golden_set.csv                     # Gerado pelo treinamento
└── mlruns/                            # Rastreamento local do MLflow
```

---

## 🚀 Instruções de Execução

### 1. Pré-requisitos
- Python 3.10+
- pip ou conda

### 2. Instalação

```bash
# Clone o repositório
cd datathon-8mlet-grupo-35

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale dependências
pip install -r requirements.txt

# Baixe a base Kaggle
# - Acesse https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
# - Clique em "Download"
# - Descompacte em data/bank-marketing.csv
```

### 3. Executar Análise Exploratória
```bash
jupyter notebook notebooks/01_eda.ipynb
# Etapa 1: análise da base, qualidade dos dados e tratamento inicial
```

### 4. Treinar Modelo
```bash
jupyter notebook notebooks/02_modelo.ipynb
# Etapas 2-4: preparação, baseline, Thompson Sampling e avaliação
# Etapa 7: logging automático em MLflow
```

### 5. Ver Experimentos MLflow
```bash
mlflow ui
# Navegue a http://localhost:5000 para ver métricas e parâmetros
```

### 6. Executar API
```bash
python api_service.py
# API rodando em http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 7. Testar Recomendação
```bash
# Em outro terminal
curl -X POST http://localhost:8000/recomenda-oferta \
  -H "Content-Type: application/json" \
  -d '{
    "idade": 35,
    "emprego": "technician",
    "saldo": 1234.56,
    "campanha": 2,
    "pdays": 15,
    "mes": "may",
    "poutcome": "success"
  }'
```

---

## 📚 Referências

1. **Thompson Sampling**
   - Russo et al. (2018) - "A Tutorial on Thompson Sampling"
   - Thompson, W. R. (1933) - "On the Likelihood that One Unknown Probability Exceeds Another..."

2. **Multi-Armed Bandit**
   - Lattimore & Szepesvári (2020) - "Bandit Algorithms"

3. **Bank Marketing Dataset**
   - Moro et al. (2014) - "A Data-Driven Approach to Predict the Success of Bank Telemarketing"

---

## 👥 Grupo 35

Desenvolvimento: Grupo 35 - Datathon 8MLet
Data: 2026
Licença: Uso educacional (PosTech)
