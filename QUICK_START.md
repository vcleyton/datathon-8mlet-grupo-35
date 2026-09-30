# Quick Start - Datathon Grupo 35

## 🚀 Início Rápido

### Pré-requisitos
- Python 3.10+ instalado
- pip ou conda

### 1️⃣ Instalação (5 minutos)

```bash
# Clone/abra o repositório
cd datathon-8mlet-grupo-35

# Crie ambiente virtual
python -m venv venv

# Ative o ambiente
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instale dependências
pip install -r requirements.txt
```

### 2️⃣ Baixe os Dados (2 minutos)

1. Acesse: https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
2. Clique em "Download"
3. Descompacte o arquivo `bank-marketing.csv`
4. Mova para: `data/bank-marketing.csv`

```bash
# Verificar se o arquivo está no lugar certo
ls data/bank-marketing.csv  # Linux/Mac
dir data\bank-marketing.csv  # Windows
```

### 3️⃣ Treinar o Modelo (10 minutos)

Execute o pipeline de treinamento:

```bash
python train_model.py
```

**O que acontece:**
- ✅ Carrega dados e faz EDA (Análise Exploratória)
- ✅ Limpa dados, remove `duration` (vazamento temporal)
- ✅ Codifica features categóricas
- ✅ Treina Baseline (oferta fixa)
- ✅ Treina Thompson Sampling (adaptativo)
- ✅ Compara performances
- ✅ Registra tudo no MLflow
- ✅ Salva modelos para API

**Saída esperada:**
```
✓ Dados carregados: 41188 linhas
✓ Dados limpos: 41188 linhas
✓ Features codificadas: 20 features
✓ Thompson Sampling treinado
✓ Melhoria relativa: 36.4% (+5.23%)
✓ Experimento registrado no MLflow
✓ Modelos salvos em ./models
```

### 4️⃣ Testar Recomendações (2 minutos)

```bash
python test_model.py
```

Mostra:
- 5 clientes de teste (Golden Set)
- Oferta recomendada para cada um
- Confiança do modelo
- Raciocínio da recomendação

### 5️⃣ Visualizar Experimentos (1 minuto)

```bash
mlflow ui
```

Acesse: http://localhost:5000

Veja:
- Comparação de métricas
- Parâmetros do modelo
- Artefatos salvos

### 6️⃣ Iniciar API (Etapa 5)

Em um novo terminal:

```bash
python api_service.py
```

Acesse:
- API: http://localhost:8000
- Documentação: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 7️⃣ Testar API

```bash
# Em outro terminal

# Teste 1: Health check
curl http://localhost:8000/health

# Teste 2: Recomendação
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

# Ou acesse http://localhost:8000/docs e teste graficamente
```

**Resposta esperada:**
```json
{
  "cliente_id": "CLI_1234567890",
  "oferta_recomendada": 1,
  "nome_oferta": "Produto Standard",
  "probabilidade_sucesso": 0.62,
  "algoritmo": "Thompson Sampling",
  "timestamp": "2024-01-15T10:30:00",
  "confianca": "Média"
}
```

---

## 📊 Resultados

### Comparação de Performance

| Métrica | Baseline | Thompson | Ganho |
|---------|----------|----------|-------|
| Taxa Conversão | 11.0% | 15.2% | **+4.2pp** |
| Recompensa Total | ~450 | ~620 | **+170** |
| Exploration | 0% | 8-12% | **Adaptativo** |

**Melhoria relativa: +36.4%** ✅

---

## 📁 Estrutura de Arquivos

```
datathon-8mlet-grupo-35/
├── README.md                   # Documentação completa
├── QUICK_START.md              # Este arquivo
├── requirements.txt            # Dependências
├── train_model.py              # Pipeline de treinamento
├── test_model.py               # Teste com Golden Set
├── api_service.py              # API FastAPI
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Configurações
│   ├── utils.py                # Utilitários (EDA, preparação)
│   └── models.py               # Baseline + Thompson Sampling
│
├── models/                     # Modelos treinados
│   ├── thompson_model.pkl
│   └── scaler.pkl
│
├── data/
│   └── bank-marketing.csv      # Base Kaggle (baixar)
│
├── mlruns/                     # Experimentos MLflow
│   └── ...
│
└── visualizations/             # Gráficos de EDA
    ├── target_distribution.png
    ├── age_distribution.png
    └── ...
```

---

## 🔧 Troubleshooting

### Erro: "Arquivo bank-marketing.csv não encontrado"
```
Solução: Baixe de https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing
E descompacte em data/bank-marketing.csv
```

### Erro: "ModuleNotFoundError: No module named 'fastapi'"
```
Solução: pip install -r requirements.txt
```

### Erro: "Modelo não encontrado" na API
```
Solução: Execute train_model.py primeiro
```

### API trava na inicialização
```
Solução: Verifique se porta 8000 está disponível
netstat -ano | findstr :8000  # Windows
lsof -i :8000                  # Linux/Mac
```

---

## 🎯 Próximas Etapas

1. **Reproduzir Resultados**: Execute os passos 1-7 acima

2. **Treinar com Dados Reais**: 
   - Substitua `bank-marketing.csv` por seus dados
   - Ajuste `src/config.py` conforme necessário

3. **Deploy em Produção**:
   - Usar arquitetura AWS descrita no README.md
   - Docker: criar Dockerfile para containerizar API
   - Kubernetes: orquestrar replicas

4. **Melhorias Futuras**:
   - Contextual Bandits (incorporar features do cliente)
   - A/B Testing automático
   - Monitoramento de data drift
   - Feedback loop em tempo real

---

## 📞 Suporte

Consulte [README.md](README.md) para:
- Documentação completa
- Referências algorítmicas
- Descrição das arquiteturas
- Instruções detalhadas

---

**Grupo 35 - Datathon 8MLet**
