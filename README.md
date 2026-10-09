# Credit Risk Prediction ML

Desenvolvido por:
- Ana Carolina Sampaio dos Santos
- Marcus Vinicius Ladeia Correa

O sistema realiza:

- Análise exploratória dos dados (EDA);
- Pré-processamento sem vazamento de dados;
- Comparação de algoritmos de classificação;
- Avaliação com métricas de negócio;
- Seleção de limiar baseada em custo financeiro;
- Disponibilização do modelo através de uma API Flask;
- Interface Web para simulação de risco de crédito.

---

# Objetivo

Uma fintech deseja apoiar sua equipe de crédito na análise de pedidos de empréstimo, estimando a probabilidade de inadimplência de cada cliente.

O modelo utiliza dados históricos de contratos para identificar padrões associados ao não pagamento e gerar uma probabilidade de risco para novos clientes.

---

# Dataset

O conjunto de dados utilizado foi gerado artificialmente através do script:

```text
gerar_dados.py
```

Arquivo utilizado:

```text
data/credito.csv
```

Características:

- 6.000 contratos
- Aproximadamente 21% de inadimplência
- Variáveis numéricas e categóricas
- Inclusão proposital de valores ausentes para simular cenários reais

## Variáveis

| Variável | Descrição |
|-----------|-----------|
| id_contrato | Identificador do contrato |
| idade | Idade do cliente |
| renda_mensal | Renda mensal |
| tempo_emprego_anos | Tempo de emprego |
| score_credito | Score de crédito |
| dividas_ativas | Dívidas em aberto |
| possui_imovel | Sim ou não |
| finalidade | Destino do empréstimo |
| valor_emprestimo | Valor solicitado |
| prazo_meses | Prazo de pagamento |
| inadimplente | Alvo (1 = não pagou, 0 = pagou) |

---

# Estrutura do Projeto

```text
credit-risk-prediction-ml/
│
├── app.py
├── config.py
├── gerar_dados.py
├── treinar.py
├── analise_credito.py
├── requirements.txt
│
├── data/
│   ├── churn.csv
│   ├── imoveis.csv
│   └── credito.csv
│
├── models/
│   ├── churn.joblib
│   ├── imoveis.joblib
│   ├── credito.joblib
│   └── metricas.json
│
├── templates/
│   └── index.html
│
├── screenshots/
│   ├── risco-alto.png
│   └── risco-baixo.png
│
└── README.md
```

---

# Tecnologias Utilizadas

- Python 3.11+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- Joblib
- Flask

---

# Análise Exploratória (EDA)

Foram realizadas análises para identificar:

- Distribuição da variável alvo;
- Valores ausentes;
- Impacto do score de crédito;
- Influência da finalidade do empréstimo;
- Relação entre posse de imóvel e inadimplência.

## Principais Resultados

- Aproximadamente 21% dos contratos ficaram inadimplentes;
- As colunas renda_mensal e tempo_emprego_anos apresentaram valores ausentes;
- Scores de crédito menores foram associados a maior inadimplência;
- Empréstimos para negócios apresentaram maior risco;
- Clientes com imóvel apresentaram menor taxa de inadimplência;
- A variável id_contrato foi descartada por não possuir valor preditivo.

---

# Pré-Processamento

Foi utilizado um Pipeline do Scikit-Learn para evitar vazamento de dados.

## Variáveis Numéricas

- Imputação pela mediana
- Padronização com StandardScaler

## Variáveis Categóricas

- Imputação pela moda
- One-Hot Encoding

---

# Modelagem

Separação dos dados:

```python
train_test_split(
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Validação:

```python
5-fold Cross Validation
```

Métrica principal:

```text
ROC AUC
```

## Algoritmos avaliados

- Regressão Logística
- Random Forest
- Gradient Boosting

---

# Avaliação do Modelo

As seguintes métricas foram avaliadas no conjunto de teste:

- Matriz de Confusão
- Acurácia
- Precisão
- Recall
- F1 Score
- ROC AUC

## Interpretação dos Erros

### Falso Positivo

O modelo prevê inadimplência para um cliente que pagaria corretamente.

Impacto:

- Crédito recusado indevidamente;
- Perda de oportunidade de lucro.

Custo estimado:

```text
R$ 1.500
```

### Falso Negativo

O modelo prevê que o cliente pagará, mas ele se torna inadimplente.

Impacto:

- Empréstimo aprovado para um cliente de alto risco;
- Prejuízo financeiro direto.

Custo estimado:

```text
R$ 8.000
```

O falso negativo representa o erro mais caro para a fintech.

---

# Análise de Limiar

As probabilidades foram obtidas utilizando:

```python
cross_val_predict(
    pipeline,
    X_treino,
    y_treino,
    cv=5,
    method="predict_proba"
)
```

Foram testados diversos limiares de decisão:

```text
0.20
0.30
0.40
0.50
0.60
```

O custo total foi calculado considerando:

```text
FN × R$ 8.000
+
FP × R$ 1.500
```

O limiar recomendado foi aquele que apresentou o menor custo financeiro total.

---

# Feature Engineering

Foi criada a variável:

```python
comprometimento_renda
```

Definição:

```python
(
    valor_emprestimo * 1.33
    / prazo_meses
    / renda_mensal
)
```

Essa variável representa a parcela estimada em relação à renda mensal do cliente.

---

# API Flask

A aplicação disponibiliza previsões através de uma API REST.

## Previsão

Endpoint:

```http
POST /api/prever/credito
```

Exemplo:

```json
{
  "idade": 35,
  "renda_mensal": 5000,
  "tempo_emprego_anos": 5,
  "score_credito": 650,
  "dividas_ativas": 1,
  "possui_imovel": "sim",
  "finalidade": "pessoal",
  "valor_emprestimo": 15000,
  "prazo_meses": 24
}
```

Resposta:

```json
{
  "probabilidade": 0.18
}
```

---

# Interface Web

A interface foi desenvolvida utilizando:

- HTML
- CSS
- JavaScript
- Flask Templates

Funcionalidades:

- Formulário de entrada de dados;
- Simulação em tempo real;
- Exibição da probabilidade de inadimplência;
- Visualização das métricas dos modelos.

---

# Executando o Projeto

## 1. Criar ambiente virtual

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

---

## 2. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## 3. Gerar os dados

```bash
python gerar_dados.py
```

---

## 4. Treinar os modelos

```bash
python treinar.py
```

---

## 5. Executar a API

```bash
python app.py
```

Abrir:

```text
http://localhost:5000
```

# Resultados Obtidos

## Comparação dos Modelos

Os modelos foram comparados utilizando validação cruzada de 5 folds e a métrica ROC AUC.

| Modelo | ROC AUC Média | Desvio Padrão |
|----------|----------:|----------:|
| Regressão Logística | 0,7781 | 0,0224 |
| Gradient Boosting | 0,7474 | 0,0261 |
| Random Forest | 0,7418 | 0,0230 |

A Regressão Logística apresentou o melhor desempenho e foi selecionada para avaliação final.

---

## Avaliação no Conjunto de Teste

| Métrica | Valor |
|----------|----------:|
| Acurácia | 0,8058 |
| Precisão | 0,5888 |
| Recall | 0,2500 |
| F1 Score | 0,3510 |
| ROC AUC | 0,7625 |

Os resultados demonstram boa capacidade de separação entre clientes adimplentes e inadimplentes, embora ainda existam oportunidades de melhoria na identificação dos casos positivos, refletidas pelo recall mais baixo.

---

## Análise de Limiar

Para a definição do limiar de decisão foram utilizadas probabilidades obtidas através de validação cruzada no conjunto de treinamento, sem utilizar o conjunto de teste.

Custos considerados:

- Falso Positivo (FP): R$ 1.500
- Falso Negativo (FN): R$ 8.000

Após selecionar o limiar de menor custo e avaliá-lo no conjunto de teste, foram obtidos:

- Falsos Positivos (FP): 314
- Falsos Negativos (FN): 78

Custo total estimado:

```text
R$ 1.095.000,00
```

# Conclusão

O projeto demonstrou todas as etapas fundamentais de um fluxo de Machine Learning aplicado a problemas de crédito: análise exploratória, preparação dos dados, treinamento, validação, avaliação, escolha de limiar de decisão e implantação em uma aplicação web. A Regressão Logística apresentou o melhor desempenho entre os modelos avaliados, alcançando ROC AUC superior a 0,76 no conjunto de teste. Além dos aspectos técnicos, foram considerados fatores de negócio, custos associados aos erros de classificação e questões éticas relacionadas ao uso de sistemas automatizados para concessão de crédito.

---

# Ética e LGPD

Modelos de crédito podem reproduzir vieses históricos presentes nos dados e gerar discriminação contra determinados grupos sociais.

Mesmo que aumentassem o desempenho do modelo, atributos como:

- raça;
- religião;
- gênero;
- orientação sexual;

não devem ser utilizados.

A LGPD exige:

- finalidade legítima;
- minimização de dados;
- transparência;
- segurança da informação;
- possibilidade de revisão de decisões automatizadas.

Por esse motivo, o modelo deve atuar como apoio à decisão humana e ser constantemente monitorado.

---
