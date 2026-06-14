# Avaliador de Anonimização e Trade-off de Privacidade

Este projeto é o trabalho final desenvolvido para a disciplina de Proteção da Privacidade em Machine Learning (PPML) do curso de Ciência da Computação da UFMG. Trata-se de uma aplicação interativa que não apenas aplica técnicas de anonimização em bases de dados, mas também permite a avaliação visual do trade-off entre a garantia de privacidade e a perda de utilidade da informação.

## 🎯 Objetivo

Fornecer uma interface amigável para aplicar modelos sintáticos de privacidade (k-anonimato, l-diversidade e t-closeness) através de operações de generalização e supressão. A ferramenta foca na transparência do processo, gerando métricas visuais que auxiliam na tomada de decisão sobre o nível ideal de anonimização antes do treinamento de modelos de Machine Learning.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.13.13
* **Interface:** Streamlit
* **Manipulação de Dados:** Pandas, NumPy
* **Visualização:** Plotly

## Estrutura de Arquivos

```PlainText
Anonimizador_PPML/
├── app.py                 # Ponto de entrada da aplicação Streamlit
├── core/                  # Lógica de negócio e algoritmos
│   ├── __init__.py
│   ├── profiler.py        # Módulo de perfilamento e classificação de atributos
│   ├── anonymizer.py      # Motor de transformação (Generalização, Supressão e k-anonimato)
│   └── evaluator.py       # Cálculo de l-diversidade, t-closeness, utilidade e risco
├── ui/                    # Componentes visuais do Streamlit
│   ├── __init__.py
│   ├── views.py           # Funções que renderizam abas/seções específicas
│   └── charts.py          # Funções para gerar gráficos (Plotly/Altair)
├── utils/                 # Ferramentas auxiliares
│   ├── __init__.py
│   └── arquivos.py         # Carregamento e exportação segura de arquivos
└── requirements.txt
```

## ✅ Checklist de Implementação

### 1. Módulo de Data Profiling e Classificação de Atributos

* [X] Criar interface de upload de arquivos CSV no Streamlit.

* [X] Implementar visualização prévia das primeiras linhas do dataset (Data Profiling básico).
* [X] Desenvolver componente (ex: `st.data_editor` ou `st.selectbox` por coluna) para classificar os atributos em:

- Identificadores Diretos (DI)
- Quase-Identificadores (QI)
- Atributos Sensíveis (SA)
- Atributos Não Sensíveis (NSA)

### 2. Motor de Transformação: Generalização e Supressão

* [X] Criar função de supressão imediata (drop ou mascaramento) para colunas classificadas como Identificadores Diretos (DI).

* [X] Implementar interface para o usuário definir regras de generalização para os Quase-Identificadores (QI):
  * [X] Suporte a generalização numérica (ex: agrupamento em faixas/bins, como idade 20-29).
  * [X] Suporte a supressão parcial de strings (ex: ocultar os 3 últimos dígitos de um CEP ou CPF mascarado).
* [X] Aplicar as transformações no DataFrame em memória.

### 3. Pipeline de Modelos Sintáticos de Privacidade

* [X] **k-anonimato:**
  * [X] Adicionar slider na interface para definição do parâmetro `k`.
  * [X] Implementar algoritmo para agrupar as classes de equivalência com base nos QIs.
  * [X] Validar e suprimir/generalizar mais os registros que não atingem o valor de `k`.

* [X] **l-diversidade:**
  * [X] Adicionar input numérico para o parâmetro `l`.
  * [X] Criar função que verifica se cada classe de equivalência possui pelo menos `l` valores distintos no Atributo Sensível (SA).
* [X] **t-closeness:**
  * [X] Adicionar input (float de 0.0 a 1.0) para o parâmetro `t`.
  * [X] Implementar cálculo de distância (ex: *Earth Mover's Distance*) entre a distribuição do SA na classe de equivalência e a distribuição global. Na implementação atual, usa-se distância de variação total.

### 4. Dashboard de Trade-off e Risco Adversarial

* [X] Desenvolver o cálculo da "Perda de Utilidade" (Loss Metric) comparando a base original e a anonimizada.

* [X] Criar gráfico interativo mostrando o nível de distorção dos dados vs. o ganho de privacidade.
* [X] Construir a tabela de "Risco Adversarial", destacando em vermelho as classes de equivalência vulneráveis a ataques de ligação/homogeneidade antes das métricas e o status seguro após a transformação.

### 5. Exportador e Relatório de Conformidade (LGPD Ready)

* [ ] Implementar botão de download (`st.download_button`) para exportar o CSV final transformado.

* [ ] Gerar dinamicamente um bloco de texto/markdown sumarizando os parâmetros aplicados (ex: "$k=5$, $l=3$, $t=0.2$ em $X$ registros").
* [ ] Mostrar um breve relatório de conformidade apontando que as restrições aos identificadores diretos e quase-identificadores foram mitigadas.

## 🚀 Como Executar Localmente

1. Clone o repositório:

```bash
   git clone [https://github.com/seu-usuario/seu-repositorio.git](https://github.com/seu-usuario/seu-repositorio.git)
```

1. Crie e ative um ambiente virtual (opcional, mas recomendado):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

```

1. Instale as dependências:

```bash
pip install -r requirements.txt

```

1. Execute a aplicação Streamlit:

```bash
streamlit run app.py

```
