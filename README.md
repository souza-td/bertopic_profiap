# BERTopic-PROFIAP

Mapeamento automático de 1.258 dissertações do Mestrado Profissional em Administração Pública (2015–2023) com BERTopic, identificando 24 tópicos que abrangem áreas como qualidade de vida no trabalho, sustentabilidade em compras públicas, teletrabalho, inclusão e transparência. O pipeline integra embeddings multilíngues, redução de dimensionalidade e clusterização por densidade, com rótulos gerados por modelo de linguagem para facilitar a interpretação. Inclui um dashboard interativo em Quarto com visualizações sobre evolução temporal e frequência de termos, apoiando pesquisadores e gestores na análise e formulação de políticas.

Dashboard interativo disponível em: https://thiago-dsouza.quarto.pub/bertopic-profiap

## 1. Executado em

| Componente | Versão |
|------------|--------|
| Sistema operacional | Ubuntu 22.04 LTS |
| CPU | Intel Core i5-13420H (ou equivalente) |
| GPU | NVIDIA RTX 3050 Laptop, 6 GB VRAM |
| RAM | 16 GB |


## 2. Ambiente Python

```bash
# Instalar Python 3.10.12 com pyenv (opcional)
pyenv install 3.10.12
pyenv local 3.10.12

# Clonar repositório
git clone https://github.com/usuario/experimento-bertopic.git
cd experimento-bertopic

# Instalar dependências
poetry install
```

## 3. Estrutura básica do repositório
```bash

.
├── 00_dados_sucupira.py                                # Opicional: Tratamento de dados da Plataforma Sucupira (CSV disponível no repositório)
├── 01_expr_bertopic_profiap.py                         # Executa o experimento e faz o cálculo das métricas
├── 02_train_model_with_gpt_representation.py           # Após seleção dos hiperparametros insere as representações usando LLM
├── data
│   ├── custom_prompt_bertopic_openai_description.txt
│   ├── custom_prompt_bertopic_openai.txt
│   ├── octis
│   └── raw
├── models
│   └── expr_bertopic_profiap
├── notebooks
│   ├── 01_EDA_metricas.ipynb
│   ├── 02_EDA_visualizacoes.ipynb
│   ├── 03_EDA_visualizacoes_with_llm_labels.ipynb
│   └── dashboard.qmd
├── pyproject.toml
├── README.md
└── scripts
    └── download_spacy_model.sh
```

## 4. Como reproduzir rapidamente

```bash
# 1. Instalar dependências
poetry install

# 2. Baixar modelo spaCy
poetry run python -m spacy download pt_core_news_md

# 3. Treinar modelos
poetry run python 01_expr_bertopic_profiap.py

# 4. Gerar labels com GPT
poetry run python 02_train_model_with_gpt_representation.py
```

Opcional: Dashboard em Quarto Markdown
```bash
# 1. Preview Quato Dashboard
# Necessário R e Quarto instalados
poetry run quarte preview ./notebooks/dashboard.qmd
```
