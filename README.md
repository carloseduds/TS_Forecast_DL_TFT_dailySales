# Previsão de Vendas no Varejo com Modelos de Série Temporal

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## 📌 Visão Geral

Este projeto implementa uma **solução end-to-end de previsão de vendas no varejo** utilizando abordagens modernas de séries temporais. São avaliados modelos como **Prophet**, **LSTM** e o **Temporal Fusion Transformer (TFT)**, dentro de um fluxo robusto e modularizado, pronto para ambientes de produção e pipelines MLOps.

O repositório demonstra:
- **Automação** de todo ciclo: preparação de dados, treinamento, tuning, deploy, previsão e avaliação;
- Uso intensivo de **PyTorch Lightning**, **MLflow** e **Optuna** (tracking, experimentação, automação de busca de hiperparâmetros);
- **Pipeline reproduzível** com Docker, Makefile, requirements e orquestração via scripts e notebooks;
- Organização profissional seguindo padrões do Cookiecutter Data Science, pronto para integração em squads de dados.

---

## 📂 Estrutura do Projeto

```
.
├── main/
│   ├── __init__.py           # Inicializador de módulo Python
│   ├── app.py                # (Exemplo: FastAPI - se usado)
│   ├── conf.py               # Configurações globais e paths
│   ├── DataProcessor.py      # Classe de ETL, ingestão, splits, preparação, persistência
│   ├── evaluation.py         # Script CLI para avaliação do modelo (test set, métricas)
│   ├── Model.py              # Definição da classe do modelo, treino, tuning, save/load
│   ├── predict.py            # Script CLI para previsão usando modelo salvo
│   ├── setup_db.py           # Setup de banco de dados e tabelas
│   ├── train.py              # Script CLI para pipeline de treino/tuning/salvamento
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements.txt
├── README.md
├── models/
│   ├── final/                # Modelos finais (.ckpt)
│   ├── study/                # Resultados de tuning (Optuna, etc)
├── logs/                     # Logs estruturados (loguru)
├── notebooks/                # Análises exploratórias, validação manual, sanity check
├── data/                     # Estrutura de dados (raw, interim, processed)
├── .env
├── .gitignore
└── LICENSE
```

---

## 🚀 Como Executar

### Requisitos

- **Docker** e **Docker Compose** (ambiente recomendado)
- Python 3.10+ (execução manual/local)

### 1. Clonando o Repositório

```bash
git clone https://github.com/carloseduds/TS_Forecast_DL_TFT_dailySales.git
cd TS_Forecast_DL_TFT_dailySales
```

### 2. Subindo a Infraestrutura

#### Principais comandos do Makefile

- `make init`: Inicializa o ambiente, cria diretórios necessários, checa dependências e redes, **constrói do zero** e sobe todos os containers (útil na primeira execução ou quando deseja garantir reprodutibilidade total).
- `make run`: Sobe os containers do projeto já existentes, sem forçar rebuild (útil para reativar rapidamente após um shutdown).
- `make build`: Apenas constrói as imagens Docker sem subir containers, útil para preparar builds customizados.
- `make clean`: Para e remove todos os containers e volumes associados ao projeto.
- `make destroy`: Remove containers, volumes e diretórios de dados/notebooks/logs criados pelo projeto (atenção: pode apagar dados locais).

```bash
make init     # Inicializa ambiente e sobe tudo do zero (build+up)
make run      # Sobe containers já existentes
make clean    # Remove containers e volumes
make destroy  # Limpa tudo (inclusive diretórios do projeto)
```

### 3. Acessos e Credenciais

Após subir a infraestrutura, acesse as principais interfaces do projeto:

- **Jupyter Notebook:** [http://localhost:8888](http://localhost:8888)  
  Senha: `secret`

- **MLflow UI:** [http://localhost:5000](http://localhost:5000)

- **MinIO Console:** [http://localhost:9001](http://localhost:9001)  
  Usuário: `minio`  
  Senha: `minio123`


### 4. Pipeline de Modelagem (Exemplos de uso)

#### Treinar o Modelo e Otimizar os Hiperparâmetros

```bash
docker exec forecasting_jupyter python main/train.py --tune
```

#### Treinar o Modelo Final

```bash
docker exec forecasting_jupyter python main/train.py --train-best
```

#### Fazer Previsão

```bash
docker exec forecasting_jupyter python main/predict.py
```

#### Avaliar Modelo

```bash
docker exec forecasting_jupyter python main/evaluation.py
```

---

## 🧩 **Arquitetura do Código e Scripts**

| Arquivo                | Função/Responsabilidade                                                                                                    |
|------------------------|---------------------------------------------------------------------------------------------------------------------------|
| `main/DataProcessor.py`| ETL, splits, processamento, persistência SQL/CSV, funções de transformação, gerenciamento de features e splits EDA        |
| `main/Model.py`        | Definição e controle do TFT, integração com MLflow/Optuna, treino, tuning, checkpointing, prediction, save/load           |
| `main/train.py`        | CLI para treino: tuning, treino final, logging, callbacks, integração robusta (utiliza DataProcessor e Model)             |
| `main/predict.py`      | CLI para geração de previsões com modelo treinado (ckpt)                                                                  |
| `main/evaluation.py`   | CLI para avaliação automatizada: gera previsões, calcula métricas e faz log no MLflow                                     |
| `main/conf.py`         | Centraliza paths, variáveis globais, setup de logging                                                                     |
| `main/setup_db.py`     | Inicialização e checagem do banco de dados Postgres                                                                       |

> **Diferenciais:**  
> - **Padronização** dos scripts CLI para cada etapa;
> - **Separa lógica de dados/modelo/execução** (facilita MLOps e integração em pipelines);
> - **Tracking** de experimentos, checkpoints e artefatos com MLflow;
> - **Logging estruturado** (loguru).

---

## 🔬 Tecnologias Utilizadas

- **PyTorch Lightning, Pytorch Forecasting:** Frameworks para deep learning e séries temporais
- **MLflow:** Tracking de experimentos, tuning, artifacts
- **Optuna:** Otimização automática de hiperparâmetros (com logging MLflow)
- **Docker:** Empacotamento, reprodutibilidade
- **SQLAlchemy + Postgres:** Persistência, ingestão, histórico de previsões
- **Loguru:** Logging moderno
- **Typer:** Scripts de linha de comando com UX profissional
- **JupyterLab:** Notebooks para exploração manual

---

## 📚 Recomendações e Extensões

- Pipeline pronto para integração com **CI/CD** (GitHub Actions, GitLab CI, etc)
- Pode ser plugado em infra de nuvem (AWS, Azure, GCP) ou serviços gerenciados (MLflow Tracking Server)
- Fácil adaptação para outros domínios (previsão de demanda, risco, etc)
- Estrutura modular para incluir **testes unitários** (`pytest`), alertas de drift, explainability (SHAP, etc)

---

## Autor

* **Autor:** Carlos Eduardo Correa
* **Base/Inspiração para Infraestrutura:**
  * [Yong Liu](https://github.com/PacktPublishing/Practical-Deep-Learning-at-Scale-with-MLFlow.git)
  * [Natu Lauchande](https://github.com/PacktPublishing/Machine-Learning-Engineering-with-Mlflow.git)
  * [cookiecutter-ds-docker](https://github.com/sertansenturk/cookiecutter-ds-docker)
  * Projetos-padrão de MLflow + Docker

---


## 📜 Licença

MIT
