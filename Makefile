SHELL := /bin/bash
.DEFAULT_GOAL := help

.PHONY: init run build destroy clean down-all prune test lint tox dirs scipy tensorflow pyspark \
    test-store-permissions clean-python fix-line-endings create-networks convert-lf help check-deps

HELP_PADDING = 28
bold := $(shell tput bold)
sgr0 := $(shell tput sgr0)
padded_str := %-$(HELP_PADDING)s
pretty_command := $(bold)$(padded_str)$(sgr0)

ifeq (,$(wildcard .env))
$(error Arquivo .env não encontrado! Crie um .env na raiz do projeto.)
endif
include .env

export

MAKEFILE_DIR = $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
BUILDKIT = 1
BUILD_OPTS =
RUN_OPTS =
UP_OPTS =
DOWN_OPTS = --remove-orphans
DOWN_ALL_OPTS = ${DOWN_OPTS} --rmi all -v

HOST_USERNAME := $(shell id -u -n)
JUPYTER_UID := $(shell id -u)
JUPYTER_USERNAME := $(shell id -u -n)
POSTGRES_UID := $(shell id -u)
POSTGRES_GID := $(shell id -g)

JUPYTER_BASE_IMAGE := ${JUPYTER_SCIPY_IMAGE}
JUPYTER_BASE_VERSION := ${JUPYTER_SCIPY_VERSION}
JUPYTER_CHOWN_EXTRA := "/data"

DATA_DIRS = data data/processed data/minio_data data/interim
MODELS_DIR = models models/final models/study models/params
NOTEBOOKS_DIR = notebooks
LOGS_DIR = logs

check-deps: ## Checa dependências do sistema (docker, docker-compose, dos2unix)
	@command -v docker >/dev/null 2>&1 || { echo "docker não instalado!"; exit 1; }
	@command -v docker-compose >/dev/null 2>&1 || { echo "docker-compose não instalado!"; exit 1; }
	@command -v dos2unix >/dev/null 2>&1 || { echo "dos2unix não instalado!"; exit 1; }

init: check-deps convert-lf dirs create-networks ## Inicializa ambiente, containers e redes
	docker-compose build --no-cache && docker-compose up -d

run: check-deps convert-lf dirs create-networks ## Sobe containers já existentes
	docker-compose up -d

build: check-deps convert-lf dirs ## Constrói imagem Docker
	DOCKER_BUILDKIT=$(BUILDKIT) COMPOSE_DOCKER_CLI_BUILD=$(BUILDKIT) \
	docker-compose build $(BUILD_OPTS)

destroy: clean ## Remove diretórios e containers do projeto
	@if [ -d $(NOTEBOOKS_DIR) ] && [ -z "$$(ls -A $(NOTEBOOKS_DIR))" ]; then \
	  rm -rf $(NOTEBOOKS_DIR); \
	else \
	  echo "Diretório $(NOTEBOOKS_DIR) não está vazio, não será removido!"; \
	fi
	rm -rf $(LOGS_DIR) $(DATA_DIRS)

clean: ## Remove containers e volumes relacionados ao projeto
	docker-compose down $(DOWN_OPTS)

down-all: ## Remove tudo relacionado ao projeto (containers, volumes, imagens)
	docker-compose down $(DOWN_ALL_OPTS)

prune: ## Remove imagens e containers com label do projeto
	docker container prune -f --filter "label=mlops-pytemplate"
	docker image prune -f --filter "label=mlops-pytemplate"

test: ## Executa os testes
	docker-compose run --rm jupyter pytest

lint: ## Roda o linter
	docker-compose run --rm jupyter pylint mlops-pytemplate

tox: ## Roda o tox
	docker-compose run --rm jupyter tox

dirs: ## Cria diretórios necessários
	mkdir -p src/workbench
	echo '__version__ = "0.1.0.dev0"' > src/workbench/__init__.py
	mkdir -p $(NOTEBOOKS_DIR)
	mkdir -p $(LOGS_DIR)
	mkdir -p $(MODELS_DIR)
	@for dir in $(DATA_DIRS); do mkdir -p $$dir; done

## ======= BASES DE CONTAINER =======
scipy: JUPYTER_BASE_IMAGE=${JUPYTER_SCIPY_IMAGE}
scipy: JUPYTER_BASE_VERSION=${JUPYTER_SCIPY_VERSION}
scipy: build run

tensorflow: JUPYTER_BASE_IMAGE=${JUPYTER_TENSORFLOW_IMAGE}
tensorflow: JUPYTER_BASE_VERSION=${JUPYTER_TENSORFLOW_VERSION}
tensorflow: build run

pyspark: JUPYTER_BASE_IMAGE=${JUPYTER_PYSPARK_IMAGE}
pyspark: JUPYTER_BASE_VERSION=${JUPYTER_PYSPARK_VERSION}
pyspark: build run

## ======= UTILITÁRIOS E OUTROS =======
test-store-permissions:
	@if [ $$(find data ! -user ${HOST_USERNAME} | wc -l) -gt 0 ]; then \
		echo "Found files and/or folders with wrong permission:" ; \
		find data ! -user ${HOST_USERNAME} -printf '%p (%u)\n' ; \
		exit 1 ; \
	else \
		exit 0 ; \
	fi

clean-python: ## Limpa arquivos temporários Python
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.ipynb_checkpoints' -exec rm -fr {} +
	rm -rf build/ dist/ .eggs/ .pytest_cache .tox/ htmlcov/
	find . -name '*.egg-info' -exec rm -rf {} +
	rm -f .coverage

fix-line-endings: ## Converte arquivos .sh para o formato Unix (evita erro de '\r')
	find . -type f -name "*.sh" -exec dos2unix {} \;

create-networks: ## Cria redes docker se não existirem
	@docker network inspect frontend >/dev/null 2>&1 || docker network create frontend
	@docker network inspect backend >/dev/null 2>&1 || docker network create backend
	@docker network inspect storage >/dev/null 2>&1 || docker network create storage
	
convert-lf:
	find . -type f -name '*.sh' -exec sed -i 's/\r$$//' {} +
## ======= AJUDA =======

help: ## Lista todos os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
