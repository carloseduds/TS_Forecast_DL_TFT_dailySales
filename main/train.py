"""
Script para treinar o modelo com ajuste de hiperparâmetros.
"""

import json

import typer
from conf import DEFAULT_LOG_PATH
from loguru import logger
from model import TFTModel
from utils.extract_config import configfile

from main.data_processor import DataProcessor

app = typer.Typer()

config = configfile()
BEST_PARAMS_FILE = config.get("model", "best_params_path")

logger.add(f"{DEFAULT_LOG_PATH}/train.log", rotation="00:00", retention="90 days", level="INFO")


@app.command()
def main(tune: bool = False, train_best: bool = False):
    """
    Executa o treinamento do modelo com opções para
    ajuste de hiperparâmetros e treinamento final.

    Parâmetros:
    --tune: Executa apenas o ajuste de hiperparâmetros.
    --train-best: Treina o modelo final com os
                  melhores hiperparâmetros encontrados.
    """
    logger.info("Iniciando treinamento do modelo...")

    try:
        # Obtém os dados para treinamento
        data_processor = DataProcessor()
        training_df, validation_df, test_df = data_processor.get_data_for_training()
        logger.info("Treinamento: dados carregados com sucesso.")

        model = TFTModel(config)

        best_params = None

        if tune:
            logger.info("Iniciando ajuste de hiperparâmetros...")
            study = model.hyperparameter_tuning(training_df, validation_df, test_df, config)
            logger.info("Ajuste de hiperparâmetros concluído.")
            best_params = study.best_trial.params

            with open(BEST_PARAMS_FILE, "w") as f:
                json.dump(best_params, f)
            logger.info(f"Melhores hiperparâmetros salvos em {BEST_PARAMS_FILE}")

        if train_best:
            if best_params is None:
                # Tenta carregar melhores hiperparâmetros salvos
                try:
                    with open(BEST_PARAMS_FILE, "r") as f:
                        best_params = json.load(f)
                    logger.info(f"Hiperparâmetros carregados de {BEST_PARAMS_FILE}")
                except FileNotFoundError:
                    logger.error(
                        "Arquivo de melhores hiperparâmetros não encontrado. Execute com --tune antes."
                    )
                    return

            logger.info("Iniciando treino com os melhores hiperparâmetros...")
            TFTModel.retrain_with_best_params(
                training_df, validation_df, test_df, config, best_params
            )
            logger.info("Treinamento final concluído.")

        if not tune and not train_best:
            logger.warning("Nenhuma opção especificada. Execute com `--tune` ou `--train-best`.")

    except Exception as e:
        logger.error(f"Erro durante o treinamento: {e}")


if __name__ == "__main__":
    app()
