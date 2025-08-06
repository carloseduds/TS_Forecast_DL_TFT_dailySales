"""
Script para realizar previsões.
"""

import torch
import typer
from lightning.pytorch import Trainer
from loguru import logger
from model import TFTModel
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from utils.extract_config import configfile

from main.conf import DEFAULT_LOG_PATH
from main.data_processor import DataProcessor

app = typer.Typer()

logger.add(f"{DEFAULT_LOG_PATH}/predict.log", rotation="00:00", retention="90 days", level="INFO")


@app.command()
def main():
    """Executa a previsão utilizando o modelo treinado."""
    logger.info("Iniciando processo de previsão...")

    try:
        config = configfile()

        data_processor = DataProcessor()
        train_df, val_df, test_df = data_processor.get_data_for_training()

        df_for_prediction = data_processor.get_data_for_prediction()
        logger.info("Previsão: dados carregados com sucesso.")

        model = TFTModel(config)
        model._build_datasets(train_df, val_df, test_df)

        loaded_model = TemporalFusionTransformer.load_from_checkpoint(
            config.get("model", "model_final_path")
        )
        model.model = loaded_model

        logger.info(f'Modelo carregado de {config.get("model", "model_final_path")}')

        predictions_df = model.predict(model.test_dataloader, df_for_prediction)
        logger.info("Previsão: previsões realizadas com sucesso.")

        data_processor.write_to_db(predictions_df, "prediction")
        logger.info("Previsão: previsões armazenadas no banco de dados.")

        logger.success("Processo de previsão concluído com sucesso!")

    except Exception as e:
        logger.error(f"Erro ao executar previsão: {e}")


if __name__ == "__main__":
    app()
