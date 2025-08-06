"""
Script para avaliar a performance do modelo.
"""

from datetime import datetime

import mlflow
import typer
from loguru import logger
from model import TFTModel
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from utils.compute_metrics import compute_eval_data_ratio, compute_wmae, compute_wmape
from utils.extract_config import configfile

from main.conf import DEFAULT_LOG_PATH
from main.data_processor import DataProcessor

app = typer.Typer()

logger.add(
    f"{DEFAULT_LOG_PATH}/evaluation.log", rotation="00:00", retention="90 days", level="INFO"
)


@app.command()
def main():
    """Executa a avaliação do modelo."""
    logger.info("Iniciando avaliação do modelo...")

    try:
        config = configfile()
        MLFLOW_TRACKING_URI = config.get("optimization", "mlflow_tracking_uri")
        MLFLOW_EVAL_EXPERIMENT_NAME = config.get("optimization", "mlflow_eval_experiment_name")

        data_processor = DataProcessor()
        evaluation_df, training_df = data_processor.get_data_for_eval()
        logger.info("Avaliação: dados de avaliação carregados com sucesso.")

        model = TFTModel(config)

        train_df, val_df, test_df = data_processor.get_data_for_training()
        model._build_datasets(train_df, val_df, test_df)

        loaded_model = TemporalFusionTransformer.load_from_checkpoint(
            config.get("model", "model_final_path")
        )
        model.model = loaded_model
        logger.info(f'Modelo carregado de {config.get("model", "model_final_path")}')

        predictions_df = model.predict(model.test_dataloader, evaluation_df)
        predictions_df = data_processor.create_idx_prediction(predictions_df, evaluation_df)
        logger.info("Previsões geradas em tempo real.")

        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EVAL_EXPERIMENT_NAME)
        experiment = mlflow.get_experiment_by_name(MLFLOW_EVAL_EXPERIMENT_NAME)

        with mlflow.start_run(experiment_id=experiment.experiment_id):
            mlflow.log_param("date", datetime.now().strftime("%Y-%m-%d"))

            wmae = compute_wmae(evaluation_df, predictions_df, "sales", "pred")
            wmape = compute_wmape(evaluation_df, predictions_df, "sales", "pred")
            logger.info("Avaliação: métricas do modelo calculadas.")

            mean_ratio, stdev_ratio = compute_eval_data_ratio(evaluation_df, predictions_df)
            logger.info("Avaliação: métricas dos dados calculadas.")

            mlflow.log_metric("wmae", wmae)
            mlflow.log_metric("wmape", wmape)
            mlflow.log_metric("mean_ratio", mean_ratio)
            mlflow.log_metric("stdev_ratio", stdev_ratio)
            logger.info("Avaliação: todas as métricas registradas no MLflow.")

        logger.success("Avaliação do modelo concluída com sucesso!")

    except Exception as e:
        logger.error(f"Erro durante a avaliação: {e}")


if __name__ == "__main__":
    app()
