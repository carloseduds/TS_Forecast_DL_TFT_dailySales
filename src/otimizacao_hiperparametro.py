import logging
import warnings

import lightning.pytorch as pl
import numpy as np
import optuna
import pandas as pd
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE, MAPE, RMSE, SMAPE, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

from src.treinamento_modelo import calcular_wmae_wmape, expanding_window_split

# Reduzindo logs do Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Reduzindo logs do PyTorch Lightning
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


class ModifyHPMetric(Callback):
    """
    Callback para salvar a melhor métrica de validação (val_loss) em hp_metric.
    """

    def __init__(self):
        super().__init__()
        self.best_metric = None

    def on_validation_end(self, trainer, pl_module):
        """
        No final de cada validação, grava a menor val_loss encontrada em hp_metric.
        """
        metrics = trainer.callback_metrics
        print(f"METRICAS DISPONÍVEIS: {metrics}")  # Debugging

        # Primeiro, verifica se 'val_loss' existe
        if "val_loss" in metrics:
            val_loss = metrics["val_loss"]
        elif "sanity_check_loss" in metrics:
            val_loss = metrics["sanity_check_loss"]
        else:
            print("Nenhuma métrica de perda encontrada! Pulando atualização de hp_metric.")
            return  # Sai da função se não houver métrica válida

        # Atualiza `hp_metric` com a menor perda encontrada
        if self.best_metric is None:
            self.best_metric = val_loss
        else:
            if val_loss < self.best_metric:
                self.best_metric = val_loss

        trainer.logger.log_metrics({"hp_metric": self.best_metric})


def otimizar_tft(trial, dataset, lr_min, lr_max, cv=3, epochs=5):
    """
    Função objetivo para o Optuna otimizar os hiperparâmetros do TFT usando validação cruzada Walk-Forward.
    """

    # Lista para armazenar os resultados da validação cruzada
    wmae_lista = []

    # Espaço de busca dos hiperparâmetros
    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", lr_min, lr_max),
        "hidden_size": trial.suggest_int("hidden_size", 32, 256, step=32),
        "attention_head_size": trial.suggest_int("attention_head_size", 1, 8, step=1),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        "hidden_continuous_size": trial.suggest_int("hidden_continuous_size", 32, 256, step=32),
        "reduce_on_plateau_patience": trial.suggest_int("reduce_on_plateau_patience", 2, 6),
    }

    # Executa o treino e teste para cada split da validação cruzada
    for i in range(cv):
        training_df, validation_df, test_df = expanding_window_split(dataset, i, validation=True)

        # Criando TimeSeriesDataSet para treino
        dados_treino = TimeSeriesDataSet(
            training_df,
            time_idx="time_idx",
            target="sales",
            group_ids=["series_id"],
            min_encoder_length=168 // 2,
            max_encoder_length=168,
            static_categoricals=["series_id"],
            min_prediction_length=1,
            max_prediction_length=28,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["sales"],
        )

        # Criando TimeSeriesDataSet para validação
        dados_valid = TimeSeriesDataSet.from_dataset(
            dados_treino,
            pd.concat([training_df, validation_df]).reset_index(drop=True),
            predict=True,
            stop_randomization=True,
        )

        # Criando DataLoaders
        dl_treino = dados_treino.to_dataloader(
            train=True, batch_size=32, num_workers=2, pin_memory=True
        )
        dl_valid = dados_valid.to_dataloader(
            train=False, batch_size=32, num_workers=2, pin_memory=True
        )

        # Criando o callback para registrar `hp_metric`
        hp_callback = ModifyHPMetric()

        # Callbacks
        early_stop_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1)

        # Logger para TensorBoard (desativado para evitar excesso de logs)
        logger = TensorBoardLogger(
            "../reports/lightning_logs", name=f"tft_optuna_cv_{i}", version=i
        )

        # Configuração do Trainer
        trainer = pl.Trainer(
            max_epochs=epochs,
            accelerator="gpu",
            enable_model_summary=False,
            enable_progress_bar=False,
            gradient_clip_val=0.1,
            callbacks=[hp_callback, early_stop_callback, checkpoint_callback],
            logger=logger,
        )

        # Criando e treinando o modelo TFT com os hiperparâmetros sugeridos pelo Optuna
        modelo_tft = TemporalFusionTransformer.from_dataset(
            dados_treino,
            learning_rate=params["learning_rate"],
            hidden_size=params["hidden_size"],
            attention_head_size=params["attention_head_size"],
            dropout=params["dropout"],
            hidden_continuous_size=params["hidden_continuous_size"],
            output_size=3,
            loss=QuantileLoss([0.1, 0.5, 0.9]),
            reduce_on_plateau_patience=params["reduce_on_plateau_patience"],
            logging_metrics=[MAE(), MAPE(), RMSE(), SMAPE()],
        )

        trainer.fit(modelo_tft, train_dataloaders=dl_treino, val_dataloaders=dl_valid)

        # Carregar o melhor modelo salvo pelo checkpoint
        best_model_path = trainer.checkpoint_callback.best_model_path
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

        # Fazendo previsões no conjunto de validação
        predictions, _, index, _, _ = best_model.predict(dl_valid, return_index=True)

        # Criando um DataFrame com as previsões e os respectivos `series_id` e `time_idx`
        time_idx_start = index.loc[0, "time_idx"]
        time_idx_end = time_idx_start + len(predictions[0])

        predictions_df_wide = pd.DataFrame(
            predictions.cpu().numpy(), columns=range(time_idx_start, time_idx_end)
        )
        predictions_df_wide["series_id"] = index["series_id"]
        predictions_df = predictions_df_wide.melt(id_vars=["series_id"])
        predictions_df.rename(columns={"variable": "time_idx"}, inplace=True)

        # Fazendo o merge das previsões com o DataFrame de validação
        validation_df = validation_df.merge(
            predictions_df, on=["series_id", "time_idx"], how="left"
        )
        validation_df.rename(columns={"value": "tft_pred"}, inplace=True)

        # Calcular o WMAE como métrica de avaliação
        wmae = calcular_wmae_wmape(training_df, validation_df, "sales", "tft_pred")[0]
        wmae_lista.append(wmae)

    # Retorna a média do WMAE das rodadas de validação cruzada
    return np.mean(wmae_lista)
