import itertools
from datetime import datetime

import lightning.pytorch as pl
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAPE, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from utils.compute_metrics import compute_wmae
from utils.mlflow_callbacks import GPUUsageLogger, MLflowMetricsLogger, SystemUsageLogger


class TFTModel:
    """
    Modelo Temporal Fusion Transformer com integração MLflow e Optuna.
    """

    def __init__(self, cfg):
        """
        cfg: objeto ConfigParser do configfile()
        """
        self.cfg = cfg
        self.model = None
        self.trainer = None
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.best_model = None

    def _build_datasets(self, train_df, val_df, test_df):
        quantis = [0.1, 0.5, 0.9]
        context_length = int(self.cfg.get("model", "context_length"))
        prediction_length = int(self.cfg.get("model", "prediction_length"))

        base_dataset = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target="sales",
            group_ids=["series_id"],
            min_encoder_length=context_length // 2,
            max_encoder_length=context_length,
            static_categoricals=["series_id"],
            min_prediction_length=1,
            max_prediction_length=prediction_length,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["sales"],
            target_normalizer=GroupNormalizer(groups=["series_id"]),
        )

        val_data = TimeSeriesDataSet.from_dataset(
            base_dataset,
            pd.concat([train_df, val_df]).reset_index(drop=True),
            min_prediction_idx=val_df["time_idx"].min(),
            predict=True,
            stop_randomization=True,
        )
        test_data = TimeSeriesDataSet.from_dataset(
            base_dataset,
            pd.concat([train_df, val_df, test_df]).reset_index(drop=True),
            min_prediction_idx=test_df["time_idx"].min(),
            predict=True,
            stop_randomization=True,
        )

        batch_size = (
            int(self.cfg.get("model", "batch_size"))
            if self.cfg.has_option("model", "batch_size")
            else 64
        )

        self.train_dataloader = base_dataset.to_dataloader(
            train=True, batch_size=batch_size, num_workers=2
        )
        self.val_dataloader = val_data.to_dataloader(
            train=False, batch_size=batch_size, num_workers=2
        )
        self.test_dataloader = test_data.to_dataloader(
            train=False, batch_size=batch_size, num_workers=2
        )

    def build_model(self, params):
        quantis = [0.1, 0.5, 0.9]
        model = TemporalFusionTransformer.from_dataset(
            self.train_dataloader.dataset,
            learning_rate=params["learning_rate"],
            hidden_size=params["hidden_size"],
            attention_head_size=params["attention_head_size"],
            dropout=params["dropout"],
            hidden_continuous_size=params["hidden_continuous_size"],
            output_size=3,
            loss=QuantileLoss(quantis),
            reduce_on_plateau_patience=params.get("reduce_on_plateau_patience", 3),
            logging_metrics=[MAPE()],
        )
        return model

    def fit(self, train_df, val_df, test_df, params, max_epochs=10):
        self._build_datasets(train_df, val_df, test_df)
        self.model = self.build_model(params)

        pl_seed = params.get("seed", 42)
        torch.manual_seed(pl_seed)
        np.random.seed(pl_seed)

        earlystopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        checkpoint = ModelCheckpoint(monitor="val_loss")
        tensorboard_log_dir = (
            self.cfg.get("model", "tensorboard_log_dir")
            if self.cfg.has_option("model", "tensorboard_log_dir")
            else "./logs/tensorboard"
        )
        tb_logger = TensorBoardLogger(tensorboard_log_dir)
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        max_epochs = int(max_epochs)

        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator=accelerator,
            gradient_clip_val=0.1,
            callbacks=[
                earlystopping,
                checkpoint,
                MLflowMetricsLogger(),
                SystemUsageLogger(),
                GPUUsageLogger(),
            ],
            logger=tb_logger,
        )
        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_dataloader,
            val_dataloaders=self.val_dataloader,
        )
        self.best_model = self.trainer.checkpoint_callback.best_model_path
        return self.best_model

    def predict(self, dataloader, df_reference=None):
        predictions, _, index, _, _ = self.model.predict(dataloader, return_index=True)

        time_idx_start = index.loc[0, "time_idx"]
        n_steps = predictions.shape[1]
        time_idxs = range(time_idx_start, time_idx_start + n_steps)
        predictions_df_wide = pd.DataFrame(predictions.cpu().numpy(), columns=time_idxs)
        predictions_df_wide["series_id"] = index["series_id"].values

        predictions_df = predictions_df_wide.melt(
            id_vars="series_id", var_name="time_idx", value_name="pred"
        )
        predictions_df["time_idx"] = predictions_df["time_idx"].astype(int)

        predictions_df[["store_id", "cat_id"]] = predictions_df["series_id"].str.rsplit(
            "_", n=1, expand=True
        )

        pred_df = predictions_df.copy()
        pred_df["creation_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        START_DATE = pd.to_datetime("2011-01-29")
        pred_df["date"] = START_DATE + pd.to_timedelta(pred_df["time_idx"], unit="D")
        pred_df["date"] = pred_df["date"].dt.date

        result_cols = ["store_id", "cat_id", "date", "pred", "creation_time"]
        pred_df = pred_df[result_cols]

        return pred_df

    @staticmethod
    def objective(trial, train_df, val_df, test_df, cfg):
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "hidden_size": trial.suggest_int("hidden_size", 32, 256, step=32),
            "attention_head_size": trial.suggest_int("attention_head_size", 1, 8, step=1),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.05),
            "hidden_continuous_size": trial.suggest_int(
                "hidden_continuous_size", 32, 256, step=32
            ),
            "reduce_on_plateau_patience": trial.suggest_int("reduce_on_plateau_patience", 2, 6),
            "seed": int(cfg.get("model", "seed")),
        }
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            model_runner = TFTModel(cfg)
            max_epochs = (
                int(cfg.get("model", "max_epochs"))
                if cfg.has_option("model", "max_epochs")
                else 10
            )
            model_runner.fit(train_df, val_df, test_df, params, max_epochs=max_epochs)
            val_pred = model_runner.predict(model_runner.val_dataloader, val_df)
            val_wmae = compute_wmae(val_df, val_pred, "sales", "pred")
            mlflow.log_metric("val_wmae", val_wmae)
            # mlflow.log_artifact(best_model)
            model_runner.trainer.save_checkpoint(cfg.get("model", "model_study_path"))
        return val_wmae

    @staticmethod
    def hyperparameter_tuning(train_df, val_df, test_df, cfg):
        mlflow.set_tracking_uri(cfg.get("optimization", "mlflow_tracking_uri"))
        mlflow.set_experiment(cfg.get("optimization", "mlflow_training_experiment_name"))
        from optuna.integration import MLflowCallback

        mlflow_callback = MLflowCallback(
            tracking_uri=cfg.get("optimization", "mlflow_tracking_uri"), metric_name="val_wmae"
        )
        n_trials = int(cfg.get("optimization", "optimization_trials"))
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: TFTModel.objective(trial, train_df, val_df, test_df, cfg),
            n_trials=n_trials,
            callbacks=[mlflow_callback],
        )
        logger.success(f"""Modelo Study salvo em: {cfg.get("model", "model_study_path")}""")
        return study

    @staticmethod
    def retrain_with_best_params(train_df, val_df, test_df, cfg, best_params):
        mlflow.set_tracking_uri(cfg.get("optimization", "mlflow_tracking_uri"))
        mlflow.set_experiment(cfg.get("optimization", "mlflow_training_experiment_name"))
        max_epochs = (
            int(cfg.get("model", "max_epochs")) if cfg.has_option("model", "max_epochs") else 10
        )
        with mlflow.start_run(run_name="final_model"):
            model_runner = TFTModel(cfg)
            model_runner.fit(train_df, val_df, test_df, best_params, max_epochs=max_epochs)
            mlflow.log_params(best_params)
            # mlflow.log_artifact(best_model)
            mlflow.pytorch.log_model(model_runner.model, artifact_path="model")
            test_pred = model_runner.predict(model_runner.test_dataloader, test_df)
            test_wmae = compute_wmae(test_df, test_pred, "sales", "pred")
            mlflow.log_metric("test_wmae", test_wmae)
            model_runner.trainer.save_checkpoint(cfg.get("model", "model_final_path"))
            logger.success(f"""Modelo final salvo em: {cfg.get("model", "model_final_path")}""")
