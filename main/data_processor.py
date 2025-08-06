"""
Classe para processamento de dados utilizados na previsão e avaliação do modelo.
"""

import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from conf import DEFAULT_LOG_PATH
from loguru import logger
from sqlalchemy import create_engine
from utils.extract_config import configfile

# Suprime o alerta SettingWithCopyWarning do Pandas
pd.options.mode.chained_assignment = None

# Obtém as configurações do arquivo de configuração
configfile = configfile()
DATABASE = configfile.get("database", "name")
USER = configfile.get("database", "user")
PASSWORD = configfile.get("database", "password")
HOST = configfile.get("database", "host")
PORT = configfile.get("database", "port", fallback="5432")

DB_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"
PREDICTION_LENGTH = int(configfile.get("model", "prediction_length"))
CONTEXT_LENGTH = int(configfile.get("model", "context_length"))

# Configuração do log
logger.add(
    f"{DEFAULT_LOG_PATH}/data_processor.log", rotation="00:00", retention="90 days", level="INFO"
)


class DataProcessor:
    """
    Classe responsável pelo processamento de dados.
    """

    def __init__(self):
        self.engine = create_engine(DB_URL)

    def get_data_for_prediction(self):
        """
        Obtém os dados mais recentes para previsão.

        Retorna:
        prediction_df (pandas.DataFrame): DataFrame com os dados processados para previsão.
        """
        date_df = self.read_from_db("SELECT MAX(date) AS date FROM sale_clean")
        max_date = pd.to_datetime(date_df.at[0, "date"], format="%Y-%m-%d")
        filter_date = max_date - pd.Timedelta(f"{CONTEXT_LENGTH} days")
        logger.info(f"Data de filtro extraída: {filter_date}")

        sales_df = self.read_from_db(f"SELECT * FROM sale_clean WHERE date >= '{filter_date}'")
        sales_df["date"] = pd.to_datetime(sales_df["date"], format="%Y-%m-%d")
        sales_df["series_id"] = sales_df["store_id"] + "_" + sales_df["cat_id"]
        sales_df["time_idx"] = (sales_df["date"] - sales_df["date"].min()).dt.days

        test_date = max_date - pd.Timedelta(f"{PREDICTION_LENGTH - 1} days")
        future_df = sales_df.loc[sales_df.date >= test_date].copy()
        future_df["time_idx"] += PREDICTION_LENGTH
        future_df["date"] += pd.Timedelta(f"{PREDICTION_LENGTH} days")
        future_df["sales"] = 0

        prediction_df = pd.concat([sales_df, future_df]).reset_index(drop=True)
        logger.info("Dados processados para previsão.")
        return prediction_df

    def read_from_db(self, sql_script, **kwargs):
        """
        Executa uma query SQL e retorna um DataFrame.
        """
        return pd.read_sql(sql_script, self.engine, **kwargs)

    def write_to_db(self, df, table_name, if_exists="append", index=False):
        """
        Escreve um DataFrame no banco de dados.
        """
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=index)
        logger.info(f"Dados escritos na tabela {table_name}.")

    def get_data_for_training(self, from_date=None, split_num=0):
        """
        Obtém os dados para treinamento.
        """
        query = "SELECT * FROM sale_clean"
        if from_date:
            query += f" WHERE date >= '{from_date}'"
        sales_df = self.read_from_db(query)
        sales_df["date"] = pd.to_datetime(sales_df["date"], format="%Y-%m-%d")
        sales_df["series_id"] = sales_df["store_id"] + "_" + sales_df["cat_id"]
        sales_df["time_idx"] = (sales_df["date"] - sales_df["date"].min()).dt.days
        return self._expanding_window_split(sales_df, split_num)

    def get_data_for_eval(self):
        """
        Obtém os dados necessários para avaliação.
        """
        date_df = self.read_from_db("SELECT MAX(date) AS date FROM sale_clean")
        max_date = pd.to_datetime(date_df.at[0, "date"], format="%Y-%m-%d")
        filter_date = max_date - pd.Timedelta(f"{CONTEXT_LENGTH+PREDICTION_LENGTH} days")
        logger.info(f"Data de filtro extraída: {filter_date}")

        sales_df = self.read_from_db(f"SELECT * FROM sale_clean WHERE date >= '{filter_date}'")
        sales_df["date"] = pd.to_datetime(sales_df["date"], format="%Y-%m-%d")
        sales_df["series_id"] = sales_df["store_id"] + "_" + sales_df["cat_id"]
        sales_df["time_idx"] = (sales_df["date"] - sales_df["date"].min()).dt.days

        test_date = max_date - pd.Timedelta(f"{PREDICTION_LENGTH - 1} days")
        evaluation_df = sales_df.loc[sales_df.date >= test_date]
        training_df = sales_df.copy()
        training_df.loc[sales_df.date >= test_date, "sales"] = 0

        logger.info("Dados processados para previsão.")
        return evaluation_df, training_df

    def create_idx_prediction(self, df_pred, df_eval):
        """
        Junta as previsões com o DataFrame de avaliação, garantindo alinhamento dos índices.
        """
        if "creation_time" in df_pred.columns:
            df_pred = df_pred.drop(columns=["creation_time"])

        df_pred["date"] = pd.to_datetime(df_pred["date"])
        df_eval["date"] = pd.to_datetime(df_eval["date"])

        df_pred["series_id"] = df_pred["store_id"] + "_" + df_pred["cat_id"]

        result = pd.merge(
            df_eval[["store_id", "cat_id", "date", "sales", "series_id", "time_idx"]],
            df_pred[["store_id", "cat_id", "date", "pred"]],
            on=["store_id", "cat_id", "date"],
            how="left",
        )

        return result

    def _expanding_window_split(
        self, df, split_num, prediction_length=PREDICTION_LENGTH, validation=True
    ):
        """
        Divide os dados em treino, validação e teste usando a abordagem expanding window.
        """
        df = df.copy().reset_index(drop=False)
        df["series_id"] = df["store_id"] + "_" + df["cat_id"]
        test_list, validation_list, training_list = [], [], []

        test_offset = prediction_length * ((split_num + 1) * 2 - 1)
        val_offset = prediction_length * (split_num + 1) * 2
        test_upper_offset = prediction_length * (split_num * 2)

        for series in df["series_id"].unique():
            df_series = df[df["series_id"] == series]
            max_date, min_date = df_series["date"].max(), df_series["date"].min()

            test_lower_date = max_date - pd.Timedelta(f"{test_offset} days")
            test_upper_date = max_date - pd.Timedelta(f"{test_upper_offset} days")
            val_lower_date = max_date - pd.Timedelta(f"{val_offset} days")

            df_series_test = df_series.query(
                "date > @test_lower_date and date <= @test_upper_date"
            )
            df_series_val = (
                df_series.query("date > @val_lower_date and date <= @test_lower_date")
                if validation
                else pd.DataFrame()
            )
            df_series_train = (
                df_series.query("date <= @val_lower_date")
                if validation
                else df_series.query("date <= @test_lower_date")
            )

            test_list.append(df_series_test)
            validation_list.append(df_series_val)
            training_list.append(df_series_train)

        return (
            pd.concat(training_list, ignore_index=True),
            pd.concat(validation_list, ignore_index=True),
            pd.concat(test_list, ignore_index=True),
        )
