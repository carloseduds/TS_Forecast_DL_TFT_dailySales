"""
Funções auxiliares para cálculo de métricas de erro.
"""

import numpy as np
import pandas as pd


def compute_mae(
    training_df: pd.DataFrame, prediction_test_df: pd.DataFrame, y: str, y_hat: str, series_id: str
):
    """
    Calcula o erro médio absoluto (MAE) para uma série temporal específica.

    Parâmetros:
    - training_df (pd.DataFrame): DataFrame com os dados de treinamento.
    - prediction_test_df (pd.DataFrame): DataFrame com previsões e valores reais de teste.
    - y (str): Nome da coluna de vendas reais.
    - y_hat (str): Nome da coluna de previsões.
    - series_id (str): ID da série temporal.

    Retorna:
    - mae (float): Erro médio absoluto.
    - total_sales (float): Total de vendas dos últimos 28 dias do treinamento.
    """
    prediction_series = prediction_test_df.loc[prediction_test_df.series_id == series_id].copy()
    training_series = training_df.loc[training_df.series_id == series_id].copy()
    training_series.sort_values(by="date", ascending=False, inplace=True)
    prediction_series["abs_error"] = (prediction_series[y_hat] - prediction_series[y]).abs()
    mae = prediction_series["abs_error"].mean()
    total_sales = training_series[:28]["sales"].sum()
    return mae, total_sales


def compute_wmae(training_df: pd.DataFrame, prediction_test_df: pd.DataFrame, y: str, y_hat: str):
    """
    Calcula o erro médio absoluto ponderado (WMAE).
    """
    series_list = prediction_test_df.series_id.unique()
    sales_list, mae_list = [], []

    for series in series_list:
        mae_series, total_sales_series = compute_mae(
            training_df, prediction_test_df, y, y_hat, series
        )
        mae_list.append(mae_series)
        sales_list.append(total_sales_series)

    overall_sales = np.sum(sales_list)
    weights = [s / overall_sales for s in sales_list]
    wmae = np.sum([m * w for m, w in zip(mae_list, weights)])
    return wmae


def compute_mape(
    training_df: pd.DataFrame, prediction_test_df: pd.DataFrame, y: str, y_hat: str, series_id: str
):
    """
    Calcula o erro percentual absoluto médio (MAPE) para uma série temporal específica.
    """
    training_series = training_df.loc[training_df.series_id == series_id].copy()
    training_series.sort_values(by="date", ascending=False, inplace=True)
    prediction_series = prediction_test_df.loc[prediction_test_df.series_id == series_id].copy()
    prediction_series["abs_pct_error"] = (
        (prediction_series[y] - prediction_series[y_hat]) / prediction_series[y]
    ).abs()
    mape = prediction_series["abs_pct_error"].mean()
    total_sales = training_series[:28]["sales"].sum()
    return mape, total_sales


def compute_wmape(training_df: pd.DataFrame, prediction_test_df: pd.DataFrame, y: str, y_hat: str):
    """
    Calcula o erro percentual absoluto médio ponderado (WMAPE).
    """
    series_list = prediction_test_df.series_id.unique()
    sales_list, mape_list = [], []

    for series in series_list:
        mape_series, total_sales_series = compute_mape(
            training_df, prediction_test_df, y, y_hat, series
        )
        mape_list.append(mape_series)
        sales_list.append(total_sales_series)

    overall_sales = np.sum(sales_list)
    weights = [s / overall_sales for s in sales_list]
    wmape = np.sum([m * w for m, w in zip(mape_list, weights)])
    return wmape


def compute_eval_data_ratio(training_df: pd.DataFrame, evaluation_df: pd.DataFrame):
    """
    Calcula métricas de avaliação comparando os dados de treino e avaliação.
    """
    training_stats = training_df.groupby("series_id").agg(
        mean_train_sales=("sales", np.mean), stdev_train_sales=("sales", np.std)
    )
    eval_stats = evaluation_df.groupby("series_id").agg(
        mean_eval_sales=("sales", np.mean), stdev_eval_sales=("sales", np.std)
    )

    ratio_df = training_stats.join(eval_stats)
    ratio_df["mean_sales_ratio"] = ratio_df["mean_eval_sales"] / ratio_df["mean_train_sales"]
    ratio_df["stdev_sales_ratio"] = ratio_df["stdev_eval_sales"] / ratio_df["stdev_train_sales"]

    mean_sales_ratio = ratio_df["mean_sales_ratio"].mean()
    stdev_sales_ratio = ratio_df["stdev_sales_ratio"].mean()

    return mean_sales_ratio, stdev_sales_ratio
