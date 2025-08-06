import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from prophet import Prophet
from pytorch_forecasting import RecurrentNetwork, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, MAPE, QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer
from tqdm import tqdm

# ==============================================================================
# 🔹 FUNÇÃO DE DIVISÃO EXPANDING WINDOW
# ==============================================================================


def expanding_window_split(df, split_num, prediction_length=28, validation=True):
    """
    Implementa a divisão dos dados em treino, validação e teste usando a abordagem expanding window.

    Parâmetros:
        - df (DataFrame): Conjunto de dados contendo séries temporais.
        - split_num (int): Índice do split para controle do deslocamento da janela de previsão.
        - prediction_length (int): Número de dias previstos por janela.
        - validation (bool): Se True, inclui um conjunto de validação.

    Retorna:
        - training_df (DataFrame): Dados de treinamento.
        - validation_df (DataFrame): Dados de validação (se `validation=True`).
        - test_df (DataFrame): Dados de teste.
    """
    df = df.copy().reset_index(drop=False)

    # Criar identificador único da série caso não exista
    if "series_id" not in df.columns:
        df["series_id"] = df["store_id"] + "_" + df["cat_id"]

    test_list, validation_list, training_list = [], [], []

    # Calcular deslocamentos de datas
    test_offset = prediction_length * ((split_num + 1) * 2 - 1)
    val_offset = prediction_length * (split_num + 1) * 2
    test_upper_offset = prediction_length * (split_num * 2)

    for series in df["series_id"].unique():
        df_series = df[df["series_id"] == series]
        max_date, min_date = df_series["date"].max(), df_series["date"].min()

        test_lower_date = max_date - pd.Timedelta(f"{test_offset} days")
        test_upper_date = max_date - pd.Timedelta(f"{test_upper_offset} days")
        val_lower_date = max_date - pd.Timedelta(f"{val_offset} days")

        if min(test_lower_date, test_upper_date) < min_date:
            raise ValueError(
                f"A série '{series}' não tem dados suficientes para o split {split_num}. "
                f"Data mínima: {min_date}, necessário: {test_lower_date}."
            )

        df_series_test = df_series.query("date > @test_lower_date and date <= @test_upper_date")
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


# ==============================================================================
# 🔹 FUNÇÕES DE PREVISÃO (Naïve e sNaïve)
# ==============================================================================


def previsao_naive(dados_treino, dados_teste):
    """
    Aplica previsão ingênua (Naïve), onde a previsão é simplesmente o último valor observado.

    Parâmetros:
        - dados_treino (DataFrame): Conjunto de dados de treinamento.
        - dados_teste (DataFrame): Conjunto de dados de teste.

    Retorna:
        - DataFrame contendo os dados de teste com a previsão Naïve.
    """
    ultima_observacao = (
        dados_treino.groupby("series_id").last().reset_index()[["series_id", "sales"]]
    )
    ultima_observacao.rename(columns={"sales": "naive_pred"}, inplace=True)
    return dados_teste.merge(ultima_observacao, on="series_id", how="left")


def previsao_snaive(dados_treino, dados_teste):
    """
    Aplica previsão sazonal ingênua (sNaïve), onde a previsão futura é baseada no valor
    do mesmo dia da semana anterior.

    Parâmetros:
        - dados_treino (DataFrame): Dados de treinamento.
        - dados_teste (DataFrame): Dados de teste.

    Retorna:
        - DataFrame contendo os dados de teste com a previsão sazonal Naïve.
    """
    dados_treino["dayofweek"] = dados_treino["date"].dt.weekday
    dados_teste["dayofweek"] = dados_teste["date"].dt.weekday

    ultimos_dados = (
        dados_treino.sort_values(by=["series_id", "date"], ascending=[True, False])
        .groupby(["series_id", "dayofweek"])
        .first()
        .reset_index()[["series_id", "dayofweek", "sales"]]
    )

    ultimos_dados.rename(columns={"sales": "snaive_pred"}, inplace=True)

    return dados_teste.merge(ultimos_dados, on=["series_id", "dayofweek"], how="left")


# ==============================================================================
# 🔹 FUNÇÕES DE PLOTAR PREVISAO
# ==============================================================================


def plotar_previsao(dados_previsao_teste, id_serie, yhat):
    """
    Plota as previsões em comparação com as vendas reais para uma série temporal específica.

    Parâmetros:
        dados_previsao_teste (DataFrame): Conjunto de dados contendo as previsões e as vendas reais.
        id_serie (str ou int): Identificador da série temporal a ser plotada.
        yhat (str): Nome da coluna contendo as previsões.

    Retorna:
        None: Exibe o gráfico com a comparação entre vendas reais e previsões.
    """
    dados_serie = dados_previsao_teste.loc[dados_previsao_teste.series_id == id_serie]
    plt.figure(figsize=(15, 5))
    sns.lineplot(x=dados_serie.date, y=dados_serie.sales, label="Sales")
    sns.lineplot(x=dados_serie.date, y=dados_serie[yhat], label=yhat)
    plt.xlabel("Data")
    plt.ylabel("Vendas")
    plt.title(f"Vendas Reais vs. Previsões para a Série {id_serie}")
    plt.legend()
    plt.show()


# ==============================================================================
# 🔹 FUNÇÕES DE AVALIAÇÃO (WMAE e WMAPE)
# ==============================================================================


def calcular_wmae_wmape(dados_treinamento, dados_teste, verdadeiro, previsto):
    """
    Calcula o Erro Absoluto Médio Ponderado (WMAE) e o Erro Percentual Absoluto Médio Ponderado (WMAPE).

    Parâmetros:
        - dados_treinamento (DataFrame): Dados de treino.
        - dados_teste (DataFrame): Dados de teste.
        - verdadeiro (str): Nome da coluna com os valores reais.
        - previsto (str): Nome da coluna com as previsões.

    Retorna:
        - Tuple (wmae, wmape)
    """
    series_unicas = dados_teste["series_id"].unique()

    lista_vendas, lista_mae, lista_mape = [], [], []

    for serie in series_unicas:
        dados_teste_serie = dados_teste[dados_teste["series_id"] == serie].copy()
        dados_treinamento_serie = dados_treinamento[
            dados_treinamento["series_id"] == serie
        ].sort_values(by="date", ascending=False)

        dados_teste_serie["erro_absoluto"] = (
            dados_teste_serie[previsto] - dados_teste_serie[verdadeiro]
        ).abs()
        mae = dados_teste_serie["erro_absoluto"].mean()

        dados_teste_serie["erro_pct_absoluto"] = (
            (dados_teste_serie[verdadeiro] - dados_teste_serie[previsto])
            / dados_teste_serie[verdadeiro]
        ).abs()
        mape = dados_teste_serie["erro_pct_absoluto"].mean()

        vendas_totais = dados_treinamento_serie.head(28)["sales"].sum()

        lista_mae.append(mae)
        lista_mape.append(mape)
        lista_vendas.append(vendas_totais)

    vendas_totais_geral = np.sum(lista_vendas)

    if vendas_totais_geral == 0:
        return 0, 0

    pesos = np.array(lista_vendas) / vendas_totais_geral
    return np.dot(lista_mae, pesos), np.dot(lista_mape, pesos)


# ==============================================================================
# 🔹 FUNÇÃO DE PREVISÃO COM PROPHET
# ==============================================================================


def previsao_prophet(
    dados_treinamento,
    dados_teste,
    cv,
    sazonalidade_mensal=True,
    escala_ponto_mudanca=0.05,
    intervalo_ponto_mudanca=0.8,
):
    """
    Treina e faz previsões de vendas usando o modelo Prophet.

    Parâmetros:
        - dados_treinamento (DataFrame): Conjunto de dados de treino.
        - dados_teste (DataFrame): Conjunto de dados de teste.
        - cv (int): Índice da validação cruzada.
        - sazonalidade_mensal (bool): Adiciona sazonalidade semanal e mensal ao modelo.
        - escala_ponto_mudanca (float): Define a flexibilidade do modelo às mudanças de tendência.
        - intervalo_ponto_mudanca (float): Percentual dos dados usado para detectar mudanças de tendência.

    Retorna:
        - DataFrame contendo as previsões do Prophet.
    """
    # Renomeia colunas para o formato esperado pelo Prophet
    dados_treinamento = dados_treinamento.rename(columns={"sales": "y", "date": "ds"})

    series_unicas = dados_treinamento["series_id"].unique()
    lista_previsoes = []

    for serie in tqdm(series_unicas, desc=f"Previsão Prophet - CV{cv}"):
        dados_treinamento_serie = dados_treinamento[dados_treinamento["series_id"] == serie]

        # Inicializa o modelo Prophet
        modelo = Prophet(
            yearly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=escala_ponto_mudanca,
            changepoint_range=intervalo_ponto_mudanca,
        )

        # Adiciona sazonalidades, se ativado
        if sazonalidade_mensal:
            modelo.add_seasonality(name="semanal", period=7, fourier_order=3)
            modelo.add_seasonality(name="mensal", period=30.5, fourier_order=5)

        # Treina o modelo
        modelo.fit(dados_treinamento_serie)

        # Cria o período de previsão (28 dias no futuro)
        futuro = modelo.make_future_dataframe(periods=28, include_history=False)
        previsao = modelo.predict(futuro)[["ds", "yhat"]]

        # Adiciona o ID da série e salva a previsão
        previsao["series_id"] = serie
        lista_previsoes.append(previsao)

    # Une todas as previsões em um único DataFrame
    previsoes_prophet = pd.concat(lista_previsoes)

    # Renomeia colunas de volta ao formato original
    previsoes_prophet = previsoes_prophet.rename(columns={"ds": "date", "yhat": "prophet_pred"})

    # Junta as previsões com os dados de teste
    dados_teste_prophet = dados_teste.merge(
        previsoes_prophet, on=["series_id", "date"], how="left"
    )

    return dados_teste_prophet


# ==============================================================================
# 🔹 MODELO LSTM PARA PREVISÃO DE SÉRIES TEMPORAIS
# ==============================================================================


def modelagem_lstm(
    training_df,
    validation_df,
    test_df,
    max_prediction_length=28,
    max_encoder_length=28 * 5,
    batch_size=32,
    max_epochs=6,
    hidden_size=10,
    rnn_layers=2,
    dropout=0.1,
):
    """
    Cria e retorna um modelo LSTM para previsão de séries temporais.

    Parâmetros:
        - training_df (DataFrame): Dados de treino.
        - validation_df (DataFrame): Dados de validação.
        - test_df (DataFrame): Dados de teste.
        - max_prediction_length (int): Número máximo de dias para previsão.
        - max_encoder_length (int): Comprimento da janela de entrada.
        - batch_size (int): Tamanho do lote para o treinamento.
        - max_epochs (int): Número máximo de épocas de treinamento.
        - hidden_size (int): Número de unidades ocultas no LSTM.
        - rnn_layers (int): Número de camadas no LSTM.
        - dropout (float): Taxa de dropout.

    Retorna:
        - model (RecurrentNetwork): Modelo treinado.
        - trainer (pl.Trainer): Treinador do PyTorch Lightning.
        - train_dataloader, val_dataloader, test_dataloader (DataLoaders): Datasets preparados.
    """
    val_idx = validation_df["time_idx"].min()
    test_idx = test_df["time_idx"].min()

    training_data = TimeSeriesDataSet(
        training_df,
        time_idx="time_idx",
        target="sales",
        group_ids=["series_id"],
        time_varying_unknown_reals=["sales"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        target_normalizer=GroupNormalizer(groups=["series_id"]),
    )

    validation_data = TimeSeriesDataSet.from_dataset(
        training_data,
        pd.concat([training_df, validation_df]).reset_index(drop=True),
        min_prediction_idx=val_idx,
    )
    test_data = TimeSeriesDataSet.from_dataset(
        training_data,
        pd.concat([training_df, validation_df, test_df]).reset_index(drop=True),
        min_prediction_idx=test_idx,
    )

    train_dataloader = training_data.to_dataloader(train=True, batch_size=batch_size)
    val_dataloader = validation_data.to_dataloader(train=False, batch_size=batch_size)
    test_dataloader = test_data.to_dataloader(train=False, batch_size=batch_size)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    logger = TensorBoardLogger("training_logs")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        gradient_clip_val=0.1,
        callbacks=[checkpoint_callback],
        logger=logger,
    )

    model = RecurrentNetwork.from_dataset(
        training_data,
        hidden_size=hidden_size,
        rnn_layers=rnn_layers,
        dropout=dropout,
        logging_metrics=[MAE(), MAPE()],
    )

    return model, trainer, train_dataloader, val_dataloader, test_dataloader


# ==============================================================================
# 🔹 PREVISAO LSTM PARA PREVISÃO DE SÉRIES TEMPORAIS
# ==============================================================================


def previsao_lstm(trainer, model, train_dataloader, val_dataloader, test_dataloader, test_df):
    """
    Gera previsões usando um modelo LSTM treinado e retorna um DataFrame com os valores reais e previstos.

    Parâmetros:
        - trainer (pl.Trainer): Treinador PyTorch Lightning contendo o modelo treinado.
        - test_dataloader (DataLoader): DataLoader contendo os dados de teste.
        - test_df (DataFrame): DataFrame original dos dados de teste.

    Retorna:
        - DataFrame contendo as previsões LSTM adicionadas aos dados de teste.
    """
    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Carregar o melhor modelo salvo durante o treinamento
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = RecurrentNetwork.load_from_checkpoint(best_model_path)

    # Fazer previsões com o modelo treinado
    predictions, _, index, _, _ = best_model.predict(test_dataloader, return_index=True)

    # Criar um DataFrame com as previsões e os respectivos `series_id` e `time_idx`
    time_idx_start = index.loc[0, "time_idx"]
    time_idx_end = time_idx_start + len(predictions[0])

    predictions_df_wide = pd.DataFrame(
        predictions.cpu().numpy(), columns=range(time_idx_start, time_idx_end)
    )
    predictions_df_wide["series_id"] = index["series_id"]
    predictions_df = predictions_df_wide.melt(id_vars=["series_id"])
    predictions_df.rename(columns={"variable": "time_idx", "value": "lstm_pred"}, inplace=True)

    # Fazer o merge das previsões com os dados originais de teste
    lstm_test_df = test_df.merge(predictions_df, on=["series_id", "time_idx"], how="left")

    return lstm_test_df


# ==============================================================================
# 🔹 TREINAMENTO E PREVISÃO COM TEMPORAL FUSION TRANSFORMER (TFT)
# ==============================================================================


def previsao_tft(training_df, validation_df, test_df, i, max_epochs=6):
    """
    Treina o modelo TFT e faz previsões para um conjunto de teste.

    Parâmetros:
        - training_df (DataFrame): Dados de treino.
        - validation_df (DataFrame): Dados de validação.
        - test_df (DataFrame): Dados de teste.
        - i (int): Índice da rodada de validação cruzada.
        - max_epochs (int): Número máximo de épocas de treinamento.

    Retorna:
        - test_df com as previsões adicionadas.
    """
    val_idx = validation_df["time_idx"].min()
    test_idx = test_df["time_idx"].min()

    quantis = [0.1, 0.5, 0.9]

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

    validation_data = TimeSeriesDataSet.from_dataset(
        dados_treino,
        pd.concat([training_df, validation_df]).reset_index(drop=True),
        min_prediction_idx=val_idx,
        predict=True,
        stop_randomization=True,
    )

    test_data = TimeSeriesDataSet.from_dataset(
        dados_treino,
        pd.concat([training_df, validation_df, test_df]).reset_index(drop=True),
        min_prediction_idx=test_idx,
        predict=True,
        stop_randomization=True,
    )

    # Criando DataLoaders
    dl_treino = dados_treino.to_dataloader(
        train=True, batch_size=32, num_workers=2, pin_memory=True
    )
    dl_valid = validation_data.to_dataloader(
        train=False, batch_size=32, num_workers=2, pin_memory=True
    )
    dl_teste = test_data.to_dataloader(train=False, batch_size=32, num_workers=2, pin_memory=True)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        gradient_clip_val=0.1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1),
        ],
        logger=TensorBoardLogger("../reports/lightning_logs", name=f"tft_cv_fold_{i}"),
    )

    modelo_tft = TemporalFusionTransformer.from_dataset(
        dados_treino,
        learning_rate=0.00023,
        hidden_size=128,
        dropout=0.1,
        loss=QuantileLoss(quantis),
    )

    trainer.fit(
        modelo_tft,
        train_dataloaders=dl_treino,
        val_dataloaders=dl_valid,
    )
    # Carregar o melhor modelo salvo durante o treinamento
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    # Fazer previsões com o modelo treinado
    predictions, _, index, _, _ = best_model.predict(dl_teste, return_index=True)

    # Criar DataFrame com previsões
    time_idx_start = index.loc[0, "time_idx"]
    time_idx_end = time_idx_start + len(predictions[0])

    predictions_df_wide = pd.DataFrame(
        predictions.cpu().numpy(), columns=range(time_idx_start, time_idx_end)
    )
    predictions_df_wide["series_id"] = index["series_id"]
    predictions_df = predictions_df_wide.melt(id_vars=["series_id"])
    predictions_df.rename(columns={"variable": "time_idx", "value": "tft_pred"}, inplace=True)

    # Fazer merge com test_df
    test_df_previsto = test_df.merge(predictions_df, on=["series_id", "time_idx"], how="left")

    return test_df_previsto
