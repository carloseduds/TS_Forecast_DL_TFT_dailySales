import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.stattools import adfuller


def testa_estacionaridade(serie, nome="Série", window=12, plot=True):
    """
    Testa a estacionariedade de uma série temporal utilizando estatísticas móveis e o teste Dickey-Fuller.

    Parâmetros:
        serie (pd.Series): Série temporal a ser analisada (com índice datetime).
        nome (str): Nome da série para identificação na tabela de resultados.
        window (int): Tamanho da janela para cálculo da média móvel e desvio padrão.
        plot (bool): Se True, exibe os gráficos de estatísticas móveis. Se False, apenas executa o teste Dickey-Fuller.

    Retorno:
        pd.DataFrame contendo os resultados do teste Dickey-Fuller.
    """
    if not isinstance(serie, pd.Series):
        raise ValueError("A entrada deve ser uma pandas Series.")

    if not np.issubdtype(serie.index.dtype, np.datetime64):
        raise ValueError("O índice da série deve estar no formato datetime.")

    if plot:
        rolmean = serie.rolling(window=window).mean()
        rolstd = serie.rolling(window=window).std()

        plt.figure(figsize=(10, 4))
        plt.plot(serie, color="blue", label="Série Original", alpha=0.7)
        plt.plot(rolmean, color="red", label=f"Média Móvel ({window})")
        plt.plot(rolstd, color="black", linestyle="dashed", label=f"Desvio Padrão ({window})")
        plt.legend(loc="best")
        plt.title(f"Estatísticas Móveis - Média e Desvio Padrão para o Grupo: {nome}")
        plt.grid()
        plt.show()

    dfteste = adfuller(serie.dropna(), autolag="AIC")
    resultados = pd.DataFrame(
        {
            "Série": [nome],
            "Estatística do Teste": [dfteste[0]],
            "Valor-p": [dfteste[1]],
            "Número de Lags": [dfteste[2]],
            "Número de Observações": [dfteste[3]],
            "Valor Crítico (1%)": [dfteste[4]["1%"]],
            "Valor Crítico (5%)": [dfteste[4]["5%"]],
            "Valor Crítico (10%)": [dfteste[4]["10%"]],
            "Estacionária?": [dfteste[1] < 0.05],
        }
    )
    return resultados


def detectar_outliers(dataset):
    """
    Detecta outliers utilizando o intervalo interquartil (IQR).

    Parâmetros:
        dataset (DataFrame): Conjunto de dados contendo 'sales_detrend'.

    Retorno:
        DataFrame atualizado com a coluna 'anomaly' indicando os outliers.
    """
    Q1 = dataset["sales_detrend"].quantile(0.25)
    Q3 = dataset["sales_detrend"].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 3 * IQR
    limite_superior = Q3 + 3 * IQR
    dataset["anomaly"] = (dataset["sales_detrend"] <= limite_inferior) | (
        dataset["sales_detrend"] >= limite_superior
    )
    return dataset


def processar_grupo(grupo):
    """
    Aplica remoção de tendência e detecção de outliers para cada grupo de store_id e cat_id.

    Parâmetros:
        grupo (DataFrame): Subconjunto dos dados agrupado por 'store_id' e 'cat_id'.

    Retorno:
        DataFrame processado com outliers detectados.
    """
    grupo = grupo.copy()
    grupo["sales_detrend"] = signal.detrend(grupo["sales"].values)
    return detectar_outliers(grupo)


def visualizar_outliers(dataset, loja, categoria):
    """
    Visualiza os outliers de uma loja e categoria específica.
    """
    outlier_series_df = dataset.loc[
        (dataset["store_id"] == loja) & (dataset["cat_id"] == categoria)
    ].copy()

    if outlier_series_df.empty:
        print(f"Nenhum dado encontrado para Loja: {loja}, Categoria: {categoria}.")
        return

    outliers = outlier_series_df.loc[outlier_series_df["anomaly"], ["sales"]]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(outlier_series_df.index, outlier_series_df["sales"], color="black", label="Normal")

    if not outliers.empty:
        ax.scatter(outliers.index, outliers["sales"], color="red", label="Anomalia")

    plt.title(f"Outliers na Loja: {loja} | Categoria: {categoria}")
    plt.xlabel("Data")
    plt.ylabel("Vendas")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualizar_outliers_grid(dataset, lojas, categorias):
    """
    Gera uma grade de gráficos visualizando os outliers por loja e categoria.

    Parâmetros:
        dataset (DataFrame): Conjunto de dados contendo 'store_id', 'cat_id', 'sales' e 'anomaly'.
        lojas (list): Lista de identificadores de lojas.
        categorias (list): Lista de categorias de produtos.
    """
    n_linhas = len(lojas)
    n_colunas = len(categorias)

    fig, eixos = plt.subplots(n_linhas, n_colunas, figsize=(30, 30), sharex=False, sharey=True)
    fig.suptitle("Visualização de Outliers por Categoria e Loja", fontsize=16)

    for i, loja in enumerate(lojas):
        for j, categoria in enumerate(categorias):
            eixo = eixos[i, j]
            outlier_series_df = dataset.loc[
                (dataset["store_id"] == loja) & (dataset["cat_id"] == categoria)
            ].copy()
            outliers = outlier_series_df.loc[outlier_series_df["anomaly"], ["sales"]]
            eixo.plot(outlier_series_df.index, outlier_series_df["sales"], color="black")
            if not outliers.empty:
                eixo.scatter(outliers.index, outliers["sales"], color="red")
            eixo.set_title(f"{loja} - {categoria}", fontsize=10)
            eixo.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def visualizar_sales_imputed(dataset, loja, categoria):
    """
    Visualiza a série temporal de vendas, destacando os valores imputados.

    Parâmetros:
        dataset (DataFrame): Conjunto de dados contendo 'store_id', 'cat_id', 'sales' e 'anomaly'.
        loja (str): Identificador da loja.
        categoria (str): Categoria do produto.
    """
    sales_series_df = dataset.loc[
        (dataset["store_id"] == loja) & (dataset["cat_id"] == categoria)
    ].copy()

    if sales_series_df.empty:
        print(f"Nenhum dado encontrado para Loja: {loja}, Categoria: {categoria}.")
        return

    outliers = sales_series_df.loc[sales_series_df["anomaly"], ["sales"]]

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(
        sales_series_df.index, sales_series_df["sales"], color="black", label="Normal", alpha=0.5
    )
    if not outliers.empty:
        ax.scatter(outliers.index, outliers["sales"], color="red", label="Outlier Imputed")
    plt.title(f"Loja: {loja} | Categoria: {categoria}")
    plt.xlabel("Data")
    plt.ylabel("Vendas")
    plt.legend()
    plt.grid(True)
    plt.show()
