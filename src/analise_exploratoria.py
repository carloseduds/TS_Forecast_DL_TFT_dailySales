import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import savgol_filter


def visualizar_grafico(df, loja, categoria, data_alvo=None):
    """
    Visualiza os outliers de uma loja e categoria específica.

    Parâmetros:
        df (DataFrame): DataFrame contendo os dados de vendas.
        loja (str): Identificador da loja.
        categoria (str): Identificador da categoria de produto.
        data_alvo (str, opcional): Data a ser destacada no gráfico.
    """

    df_serie = df.loc[
        (df["store_id"] == loja) & (df["cat_id"] == categoria)
    ].copy()

    if df_serie.empty:
        print(f"Nenhum dado encontrado para Loja: {loja}, Categoria: {categoria}.")
        return

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(df_serie.index, df_serie["sales"], color="black", label="Normal")

    if data_alvo:
        data_alvo = pd.to_datetime(data_alvo)
        if data_alvo in df_serie.index:
            ax.axvline(
                data_alvo,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Quebra: {data_alvo.date()}",
            )
        else:
            print(f"A data {data_alvo.date()} não está presente nos dados.")

    plt.title(f"Gráfico da Loja: {loja} | Categoria: {categoria}")
    plt.xlabel("Data")
    plt.ylabel("Vendas")
    plt.legend()
    plt.grid(True)
    plt.show()


def visualizar_grafico_grid(df, lojas, categorias):
    """
    Gera uma grade de gráficos visualizando os outliers por loja e categoria.

    Parâmetros:
        df (DataFrame): DataFrame contendo 'store_id', 'cat_id', 'sales' e 'anomaly'.
        lojas (list): Lista de identificadores de lojas.
        categorias (list): Lista de categorias de produtos.
    """
    n_linhas = len(lojas)
    n_colunas = len(categorias)

    fig, eixos = plt.subplots(n_linhas, n_colunas, figsize=(30, 30), sharex=False, sharey=True)
    fig.suptitle("Visualização de Gráficos por Categoria e Loja", fontsize=16)

    for i, loja in enumerate(lojas):
        for j, categoria in enumerate(categorias):
            eixo = eixos[i, j]
            df_serie = df.loc[
                (df["store_id"] == loja) & (df["cat_id"] == categoria)
            ].copy()
            eixo.plot(df_serie.index, df_serie["sales"], color="black")
            eixo.set_title(f"{loja} - {categoria}", fontsize=10)
            eixo.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def remover_tendencia(df, loja, categoria):
    """
    Remove a tendência da série temporal de vendas utilizando o filtro Savitzky-Golay.

    Parâmetros:
        df (DataFrame): DataFrame contendo os dados de vendas.
        loja (str): Identificador da loja.
        categoria (str): Categoria do produto.

    Retorno:
        tuple: (array de vendas sem tendência, array da tendência, DataFrame filtrado)
    """
    df_filtro = df.loc[
        (df.store_id == loja) & (df.cat_id == categoria)
    ]
    tendencia = savgol_filter(df_filtro["sales"].values, 365, 1)
    vendas_sem_tendencia = (df_filtro["sales"] - tendencia).values
    return vendas_sem_tendencia, tendencia, df_filtro


def extrair_periodicidade(vendas_sem_tendencia, calcular_periodo=False):
    """
    Identifica os principais valores de frequência na série temporal por meio do periodograma.

    Parâmetros:
        vendas_sem_tendencia (array): Série temporal de vendas sem tendência.
        calcular_periodo (bool, opcional): Se True, retorna também a frequência dominante e o período correspondente.

    Retorno:
        DataFrame contendo os três principais valores de frequência e suas densidades espectrais.
    """
    frequencias, espectro_potencia = signal.periodogram(vendas_sem_tendencia)
    frequencias_validas = frequencias[frequencias > 0]
    espectro_potencia_validos = espectro_potencia[frequencias > 0]

    dados = {
        "intervalo": 1 / frequencias_validas,
        "densidade_espectral": espectro_potencia_validos,
    }
    df_periodograma = pd.DataFrame(dados)
    df_periodograma.sort_values(by="densidade_espectral", ascending=False, inplace=True)

    if calcular_periodo:
        frequencia_dominante = frequencias[np.argmax(espectro_potencia)]
        periodo_dominante = 1 / frequencia_dominante if frequencia_dominante > 0 else None
        return df_periodograma.head(3), periodo_dominante

    return df_periodograma.head(3)


def plotar_tendencia(df_filtro, tendencia, vendas_sem_tendencia):
    """Plota as vendas originais, a tendência identificada e a série sem tendência."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].plot(df_filtro.index.values, df_filtro["sales"].values, label="Vendas Originais")
    ax[0].plot(df_filtro.index.values, tendencia, label="Tendência", linestyle="dashed")
    ax[0].set_title("Vendas com Linha de Tendência")
    ax[0].set_ylabel("Vendas")
    ax[0].legend()
    ax[1].plot(df_filtro.index.values, vendas_sem_tendencia, color="orange")
    ax[1].set_title("Vendas Sem Tendência")
    plt.show()


def plotar_periodicidade(vendas_sem_tendencia):
    """Plota o periodograma da série temporal sem tendência."""
    frequencias, espectro_potencia = signal.periodogram(vendas_sem_tendencia)
    plt.figure(figsize=(10, 5))
    plt.plot(frequencias, espectro_potencia, color="purple")
    plt.title("Periodograma - Identificação de Sazonalidade")
    plt.xlabel("Frequência")
    plt.ylabel("Densidade de Potência")
    plt.grid(True)
    plt.show()
