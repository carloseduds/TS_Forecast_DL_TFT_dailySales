"""
Script para configurar a API REST para servir previsões.
"""

import datetime
import json
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI
from loguru import logger

from main.conf import DEFAULT_LOG_PATH
from main.data_processor import DataProcessor
from main.utils.extract_config import configfile

# Obtém as configurações do arquivo de configuração
config = configfile()
DATABASE = config.get("database", "name")
USER = config.get("database", "user")
PASSWORD = config.get("database", "password")
HOST = config.get("database", "host")
PORT = config.get("database", "port", fallback="5432")

DB_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"

# Configuração do log
logger.add(f"{DEFAULT_LOG_PATH}/api.log", rotation="00:00", retention="90 days", level="INFO")

# Criação da instância FastAPI
description = "Esta API retorna as previsões de vendas."
app = FastAPI(title="Previsões de Vendas", description=description)


@app.get("/predictions/")
def get_predictions(
    start: Optional[datetime.date] = None,
    end: Optional[datetime.date] = None,
    category: Optional[str] = None,
    store: Optional[str] = None,
):
    """
    Recupera os dados de previsão do banco de dados com base nos parâmetros da consulta.

    Parâmetros:
    - start (datetime.date): Data de início para a consulta. Opcional.
    - end (datetime.date): Data final para a consulta. Opcional.
    - category (str): Categoria para filtrar. Opcional.
    - store (str): Loja para filtrar. Opcional.

    Retorna:
    - parsed (JSON): Dados de previsão e esquema relevante.
    """
    start_time = time.time()
    dataprocessor = DataProcessor()

    distinct_on_query_part = "DISTINCT ON (store_id, cat_id, date)"
    order_by_query_part = "ORDER BY store_id, cat_id, date, creation_time DESC"

    # Lógica para consulta dos dados de previsão com base nos parâmetros
    query = f"SELECT {distinct_on_query_part} * FROM prediction "
    conditions = []

    if start:
        conditions.append(f"date >= '{start}'")
    if end:
        conditions.append(f"date <= '{end}'")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += f" {order_by_query_part} LIMIT 100"

    predictions_df = dataprocessor.read_from_db(query, parse_dates=["date", "creation_time"])

    if category:
        predictions_df = predictions_df.loc[predictions_df.cat_id == category]
    if store:
        predictions_df = predictions_df.loc[predictions_df.store_id == store]

    # Converte o DataFrame para JSON
    result = predictions_df.to_json(orient="table", index=False)
    parsed = json.loads(result)

    logger.info(f"Tempo de execução: {time.time()-start_time:.2f} segundos")
    return parsed


if __name__ == "__main__":
    # Inicia a API
    uvicorn.run("main.app:app", host="127.0.0.1", port=8000, log_level="info")
