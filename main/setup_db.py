"""
Script para configurar o banco de dados para o exercício.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import psycopg2
import typer
from conf import DEFAULT_LOG_PATH, PROCESSED_DATA_DIR
from loguru import logger
from sqlalchemy import create_engine
from utils.extract_config import configfile

app = typer.Typer()

# Obtém as configurações do arquivo de configuração
config = configfile()
DATABASE = config.get("database", "name")
USER = config.get("database", "user")
PASSWORD = config.get("database", "password")
HOST = config.get("database", "host")
PORT = config.get("database", "port", fallback="5432")

DB_URL = f"postgresql://{USER}:{PASSWORD}@{HOST}:{PORT}/{DATABASE}"

# Configuração do log
logger.add(f"{DEFAULT_LOG_PATH}/setup_db.log", rotation="00:00", retention="90 days", level="INFO")


@app.command()
def main():
    """Executa a configuração inicial do banco de dados."""
    logger.info("Iniciando configuração do banco de dados...")

    try:
        # Estabelecendo a conexão com o banco de dados
        conn = psycopg2.connect(
            database=DATABASE, user=USER, password=PASSWORD, host=HOST, port=PORT
        )
        cursor = conn.cursor()
        logger.info("Conexão com o banco de dados estabelecida com sucesso.")

        # Excluindo tabelas se já existirem
        cursor.execute("DROP TABLE IF EXISTS SALE_CLEAN, PREDICTION")

        # Criando tabelas
        create_pred_sql = """
        CREATE TABLE PREDICTION(
            STORE_ID TEXT NOT NULL,
            CAT_ID TEXT NOT NULL,
            DATE DATE NOT NULL,
            PRED FLOAT NOT NULL,
            CREATION_TIME TIMESTAMP NOT NULL,
            PRIMARY KEY(DATE, STORE_ID, CAT_ID, CREATION_TIME)
        )"""

        create_sale_sql = """
        CREATE TABLE SALE_CLEAN(
            STORE_ID TEXT NOT NULL,
            CAT_ID TEXT NOT NULL,
            DATE DATE NOT NULL,
            SALES FLOAT NOT NULL,
            IN_TRAINING BOOLEAN,
            PRIMARY KEY(DATE, STORE_ID, CAT_ID)
        )"""

        cursor.execute(create_pred_sql)
        logger.info("Tabela 'PREDICTION' criada com sucesso.")

        cursor.execute(create_sale_sql)
        logger.info("Tabela 'SALE_CLEAN' criada com sucesso.")

        conn.commit()
        conn.close()

        # Criando conexão com SQLAlchemy
        engine = create_engine(DB_URL)

        # Escrevendo os dados processados na tabela SALE_CLEAN
        sales_df = pd.read_parquet(PROCESSED_DATA_DIR / "sales_cleaning_processed.parquet")
        sales_df.to_sql("sale_clean", engine, if_exists="append", index=False)
        logger.info("Dados de vendas carregados na tabela 'SALE_CLEAN'.")

    except Exception as e:
        logger.error(f"Erro durante a configuração do banco de dados: {e}")


if __name__ == "__main__":
    app()
