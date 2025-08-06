"""
Função auxiliar para manipular a extração de configurações.
"""

import configparser
from pathlib import Path

from conf import BASE_CONFIG_PATH

# Caminho padrão do arquivo de configuração
BASE_CONFIG_PATH_COMPLETE = f"{BASE_CONFIG_PATH}/config.ini"


def configfile(base_config_path: Path = BASE_CONFIG_PATH_COMPLETE):
    """
    Lê as configurações do arquivo base e retorna um objeto ConfigParser.

    Exemplo de uso:
    ```python
    config = configfile()
    database_url = config.get('database', 'database_url')
    ```

    Parâmetros:
    - base_config_path (Path): Caminho do arquivo de configuração.

    Retorna:
    - config (ConfigParser): Objeto que permite acesso aos parâmetros do config.ini.
    """
    config = configparser.ConfigParser()
    config.read(base_config_path)
    return config


if __name__ == "__main__":
    config = configfile()
