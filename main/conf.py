from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__name__).resolve().parents[0]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DEFAULT_LOG_PATH = PROJ_ROOT / "logs" / "logfile"
MODEL_FINAL_DIR = PROJ_ROOT / "models" / "final"
MODEL_STUDY_DIR = PROJ_ROOT / "models" / "study"
LIGHTLING_TRAINING_DIR = PROJ_ROOT / "reports" / "training"
PROCESSED_DATA_DIR = PROJ_ROOT / "data" / "processed"
BASE_CONFIG_PATH = PROJ_ROOT / "main" / "config"
logger.info(f"PROJ_ROOT path is: {BASE_CONFIG_PATH}")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove()
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
