import pandas as pd
from functools import lru_cache
from src.config import DATA_DIR


@lru_cache(maxsize=1)
def load_drains() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "drains.csv", parse_dates=["last_cleaned"])


@lru_cache(maxsize=1)
def load_sensors() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sensors.csv", parse_dates=["timestamp"])

# NOTE: cached objects are shared. Always .copy() before modifying.