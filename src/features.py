"""build_features(sensors, drains) -> DataFrame with FEATURE_COLS (+ metadata columns).
Uses only PAST readings per drain (rolling windows look backwards), so no leakage."""
import numpy as np
import pandas as pd
from src.config import *

META = ["drain_id", "name", "lat", "lon", "location_type", "elevation_m", "criticality"]


def build_features(sensors: pd.DataFrame, drains: pd.DataFrame) -> pd.DataFrame:
    df = sensors.sort_values(["drain_id", "timestamp"]).copy()
    df[BASE_COLS] = df.groupby("drain_id")[BASE_COLS].transform(lambda s: s.ffill().bfill())
    df[BASE_COLS] = df[BASE_COLS].fillna(0)                           # sensor dropout handling
    g = df.groupby("drain_id")
    h = STEPS_PER_HOUR
    roll = lambda col, n: g[col].transform(lambda s: s.rolling(n, min_periods=1).mean())
    df["level_mean_1h"] = roll("water_level_cm", h)
    df["level_trend_1h"] = g["water_level_cm"].transform(lambda s: s.diff(h)).fillna(0)
    df["flow_mean_1h"] = roll("flow_rate_lps", h)
    df["flow_trend_1h"] = g["flow_rate_lps"].transform(lambda s: s.diff(h)).fillna(0)
    df["turb_mean_1h"] = roll("turbidity_ntu", h)
    df["rain_mean_3h"] = roll("rainfall_mmhr", 3 * h)
    df["level_flow_ratio"] = df["water_level_cm"] / (df["flow_rate_lps"] + 1)   # high = water backing up

    d = drains[META + ["age_years", "past_blockages", "last_cleaned"]]
    df = df.merge(d, on="drain_id", how="left")
    df["days_since_cleaned"] = (df["timestamp"] - df["last_cleaned"]).dt.total_seconds() / 86400
    for t in LOCATION_TYPES:
        df[f"loc_{t}"] = (df["location_type"] == t).astype(int)
    return df.reset_index(drop=True)


def residual_X(df: pd.DataFrame) -> np.ndarray:
    """Inputs for 'expected water level given flow and rain' (flow/level mismatch detector)."""
    f = df["flow_rate_lps"].clip(lower=0)
    return np.column_stack([f, np.sqrt(f), df["rain_mean_3h"]])