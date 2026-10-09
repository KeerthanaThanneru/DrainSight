"""Flood-risk score (0-100) = blockage index x rain factor x exposure factor, with plain-language reasons."""
import numpy as np
import pandas as pd
from src.predict import explain_row

LEVELS = [(70, "Critical"), (45, "High"), (20, "Medium"), (0, "Low")]
LEVEL_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
RAIN_SAT = 50.0            # mm/hr at which the rain factor saturates at 1.0


def level_for(score: float) -> str:
    for thr, name in LEVELS:
        if score >= thr:
            return name
    return "Low"


def simple_rain_forecast(sensors: pd.DataFrame, as_of=None) -> float:
    """Next-3h rain estimate (mm/hr): last 30 min average plus its trend vs the 30 min before.
    Replace with Open-Meteo (NICE) by returning its hourly precipitation instead."""
    rain = sensors.groupby("timestamp")["rainfall_mmhr"].first().sort_index()
    if as_of is not None:
        rain = rain[rain.index <= pd.Timestamp(as_of)]
    last = rain.iloc[-3:].mean()
    prev = rain.iloc[-6:-3].mean() if len(rain) >= 6 else last
    return float(max(0.0, last + (last - prev)))


def _why(r, fc: float) -> str:
    out = []
    if r["f_block"] >= 0.5:
        out.append(f"Blockage index {r['f_block'] * 100:.0f}/100")
    if r["f_block"] >= 0.35:
        out += explain_row(r, k=3)                                  # model drivers (SHAP-lite)
    if r["is_anomaly"]:
        out.append(f"Water level {r['residual_z']:.1f}σ above what flow explains (backing up)")
    if fc >= 10:
        out.append(f"Forecast rain {fc:.0f} mm/hr")
    if r["elevation_m"] < 8:
        out.append(f"Low-lying site ({r['elevation_m']:.0f} m)")
    if r["criticality"] >= 4:
        out.append(f"High-criticality area ({str(r['location_type']).replace('_', ' ')})")
    return " • ".join(out) if out else "Readings normal"


def compute_flood_risk(scored: pd.DataFrame, rain_forecast: float) -> pd.DataFrame:
    """scored = output of predict.score_at(). Returns a copy sorted by flood_score (desc) with:
    f_block, f_rain, f_exposure, flood_score, risk_level, rain_forecast, why."""
    df = scored.copy()
    df["f_block"] = np.maximum(df["p_block"], 0.7 * df["anomaly_score"])      # model OR sensor-mismatch evidence
    f_rain = float(np.clip(rain_forecast / RAIN_SAT, 0, 1))
    low_lying = np.clip(1 - (df["elevation_m"] - 2) / 20, 0, 1)
    crit = (df["criticality"] - 1) / 4
    df["f_rain"] = f_rain
    df["f_exposure"] = 0.5 * low_lying + 0.5 * crit
    df["flood_score"] = (100 * df["f_block"] * (0.4 + 0.6 * f_rain) * (0.6 + 0.4 * df["f_exposure"])).clip(0, 100)
    df["risk_level"] = df["flood_score"].map(level_for)
    df["rain_forecast"] = rain_forecast
    df["why"] = df.apply(lambda r: _why(r, rain_forecast), axis=1)
    return df.sort_values("flood_score", ascending=False).reset_index(drop=True)