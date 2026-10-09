"""Inference helpers. No Streamlit imports here, so everything is unit-testable.
In the app, wrap calls with @st.cache_data if they get slow."""
from functools import lru_cache
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from src.config import *
from src.features import build_features, residual_X
from src.image_utils import extract_features


@lru_cache(maxsize=1)
def load_risk():
    return joblib.load(MODELS_DIR / "risk_rf.joblib")


@lru_cache(maxsize=1)
def load_anomaly():
    return joblib.load(MODELS_DIR / "anomaly_iso.joblib")


@lru_cache(maxsize=1)
def load_image_model():
    path = MODELS_DIR / "image_clf.joblib"
    return joblib.load(path) if path.exists() else None


def anomaly_scores(df: pd.DataFrame, bundle) -> tuple[np.ndarray, np.ndarray]:
    """0-1 anomaly score = max(IsolationForest rarity, flow-vs-level residual).
    Residual z>2 means water level is far above what the flow explains (backing up)."""
    d = bundle["iso"].decision_function(df[ANOMALY_COLS])
    p = np.searchsorted(bundle["ref"], d) / len(bundle["ref"])       # share of healthy rows that look MORE anomalous
    iso_score = np.clip(1 - p / 0.05, 0, 1)                          # bottom 5% of healthy -> flagged
    z = (df["water_level_cm"] - bundle["ridge"].predict(residual_X(df))) / bundle["resid_std"]
    res_score = np.clip((z.to_numpy() - 2) / 4, 0, 1)
    return np.maximum(iso_score, res_score), z.to_numpy()


def predict_risk(feat: pd.DataFrame) -> pd.DataFrame:
    """feat = output of build_features. Adds p_raw, p_block, pred_block, anomaly_score, residual_z, is_anomaly.
    p_block is threshold-normalised: 0.5 == the tuned decision boundary. (The RF is trained with
    class weights, so raw probabilities are inflated and not literal chances.)"""
    risk = load_risk()
    out = feat.copy()
    raw = risk["model"].predict_proba(out[FEATURE_COLS])[:, 1]
    thr = risk["threshold"]
    out["p_raw"] = raw
    out["p_block"] = np.where(raw < thr, 0.5 * raw / thr, 0.5 + 0.5 * (raw - thr) / (1 - thr + 1e-9))
    out["pred_block"] = (raw >= thr).astype(int)
    out["anomaly_score"], out["residual_z"] = anomaly_scores(out, load_anomaly())
    out["is_anomaly"] = out["anomaly_score"] >= 0.5
    return out


def score_at(sensors: pd.DataFrame, drains: pd.DataFrame, as_of=None, window_h: int = 6) -> pd.DataFrame:
    """One scored row per drain at time `as_of` (default: latest). Fast: only uses a short window."""
    as_of = pd.Timestamp(sensors.timestamp.max() if as_of is None else as_of)
    win = sensors[(sensors.timestamp <= as_of) & (sensors.timestamp > as_of - pd.Timedelta(hours=window_h))]
    feat = build_features(win, drains)
    last = feat.sort_values("timestamp").groupby("drain_id").tail(1)
    return predict_risk(last).reset_index(drop=True)


def explain_row(row, k: int = 3):
    """SHAP-lite: contribution = importance x direction x z-score of the feature.
    Returns up to k human-readable drivers that push risk UP."""
    r = load_risk()
    out = []
    for c in FEATURE_COLS:
        if c not in FRIENDLY:
            continue
        z = np.clip((row[c] - r["stats"]["mean"][c]) / r["stats"]["std"][c], -4, 4)
        score = r["importance"][c] * r["direction"][c] * z
        if score > 0:
            label, unit = FRIENDLY[c]
            out.append((score, f"{label}: {row[c]:.1f} {unit} (typical {r['stats']['mean'][c]:.1f})"))
    return [txt for _, txt in sorted(out, reverse=True)[:k]]


def severity_label(pct: float) -> str:
    return "Low" if pct < 25 else "Moderate" if pct < 60 else "Severe"


def classify_image(img: Image.Image):
    """-> (label, confidence 0-1, severity_pct 0-100). severity = 50*P(partial) + 100*P(blocked)."""
    m = load_image_model()
    if m is None:
        raise FileNotFoundError("models/image_clf.joblib missing: run `python -m src.train_image`")
    proba = m.predict_proba(extract_features(img).reshape(1, -1))[0]
    p = dict(zip(m.classes_, proba))
    label = max(p, key=p.get)
    return label, float(p[label]), float(100 * (0.5 * p.get("partial", 0) + p.get("blocked", 0)))