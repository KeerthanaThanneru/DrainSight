"""Trains RandomForest blockage-risk model + IsolationForest/residual anomaly detector.
Run: python -m src.train_risk"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import (precision_score, recall_score, f1_score, fbeta_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix, precision_recall_curve)
from src.config import *
from src.data_io import load_drains, load_sensors
from src.features import build_features, residual_X
from src.predict import anomaly_scores


def add_target(df, horizon=HORIZON_STEPS):
    """y=1 if the drain is blocked at any point in [t, t+horizon]."""
    df = df.sort_values(["drain_id", "timestamp"]).copy()
    fwd_max = lambda s: pd.Series(
        pd.Series(s.to_numpy()[::-1]).rolling(horizon + 1, min_periods=1).max().to_numpy()[::-1],
        index=s.index)
    df["y"] = df.groupby("drain_id")["is_blocked"].transform(fwd_max).astype(int)
    return df


def evaluate(y, p, thr):
    pred = (p >= thr).astype(int)
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if y.nunique() > 1 else None,
        "pr_auc": float(average_precision_score(y, p)) if y.sum() > 0 else None,
        "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    df = add_target(build_features(load_sensors(), load_drains()))

    # time-based split 60 / 20 / 20  (train / validation for threshold / test)
    ts = np.sort(df.timestamp.unique())
    t1, t2 = ts[int(len(ts) * 0.6)], ts[int(len(ts) * 0.8)]
    train, val, test = df[df.timestamp < t1], df[(df.timestamp >= t1) & (df.timestamp < t2)], df[df.timestamp >= t2]
    print(f"positive rate  train={train.y.mean():.1%}  val={val.y.mean():.1%}  test={test.y.mean():.1%}")

    # ---- a) blockage-risk model ----
    rf = RandomForestClassifier(n_estimators=120, max_depth=10, min_samples_leaf=20,
                                class_weight="balanced_subsample", n_jobs=-1, random_state=SEED)
    rf.fit(train[FEATURE_COLS], train.y)

    # false negatives are costly -> choose the threshold that maximises F2 (recall-weighted) on validation
    p_val = rf.predict_proba(val[FEATURE_COLS])[:, 1]
    thr = 0.5
    if val.y.sum() > 0:
        prec, rec, thrs = precision_recall_curve(val.y, p_val)
        f2 = 5 * prec * rec / (4 * prec + rec + 1e-9)
        thr = float(thrs[np.argmax(f2[:-1])])

    p_test = rf.predict_proba(test[FEATURE_COLS])[:, 1]
    m_tuned, m_default = evaluate(test.y, p_test, thr), evaluate(test.y, p_test, 0.5)

    X = train[FEATURE_COLS]
    risk_bundle = {
        "model": rf, "threshold": thr,
        "importance": dict(zip(FEATURE_COLS, map(float, rf.feature_importances_))),
        "stats": {"mean": X.mean().to_dict(), "std": X.std().replace(0, 1).to_dict()},
        "direction": np.sign(X.corrwith(train.y)).fillna(0).to_dict(),   # + means higher value -> more risk
    }
    joblib.dump(risk_bundle, MODELS_DIR / "risk_rf.joblib", compress=3)

    # ---- b) anomaly detector: trained on HEALTHY rows only ----
    healthy = train[train.y == 0]
    iso = IsolationForest(n_estimators=100, max_samples=2048, contamination=0.01, random_state=SEED)
    iso.fit(healthy[ANOMALY_COLS])
    ridge = Ridge(alpha=1.0).fit(residual_X(healthy), healthy.water_level_cm)
    resid_std = float(np.std(healthy.water_level_cm - ridge.predict(residual_X(healthy))))
    d = iso.decision_function(healthy[ANOMALY_COLS])
    ref = np.sort(np.random.default_rng(SEED).choice(d, size=min(5000, len(d)), replace=False))
    anom_bundle = {"iso": iso, "ridge": ridge, "resid_std": resid_std, "ref": ref}
    joblib.dump(anom_bundle, MODELS_DIR / "anomaly_iso.joblib", compress=3)

    a_score, _ = anomaly_scores(test, anom_bundle)
    anom_auc = float(roc_auc_score(test.is_blocked, a_score)) if test.is_blocked.nunique() > 1 else None

    report = {"threshold": thr, "test_at_tuned_threshold": m_tuned, "test_at_0.5": m_default,
              "anomaly_auc_vs_current_blockage": anom_auc,
              "top_features": sorted(risk_bundle["importance"].items(), key=lambda kv: -kv[1])[:8]}
    (MODELS_DIR / "metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()