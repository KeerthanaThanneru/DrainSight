"""Rule-based alerts, task queue, notifications. No Streamlit imports (testable)."""
import numpy as np
import pandas as pd
import requests
from src.config import DATA_DIR
from src.risk import LEVEL_ORDER

ALERT_COLS = ["alert_id", "created_at", "drain_id", "name", "location_type", "risk_level", "priority",
              "flood_score", "message", "action", "status", "assigned_to", "resolved_at", "sent_via", "source"]
CREWS = ["Crew A (North)", "Crew B (South)", "Crew C (East)", "Crew D (West)", "Rapid Response"]
RULES = {   # level -> (priority, recommended action)
    "Critical": ("P1", "Dispatch crew immediately (within 1 hour), clear the blockage, consider traffic diversion."),
    "High": ("P2", "Schedule a crew within 4 hours; pre-position a pump if heavy rain is forecast."),
    "Medium": ("P3", "Add to the next inspection round (24 h); visually check the grate and sensor."),
}


def empty_alerts() -> pd.DataFrame:
    return pd.DataFrame(columns=ALERT_COLS)


def needs_alert(row) -> bool:
    """Critical/High always alert. Medium alerts only if the model itself predicts a blockage."""
    lvl = row["risk_level"]
    return lvl in ("Critical", "High") or (lvl == "Medium" and bool(row.get("pred_block", 0)))


def build_message(row, priority: str, action: str) -> str:
    return (f"[{priority}] {row['risk_level'].upper()} flood risk at {row['name']} "
            f"({row['drain_id']}, {str(row['location_type']).replace('_', ' ')}). "
            f"Score {row['flood_score']:.0f}/100. Why: {row['why']}. Action: {action}")


def generate_alerts(risk_df: pd.DataFrame, existing: pd.DataFrame | None = None, now=None):
    """Returns (all_alerts, new_alerts). One open alert per drain: a worse level escalates it in place."""
    alerts = empty_alerts() if existing is None else existing.copy()
    now = pd.Timestamp(risk_df["timestamp"].max() if now is None else now)   # simulation time
    new_rows = []
    for _, r in risk_df.iterrows():
        if not needs_alert(r):
            continue
        pr, action = RULES[r["risk_level"]]
        msg = build_message(r, pr, action)
        open_here = alerts[alerts["status"].isin(["Open", "Assigned"]) & (alerts["drain_id"] == r["drain_id"])]
        if len(open_here):
            i = open_here.index[0]
            if LEVEL_ORDER[r["risk_level"]] > LEVEL_ORDER[alerts.at[i, "risk_level"]]:      # escalation
                for col, val in [("risk_level", r["risk_level"]), ("priority", pr), ("flood_score", round(r["flood_score"], 1)),
                                 ("message", msg), ("action", action)]:
                    alerts.at[i, col] = val
            continue
        new_rows.append({
            "alert_id": f"A{len(alerts) + len(new_rows) + 1:04d}", "created_at": now, "drain_id": r["drain_id"],
            "name": r["name"], "location_type": r["location_type"], "risk_level": r["risk_level"], "priority": pr,
            "flood_score": round(r["flood_score"], 1), "message": msg, "action": action, "status": "Open",
            "assigned_to": "", "resolved_at": "", "sent_via": "", "source": "AI model"})
    new = pd.DataFrame(new_rows, columns=ALERT_COLS)
    if len(new):
        alerts = pd.concat([alerts, new], ignore_index=True)
    alerts["flood_score"] = pd.to_numeric(alerts["flood_score"])
    return alerts, new


def make_manual_alert(alerts, drain, risk_level: str, message: str, now, source="CCTV image"):
    """Ticket raised by a person or by the image classifier. `drain` is a row of the risk DataFrame."""
    pr, action = RULES[risk_level]
    row = {"alert_id": f"A{len(alerts) + 1:04d}", "created_at": pd.Timestamp(now), "drain_id": drain["drain_id"],
           "name": drain["name"], "location_type": drain["location_type"], "risk_level": risk_level, "priority": pr,
           "flood_score": round(float(drain["flood_score"]), 1), "message": f"[{pr}] {message}", "action": action,
           "status": "Open", "assigned_to": "", "resolved_at": "", "sent_via": "", "source": source}
    return pd.concat([alerts, pd.DataFrame([row], columns=ALERT_COLS)], ignore_index=True)


def assign_alert(alerts, alert_id, crew):
    out = alerts.copy()
    m = out["alert_id"] == alert_id
    out.loc[m, ["status", "assigned_to"]] = ["Assigned", crew]
    return out


def resolve_alert(alerts, alert_id, now):
    out = alerts.copy()
    m = out["alert_id"] == alert_id
    out.loc[m, ["status", "resolved_at"]] = ["Resolved", str(now)]
    return out


def send_alert(alert, telegram_cfg: dict | None = None, timeout: int = 8) -> str:
    """Telegram if configured and reachable, otherwise 'simulated' (always works for the demo).
    Returns the channel used."""
    if telegram_cfg and telegram_cfg.get("token") and telegram_cfg.get("chat_id"):
        try:
            r = requests.post(f"https://api.telegram.org/bot{telegram_cfg['token']}/sendMessage",
                              json={"chat_id": telegram_cfg["chat_id"], "text": alert["message"]}, timeout=timeout)
            if r.ok:
                return "telegram"
        except requests.RequestException:
            pass
    return "simulated"


def alerts_to_csv(alerts: pd.DataFrame) -> bytes:
    return alerts.to_csv(index=False).encode("utf-8")


def save_alerts(alerts: pd.DataFrame):
    """Best-effort audit trail. Streamlit Cloud's disk is ephemeral, so the in-app CSV export is the real record."""
    try:
        alerts.to_csv(DATA_DIR / "alerts_log.csv", index=False)
    except OSError:
        pass