"""Shared UI helpers: session state, cached risk computation, map, KPI row, tables."""
import json
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from src.config import MODELS_DIR
from src.data_io import load_sensors, load_drains
from src.live_sim import LiveSim
from src.predict import score_at
from src.risk import compute_flood_risk, simple_rain_forecast
from src.alerts import empty_alerts, generate_alerts, send_alert, save_alerts

LEVELS_ORDER = ["Critical", "High", "Medium", "Low"]
LEVEL_COLORS = {"Low": (46, 160, 67), "Medium": (235, 190, 0), "High": (245, 130, 20), "Critical": (214, 39, 40)}
LEVEL_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}


def telegram_cfg():
    try:
        return {"token": st.secrets["telegram"]["token"], "chat_id": st.secrets["telegram"]["chat_id"]}
    except Exception:
        return None                                   # no secrets -> simulated sending


def load_json(name: str):
    p = MODELS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def get_sim() -> LiveSim:
    if "sim" not in st.session_state:
        st.session_state.sim = LiveSim(load_sensors(), load_drains())
        st.session_state.alerts = empty_alerts()
        st.session_state.risk_tick = -1
    return st.session_state.sim


def get_state() -> pd.DataFrame:
    """Scored + risk-rated DataFrame for the current sim state. Recomputed only when the sim changes;
    this is also where alerts are generated (and P1 auto-sent if enabled)."""
    sim = get_sim()
    if st.session_state.risk_tick != sim.tick:
        risk = compute_flood_risk(score_at(sim.history, sim.drains), simple_rain_forecast(sim.history))
        alerts, new = generate_alerts(risk, st.session_state.alerts)
        if st.session_state.get("auto_send", False) and len(new):
            tg = telegram_cfg()
            for _, a in new[new["priority"] == "P1"].iterrows():
                alerts.loc[alerts["alert_id"] == a["alert_id"], "sent_via"] = send_alert(a, tg)
        st.session_state.update(risk=risk, alerts=alerts, new_alerts=new, risk_tick=sim.tick)
        save_alerts(alerts)
    return st.session_state.risk


def sidebar_status(risk: pd.DataFrame):
    sim = get_sim()
    sb = st.sidebar
    sb.markdown(f"**🕒 Sim time:** {sim.now:%d %b %H:%M}")
    for lvl in LEVELS_ORDER:
        sb.write(f"{LEVEL_EMOJI[lvl]} {lvl}: **{int((risk['risk_level'] == lvl).sum())}**")
    sb.divider()
    if sb.button("↺ Reset demo"):
        for k in ["sim", "alerts", "risk_tick", "risk"]:
            st.session_state.pop(k, None)
        st.rerun()


def kpis(risk: pd.DataFrame, alerts: pd.DataFrame):
    open_n = int(alerts["status"].isin(["Open", "Assigned"]).sum()) if len(alerts) else 0
    c = st.columns(5)
    c[0].metric("Drains monitored", len(risk))
    c[1].metric("Active alerts", open_n)
    c[2].metric("High / Critical", int(risk["risk_level"].isin(["High", "Critical"]).sum()))
    c[3].metric("Sensor anomalies", int(risk["is_anomaly"].sum()))
    c[4].metric("Forecast rain", f"{risk['rain_forecast'].iloc[0]:.0f} mm/hr")


def risk_map(risk: pd.DataFrame, height: int = 520):
    d = risk.copy()
    rgb = np.array(d["risk_level"].map(LEVEL_COLORS).tolist())
    d["r"], d["g"], d["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    d["radius"] = 60 + 25 * d["criticality"]
    d["score"] = d["flood_score"].round(0).astype(int)
    layer = pdk.Layer("ScatterplotLayer", data=d[["drain_id", "name", "lat", "lon", "r", "g", "b", "radius",
                                                   "risk_level", "score", "why"]],
                      get_position="[lon, lat]", get_fill_color="[r, g, b, 215]", get_radius="radius",
                      radius_min_pixels=5, radius_max_pixels=18, pickable=True,
                      stroked=True, get_line_color=[40, 40, 40], line_width_min_pixels=1)
    view = pdk.ViewState(latitude=float(d["lat"].mean()), longitude=float(d["lon"].mean()), zoom=11.3)
    tip = {"html": "<b>{name}</b> ({drain_id})<br/>{risk_level} · {score}/100<br/><i>{why}</i>",
           "style": {"backgroundColor": "#222", "color": "white", "fontSize": "12px"}}
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tip, map_style="light"), height=height)


def risk_table(risk: pd.DataFrame, n: int = 10):
    t = risk.head(n).copy()
    t["Level"] = t["risk_level"].map(lambda l: f"{LEVEL_EMOJI[l]} {l}")
    t = t[["drain_id", "name", "Level", "flood_score", "why"]].rename(
        columns={"drain_id": "ID", "name": "Drain", "flood_score": "Score", "why": "Why"})
    st.dataframe(t, hide_index=True, column_config={
        "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f")})