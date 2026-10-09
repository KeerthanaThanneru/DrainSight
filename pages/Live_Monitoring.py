import plotly.express as px
import streamlit as st
from src.ui_helpers import get_sim, get_state, sidebar_status, risk_table

st.set_page_config(page_title="Live Monitoring", page_icon="📡", layout="wide")
st.title("📡 Live monitoring")
sim = get_sim()
sidebar_status(get_state())

for k, v in {"rain": 5, "playing": False, "speed": 1}.items():
    st.session_state.setdefault(k, v)

c1, c2, c3 = st.columns([3, 2, 2])
c1.slider("Rain intensity (mm/hr)", 0, 80, key="rain")
c2.select_slider("Speed (10-min steps per refresh)", [1, 2, 3, 6], key="speed")
c3.toggle("▶ Play live feed", key="playing")

b1, b2, b3 = st.columns(3)
b1.button("☔ Heavy storm (70 mm/hr)", on_click=lambda: st.session_state.update(rain=70))
b2.button("🌤 Clear skies", on_click=lambda: st.session_state.update(rain=0))
if b3.button("⏭ Step once"):
    sim.step(st.session_state.rain, st.session_state.speed)

with st.expander("🎬 Demo controls"):
    names = dict(zip(sim.drains["drain_id"], sim.drains["name"]))
    d1, d2 = st.columns([3, 1])
    pick = d1.selectbox("Drain", list(names), format_func=lambda i: f"{i} — {names[i]}", key="force_id")
    if d2.button("💥 Block it"):
        sim.force_block(pick, 0.9)
        st.toast(f"Debris jam started at {names[pick]}")


@st.fragment(run_every=2 if st.session_state.playing else None)
def live_panel():
    if st.session_state.playing:
        sim.step(st.session_state.rain, st.session_state.speed)
    risk = get_state()
    alerts = st.session_state.alerts
    new = st.session_state.get("new_alerts")
    if new is not None and len(new) and st.session_state.get("toasted") != sim.tick:
        st.toast(f"🚨 {len(new)} new alert(s): {', '.join(new['name'].head(2))}")
        st.session_state.toasted = sim.tick

    m = st.columns(4)
    m[0].metric("Sim time", f"{sim.now:%H:%M}")
    m[1].metric("Rain now", f"{sim.rain_series().iloc[-1]:.0f} mm/hr")
    m[2].metric("High / Critical", int(risk["risk_level"].isin(["High", "Critical"]).sum()))
    m[3].metric("Open alerts", int(alerts["status"].isin(["Open", "Assigned"]).sum()))

    cols = st.columns(3)
    rain = sim.rain_series().tail(36)
    f1 = px.area(x=rain.index, y=rain.values, labels={"x": "", "y": "Rain (mm/hr)"}, title="Rainfall, last 6 h")
    top = risk["drain_id"].head(4).tolist()
    h = sim.history[sim.history["drain_id"].isin(top) & (sim.history["timestamp"] > sim.now - __import__("pandas").Timedelta(hours=6))]
    f2 = px.line(h, x="timestamp", y="water_level_cm", color="drain_id", title="Water level (cm), top-risk drains")
    f3 = px.line(h, x="timestamp", y="flow_rate_lps", color="drain_id", title="Flow (L/s), same drains")
    for col, f in zip(cols, (f1, f2, f3)):
        f.update_layout(height=260, margin=dict(l=0, r=0, t=40, b=0), xaxis_title=None)
        col.plotly_chart(f, use_container_width=True)
    st.caption("Look for the signature: water level climbing while flow falls = blockage.")
    risk_table(risk, 10)


live_panel()