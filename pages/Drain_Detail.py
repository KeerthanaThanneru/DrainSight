import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from src.data_io import load_sensors
from src.ui_helpers import get_sim, get_state, sidebar_status, LEVEL_EMOJI

st.set_page_config(page_title="Drain Detail", page_icon="🔎", layout="wide")
sim = get_sim()
risk = get_state()
sidebar_status(risk)
st.title("🔎 Drain detail")

names = dict(zip(risk["drain_id"], risk["name"]))
ids = sorted(names)
default = ids.index(risk["drain_id"].iloc[0])                 # open on the riskiest drain
did = st.selectbox("Drain", ids, index=default, format_func=lambda i: f"{i} — {names[i]}")
row = risk[risk["drain_id"] == did].iloc[0]

st.subheader(f"{LEVEL_EMOJI[row['risk_level']]} {row['name']}: {row['risk_level']} ({row['flood_score']:.0f}/100)")
m = st.columns(5)
m[0].metric("Blockage index", f"{row['f_block'] * 100:.0f}/100")
m[1].metric("Sensor anomaly", f"{row['anomaly_score']:.2f}")
m[2].metric("Forecast rain", f"{row['rain_forecast']:.0f} mm/hr")
m[3].metric("Elevation", f"{row['elevation_m']:.0f} m")
m[4].metric("Criticality", f"{int(row['criticality'])}/5")

st.markdown("**Why this rating**")
for reason in row["why"].split(" • "):
    st.markdown(f"- {reason}")

h = sim.history[sim.history["drain_id"] == did].sort_values("timestamp")
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=h["timestamp"], y=h["is_blocked"] * h["water_level_cm"].max(), name="Blocked (simulator truth)",
                         fill="tozeroy", line_width=0, fillcolor="rgba(214,39,40,0.15)"))
fig.add_trace(go.Scatter(x=h["timestamp"], y=h["water_level_cm"], name="Water level (cm)", line_color="#1f77b4"))
fig.add_trace(go.Scatter(x=h["timestamp"], y=h["flow_rate_lps"], name="Flow (L/s)", line_color="#2ca02c"), secondary_y=True)
fig.add_trace(go.Bar(x=h["timestamp"], y=h["rainfall_mmhr"], name="Rain (mm/hr)", marker_color="rgba(100,100,100,0.25)"),
              secondary_y=True)
fig.update_layout(height=380, margin=dict(l=0, r=0, t=30, b=0), title="Last 48 h", legend_orientation="h")
st.plotly_chart(fig, use_container_width=True)

st.subheader("History")
full = load_sensors()
d = full[full["drain_id"] == did].sort_values("timestamp")
episodes = int((d["is_blocked"].diff() == 1).sum())
h1, h2, h3, h4 = st.columns(4)
h1.metric("Blockage episodes (10-day record)", episodes)
h2.metric("Registered past blockages", int(sim.drains.loc[sim.drains["drain_id"] == did, "past_blockages"].iloc[0]))
h3.metric("Drain age", f"{int(row['age_years'])} yrs")
h4.metric("Last cleaned", f"{row['last_cleaned']:%d %b %Y}")

a = st.session_state.alerts
mine = a[a["drain_id"] == did] if len(a) else a
st.markdown("**Alerts for this drain**")
st.dataframe(mine[["alert_id", "created_at", "priority", "risk_level", "status", "assigned_to"]]
             if len(mine) else pd.DataFrame({"info": ["No alerts yet"]}), hide_index=True)