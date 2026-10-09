import pandas as pd
import plotly.express as px
import streamlit as st
from src.config import STEPS_PER_HOUR
from src.predict import score_at
from src.risk import compute_flood_risk, simple_rain_forecast
from src.ui_helpers import get_sim, get_state, sidebar_status, LEVELS_ORDER, LEVEL_COLORS, LEVEL_EMOJI

st.set_page_config(page_title="What-if Simulator", page_icon="🧪", layout="wide")
sim = get_sim()
now_risk = get_state()
sidebar_status(now_risk)
st.title("🧪 What-if simulator")
st.caption("Fork the live simulation, let a storm fall for N hours, and compare. Heavier rain also washes in new debris.")


@st.cache_data(show_spinner=False)
def run_scenario(tick: int, rain: float, hours: int, _sim) -> pd.DataFrame:
    f = _sim.fork()
    f.step(rain, hours * STEPS_PER_HOUR)
    return compute_flood_risk(score_at(f.history, f.drains), simple_rain_forecast(f.history))


c1, c2 = st.columns(2)
rain = c1.slider("Storm intensity (mm/hr)", 0, 100, 40, step=5)
hours = c2.slider("Duration (hours)", 1, 6, 3)
scen = run_scenario(sim.tick, rain, hours, sim)

hi = lambda d: int(d["risk_level"].isin(["High", "Critical"]).sum())
m = st.columns(3)
m[0].metric("High/Critical now", hi(now_risk))
m[1].metric(f"High/Critical after {hours} h at {rain} mm/hr", hi(scen), delta=hi(scen) - hi(now_risk), delta_color="inverse")
m[2].metric("Average flood score", f"{scen['flood_score'].mean():.1f}", delta=f"{scen['flood_score'].mean() - now_risk['flood_score'].mean():+.1f}",
            delta_color="inverse")

dist = pd.concat([
    now_risk["risk_level"].value_counts().reindex(LEVELS_ORDER[::-1]).fillna(0).rename("Now"),
    scen["risk_level"].value_counts().reindex(LEVELS_ORDER[::-1]).fillna(0).rename("Scenario")], axis=1)
dist = dist.reset_index().melt(id_vars="risk_level", var_name="When", value_name="Drains")
fig = px.bar(dist, x="risk_level", y="Drains", color="When", barmode="group", labels={"risk_level": "Level"})
fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Drains whose risk rises most")
cmp = now_risk[["drain_id", "name", "flood_score", "risk_level"]].merge(
    scen[["drain_id", "flood_score", "risk_level", "why"]], on="drain_id", suffixes=("_now", "_scen"))
cmp["Change"] = cmp["flood_score_scen"] - cmp["flood_score_now"]
top = cmp.sort_values("Change", ascending=False).head(10)
top["Now → Scenario"] = top.apply(lambda r: f"{LEVEL_EMOJI[r['risk_level_now']]} → {LEVEL_EMOJI[r['risk_level_scen']]}", axis=1)
st.dataframe(top[["drain_id", "name", "Now → Scenario", "flood_score_now", "flood_score_scen", "Change", "why"]]
             .rename(columns={"flood_score_now": "Score now", "flood_score_scen": "Score scenario", "why": "Why (scenario)"}),
             hide_index=True)

if st.button("📈 Sweep intensity 0–80 mm/hr"):
    rows, bar = [], st.progress(0)
    for i, r in enumerate(range(0, 81, 10)):
        s = run_scenario(sim.tick, r, hours, sim)
        rows.append({"Rain (mm/hr)": r, "High/Critical drains": hi(s), "Mean score": s["flood_score"].mean()})
        bar.progress((i + 1) / 9)
    sw = pd.DataFrame(rows)
    st.plotly_chart(px.line(sw, x="Rain (mm/hr)", y="High/Critical drains", markers=True,
                            title=f"High/Critical drains after {hours} h of rain"), use_container_width=True)