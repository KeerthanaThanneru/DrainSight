import plotly.express as px
import streamlit as st
from src.ui_helpers import (get_sim, get_state, sidebar_status, kpis, risk_map, risk_table,
                            load_json, LEVEL_COLORS, LEVELS_ORDER)

st.set_page_config(page_title="DrainGuard AI", page_icon="🌧️", layout="wide")
st.title("🌧️ DrainGuard AI")
st.caption("Early drain-blockage detection and flood warning: simulated IoT + rainfall + AI risk scoring")

get_sim()
risk = get_state()
sidebar_status(risk)
kpis(risk, st.session_state.alerts)

left, right = st.columns([3, 2])
with left:
    st.subheader("City risk map")
    risk_map(risk)
with right:
    st.subheader("Risk distribution")
    counts = risk["risk_level"].value_counts().reindex(LEVELS_ORDER[::-1]).fillna(0).reset_index()
    counts.columns = ["Level", "Drains"]
    fig = px.bar(counts, x="Level", y="Drains", color="Level",
                 color_discrete_map={k: f"rgb{v}" for k, v in LEVEL_COLORS.items()})
    fig.update_layout(showlegend=False, height=220, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Top 8 at-risk drains")
    risk_table(risk, 8)

with st.expander("📊 Model performance (held-out test period)"):
    m, im = load_json("metrics.json"), load_json("image_metrics.json")
    if m:
        t = m["test_at_tuned_threshold"]
        c = st.columns(5)
        c[0].metric("Recall", f"{t['recall']:.2f}")
        c[1].metric("Precision", f"{t['precision']:.2f}")
        c[2].metric("F1", f"{t['f1']:.2f}")
        c[3].metric("PR-AUC", f"{t['pr_auc']:.2f}" if t["pr_auc"] is not None else "n/a")
        c[4].metric("Anomaly AUC", f"{m['anomaly_auc_vs_current_blockage']:.2f}"
                    if m["anomaly_auc_vs_current_blockage"] is not None else "n/a")
        st.caption("Target: a blockage within the next hour. Threshold tuned for recall (missed blockages are costly). "
                   "Time-based split, so no future leakage. Evaluated on simulated data.")
    if im:
        st.write(f"Image classifier accuracy: **{im['accuracy']:.2f}** (data source: `{im['data_source']}`)")
        if im["data_source"] == "synthetic":
            st.warning("Image model is trained on synthetic grate images: it proves the pipeline, not real-world accuracy.")