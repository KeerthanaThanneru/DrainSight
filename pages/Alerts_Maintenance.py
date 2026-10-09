import streamlit as st
from src.alerts import assign_alert, resolve_alert, send_alert, alerts_to_csv, CREWS
from src.ui_helpers import get_sim, get_state, sidebar_status, telegram_cfg

st.set_page_config(page_title="Alerts & Maintenance", page_icon="🚨", layout="wide")
sim = get_sim()
risk = get_state()
sidebar_status(risk)
st.title("🚨 Alerts & maintenance")

flash = st.session_state.pop("flash", None)
if flash:
    st.success(flash)

st.toggle("Auto-send P1 alerts (Telegram if configured, otherwise simulated)", key="auto_send")
alerts = st.session_state.alerts
active = alerts[alerts["status"].isin(["Open", "Assigned"])].sort_values(["priority", "flood_score"], ascending=[True, False])

m = st.columns(4)
m[0].metric("Open + assigned", len(active))
m[1].metric("P1 critical", int((active["priority"] == "P1").sum()))
m[2].metric("Assigned", int((alerts["status"] == "Assigned").sum()))
m[3].metric("Resolved", int((alerts["status"] == "Resolved").sum()))

tab_q, tab_log = st.tabs(["Task queue", "Full log & export"])

with tab_q:
    if active.empty:
        st.success("No open alerts. Try a heavy storm on the Live Monitoring page.")
    else:
        st.dataframe(active[["alert_id", "priority", "risk_level", "name", "flood_score", "status", "assigned_to",
                             "created_at", "sent_via", "source"]], hide_index=True,
                     column_config={"flood_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f")})
        sel = st.selectbox("Select alert", active["alert_id"].tolist(),
                           format_func=lambda a: f"{a} — {active.loc[active['alert_id'] == a, 'name'].iloc[0]}")
        a = active[active["alert_id"] == sel].iloc[0]
        st.info(a["message"])
        crew = st.selectbox("Assign to", CREWS)
        b1, b2, b3 = st.columns(3)
        if b1.button("👷 Assign crew"):
            st.session_state.alerts = assign_alert(alerts, sel, crew)
            st.session_state.flash = f"{sel} assigned to {crew}"
            st.rerun()
        if b2.button("📨 Send notification"):
            ch = send_alert(a, telegram_cfg())
            alerts.loc[alerts["alert_id"] == sel, "sent_via"] = ch
            st.session_state.flash = f"{sel} sent via {ch}" + (" (no Telegram secrets, simulated)" if ch == "simulated" else "")
            st.rerun()
        if b3.button("✅ Mark resolved (drain cleaned)"):
            st.session_state.alerts = resolve_alert(alerts, sel, sim.now)
            sim.clean_drain(a["drain_id"])                  # closes the loop: sensors recover, risk drops
            st.session_state.flash = f"{sel} resolved. Drain {a['drain_id']} cleaned in the simulation."
            st.rerun()

with tab_log:
    st.dataframe(alerts, hide_index=True)
    st.download_button("⬇ Export alerts CSV", alerts_to_csv(alerts), "alerts_log.csv", "text/csv", disabled=alerts.empty)