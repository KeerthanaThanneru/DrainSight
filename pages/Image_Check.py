import numpy as np
import streamlit as st
from PIL import Image
from src.config import IMAGE_CLASSES
from src.predict import classify_image, severity_label
from src.alerts import make_manual_alert
from src.train_image import synth_image
from src.ui_helpers import get_sim, get_state, sidebar_status, load_json

st.set_page_config(page_title="Image Check", page_icon="📷", layout="wide")
st.title("📷 CCTV / image blockage check")
sim = get_sim()
risk = get_state()
sidebar_status(risk)

im = load_json("image_metrics.json")
if im and im["data_source"] == "synthetic":
    st.warning("The image model is currently trained on synthetic grate images, so real photos may be misclassified. "
               "Use the sample generator for the demo, or retrain with real images (`python -m src.train_image`).")

left, right = st.columns(2)
with left:
    up = st.file_uploader("Upload a drain / CCTV frame", type=["jpg", "jpeg", "png"])
    cls = st.selectbox("…or generate a sample grate", IMAGE_CLASSES, index=2)
    if st.button("Generate sample"):
        st.session_state.sample_img = synth_image(cls, np.random.default_rng())
    img = Image.open(up) if up else st.session_state.get("sample_img")
    if img is not None:
        st.image(img, width=360)

with right:
    if img is None:
        st.info("Upload an image or generate a sample to run the classifier.")
    else:
        try:
            label, conf, sev = classify_image(img)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        icon = {"clear": "🟢", "partial": "🟡", "blocked": "🔴"}[label]
        st.subheader(f"{icon} {label.upper()}  ({conf:.0%} confidence)")
        st.progress(int(sev), text=f"Blockage severity: {sev:.0f}/100 ({severity_label(sev)})")
        action = {"clear": "No action needed.", "partial": "Schedule cleaning within 24 h.",
                  "blocked": "Dispatch a crew now."}[label]
        st.write(f"**Recommended action:** {action}")

        if label != "clear":
            names = dict(zip(risk["drain_id"], risk["name"]))
            pick = st.selectbox("Camera location (drain)", list(names), format_func=lambda i: f"{i} — {names[i]}")
            if st.button("🎫 Create maintenance ticket"):
                drain = risk[risk["drain_id"] == pick].iloc[0]
                level = "Critical" if label == "blocked" else "Medium"
                st.session_state.alerts = make_manual_alert(
                    st.session_state.alerts, drain, level,
                    f"CCTV flagged {label} grate at {drain['name']} ({pick}), severity {sev:.0f}/100.", sim.now)
                st.success("Ticket added to the Alerts & Maintenance queue.")