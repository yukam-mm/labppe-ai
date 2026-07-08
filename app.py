"""
LabPPE AI — Streamlit dashboard.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src import compliance, config, report, voice
from src.detector import PPEDetector

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="LabPPE AI — PPE Compliance Checker",
    page_icon="🧪",
    layout="wide",
)

CUSTOM_CSS = f"""
<style>
  .stApp {{ background: #0e1117; }}
  .banner {{
    padding: 1rem 1.25rem; border-radius: 0.6rem; font-size: 1.6rem;
    font-weight: 700; text-align: center; margin: 0.5rem 0 1rem 0;
  }}
  .granted {{ background: {config.COLOR_PRESENT_HEX}22; color: {config.COLOR_PRESENT_HEX};
              border: 1px solid {config.COLOR_PRESENT_HEX}; }}
  .denied  {{ background: {config.COLOR_MISSING_HEX}22; color: {config.COLOR_MISSING_HEX};
              border: 1px solid {config.COLOR_MISSING_HEX}; }}
  .item {{ padding: 0.5rem 0.75rem; border-radius: 0.4rem; margin-bottom: 0.4rem;
           font-size: 1.05rem; font-weight: 600; }}
  .item-present {{ background: {config.COLOR_PRESENT_HEX}1f; color: {config.COLOR_PRESENT_HEX}; }}
  .item-missing {{ background: {config.COLOR_MISSING_HEX}1f; color: {config.COLOR_MISSING_HEX}; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Model (cached across reruns)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading YOLOv8 model …")
def load_detector() -> PPEDetector:
    config.ensure_dirs()
    return PPEDetector()


detector = load_detector()


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🧪 LabPPE AI")
    st.caption("Biotech laboratory PPE compliance gate")

    mode = st.radio("Input mode", ["📁 Upload image", "📷 Webcam"], index=0)

    config.CONFIDENCE_THRESHOLD = st.slider(
        "Detection confidence", 0.10, 0.90, config.CONFIDENCE_THRESHOLD, 0.05
    )
    enable_voice = st.checkbox("🔊 Announce decision", value=True)

    st.divider()
    st.subheader("Required PPE")
    for name in config.PPE_REQUIREMENTS:
        st.write(f"• {name}")

    if detector.using_fallback:
        st.warning(
            "No trained weights found — running the generic COCO model, "
            "which cannot detect PPE. Run `python train.py` first."
        )


# --------------------------------------------------------------------------- #
# Input acquisition
# --------------------------------------------------------------------------- #
st.header("Laboratory Entry — PPE Check")

image_rgb: np.ndarray | None = None
source = "upload"

if mode == "📁 Upload image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        image_rgb = np.array(Image.open(uploaded).convert("RGB"))
else:
    source = "webcam"
    snapshot = st.camera_input("Capture from webcam")
    if snapshot is not None:
        image_rgb = np.array(Image.open(snapshot).convert("RGB"))


# --------------------------------------------------------------------------- #
# Detection + reporting
# --------------------------------------------------------------------------- #
if image_rgb is None:
    st.info("Upload or capture an image to run a compliance check.")
    st.stop()

with st.spinner("Detecting PPE …"):
    detections = detector.detect(image_rgb)
result = compliance.evaluate(detections)
annotated = detector.draw(image_rgb, detections)
record = report.build_record(result, detections, source)

# --- images side by side ---
left, right = st.columns(2)
with left:
    st.subheader("Original")
    st.image(image_rgb, use_container_width=True)
with right:
    st.subheader("Detections")
    st.image(annotated, use_container_width=True)

# --- access banner ---
if result.access_granted:
    st.markdown('<div class="banner granted">🟢 ACCESS GRANTED</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="banner denied">🔴 ACCESS DENIED</div>', unsafe_allow_html=True)

# --- checklist + score ---
col_check, col_score = st.columns([2, 1])
with col_check:
    st.subheader("PPE Checklist")
    for item in result.checklist:
        css = "item-present" if item.present else "item-missing"
        mark = "✓" if item.present else "✗"
        conf = f" — {item.confidence:.0%}" if item.confidence else ""
        st.markdown(
            f'<div class="item {css}">{mark} {item.name}{conf}</div>',
            unsafe_allow_html=True,
        )

with col_score:
    st.subheader("Compliance")
    st.metric("Score", f"{result.compliance_pct:.0f}%")
    st.progress(result.compliance_pct / 100.0)
    if result.missing:
        st.error("Missing: " + ", ".join(result.missing))
    else:
        st.success("All required PPE detected.")

# --- voice announcement ---
announcement = voice.build_announcement(result)
if enable_voice:
    components.html(voice.speak_html(announcement), height=40)
st.caption(f"🔊 {announcement}")

# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
st.divider()
st.subheader("Laboratory Report")

meta = st.columns(3)
meta[0].write(f"**Report ID:** `{record['report_id']}`")
meta[1].write(f"**Timestamp:** {record['timestamp']}")
meta[2].write(f"**Source:** {record['source']}")

st.dataframe(report.checklist_dataframe(result), use_container_width=True, hide_index=True)

md = report.to_markdown(record)
with st.expander("Preview full report"):
    st.markdown(md)

dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.download_button(
        "⬇ Download report (.md)",
        md,
        file_name=f"labppe_report_{record['report_id']}.md",
        mime="text/markdown",
    )
with dl2:
    csv = report.checklist_dataframe(result).to_csv(index=False).encode()
    st.download_button(
        "⬇ Download checklist (.csv)",
        csv,
        file_name=f"labppe_checklist_{record['report_id']}.csv",
        mime="text/csv",
    )
with dl3:
    if st.button("💾 Save to reports/ folder"):
        img_path, md_path = report.save_artifacts(record, annotated)
        st.success(f"Saved:\n- {img_path.name}\n- {md_path.name}")
