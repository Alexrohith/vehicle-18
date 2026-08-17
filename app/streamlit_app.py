import sys
import uuid
from pathlib import Path
import tempfile
from datetime import datetime

import streamlit as st
from PIL import Image

# ============================================================
# PROJECT PATH
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.explainability.gradcam import generate_gradcam

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="ClaimShield — Vehicle Claim Assessment",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DESIGN TOKENS  (from provided palette)
# ============================================================
PRIMARY = "#0F172A"      # deep navy — sidebar, headings
SECONDARY = "#64748B"    # slate — secondary text
TERTIARY = "#231500"     # near-black brown — used sparingly for emphasis
NEUTRAL = "#78777B"      # neutral gray — captions / muted labels
SURFACE = "#F5F6F8"      # app background
CARD = "#FFFFFF"
BORDER = "#E6E8EC"
DANGER = "#D32F2F"
DANGER_BG = "#FDECEC"
SAFE = "#2E7D32"
SAFE_BG = "#EAF6EA"

if "history" not in st.session_state:
    st.session_state.history = []
if "claim_id" not in st.session_state:
    st.session_state.claim_id = f"CLM-{datetime.now().year}-{str(uuid.uuid4().int)[:5]}"

# ============================================================
# GLOBAL CSS
# ============================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    h1, h2, h3, .headline {{
        font-family: 'IBM Plex Sans', sans-serif !important;
    }}

    .stApp {{
        background-color: {SURFACE};
    }}
    #MainMenu, footer, header {{visibility: hidden;}}

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background-color: {PRIMARY};
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.6rem;
    }}
    .brand {{
        color: #FFFFFF;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 700;
        font-size: 20px;
        margin-bottom: 2px;
    }}
    .brand-sub {{
        color: {SECONDARY};
        font-size: 12.5px;
        margin-bottom: 26px;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        color: #C9CDD6;
        font-size: 14.5px;
        padding: 9px 12px;
        border-radius: 8px;
        width: 100%;
        margin-bottom: 2px;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background-color: rgba(255,255,255,0.06);
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] input:checked + div {{
        color: #FFFFFF !important;
        font-weight: 600;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {{
        background-color: rgba(255,255,255,0.10);
        border-left: 3px solid #4C8DFF;
    }}
    .sidebar-footer {{
        position: fixed;
        bottom: 22px;
        color: {SECONDARY};
        font-size: 12.5px;
        line-height: 1.9;
    }}
    .dot {{
        height: 7px; width: 7px; border-radius: 50%;
        background-color: #34C759; display: inline-block; margin-right: 6px;
    }}

    /* ---------- Top bar ---------- */
    .page-title {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 700;
        font-size: 26px;
        color: {PRIMARY};
        margin-bottom: 0px;
    }}
    .page-subtitle {{
        color: {SECONDARY};
        font-size: 14.5px;
        margin-top: 2px;
    }}
    .claim-badge {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 8px 14px;
        color: {SECONDARY};
        font-size: 13px;
        font-weight: 500;
        float: right;
    }}

    /* ---------- Cards ---------- */
    .card {{
        background-color: {CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 18px;
    }}
    .card-title {{
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        font-size: 16px;
        color: {PRIMARY};
        margin-bottom: 14px;
    }}

    /* ---------- Badges ---------- */
    .fraud-badge {{
        background-color: {DANGER_BG};
        color: {DANGER};
        border: 1px solid #F3B9B9;
        border-radius: 8px;
        padding: 7px 14px;
        font-weight: 600;
        font-size: 13.5px;
        float: right;
    }}
    .safe-badge {{
        background-color: {SAFE_BG};
        color: {SAFE};
        border: 1px solid #B8E0B9;
        border-radius: 8px;
        padding: 7px 14px;
        font-weight: 600;
        font-size: 13.5px;
        float: right;
    }}

    /* ---------- Metric bars ---------- */
    .metric-label {{
        display: flex; justify-content: space-between;
        font-size: 13px; color: {SECONDARY}; margin-bottom: 4px;
    }}
    .metric-value {{ font-weight: 700; color: {PRIMARY}; }}
    .bar-track {{
        height: 7px; background-color: #EEF0F3; border-radius: 4px; overflow: hidden; margin-bottom: 16px;
    }}
    .bar-fill-danger {{ height: 100%; background-color: {DANGER}; }}
    .bar-fill-dark {{ height: 100%; background-color: {PRIMARY}; }}

    /* ---------- Summary box ---------- */
    .summary-box {{
        background-color: #F3F6FB;
        border: 1px solid #DDE6F2;
        border-radius: 10px;
        padding: 16px 18px;
        font-size: 13.8px;
        color: #33394A;
        line-height: 1.55;
    }}
    .summary-note {{
        color: {NEUTRAL};
        font-size: 12px;
        font-style: italic;
        margin-top: 10px;
    }}

    /* ---------- Tech table ---------- */
    .tech-table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
    .tech-table td {{ padding: 8px 4px; border-bottom: 1px solid {BORDER}; }}
    .tech-key {{ color: {SECONDARY}; }}
    .tech-val {{ color: {PRIMARY}; font-weight: 500; }}
    .pill {{
        background-color: {SAFE_BG}; color: {SAFE}; border-radius: 20px;
        padding: 2px 10px; font-size: 12px; font-weight: 600;
    }}

    div.stButton > button {{
        background-color: {PRIMARY};
        color: #FFFFFF;
        border-radius: 9px;
        border: none;
        padding: 10px 0;
        font-weight: 600;
    }}
    div.stButton > button:hover {{
        background-color: #1E293B;
        color: #FFFFFF;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<div class="brand">🛡️ ClaimShield</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-sub">Vehicle Claims Intelligence</div>', unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["Dashboard", "Active Investigations", "Fraud Database", "Compliance", "Reports"],
        index=1,
        label_visibility="collapsed",
    )

    st.markdown(
        f"""
        <div class="sidebar-footer">
        <span class="dot"></span>System Operational<br>
        &nbsp;&nbsp;&nbsp;Support
        </div>
        """,
        unsafe_allow_html=True,
    )

if page != "Active Investigations":
    st.markdown(f'<div class="page-title">{page}</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">This section is not part of the current build.</div>', unsafe_allow_html=True)
    st.stop()

# ============================================================
# TOP BAR
# ============================================================
top_left, top_right = st.columns([3, 1])
with top_left:
    st.markdown('<div class="page-title">Vehicle Claim Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Review vehicle damage evidence and assess potential fraud risk.</div>', unsafe_allow_html=True)
with top_right:
    st.markdown(f'<div class="claim-badge"># Claim ID: {st.session_state.claim_id}</div>', unsafe_allow_html=True)

st.write("")
left_col, right_col = st.columns([1, 1.15])

# ============================================================
# LEFT — CLAIM EVIDENCE
# ============================================================
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 Claim Evidence</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drag & drop evidence — Supported: JPG, PNG (Max 10MB)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_container_width=True)
        size_kb = uploaded_file.size / 1024
        st.caption(
            f"✅ **{uploaded_file.name}** · {image.width} × {image.height} · {size_kb:.1f} KB · Upload complete"
        )

    analyze = st.button("🔍 Analyze Claim", type="primary", use_container_width=True, disabled=image is None)
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# RUN MODEL
# ============================================================
if analyze and image is not None:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.save(temp_file.name, format="JPEG")
        temp_path = Path(temp_file.name)

    with st.spinner("Running ConvNeXt-Tiny analysis..."):
        try:
            result = generate_gradcam(temp_path)
        except Exception as e:
            st.error(f"Model inference failed: {e}")
            st.stop()

    st.session_state.result = result
    st.session_state.processing_id = f"PRC-{str(uuid.uuid4().int)[:4]}"
    st.session_state.exec_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

# ============================================================
# RIGHT — ASSESSMENT RESULT
# ============================================================
with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.markdown('<div class="card-title">Assessment Result</div>', unsafe_allow_html=True)
        st.caption("Automated fraud detection engine analysis")
        st.info("Upload evidence and run **Analyze Claim** to generate an assessment.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        result = st.session_state.result
        prediction = result["prediction"]
        fraud_probability = result["fraud_probability"]
        confidence = result["confidence"]
        gradcam_path = Path(result["gradcam_path"])
        is_fraud = prediction == "Fraud"

        badge_html = (
            f'<div class="fraud-badge">▲ POTENTIAL FRAUD</div>'
            if is_fraud
            else f'<div class="safe-badge">✔ NON-FRAUD</div>'
        )
        st.markdown(f'<div class="card-title">Assessment Result</div>{badge_html}', unsafe_allow_html=True)
        st.caption("Automated fraud detection engine analysis")
        st.write("")

        st.markdown(
            f"""
            <div class="metric-label"><span>Fraud Probability</span><span class="metric-value">{fraud_probability:.1%}</span></div>
            <div class="bar-track"><div class="bar-fill-danger" style="width:{fraud_probability*100:.1f}%;"></div></div>
            <div class="metric-label"><span>Model Confidence</span><span class="metric-value">{confidence:.1%}</span></div>
            <div class="bar-track"><div class="bar-fill-dark" style="width:{confidence*100:.1f}%;"></div></div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Visual Evidence**")
        ev_col1, ev_col2 = st.columns(2)
        with ev_col1:
            st.image(image, caption="Original Evidence", use_container_width=True)
        with ev_col2:
            if gradcam_path.exists():
                st.image(Image.open(gradcam_path), caption="Model Attention (Heatmap)", use_container_width=True)
            else:
                st.warning("Grad-CAM not generated.")

        st.write("")
        st.markdown("**Assessment Summary**")
        if is_fraud:
            summary = (
                f"The model identified visual patterns consistent with the Fraud class "
                f"at {fraud_probability:.1%} probability. The Grad-CAM heatmap highlights the image "
                f"regions that most strongly influenced this prediction — review these areas alongside "
                f"the claim history before making a determination."
            )
        else:
            summary = (
                f"No strong fraud signal was detected. The model's non-fraud confidence is "
                f"{1 - fraud_probability:.1%}. This result should still be treated as AI-assisted "
                f"screening rather than a final decision."
            )
        st.markdown(
            f"""
            <div class="summary-box">{summary}</div>
            <div class="summary-note">Automated assessment is decision support only — final adjudication requires human analyst review.</div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown("**Technical Execution Details**")
        st.markdown(
            f"""
            <table class="tech-table">
            <tr><td class="tech-key">Processing ID</td><td class="tech-val">{st.session_state.processing_id}</td></tr>
            <tr><td class="tech-key">Execution Date</td><td class="tech-val">{st.session_state.exec_time}</td></tr>
            <tr><td class="tech-key">Architecture</td><td class="tech-val">ConvNeXt-Tiny</td></tr>
            <tr><td class="tech-key">Explainability Map</td><td class="tech-val">Grad-CAM</td></tr>
            <tr><td class="tech-key">Pipeline Status</td><td class="tech-val"><span class="pill">● Completed</span></td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)