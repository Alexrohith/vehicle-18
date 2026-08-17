import sys
import uuid
from pathlib import Path
import tempfile
from datetime import datetime

import streamlit as st
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as ReportLabImage,
)

# ============================================================
# PROJECT PATH
# ============================================================
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.explainability.gradcam import generate_gradcam
from src.inference.damage_detector import VehicleDamageDetector

# ============================================================
# YOLO DAMAGE DETECTOR
# ============================================================

@st.cache_resource
def load_damage_detector():
    return VehicleDamageDetector(
        model_path="models/best.pt",
        confidence=0.25,
        iou=0.45,
    )


damage_detector = load_damage_detector()


# ============================================================
# PDF CLAIM REPORT GENERATOR
# ============================================================

def generate_claim_report_pdf(
    output_path,
    claim_id,
    processing_id,
    execution_time,
    original_image_path,
    annotated_image_path,
    gradcam_path,
    fraud_label,
    fraud_probability,
    model_confidence,
    damage_result,
):
    """Generate the final vehicle claim assessment PDF."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ClaimReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "ClaimReportSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontSize=9,
        leading=12,
        spaceAfter=14,
    )

    section_style = ParagraphStyle(
        "ClaimReportSection",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=6,
    )

    body_style = ParagraphStyle(
        "ClaimReportBody",
        parent=styles["BodyText"],
        fontSize=8.5,
        leading=11,
    )

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Vehicle Claim Assessment Report",
    )

    story = []

    story.append(
        Paragraph(
            "VEHICLE CLAIM ASSESSMENT REPORT",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "AI-assisted vehicle insurance claim assessment",
            subtitle_style,
        )
    )

    # --------------------------------------------------------
    # Claim information
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "1. CLAIM INFORMATION",
            section_style,
        )
    )

    claim_rows = [
        ["Claim ID", str(claim_id)],
        ["Processing ID", str(processing_id)],
        ["Assessment Date", str(execution_time)],
        ["Fraud Model", "ConvNeXt-Tiny"],
        ["Damage Model", "YOLOv11n"],
        ["Explainability", "Grad-CAM"],
    ]

    claim_table = Table(
        claim_rows,
        colWidths=[
            50 * mm,
            120 * mm,
        ],
    )

    claim_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(claim_table)

    # --------------------------------------------------------
    # Fraud assessment
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "2. FRAUD ASSESSMENT",
            section_style,
        )
    )

    fraud_rows = [
        ["Classification", str(fraud_label)],
        ["Fraud Probability", f"{fraud_probability:.2f}%"],
        ["Model Confidence", f"{model_confidence:.2f}%"],
    ]

    fraud_table = Table(
        fraud_rows,
        colWidths=[
            50 * mm,
            120 * mm,
        ],
    )

    fraud_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(fraud_table)

    # --------------------------------------------------------
    # Damage assessment
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "3. DAMAGE ASSESSMENT",
            section_style,
        )
    )

    damage_detected = bool(
        damage_result.get(
            "damage_detected",
            False,
        )
    )

    damage_count = int(
        damage_result.get(
            "damage_count",
            0,
        )
    )

    damage_coverage = float(
        damage_result.get(
            "damage_coverage_percentage",
            0.0,
        )
    )

    overall_severity = str(
        damage_result.get(
            "overall_severity",
            "N/A",
        )
    )

    overall_score = float(
        damage_result.get(
            "overall_severity_score",
            0.0,
        )
    )

    damage_rows = [
        [
            "Damage Detected",
            "YES" if damage_detected else "NO",
        ],
        [
            "Damage Regions",
            str(damage_count),
        ],
        [
            "Damage Coverage",
            f"{damage_coverage:.2f}%",
        ],
        [
            "Overall Severity",
            overall_severity,
        ],
        [
            "Severity Score",
            f"{overall_score:.2f}/100",
        ],
    ]

    damage_table = Table(
        damage_rows,
        colWidths=[
            50 * mm,
            120 * mm,
        ],
    )

    damage_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(damage_table)

    # --------------------------------------------------------
    # Individual damage detections
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "4. DETECTED DAMAGE REGIONS",
            section_style,
        )
    )

    detections = damage_result.get(
        "detections",
        [],
    )

    if detections:

        detection_rows = [
            [
                "#",
                "Damage Type",
                "Confidence",
                "Area",
                "Severity",
                "Score",
            ]
        ]

        for index, damage in enumerate(
            detections,
            start=1,
        ):

            detection_rows.append(
                [
                    str(index),
                    str(
                        damage.get(
                            "damage_type",
                            "Unknown",
                        )
                    ),
                    (
                        f"{float(damage.get('confidence', 0))*100:.2f}%"
                    ),
                    (
                        f"{float(damage.get('area_percentage', 0)):.2f}%"
                    ),
                    str(
                        damage.get(
                            "severity",
                            "Unknown",
                        )
                    ),
                    (
                        f"{float(damage.get('severity_score', 0)):.2f}/100"
                    ),
                ]
            )

        detection_table = Table(
            detection_rows,
            colWidths=[
                9 * mm,
                47 * mm,
                30 * mm,
                25 * mm,
                30 * mm,
                29 * mm,
            ],
            repeatRows=1,
        )

        detection_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 7),
                    ("ALIGN", (0, 0), (0, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 3),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )

        story.append(detection_table)

    else:

        story.append(
            Paragraph(
                "No damage regions were detected by YOLO.",
                body_style,
            )
        )

    # --------------------------------------------------------
    # Visual evidence
    # --------------------------------------------------------

    story.append(
        Paragraph(
            "5. VISUAL EVIDENCE",
            section_style,
        )
    )

    evidence = []

    for label, path in [
        (
            "Original Image",
            original_image_path,
        ),
        (
            "YOLO Damage Detection",
            annotated_image_path,
        ),
        (
            "ConvNeXt Grad-CAM",
            gradcam_path,
        ),
    ]:

        if path and Path(path).exists():

            evidence.append(
                [
                    Paragraph(
                        label,
                        ParagraphStyle(
                            "EvidenceLabel",
                            parent=body_style,
                            alignment=TA_CENTER,
                            fontName="Helvetica-Bold",
                            fontSize=7.5,
                        ),
                    ),
                    ReportLabImage(
                        str(path),
                        width=105 * mm,
                        height=65 * mm,
                    ),
                ]
            )

    if evidence:

        evidence_table = Table(
            evidence,
            colWidths=[
                38 * mm,
                120 * mm,
            ],
        )

        evidence_table.setStyle(
            TableStyle(
                [
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (1, 0), (1, -1), "CENTER"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )

        story.append(evidence_table)

    story.append(Spacer(1, 10))

    story.append(
        Paragraph(
            "<b>Important:</b> This document is an AI-assisted claim "
            "assessment. Model outputs are decision-support information "
            "and should be reviewed with the complete claim documentation "
            "and by an authorized human analyst before final adjudication.",
            ParagraphStyle(
                "Disclaimer",
                parent=body_style,
                fontSize=7.5,
                leading=10,
            ),
        )
    )

    doc.build(story)

    return output_path


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

    with st.spinner("Running ConvNeXt-Tiny + YOLO damage analysis..."):
        try:
            # Main fraud classifier + Grad-CAM
            result = generate_gradcam(temp_path)

            # YOLO vehicle damage detector
            damage_result = damage_detector.predict(image)

            # YOLO annotated image
            damage_annotated_image = damage_detector.annotate(image)

        except Exception as e:
            st.error(f"Model inference failed: {e}")
            st.stop()

    st.session_state.result = result
    st.session_state.damage_result = damage_result
    st.session_state.damage_annotated_image = damage_annotated_image
    st.session_state.processing_id = f"PRC-{str(uuid.uuid4().int)[:4]}"
    st.session_state.exec_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Save evidence files for the final PDF report.
    report_evidence_dir = (
        ROOT_DIR / "artifacts" / "reports" /
        st.session_state.processing_id
    )
    report_evidence_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_original_path = (
        report_evidence_dir / "original.jpg"
    )
    image.save(
        report_original_path,
        format="JPEG",
        quality=95,
    )

    report_annotated_path = (
        report_evidence_dir / "yolo_annotated.jpg"
    )

    try:
        import cv2

        if damage_annotated_image is not None:
            cv2.imwrite(
                str(report_annotated_path),
                damage_annotated_image,
            )

    except Exception:
        report_annotated_path = None

    st.session_state.report_original_image_path = str(
        report_original_path
    )

    st.session_state.report_annotated_image_path = (
        str(report_annotated_path)
        if report_annotated_path is not None
        and report_annotated_path.exists()
        else None
    )

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

        # ========================================================
        # YOLO DAMAGE ANALYSIS
        # ========================================================

        st.write("")
        st.markdown("**🚗 Vehicle Damage Analysis**")

        damage_result = st.session_state.get(
            "damage_result"
        )

        if damage_result is None:

            st.info(
                "Vehicle damage analysis was not generated."
            )

        else:

            damage_detected = damage_result[
                "damage_detected"
            ]

            damage_count = damage_result[
                "damage_count"
            ]

            overall_severity = damage_result[
                "overall_severity"
            ]

            overall_score = damage_result[
                "overall_severity_score"
            ]

            damage_coverage = damage_result.get(
                "damage_coverage_percentage",
                0.0
            )

            if damage_detected:

                dcol1, dcol2, dcol3, dcol4 = st.columns(4)

                with dcol1:
                    st.metric(
                        "Damage Regions",
                        damage_count
                    )

                with dcol2:
                    st.metric(
                        "Damage Coverage",
                        f"{damage_coverage:.2f}%"
                    )

                with dcol3:
                    st.metric(
                        "Estimated Severity",
                        overall_severity
                    )

                with dcol4:
                    st.metric(
                        "Severity Score",
                        f"{overall_score:.1f}%"
                    )

                # --------------------------------------------
                # Annotated image
                # --------------------------------------------

                st.markdown(
                    "**Damage Localization**"
                )

                annotated_image = st.session_state.get(
                    "damage_annotated_image"
                )

                if annotated_image is not None:

                    try:
                        import cv2

                        annotated_rgb = cv2.cvtColor(
                            annotated_image,
                            cv2.COLOR_BGR2RGB
                        )

                        st.image(
                            annotated_rgb,
                            caption=(
                                "YOLO Vehicle Damage "
                                "Detection"
                            ),
                            use_container_width=True
                        )

                    except Exception:
                        st.image(
                            annotated_image,
                            caption=(
                                "YOLO Vehicle Damage "
                                "Detection"
                            ),
                            use_container_width=True
                        )

                # --------------------------------------------
                # Individual detections
                # --------------------------------------------

                st.markdown(
                    "**Detected Damage Details**"
                )

                for index, damage in enumerate(
                    damage_result["detections"],
                    start=1
                ):

                    with st.expander(
                        f"Damage #{index} — "
                        f"{damage['damage_type']}"
                    ):

                        damage_col1, damage_col2 = (
                            st.columns(2)
                        )

                        with damage_col1:

                            st.write(
                                f"**Damage Type:** "
                                f"{damage['damage_type']}"
                            )

                            st.write(
                                f"**Confidence:** "
                                f"{damage['confidence']:.2%}"
                            )

                            st.write(
                                f"**Affected Area:** "
                                f"{damage['area_percentage']:.2f}%"
                            )

                        with damage_col2:

                            st.write(
                                f"**Severity:** "
                                f"{damage['severity']}"
                            )

                            st.write(
                                f"**Severity Score:** "
                                f"{damage['severity_score']:.1f}/100"
                            )

                            st.write(
                                f"**Bounding Box:** "
                                f"{damage['bounding_box']}"
                            )

            else:

                st.success(
                    "No vehicle damage was detected "
                    "by YOLO."
                )

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
            <tr><td class="tech-key">Fraud Model</td><td class="tech-val">ConvNeXt-Tiny</td></tr>
            <tr><td class="tech-key">Damage Model</td><td class="tech-val">YOLOv11n</td></tr>
            <tr><td class="tech-key">Explainability Map</td><td class="tech-val">Grad-CAM</td></tr>
            <tr><td class="tech-key">Damage Localization</td><td class="tech-val">YOLO Bounding Boxes</td></tr>
            <tr><td class="tech-key">Pipeline Status</td><td class="tech-val"><span class="pill">● Completed</span></td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# FINAL CLAIM ASSESSMENT REPORT
# ============================================================

st.markdown("---")
st.markdown("## 📄 Final Claim Assessment Report")

if "result" in st.session_state and "damage_result" in st.session_state:

    report_result = st.session_state.result
    report_damage = st.session_state.damage_result

    report_claim_id = st.session_state.get(
        "claim_id",
        "CLM-UNKNOWN",
    )

    report_processing_id = st.session_state.get(
        "processing_id",
        "PRC-UNKNOWN",
    )

    report_exec_time = st.session_state.get(
        "exec_time",
        "N/A",
    )

    report_fraud_label = report_result.get(
        "prediction",
        "N/A",
    )

    report_fraud_probability = float(
        report_result.get(
            "fraud_probability",
            0.0,
        )
    ) * 100

    report_confidence = float(
        report_result.get(
            "confidence",
            0.0,
        )
    ) * 100

    report_original = st.session_state.get(
        "report_original_image_path",
    )

    report_annotated = st.session_state.get(
        "report_annotated_image_path",
    )

    report_gradcam = report_result.get(
        "gradcam_path",
    )

    report_dir = (
        ROOT_DIR /
        "artifacts" /
        "reports"
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path = (
        report_dir /
        f"{report_claim_id}_claim_report.pdf"
    )

    # Summary cards
    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.metric(
            "Fraud Classification",
            str(report_fraud_label),
        )

    with r2:
        st.metric(
            "Fraud Probability",
            f"{report_fraud_probability:.2f}%",
        )

    with r3:
        st.metric(
            "Damage Coverage",
            f"{float(report_damage.get('damage_coverage_percentage', 0.0)):.2f}%",
        )

    with r4:
        st.metric(
            "Damage Severity",
            str(
                report_damage.get(
                    "overall_severity",
                    "N/A",
                )
            ),
        )

    st.markdown(
        """
        Generate a professional PDF containing the fraud assessment,
        YOLO damage analysis, severity scores, bounding-box evidence,
        and Grad-CAM visual evidence.
        """
    )

    b1, b2 = st.columns(2)

    with b1:

        if st.button(
            "📄 Generate Claim Report",
            type="primary",
            use_container_width=True,
        ):

            try:

                generate_claim_report_pdf(
                    output_path=report_path,
                    claim_id=report_claim_id,
                    processing_id=report_processing_id,
                    execution_time=report_exec_time,
                    original_image_path=report_original,
                    annotated_image_path=report_annotated,
                    gradcam_path=report_gradcam,
                    fraud_label=report_fraud_label,
                    fraud_probability=report_fraud_probability,
                    model_confidence=report_confidence,
                    damage_result=report_damage,
                )

                st.session_state.claim_report_path = str(
                    report_path
                )

                st.success(
                    "✅ Claim assessment report generated."
                )

            except Exception as exc:

                st.error(
                    f"Report generation failed: {exc}"
                )

    with b2:

        saved_report = st.session_state.get(
            "claim_report_path",
        )

        if saved_report and Path(
            saved_report
        ).exists():

            with open(
                saved_report,
                "rb",
            ) as report_file:

                st.download_button(
                    "⬇️ Download Claim Report",
                    data=report_file.read(),
                    file_name=Path(
                        saved_report
                    ).name,
                    mime="application/pdf",
                    use_container_width=True,
                )

    st.markdown("### Report Contents")

    st.markdown(
        """
        - Claim identification and execution details
        - ConvNeXt-Tiny fraud classification
        - Fraud probability and model confidence
        - YOLOv11 damage detection
        - Damage region count
        - Overall damage coverage percentage
        - Overall damage severity and severity score
        - Individual damage types and confidence
        - Individual bounding-box area percentages
        - YOLO annotated evidence image
        - ConvNeXt Grad-CAM evidence
        """
    )

else:

    st.info(
        "Upload a vehicle image and click **Analyze Claim** "
        "to generate the final claim report."
    )


