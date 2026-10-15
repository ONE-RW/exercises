"""Student Pass/Fail predictor - Streamlit app.

Run locally: streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "student_status_pipeline.joblib"

SUBJECTS = {
    "Biology_and_Health_Sciences": "Biology & Health Sciences",
    "Chemistry": "Chemistry",
    "English": "English",
    "Entrepreneurship": "Entrepreneurship",
    "Geography_and_Environment": "Geography & Environment",
    "History_and_Citizenship": "History & Citizenship",
    "Kinyarwanda": "Kinyarwanda",
    "Mathematics": "Mathematics",
    "Physics": "Physics",
}

st.set_page_config(page_title="Student Status | Academic Insights", page_icon="◎", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#13253b; --muted:#708096; --teal:#087e79; --teal-dark:#075d5a; --line:#e5edf2; --paper:#f5f8fa; }
.stApp { background:var(--paper); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1240px; padding-top:2.3rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif; color:var(--ink); letter-spacing:-.035em; }
h1 { font-weight:800; font-size:2.5rem; margin:0 0 .3rem; }
.hero { padding:2rem 2.15rem; margin:.45rem 0 1.4rem; border-radius:24px; color:white; background:linear-gradient(115deg,#10293b 0%,#155b63 62%,#168b80 100%); position:relative; overflow:hidden; box-shadow:0 18px 42px rgba(20,67,78,.16); }
.hero:after { content:' '; position:absolute; width:250px; height:250px; border:1px solid rgba(255,255,255,.16); border-radius:50%; right:7%; top:-130px; box-shadow:0 0 0 32px rgba(255,255,255,.035),0 0 0 65px rgba(255,255,255,.025); }
.hero-kicker { color:#b9e8dc; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; margin-bottom:.6rem; }
.hero h1 { color:white; font-size:2.15rem; margin:0 0 .5rem; }
.hero p { max-width:650px; color:#d3e5e8; margin:0; font-size:1rem; line-height:1.6; }
.section-heading { margin:.8rem 0 .15rem; font-family:'Manrope',sans-serif; font-weight:800; font-size:1.25rem; letter-spacing:-.025em; }
.section-sub { margin:0 0 1rem; color:var(--muted); font-size:.9rem; }
.info-card { border:1px solid var(--line); background:white; border-radius:18px; padding:1.1rem 1.25rem; }
.result-card { border-radius:20px; padding:1.5rem; background:white; border:1px solid var(--line); box-shadow:0 12px 30px rgba(28,52,70,.07); margin:.65rem 0 1rem; }
.result-pass { border-top:4px solid #13a785; }
.result-fail { border-top:4px solid #e27469; }
.result-label { color:var(--muted); text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:700; }
.result-value { font:800 2rem 'Manrope',sans-serif; color:var(--ink); margin:.15rem 0 .35rem; }
.result-detail { color:var(--muted); font-size:.9rem; }
.stButton > button[kind="primaryFormSubmit"], .stFormSubmitButton > button { border:0; border-radius:12px; background:linear-gradient(100deg,#087e79,#119d8d); color:white; font-weight:700; padding:.65rem 1.4rem; min-height:3rem; box-shadow:0 8px 18px rgba(8,126,121,.2); }
.stButton > button[kind="primaryFormSubmit"]:hover, .stFormSubmitButton > button:hover { background:#075d5a; border:0; color:white; }
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { background:transparent; border-radius:10px 10px 0 0; color:var(--muted); font-weight:700; padding:.7rem 1.1rem; }
.stTabs [aria-selected="true"] { color:var(--teal-dark)!important; border-bottom:3px solid #0d9a8a!important; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; }
[data-testid="stMetricLabel"] { color:var(--muted); }
div[data-baseweb="input"] > div, div[data-baseweb="select"] > div { border-radius:10px; }
.small-note { font-size:.82rem; color:var(--muted); line-height:1.55; }
@media(max-width:700px) { .block-container{padding:1.2rem 1rem 2rem}.hero{padding:1.5rem;border-radius:18px}.hero h1{font-size:1.7rem}h1{font-size:2rem} }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def predict(artifact, frame: pd.DataFrame) -> pd.DataFrame:
    """Return predicted status and P(Pass) for each row of a raw-feature frame."""
    pipe = artifact["pipeline"]
    frame = frame[artifact["features"]]
    out = pd.DataFrame({"Predicted_Status": pipe.predict(frame)}, index=frame.index)
    if hasattr(pipe, "predict_proba"):
        classes = list(pipe.classes_)
        if "Pass" in classes:
            out["Probability_Pass"] = pipe.predict_proba(frame)[:, classes.index("Pass")].round(4)
    return out


try:
    artifact = load_artifact()
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH.name}. Put it next to app.py.")
    st.stop()

st.markdown('''<div class="hero"><div class="hero-kicker">Student success • O-level</div><h1>Make every mark count.</h1><p>Explore a student's likely outcome with a clear, data-informed prediction. Enter marks individually or assess a whole class from a CSV.</p></div>''', unsafe_allow_html=True)
tab_single, tab_batch = st.tabs(["◉  Individual student", "▦  Class assessment"])

with tab_single:
    st.markdown('<div class="section-heading">Student profile</div><div class="section-sub">Add the available subject marks. You can leave any unknown score blank.</div>', unsafe_allow_html=True)
    with st.form("single"):
        profile_col, guidance_col = st.columns([1.15, 2.1], gap="large")
        with profile_col:
            st.markdown("**Profile details**")
            gender = st.selectbox("Gender", ["Female", "Male", "Not specified"])
            st.markdown('<div class="small-note">Scores use a 0–100 scale. At least five marks are recommended.</div>', unsafe_allow_html=True)
        with guidance_col:
            st.markdown("**Subject marks**")
            cols = st.columns(3, gap="medium")
            marks = {}
            for i, (key, label) in enumerate(SUBJECTS.items()):
                marks[key] = cols[i % 3].number_input(
                    label, min_value=0.0, max_value=100.0, value=None,
                    step=0.5, placeholder="Not entered", key=f"mark_{key}",
                )
        submitted = st.form_submit_button("Generate prediction  →", type="primary", use_container_width=True)

    if submitted:
        row = {"Gender": np.nan if gender == "Not specified" else gender}
        row.update({k: (np.nan if v is None else float(v)) for k, v in marks.items()})
        result = predict(artifact, pd.DataFrame([row])).iloc[0]
        entered = [v for v in marks.values() if v is not None]
        status = result["Predicted_Status"]
        pass_probability = float(result["Probability_Pass"]) if "Probability_Pass" in result else None
        style = "result-pass" if status == "Pass" else "result-fail"
        st.markdown(f'<div class="result-card {style}"><div class="result-label">Prediction result</div><div class="result-value">{status}</div><div class="result-detail">Based on {len(entered)} of {len(SUBJECTS)} subject marks provided.</div></div>', unsafe_allow_html=True)
        if len(entered) < 5:
            st.warning("Fewer than five marks were entered. Treat this prediction with caution.")
        metric_cols = st.columns(3)
        metric_cols[0].metric("Marks provided", f"{len(entered)} / {len(SUBJECTS)}")
        metric_cols[1].metric("Average entered mark", f"{np.mean(entered):.1f}" if entered else "—")
        metric_cols[2].metric("Pass probability", f"{pass_probability:.1%}" if pass_probability is not None else "Unavailable")
        if pass_probability is not None:
            st.progress(pass_probability, text="Estimated probability of passing")

with tab_batch:
    st.markdown('<div class="section-heading">Assess a whole class</div><div class="section-sub">Upload a spreadsheet saved as CSV to get predictions for every student in one step.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.25, .75], gap="large")
    with left:
        st.markdown("#### 1. Prepare your file")
        st.markdown('<div class="small-note">Include one row per student and a column for every required feature. Gender values can be Female or Male; blank values are allowed.</div>', unsafe_allow_html=True)
        template = pd.DataFrame(columns=artifact["features"]).to_csv(index=False)
        st.download_button("↓  Download CSV template", template, "student_marks_template.csv", "text/csv", use_container_width=True)
    with right:
        st.markdown("#### 2. Upload class marks")
        upload = st.file_uploader("Choose a CSV file", type="csv", label_visibility="collapsed")

    with st.expander("View required column names"):
        st.code(", ".join(artifact["features"]), language=None)

    if upload is not None:
        try:
            data = pd.read_csv(upload)
            missing_cols = [c for c in artifact["features"] if c not in data.columns]
            if missing_cols:
                st.error(f"Your file is missing required columns: {', '.join(missing_cols)}")
            elif data.empty:
                st.warning("This CSV has a header but no student rows yet.")
            else:
                preds = predict(artifact, data)
                result = pd.concat([data, preds], axis=1)
                pass_count = int((preds["Predicted_Status"] == "Pass").sum())
                fail_count = len(preds) - pass_count
                st.markdown("#### Class prediction summary")
                summary_cols = st.columns(3)
                summary_cols[0].metric("Students assessed", f"{len(result):,}")
                summary_cols[1].metric("Predicted pass", f"{pass_count:,}")
                summary_cols[2].metric("Predicted fail", f"{fail_count:,}")
                chart_col, table_col = st.columns([.8, 1.2], gap="large")
                with chart_col:
                    st.markdown("**Outcome distribution**")
                    st.bar_chart(preds["Predicted_Status"].value_counts().rename_axis("Outcome"), color="#0c9184", horizontal=True)
                with table_col:
                    st.markdown("**Student results**")
                    st.dataframe(result.head(200), use_container_width=True, hide_index=True)
                st.download_button("↓  Download predictions", result.to_csv(index=False), "student_predictions.csv", "text/csv", type="primary")
        except (UnicodeDecodeError, pd.errors.ParserError) as exc:
            st.error(f"Could not read this file as a valid CSV. Check its encoding and formatting. ({exc})")
