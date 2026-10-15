"""Student Pass/Fail predictor - Streamlit app.

Run locally:  streamlit run app.py
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

st.set_page_config(page_title="Student Status Predictor", page_icon="🎓", layout="centered")


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def predict(artifact, frame: pd.DataFrame) -> pd.DataFrame:
    """Return predicted status and P(Pass) for each row of a raw-feature frame."""
    pipe = artifact["pipeline"]
    frame = frame[artifact["features"]]          # enforce training column order
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

st.title("🎓 Student Status Predictor")
st.caption("Predicts whether an O-level student will **Pass** or **Fail** from their subject marks.")

with st.sidebar:
    st.header("About the model")
    st.write(f"**Model:** {artifact['model_name']}")
    st.write(f"**CV F1 (5-fold):** {artifact['cv_f1_mean']:.3f} ± {artifact['cv_f1_std']:.3f}")
    st.write(f"**Test accuracy:** {artifact['test_accuracy']:.3f}")
    st.info(
        "Marks left blank are treated as missing and filled in automatically, "
        "the same way as during training. More missing marks means a less reliable prediction."
    )

tab_single, tab_batch = st.tabs(["Single student", "Batch (CSV)"])

# ---------------------------------------------------------------- single student
with tab_single:
    with st.form("single"):
        gender = st.selectbox("Gender", ["Female", "Male", "Not specified"])
        cols = st.columns(3)
        marks = {}
        for i, (key, label) in enumerate(SUBJECTS.items()):
            marks[key] = cols[i % 3].number_input(
                f"{label} (0-100)", min_value=0.0, max_value=100.0,
                value=None, step=0.5, placeholder="blank = missing",
            )
        submitted = st.form_submit_button("Predict", type="primary")

    if submitted:
        row = {"Gender": np.nan if gender == "Not specified" else gender}
        row.update({k: (np.nan if v is None else float(v)) for k, v in marks.items()})
        result = predict(artifact, pd.DataFrame([row])).iloc[0]

        entered = [v for v in marks.values() if v is not None]
        if len(entered) < 5:
            st.warning("Fewer than 5 marks entered, so treat this prediction with caution.")

        status = result["Predicted_Status"]
        (st.success if status == "Pass" else st.error)(f"Predicted status: **{status}**")
        if "Probability_Pass" in result:
            p = float(result["Probability_Pass"])
            st.progress(p, text=f"Probability of passing: {p:.1%}")
        if entered:
            st.metric("Average of entered marks", f"{np.mean(entered):.1f}")

# ---------------------------------------------------------------- batch
with tab_batch:
    st.write("Upload a CSV with these columns:")
    st.code(", ".join(artifact["features"]), language=None)
    template = pd.DataFrame(columns=artifact["features"]).to_csv(index=False)
    st.download_button("Download blank template", template, "template.csv", "text/csv")

    upload = st.file_uploader("CSV file", type="csv")
    if upload is not None:
        data = pd.read_csv(upload)
        missing_cols = [c for c in artifact["features"] if c not in data.columns]
        if missing_cols:
            st.error(f"Missing columns: {', '.join(missing_cols)}")
        else:
            preds = predict(artifact, data)
            result = pd.concat([data, preds], axis=1)
            st.write(f"Predicted {len(result):,} students.")
            st.bar_chart(preds["Predicted_Status"].value_counts())
            st.dataframe(result.head(200), use_container_width=True)
            st.download_button("Download predictions", result.to_csv(index=False),
                               "predictions.csv", "text/csv")
