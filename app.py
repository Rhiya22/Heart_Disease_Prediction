"""
app.py
──────
Interactive Streamlit demo for the heart disease classifier. Enter a
patient's clinical details in the sidebar and get a live prediction with
a SHAP explanation of what drove it.

Run with: streamlit run app.py

Self-trains on first load if no saved model is found (the dataset is
public Kaggle data, committed to the repo - same self-training pattern
as the churn/credit-risk/Leeds projects).
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_preprocessing import FEATURE_COLUMNS, CATEGORICAL_COLS

MODELS_DIR = "models"

st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="wide")


REQUIRED_ARTIFACTS = ["champion.joblib", "feature_columns.joblib", "defaults.joblib",
                      "encoders.joblib", "champion_meta.joblib"]


def _train():
    with st.spinner("Training the models (this happens once - after this it loads instantly)..."):
        try:
            from model_training import main as train_and_save
            train_and_save()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Automatic model training failed: {exc}")
            st.stop()


def _load_saved():
    return (
        joblib.load(os.path.join(MODELS_DIR, "champion.joblib")),
        joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib")),
        joblib.load(os.path.join(MODELS_DIR, "defaults.joblib")),
        joblib.load(os.path.join(MODELS_DIR, "encoders.joblib")),
        joblib.load(os.path.join(MODELS_DIR, "champion_meta.joblib")),
    )


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    missing = any(not os.path.exists(os.path.join(MODELS_DIR, f)) for f in REQUIRED_ARTIFACTS)
    if missing:
        _train()

    try:
        return _load_saved()
    except Exception:
        # Saved models can't be unpickled - most often because they were trained
        # with a different scikit-learn version than is installed now (e.g. a
        # models/ folder copied from another machine or an older run). Retrain
        # fresh with today's installed versions instead of failing.
        _train()
        try:
            return _load_saved()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unexpected error while loading model artifacts: {exc}")
            st.stop()


model, feature_columns, defaults, encoders, meta = load_artifacts()

st.title("❤️ Heart Disease Risk Predictor")
st.markdown(
    "Predicts the presence of heart disease from clinical data (918-patient "
    "dataset combining 5 hospital sources). Compares Decision Tree, Random "
    "Forest, and Gradient Boosting — Gradient Boosting is used here (test "
    "ROC-AUC 0.94)."
)

st.sidebar.header("Patient Details")

with st.sidebar:
    st.subheader("Demographics")
    age = st.slider("Age", 18, 100, int(defaults["Age"]))
    sex = st.selectbox("Sex", ["M", "F"], index=["M", "F"].index(defaults["Sex"]))

    st.subheader("Symptoms & ECG")
    chest_pain = st.selectbox(
        "Chest pain type", ["ATA", "NAP", "ASY", "TA"],
        index=["ATA", "NAP", "ASY", "TA"].index(defaults["ChestPainType"]),
        help="TA = Typical Angina, ATA = Atypical Angina, NAP = Non-Anginal Pain, ASY = Asymptomatic",
    )
    resting_ecg = st.selectbox(
        "Resting ECG", ["Normal", "ST", "LVH"],
        index=["Normal", "ST", "LVH"].index(defaults["RestingECG"]),
        help="ST = ST-T wave abnormality, LVH = probable/definite left ventricular hypertrophy",
    )
    st_slope = st.selectbox(
        "ST segment slope (exercise)", ["Up", "Flat", "Down"],
        index=["Up", "Flat", "Down"].index(defaults["ST_Slope"]),
        help="The single strongest predictor in this dataset — a flat or downward slope is associated with cardiac ischaemia.",
    )
    exercise_angina = st.selectbox(
        "Exercise-induced angina", ["N", "Y"],
        index=["N", "Y"].index(defaults["ExerciseAngina"]),
    )
    oldpeak = st.slider("Oldpeak (ST depression induced by exercise)", -3.0, 7.0, float(defaults["Oldpeak"]), 0.1)

    st.subheader("Vitals & Blood Work")
    resting_bp = st.slider("Resting blood pressure (mm Hg)", 80, 220, int(defaults["RestingBP"]))
    cholesterol = st.slider("Cholesterol (mg/dl)", 0, 620, int(defaults["Cholesterol"]),
                             help="0 in this dataset generally represents a missing reading, not an actual clinical value.")
    fasting_bs = st.selectbox("Fasting blood sugar > 120 mg/dl?", ["No", "Yes"],
                               index=int(defaults["FastingBS"]))
    max_hr = st.slider("Maximum heart rate achieved", 60, 210, int(defaults["MaxHR"]))

    predict_btn = st.button("Predict", width="stretch", type="primary")

col1, col2, col3 = st.columns(3)
col1.metric("Model", meta["model_name"].replace("_", " ").title())
results_path = os.path.join("results", "model_comparison.csv")
if os.path.exists(results_path):
    results_df = pd.read_csv(results_path)
    champion_row = results_df[results_df["model"] == meta["model_name"]].iloc[0]
    col2.metric("Test ROC-AUC", f"{champion_row['roc_auc']:.3f}")
    col3.metric("Test Accuracy", f"{champion_row['accuracy']:.1%}")

if not predict_btn:
    st.info("Fill in the patient details in the sidebar and click **Predict** to see a result.")
    st.stop()

raw_input = {
    "Age": age, "Sex": sex, "ChestPainType": chest_pain, "RestingBP": resting_bp,
    "Cholesterol": cholesterol, "FastingBS": 1 if fasting_bs == "Yes" else 0,
    "RestingECG": resting_ecg, "MaxHR": max_hr, "ExerciseAngina": exercise_angina,
    "Oldpeak": oldpeak, "ST_Slope": st_slope,
}
input_df = pd.DataFrame([raw_input])

encoded_input = input_df.copy()
for col in CATEGORICAL_COLS:
    encoded_input[col] = encoders[col].transform(encoded_input[col])
encoded_input = encoded_input[feature_columns]

from explainability import explain_single_prediction

result = explain_single_prediction(encoded_input, top_n=8)
prob = result["probability"]
prediction = result["prediction"]

st.markdown("---")
left, right = st.columns(2)

with left:
    st.subheader("Result")
    label = "Heart Disease Likely" if prediction == 1 else "No Heart Disease Likely"
    color = "#dc3545" if prediction == 1 else "#198754"
    icon = "⚠️" if prediction == 1 else "✅"
    st.markdown(
        f'<div style="background:{color}22;border-left:6px solid {color};'
        f'padding:14px 20px;border-radius:6px;font-size:1.2rem;font-weight:600;">'
        f'{icon} {label} (P = {prob:.1%})</div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": "P(Heart Disease)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#343a40"},
            "steps": [
                {"range": [0, 30], "color": "#d4edda"},
                {"range": [30, 60], "color": "#fff3cd"},
                {"range": [60, 100], "color": "#f8d7da"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("What drove this prediction")
    top_contrib = pd.Series(result["top_contributions"]).sort_values()
    colors = ["#C44E52" if v > 0 else "#55A868" for v in top_contrib.values]

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.barh(top_contrib.index, top_contrib.values, color=colors, alpha=0.85)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("SHAP contribution to P(Heart Disease)\nred = increases risk, green = reduces it")
    fig2.patch.set_facecolor("white")
    ax2.set_facecolor("white")
    plt.tight_layout()
    st.pyplot(fig2, width="stretch")
    plt.close(fig2)

st.markdown("---")
with st.expander("Full input details"):
    st.dataframe(input_df.T.rename(columns={0: "value"}).astype(str), width="stretch")

with st.expander("Model comparison table"):
    if os.path.exists(results_path):
        st.dataframe(pd.read_csv(results_path), width="stretch")

st.markdown("---")
st.caption(
    "Data: Heart Failure Prediction Dataset (Kaggle) — combines the Cleveland, Hungarian, "
    "Switzerland, Long Beach VA, and Stalog heart datasets, 918 patients after de-duplication."
)
