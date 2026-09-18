"""
explainability.py
──────────────────
SHAP-based explainability for the champion model (Gradient Boosting).
Since the champion is always tree-based here, shap.TreeExplainer is used
throughout - fast and exact, no approximation needed.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_preprocessing import run_preprocessing, FEATURE_COLUMNS, TARGET
from sklearn.model_selection import train_test_split

MODELS_DIR = "models"
RESULTS_DIR = "results"
RANDOM_STATE = 42


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "champion.joblib"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
    return model, feature_columns


def explain_single_prediction(input_row: pd.DataFrame, top_n: int = 8) -> dict:
    """input_row: single-row DataFrame, already encoded, with FEATURE_COLUMNS."""
    model, feature_columns = load_artifacts()
    input_row = input_row[feature_columns]

    proba = model.predict_proba(input_row)[0, 1]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_row)
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        values = shap_values.values[0, :, 1]
        base_value = shap_values.base_values[0, 1]
    else:
        values = shap_values.values[0]
        base_value = shap_values.base_values[0] if np.ndim(shap_values.base_values) else shap_values.base_values

    contributions = pd.Series(values, index=feature_columns).sort_values(key=abs, ascending=False)

    return {
        "probability": float(proba),
        "prediction": int(proba >= 0.5),
        "base_value": float(base_value),
        "all_contributions": contributions.to_dict(),
        "top_contributions": contributions.head(top_n).to_dict(),
    }


def save_summary_plot():
    model, feature_columns = load_artifacts()
    df, df_encoded, _ = run_preprocessing(save=False)
    X = df_encoded[feature_columns]
    y = df_encoded[TARGET]
    _, X_test, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    if hasattr(shap_values, "values") and shap_values.values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    fig = plt.gcf()
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "shap_summary.png")
    plt.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)
    return path


def main():
    path = save_summary_plot()
    print(f"Saved SHAP summary plot -> {path}")


if __name__ == "__main__":
    main()
