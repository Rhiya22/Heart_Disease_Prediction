"""
model_training.py
──────────────────
Trains and compares Decision Tree, Random Forest, and Gradient Boosting
on the heart disease classification task (same 3 models as the original
notebook's Objective 5), evaluates on a held-out test split, and saves:

  - models/champion.joblib        - the best model by test ROC-AUC
  - models/feature_columns.joblib - the feature column order
  - models/defaults.joblib        - median/mode value per feature, used
                                     to prefill the Streamlit app's sidebar
  - models/encoders.joblib        - LabelEncoders for the categorical
                                     columns (also saved by
                                     data_preprocessing.py)
  - results/model_comparison.csv  - accuracy/ROC-AUC/precision/recall/F1
                                     for all 3 models
"""

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
)

from data_preprocessing import run_preprocessing, TARGET, FEATURE_COLUMNS, CATEGORICAL_COLS

MODELS_DIR = "models"
RESULTS_DIR = "results"
RANDOM_STATE = 42

MODELS = {
    "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE),
}


def _evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df, df_encoded, encoders = run_preprocessing()

    X = df_encoded[FEATURE_COLUMNS]
    y = df_encoded[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    rows = []
    fitted = {}
    for name, model in MODELS.items():
        model.fit(X_train, y_train)
        metrics = _evaluate(model, X_test, y_test)
        rows.append({"model": name, **metrics})
        fitted[name] = model
        print(f"  {name:<20} acc={metrics['accuracy']:.3f}  "
              f"roc_auc={metrics['roc_auc']:.3f}  f1={metrics['f1']:.3f}")

    champion_name = max(rows, key=lambda r: r["roc_auc"])["model"]
    champion = fitted[champion_name]
    print(f"\nChampion: {champion_name}")

    joblib.dump(champion, os.path.join(MODELS_DIR, "champion.joblib"))
    joblib.dump(FEATURE_COLUMNS, os.path.join(MODELS_DIR, "feature_columns.joblib"))
    joblib.dump({"model_name": champion_name}, os.path.join(MODELS_DIR, "champion_meta.joblib"))

    # Defaults for the app's sidebar: median for numeric, mode for categorical
    # (computed on the raw, unencoded data so defaults display in human units).
    defaults = {}
    for col in FEATURE_COLUMNS:
        if col in CATEGORICAL_COLS:
            defaults[col] = df[col].mode().iloc[0]
        else:
            defaults[col] = df[col].median()
    joblib.dump(defaults, os.path.join(MODELS_DIR, "defaults.joblib"))

    results_df = pd.DataFrame(rows)
    results_df["is_champion"] = results_df["model"] == champion_name
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print(f"\nSaved results table to {RESULTS_DIR}/model_comparison.csv")


if __name__ == "__main__":
    main()
