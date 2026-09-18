"""
data_preprocessing.py
──────────────────────
Loads the Heart Failure Prediction dataset (heart.csv - 918 patients,
11 clinical features) and encodes categorical columns for modelling.

Same preprocessing as the original analysis notebook: LabelEncoder on
each categorical column, no scaling (tree-based models don't need it).
"""

import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.path.join("data", "heart.csv")
ENCODERS_PATH = os.path.join("models", "encoders.joblib")

TARGET = "HeartDisease"

CATEGORICAL_COLS = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
FEATURE_COLUMNS = [
    "Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol", "FastingBS",
    "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak", "ST_Slope",
]


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Place heart.csv at this path.")
    return pd.read_csv(path)


def encode_data(df: pd.DataFrame, encoders: dict | None = None, fit: bool = True):
    """Encodes the categorical columns. Pass `encoders` + fit=False to reuse
    encoders already fitted during training (e.g. for a single live prediction)."""
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            df[col] = encoders[col].transform(df[col])
    return df, encoders


def run_preprocessing(save: bool = True):
    df = load_raw()
    df_encoded, encoders = encode_data(df, fit=True)

    if save:
        os.makedirs("models", exist_ok=True)
        joblib.dump(encoders, ENCODERS_PATH)

    return df, df_encoded, encoders


def main():
    df, df_encoded, encoders = run_preprocessing()
    print(f"Loaded {len(df)} patients, {df_encoded.shape[1] - 1} features.")
    print(f"Heart disease rate: {df[TARGET].mean():.1%}")
    print(f"Encoders saved for: {list(encoders.keys())}")


if __name__ == "__main__":
    main()
