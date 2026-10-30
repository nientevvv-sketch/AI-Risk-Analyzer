from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from models.model_metadata import save_model_metadata

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data.csv"
MODEL_PATH = ROOT_DIR / "models" / "risk_model.joblib"
TARGET = "Bankrupt?"
RANDOM_STATE = 42


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    if TARGET not in data.columns:
        raise ValueError(f"Training data must contain the {TARGET!r} target.")

    X = data.drop(columns=[TARGET]).apply(pd.to_numeric, errors="raise")
    y = pd.to_numeric(data[TARGET], errors="raise")
    if y.nunique() != 2:
        raise ValueError(f"Target {TARGET!r} must contain exactly two classes.")

    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_model_metadata(X_train)

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Training features: {len(X.columns)}")


if __name__ == "__main__":
    main()
