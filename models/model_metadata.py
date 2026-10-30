
import json
from pathlib import Path

import numpy as np
import pandas as pd

# X_train must be a pandas DataFrame.
# fitted_model must already be trained on X_train.

MODEL_DIR = Path(__file__).resolve().parent
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def save_model_metadata(X_train: pd.DataFrame) -> dict:
    """Build and save metadata from the supplied training data."""
    features = []

    for name in X_train.columns:
        values = pd.to_numeric(X_train[name], errors="coerce").dropna()

        if values.empty:
            raise ValueError(
                f"Feature {name!r} has no valid numeric values. "
                "Categorical features need separate handling."
            )

        minimum = float(values.min())
        maximum = float(values.max())
        default = float(values.median())

        # Streamlit number_input requires min_value < max_value.
        if minimum == maximum:
            minimum -= 1.0
            maximum += 1.0

        features.append({
            "name": str(name),
            "label": str(name).replace("_", " ").title(),
            "type": "number",
            "min": minimum,
            "max": maximum,
            "default": min(max(default, minimum), maximum),
            "step": 0.01
        })

    metadata = {
        "task_name": "Company Bankruptcy Prediction",
        "positive_class": 1,
        "threshold": 0.5,
        "features": features
    }

    metadata_path = MODEL_DIR / "model_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"Metadata saved to: {metadata_path}")
    print(f"Number of features: {len(features)}")
    return metadata
