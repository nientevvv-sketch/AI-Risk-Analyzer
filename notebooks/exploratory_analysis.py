import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Benchmark script for comparing several classification models on the breast cancer dataset.
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
RESULTS_DIR = Path("results")


def load_data():
    """Load the breast cancer dataset."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    target_names = dataset.target_names
    return X, y, target_names


def build_models():
    """Create the benchmark classification models."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_estimators=200,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def run_cross_validation(X_train, y_train, models):
    """Evaluate models with stratified cross-validation."""
    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scoring = {
        "accuracy": "accuracy",
        "precision_malignant": make_scorer(
            precision_score,
            pos_label=0,
            zero_division=0,
        ),
        "recall_malignant": make_scorer(
            recall_score,
            pos_label=0,
            zero_division=0,
        ),
        "f1_malignant": make_scorer(
            f1_score,
            pos_label=0,
            zero_division=0,
        ),
    }

    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        row = {"Model": name}
        for metric in scoring:
            row[f"{metric} mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric} std"] = scores[f"test_{metric}"].std()
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_models(X_train, y_train, X_test, y_test, models):
    """Evaluate each model on the held-out test set and return metrics + predictions."""
    rows = []
    predictions = {}

    for name, model in models.items():
        fitted_model = model.fit(X_train, y_train)
        y_pred = fitted_model.predict(X_test)
        predictions[name] = y_pred

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision (malignant)": precision_score(
                    y_test,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                ),
                "Recall (malignant)": recall_score(
                    y_test,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                ),
                "F1 (malignant)": f1_score(
                    y_test,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                ),
            }
        )

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(
        by="F1 (malignant)",
        ascending=False,
    ).reset_index(drop=True)

    return results_df, predictions


def save_results(results_df, cv_df):
    """Save the model comparison and cross-validation outputs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if isinstance(cv_df, pd.DataFrame):
        results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        cv_df.to_csv(RESULTS_DIR / "cross_validation.csv", index=False)
        return

    params_df = pd.DataFrame(
        [
            {
                "Model": name,
                "Best Parameters": json.dumps(params, sort_keys=True),
            }
            for name, params in cv_df.items()
        ]
    )
    params_df.to_csv(RESULTS_DIR / "best_hyperparameters.csv", index=False)


def save_classification_reports(predictions, y_test, target_names):
    """Save classification reports for each model in CSV format."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_rows = []

    for name, y_pred in predictions.items():
        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
        for class_name in target_names:
            report_rows.append(
                {
                    "Model": name,
                    "Class": class_name,
                    "Precision": report[class_name]["precision"],
                    "Recall": report[class_name]["recall"],
                    "F1-score": report[class_name]["f1-score"],
                    "Support": report[class_name]["support"],
                }
            )

    pd.DataFrame(report_rows).to_csv(
        RESULTS_DIR / "classification_reports.csv",
        index=False,
    )


def plot_confusion_matrices(predictions, y_test, target_names):
    """Save confusion matrix plots for all model predictions."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for ax, (name, y_pred) in zip(axes.ravel(), predictions.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            labels=[0, 1],
            display_labels=target_names,
            values_format="d",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(name)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run the benchmark end-to-end."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = build_models()
    cv_df = run_cross_validation(X_train, y_train, models)
    results_df, predictions = evaluate_models(
        X_train,
        y_train,
        X_test,
        y_test,
        models,
    )
    save_results(results_df, cv_df)
    save_classification_reports(predictions, y_test, target_names)
    plot_confusion_matrices(predictions, y_test, target_names)

    MODEL_DIR = Path(__file__).resolve().parent / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = MODEL_DIR / "risk_model.joblib"
    best_model_name = results_df.iloc[0]["Model"]
    joblib.dump(models[best_model_name], MODEL_PATH)

    model = joblib.load(MODEL_PATH)

    print("Model type:", type(model))
    print("Expected features:", getattr(model, "feature_names_in_", "Not available"))

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Model size: {MODEL_PATH.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
