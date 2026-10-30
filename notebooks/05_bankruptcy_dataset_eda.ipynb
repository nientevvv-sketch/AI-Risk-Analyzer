"""Comparative ML experiment: Breast Cancer Wisconsin.

Run from the repository root:
    python comparative_experiment_fixed.py

Outputs are written to results/vv/.

Methodology:
- One stratified train/test split shared by all four models.
- Hyperparameter tuning uses the training split only.
- Repeated nested CV uses training data only and evaluates a reduced tuning grid for runtime.
- The held-out test set is used only for final evaluation and visualisations.
- In sklearn's dataset, class 0 = malignant and class 1 = benign.
- n_jobs=1 is intentional for lower memory use on Windows.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    make_scorer, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
    cross_validate, train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------ Configuration ------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
INNER_SPLITS = 3
OUTER_SPLITS = 3
OUTER_REPEATS = 1
N_JOBS = 1
RESULTS_DIR = Path("results") / "vv"
MALIGNANT_LABEL = 0


def build_models():
    """Return fresh, unfitted estimators."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def build_param_grids():
    return {
        "Logistic Regression": {
            "logisticregression__C": [0.1, 1.0, 10.0],
            "logisticregression__class_weight": [None, "balanced"],
        },
        "Decision Tree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 8],
            "min_samples_leaf": [1, 2, 5],
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 8],
            "min_samples_leaf": [1, 2, 5],
        },
        "Gradient Boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
    } 
    
def malignant_f1_scorer():
    return make_scorer(f1_score, pos_label=MALIGNANT_LABEL, zero_division=0)


def tune_models(X_train, y_train, models):
    """Tune models on training data only; return fitted best estimators and log."""
    param_grids = build_param_grids()
    cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scorer = malignant_f1_scorer()
    best_models, rows = {}, []

    for name, estimator in models.items():
        print(f"Tuning {name} ...", flush=True)
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grids[name],
            scoring=scorer,
            cv=cv,
            n_jobs=N_JOBS,
            refit=True,
            return_train_score=False,
            error_score="raise",
        )
        start = time.perf_counter()
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        best_models[name] = search.best_estimator_
        rows.append({
            "Model": name,
            "Best CV F1 (malignant)": search.best_score_,
            "Search time (sec)": elapsed,
            "Best parameters": json.dumps(search.best_params_, sort_keys=True),
        })
        print(f"  Best CV malignant F1: {search.best_score_:.4f}; time: {elapsed:.1f}s")
    return best_models, pd.DataFrame(rows)

def nested_cv_scores(X_train, y_train, models):
    """Estimate the full tuning procedure using repeated nested CV on training data."""
    # A smaller, pre-declared grid keeps nested CV practical on ordinary PCs.
    # Final model selection above still uses the full grid in build_param_grids().
    grids = {
        "Logistic Regression": {
            "logisticregression__C": [0.1, 1.0],
            "logisticregression__class_weight": [None, "balanced"],
        },
        "Decision Tree": {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5],
            "min_samples_leaf": [1, 2],
        },
        "Random Forest": {
            "n_estimators": [100],
            "max_depth": [None, 8],
            "min_samples_leaf": [1, 2],
        },
        "Gradient Boosting": {
            "n_estimators": [100],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        },
    } 
    scorer = malignant_f1_scorer()
    inner_cv = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS, n_repeats=OUTER_REPEATS, random_state=RANDOM_STATE
    )
    rows = []

    for name, estimator in models.items():
        print(f"Nested CV: {name} ...", flush=True)
        search = GridSearchCV(
            estimator=estimator, param_grid=grids[name], scoring=scorer,
            cv=inner_cv, n_jobs=N_JOBS, refit=True, error_score="raise",
        )
        result = cross_validate(
            search, X_train, y_train, cv=outer_cv, scoring=scorer,
            n_jobs=1, return_train_score=False, error_score="raise",
        )
        scores = result["test_score"]
        for i, score in enumerate(scores, start=1):
            rows.append({"Model": name, "Outer fold": i, "F1 (malignant)": float(score)})
        sd = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
        print(f"  Nested CV malignant F1: {np.mean(scores):.4f} +/- {sd:.4f}")
    return pd.DataFrame(rows) 
def evaluate_test_set(X_test, y_test, fitted_models):
    """Final descriptive evaluation on the held-out test set."""
    metric_rows, report_rows, predictions, probability_scores, error_rows = [], [], {}, {}, []
    y_malignant = (np.asarray(y_test) == MALIGNANT_LABEL).astype(int)

    for name, model in fitted_models.items():
        start = time.perf_counter()
        y_pred = model.predict(X_test)
        prediction_time = time.perf_counter() - start
        predictions[name] = y_pred

        proba = model.predict_proba(X_test)
        class_index = list(model.classes_).index(MALIGNANT_LABEL)
        malignant_proba = proba[:, class_index]
        probability_scores[name] = malignant_proba

        metric_rows.append({
            "Model": name,
            "Test Accuracy": accuracy_score(y_test, y_pred),
            "Test Precision (malignant)": precision_score(y_test, y_pred, pos_label=0, zero_division=0),
            "Test Recall (malignant)": recall_score(y_test, y_pred, pos_label=0, zero_division=0),
            "Test F1 (malignant)": f1_score(y_test, y_pred, pos_label=0, zero_division=0),
            "Test ROC-AUC (malignant)": roc_auc_score(y_malignant, malignant_proba),
            "Prediction time (ms)": prediction_time * 1000,
            "Prediction time per sample (ms)": prediction_time * 1000 / len(X_test),
        })
        report = classification_report(
            y_test, y_pred, labels=[0, 1], target_names=["malignant", "benign"],
            output_dict=True, zero_division=0,
        )
        for label in ["malignant", "benign"]:
            report_rows.append({
                "Model": name, "Class": label,
                "Precision": report[label]["precision"],
                "Recall": report[label]["recall"],
                "F1-score": report[label]["f1-score"],
                "Support": report[label]["support"],
            })

        for i, (actual, predicted, probability) in enumerate(zip(y_test, y_pred, malignant_proba)):
            if actual != predicted:
                error_rows.append({
                    "Model": name, "Test row index": str(X_test.index[i]) if hasattr(X_test, "index") else str(i),
                    "Actual class": "malignant" if actual == 0 else "benign",
                    "Predicted class": "malignant" if predicted == 0 else "benign",
                    "Predicted probability (malignant)": float(probability),
                })

    return (pd.DataFrame(metric_rows), pd.DataFrame(report_rows), predictions,
            probability_scores, pd.DataFrame(error_rows))
def save_roc_plot(y_test, probability_scores, path):
    y_malignant = (np.asarray(y_test) == MALIGNANT_LABEL).astype(int)
    plt.figure(figsize=(8, 6))
    for name, probabilities in probability_scores.items():
        fpr, tpr, _ = roc_curve(y_malignant, probabilities)
        auc = roc_auc_score(y_malignant, probabilities)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves — held-out test set (malignant class)")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion_matrices(y_test, predictions, path):
    names = list(predictions)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, name in zip(axes.flat, names):
        cm = confusion_matrix(y_test, predictions[name], labels=[0, 1])
        image = ax.imshow(cm, interpolation="nearest")
        ax.set_title(name)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_xticks([0, 1], labels=["Malignant", "Benign"])
        ax.set_yticks([0, 1], labels=["Malignant", "Benign"])
        for (r, c), value in np.ndenumerate(cm):
            ax.text(c, r, str(value), ha="center", va="center")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion matrices — held-out test set")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_stability_plot(nested_df, path):
    names = list(nested_df["Model"].drop_duplicates())
    data = [nested_df.loc[nested_df["Model"] == name, "F1 (malignant)"].to_numpy() for name in names]
    plt.figure(figsize=(9, 6))
    plt.boxplot(data, tick_labels=names, showmeans=True)
    plt.ylabel("Outer-fold F1 (malignant)")
    plt.title("Nested cross-validation stability (training data only)")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_metric_plot(metrics_df, path):
    columns = ["Test Precision (malignant)", "Test Recall (malignant)", "Test F1 (malignant)", "Test ROC-AUC (malignant)"]
    plot_df = metrics_df.set_index("Model")[columns]
    ax = plot_df.plot(kind="bar", figsize=(11, 6), ylim=(0, 1.05))
    ax.set_ylabel("Score")
    ax.set_title("Test-set metric comparison (malignant class)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading Breast Cancer Wisconsin dataset...", flush=True)
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Samples: {len(X)} | features: {X.shape[1]} | train: {len(X_train)} | test: {len(X_test)}")
    print("Class mapping: 0 = malignant, 1 = benign")

    models = build_models()
    best_models, tuning_df = tune_models(X_train, y_train, models)
    tuning_df.to_csv(RESULTS_DIR / "tuning_results.csv", index=False)

    print("\nRunning repeated nested cross-validation...", flush=True)
    nested_df = nested_cv_scores(X_train, y_train, models)
    nested_df.to_csv(RESULTS_DIR / "nested_cv_scores.csv", index=False)
    nested_summary = nested_df.groupby("Model")["F1 (malignant)"].agg(
        ["mean", "std", "min", "max"]
    ).reset_index()
    nested_summary.to_csv(RESULTS_DIR / "nested_cv_summary.csv", index=False)

    print("\nEvaluating the held-out test set...", flush=True)
    metrics_df, reports_df, predictions, probabilities, errors_df = evaluate_test_set(
        X_test, y_test, best_models
    )
    metrics_df.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    reports_df.to_csv(RESULTS_DIR / "classification_reports.csv", index=False)
    errors_df.to_csv(RESULTS_DIR / "misclassified_samples.csv", index=False)

    save_roc_plot(y_test, probabilities, RESULTS_DIR / "roc_curves.png")
    save_confusion_matrices(y_test, predictions, RESULTS_DIR / "confusion_matrices.png")
    save_stability_plot(nested_df, RESULTS_DIR / "nested_cv_stability.png")
    save_metric_plot(metrics_df, RESULTS_DIR / "test_metrics_comparison.png") 
    
    # Combined summary table; these are separate estimates, not a single score.
    summary = metrics_df.merge(nested_summary, on="Model", how="left")
    summary = summary.merge(tuning_df[["Model", "Best CV F1 (malignant)", "Search time (sec)", "Best parameters"]], on="Model", how="left")
    summary.to_csv(RESULTS_DIR / "final_summary.csv", index=False)

    print("\nTest-set results (malignant class):")
    print(metrics_df.sort_values("Test F1 (malignant)", ascending=False).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nMisclassified test rows recorded: {len(errors_df)}")
    print(f"All outputs saved to: {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
