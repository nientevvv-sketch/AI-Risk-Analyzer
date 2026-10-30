"""
Comparative ML experiment: Breast Cancer Wisconsin
=================================================

Run from the repository root:
    python comparative_experiment.py

Outputs are written to results/comparative_experiment/.

Methodology:
- One stratified train/test split, shared by all four models.
- Hyperparameter tuning is performed on the training split only.
- Repeated nested cross-validation estimates tuning/generalization stability
  using training data only.
- The held-out test set is used for final descriptive metrics and plots.
- Class 0 is malignant in sklearn's load_breast_cancer dataset.
- n_jobs=1 is intentional for lower memory use on Windows.

Dependencies:
    pip install scikit-learn pandas numpy matplotlib
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
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
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
RESULTS_DIR = Path("results") / "aa" 

MALIGNANT_LABEL = 0
MALIGNANT_NAME = "malignant"


def build_models():
    """Return fresh, unfitted estimators."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
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
            "min_samples_leaf": [1, 4],
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 8],
            "min_samples_leaf": [1, 2],
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
    """Tune each model using training data only."""
    param_grids = build_param_grids()
    cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scorer = malignant_f1_scorer()

    best_models = {}
    tuning_rows = []
    for name, estimator in models.items():
        print(f"Tuning {name} ...")
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
        tuning_rows.append(
            {
                "Model": name,
                "Best CV F1 (malignant)": search.best_score_,
                "Search time (sec)": elapsed,
                "Best parameters": json.dumps(search.best_params_, sort_keys=True),
            }
        )
        print(f"  Best CV F1 (malignant): {search.best_score_:.4f}")
        print(f"  Best parameters: {search.best_params_}")
    return best_models, pd.DataFrame(tuning_rows) 
      
def nested_cv_scores(X_train, y_train, models):
    """
    Repeated nested CV. Each outer validation fold is untouched by that fold's
    inner GridSearchCV. This estimates the performance of the tuning procedure.
    """
    param_grids = build_param_grids()
    scorer = malignant_f1_scorer()
    inner_cv = StratifiedKFold(
        n_splits=INNER_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    outer_cv = RepeatedStratifiedKFold(
        n_splits=OUTER_SPLITS,
        n_repeats=OUTER_REPEATS,
        random_state=RANDOM_STATE,
    )

    rows = []
    for name, estimator in models.items():
        print(f"Nested CV: {name} ...")
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grids[name],
            scoring=scorer,
            cv=inner_cv,
            n_jobs=N_JOBS,
            refit=True,
            error_score="raise",
        )
        result = cross_validate(
            search,
            X_train,
            y_train,
            cv=outer_cv,
            scoring=scorer,
            n_jobs=1,
            return_train_score=False,
            error_score="raise",
        )
        for fold_index, score in enumerate(result["test_score"], start=1):
            rows.append(
                {
                    "Model": name,
                    "Outer fold": fold_index,
                    "F1 (malignant)": float(score),
                }
            )
        print(
            f"  Nested CV F1: {np.mean(result['test_score']):.4f} "
            f"+/- {np.std(result['test_score'], ddof=1):.4f}"
        )

    return pd.DataFrame(rows)


def evaluate_test_set(X_test, y_test, fitted_models):
    """Evaluate fitted models on the held-out test set."""
    rows = []
    report_rows = []
    predictions = {}
    probability_scores = {}

    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        predictions[name] = y_pred

        # Find the probability column for class 0 explicitly; do not assume
        # a column position if an estimator's class order changes.
        proba = model.predict_proba(X_test)
        class_index = list(model.classes_).index(MALIGNANT_LABEL)
        malignant_proba = proba[:, class_index]
        probability_scores[name] = malignant_proba

        y_malignant = (np.asarray(y_test) == MALIGNANT_LABEL).astype(int)
        auc = roc_auc_score(y_malignant, malignant_proba)

        rows.append(
            {
                "Model": name,
                "Test Accuracy": accuracy_score(y_test, y_pred),
                "Test Precision (malignant)": precision_score(
                    y_test, y_pred, pos_label=MALIGNANT_LABEL, zero_division=0
                ),
                "Test Recall (malignant)": recall_score(
                    y_test, y_pred, pos_label=MALIGNANT_LABEL, zero_division=0
                ),
                "Test F1 (malignant)": f1_score(
                    y_test, y_pred, pos_label=MALIGNANT_LABEL, zero_division=0
                ),
                "Test ROC-AUC (malignant)": auc,
            }
        )

        report = classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["malignant", "benign"],
            output_dict=True,
            zero_division=0,
        )
                # ... (Предыдущий код из classification_report)
        for label in ["malignant", "benign"]:
            report_rows.append(
                {
                    "Model": name,
                    "Class": label,
                    "Precision": report[label]["precision"],
                    "Recall": report[label]["recall"],
                    "F1-score": report[label]["f1-score"],
                    "Support": report[label]["support"],
                }
            )

    df_metrics = pd.DataFrame(rows)
    df_reports = pd.DataFrame(report_rows)
    
    return df_metrics, df_reports, predictions, probability_scores


def plot_roc_curves(y_test, probability_scores, output_path: Path):
    """Строит и сохраняет ROC-кривые для всех моделей."""
    plt.figure(figsize=(8, 6))
    
    # Бинаризуем y_test, где 1 — это целевой класс MALIGNANT_LABEL (0)
    y_target = (np.asarray(y_test) == MALIGNANT_LABEL).astype(int)
    
    for name, proba in probability_scores.items():
        fpr, tpr, _ = roc_curve(y_target, proba)
        auc_val = roc_auc_score(y_target, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})")
        
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate (Malignant)")
    plt.ylabel("True Positive Rate (Malignant)")
    plt.title("ROC Curves - Held-out Test Set")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ------------------------------ Main Execution ------------------------------

def main():
    # Создаем директорию для результатов
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Загрузка данных
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # 2. Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    # 3. Тюнинг гиперпараметров
    models = build_models()
    best_models, df_tuning = tune_models(X_train, y_train, models)
    df_tuning.to_csv(RESULTS_DIR / "tuning_results.csv", index=False)
    
    # 4. Оценка стабильности через Nested CV
    df_nested = nested_cv_scores(X_train, y_train, models)
    df_nested.to_csv(RESULTS_DIR / "nested_cv_scores.csv", index=False)
    
    # 5. Финальный тест на отложенной выборке
    df_metrics, df_reports, _, probability_scores = evaluate_test_set(
        X_test, y_test, best_models
    )
    df_metrics.to_csv(RESULTS_DIR / "test_metrics.csv", index=False)
    df_reports.to_csv(RESULTS_DIR / "test_classification_reports.csv", index=False)
    
    # 6. Визуализация ROC-кривых
    plot_roc_curves(y_test, probability_scores, RESULTS_DIR / "roc_curves.png")
    print(f"\nExperiment complete. All outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
