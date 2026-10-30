from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import t

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, roc_auc_score, make_scorer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 10
INNER_SPLITS = 3

OUTPUT_DIR = Path("results/repeated_nested_cv")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# In sklearn's Breast Cancer dataset:
# 0 = malignant, 1 = benign
MALIGNANT_LABEL = 0


# --------------------------------------------------
# 2. Dataset
# --------------------------------------------------

# Load the breast cancer dataset and define the feature matrix and target vector.
data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target

print(f"Samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Malignant: {(y == 0).sum()}")
print(f"Benign: {(y == 1).sum()}")


# --------------------------------------------------
# 3. Scoring
# --------------------------------------------------

malignant_f1 = make_scorer(
    f1_score,
    pos_label=MALIGNANT_LABEL,
    zero_division=0,
)


def evaluate_model(model, X_test, y_test):
    """Evaluate predictions with malignant treated as the positive class."""

    y_pred = model.predict(X_test)

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=MALIGNANT_LABEL,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=MALIGNANT_LABEL,
        zero_division=0,
    )

    probabilities = model.predict_proba(X_test)
    malignant_index = list(model.classes_).index(MALIGNANT_LABEL)

    # Convert the target so malignant = 1 for ROC-AUC calculation.
    y_test_malignant = (y_test == MALIGNANT_LABEL).astype(int)
    malignant_scores = probabilities[:, malignant_index]

    auc = roc_auc_score(y_test_malignant, malignant_scores)

    return {
        "f1_malignant": f1,
        "recall_malignant": recall,
        "roc_auc_malignant": auc,
    }


# --------------------------------------------------
# 4. Models and search spaces
# --------------------------------------------------

models = {
    "Logistic Regression": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                random_state=RANDOM_STATE,
            )),
        ]),
        {
            "model__C": [0.01, 0.1, 1.0],
            "model__class_weight": [None, "balanced"],
        },
    ),

    "Decision Tree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "max_depth": [3, 5, None],
            "min_samples_leaf": [1, 2],
        },
    ),

    "Random Forest": (
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        {
            "n_estimators": [100],
            "max_depth": [None, 10],
            "min_samples_leaf": [1, 2],
        },
    ),

    "Gradient Boosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [1, 2],
        },
    ),
}


# --------------------------------------------------
# 5. Repeated nested cross-validation
# --------------------------------------------------

outer_cv = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE,
)

records = []
total_splits = N_SPLITS * N_REPEATS

for model_name, (estimator, param_grid) in models.items():
    print(f"\nEvaluating {model_name}...")

    for split_number, (train_idx, test_idx) in enumerate(
        outer_cv.split(X, y), start=1
    ):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner_cv = StratifiedKFold(
            n_splits=INNER_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE + split_number,
        )

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=malignant_f1,
            cv=inner_cv,
            refit=True,
            n_jobs=-1,
            error_score="raise",
        )

        # Hyperparameter tuning uses only the outer training data.
        search.fit(X_train, y_train)

        # The outer test fold is used strictly for evaluation.
        metrics = evaluate_model(search.best_estimator_, X_test, y_test)

        records.append({
            "model": model_name,
            "split": split_number,
            "repeat": (split_number - 1) // N_SPLITS + 1,
            **metrics,
            "best_inner_cv_f1": search.best_score_,
            "best_params": json.dumps(search.best_params_, default=str),
        })

        print(
            f"  Split {split_number:02d}/{total_splits}: "
            f"F1={metrics['f1_malignant']:.4f}, "
            f"Recall={metrics['recall_malignant']:.4f}, "
            f"AUC={metrics['roc_auc_malignant']:.4f}"
        )


# --------------------------------------------------
# 6. Save per-split results
# --------------------------------------------------

results = pd.DataFrame(records)
results.to_csv(
    OUTPUT_DIR / "repeated_cv_results.csv",
    index=False,
)


# --------------------------------------------------
# 7. Summary and approximate corrected 95% intervals
# --------------------------------------------------

def corrected_cv_interval(values, n_splits=N_SPLITS):
    """
    Approximate corrected repeated-CV interval.

    This correction accounts for dependence between overlapping training sets.
    It is not an exact confidence interval.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    mean = values.mean()
    sd = values.std(ddof=1)

    # Nadeau-Bengio-style correction:
    # test_size / train_size = 1 / (n_splits - 1)
    standard_error = sd * np.sqrt(
        (1 / n) + (1 / (n_splits - 1))
    )

    critical_value = t.ppf(0.975, df=n - 1)
    margin = critical_value * standard_error

    return mean, sd, max(0.0, mean - margin), min(1.0, mean + margin)


summary_records = []

metric_names = [
    "f1_malignant",
    "recall_malignant",
    "roc_auc_malignant",
]

for model_name, group in results.groupby("model", sort=False):
    row = {"model": model_name}

    for metric in metric_names:
        mean, sd, lower, upper = corrected_cv_interval(group[metric])

        row[f"{metric}_mean"] = mean
        row[f"{metric}_sd"] = sd
        row[f"{metric}_ci95_lower_approx"] = lower
        row[f"{metric}_ci95_upper_approx"] = upper

    summary_records.append(row)

summary = pd.DataFrame(summary_records)
summary.to_csv(OUTPUT_DIR / "repeated_cv_summary.csv", index=False)

print("\n=== Repeated Nested CV Summary ===")
print(summary.round(4).to_string(index=False))


# --------------------------------------------------
# 8. Visualize F1 distribution
# --------------------------------------------------

model_order = list(models.keys())
plot_data = [
    results.loc[results["model"] == name, "f1_malignant"].to_numpy()
    for name in model_order
]

fig, ax = plt.subplots(figsize=(10, 6))

ax.boxplot(plot_data, tick_labels=model_order, showmeans=True)
ax.set_title("Repeated Nested CV: Malignant F1-score")
ax.set_ylabel("F1-score")
ax.set_ylim(0, 1.02)
ax.tick_params(axis="x", rotation=15)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "f1_stability.png", dpi=200)
plt.close(fig)


# --------------------------------------------------
# 9. Visualize approximate corrected intervals
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

for position, row in summary.iterrows():
    mean = row["f1_malignant_mean"]
    lower = row["f1_malignant_ci95_lower_approx"]
    upper = row["f1_malignant_ci95_upper_approx"]

    ax.errorbar(
        mean,
        position,
        xerr=[[mean - lower], [upper - mean]],
        fmt="o",
        capsize=4,
    )

ax.set_yticks(range(len(summary)))
ax.set_yticklabels(summary["model"])
ax.invert_yaxis()
ax.set_xlim(0, 1.02)
ax.set_xlabel("Malignant F1-score")
ax.set_title("Mean F1 with Approximate Corrected 95% Intervals")
ax.grid(axis="x", alpha=0.3)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "f1_corrected_intervals.png", dpi=200)
plt.close(fig)

print(f"\nSaved results to: {OUTPUT_DIR.resolve()}")
