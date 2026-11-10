
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# -------------------------
# 1. Load data
# -------------------------
DATA_PATH = Path("default of credit card clients.xls")
TARGET = "default payment next month"

df = pd.read_excel(DATA_PATH, header=1)
df = df.drop(columns=["ID"])

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Categorical variables
categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numeric_features = [
    col for col in X.columns
    if col not in categorical_features
]

# -------------------------
# 2. Train/test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
print("Train default rate:", round(y_train.mean(), 4))
print("Test default rate:", round(y_test.mean(), 4))

# -------------------------
# 3. Preprocessing
# -------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features),
])

# -------------------------
# 4. Models
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        random_state=42,
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        random_state=42,
    ),
}

# -------------------------
# 5. Cross-validation
# -------------------------
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "recall": "recall",
}

results = []
fitted_models = {}

for name, model in models.items():
    print(f"\nTraining: {name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    cv_result = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    pipeline.fit(X_train, y_train)
    fitted_models[name] = pipeline

    # Positive class = default (1)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "model": name,
        "cv_roc_auc_mean": cv_result["test_roc_auc"].mean(),
        "cv_roc_auc_std": cv_result["test_roc_auc"].std(),
        "cv_pr_auc_mean": cv_result["test_pr_auc"].mean(),
        "cv_f1_mean": cv_result["test_f1"].mean(),
        "cv_recall_mean": cv_result["test_recall"].mean(),
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(
            y_test, y_pred, zero_division=0
        ),
        "test_recall": recall_score(
            y_test, y_pred, zero_division=0
        ),
        "test_f1": f1_score(
            y_test, y_pred, zero_division=0
        ),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
        "test_pr_auc": average_precision_score(y_test, y_proba),
    })

# -------------------------
# 6. Results
# -------------------------
results_df = pd.DataFrame(results).sort_values(
    "cv_pr_auc_mean",
    ascending=False,
)

print("\n=== MODEL COMPARISON ===")
print(results_df.round(4).to_string(index=False))

output_path = Path("reports")
output_path.mkdir(parents=True, exist_ok=True)

results_df.to_csv(
    output_path / "baseline_model_comparison.csv",
    index=False,
)

print("\nSaved: reports/baseline_model_comparison.csv")

