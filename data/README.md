# AI Risk Analyzer
### Financial Risk Prediction, Model Explainability & Decision Analysis

AI Risk Analyzer is a machine learning project focused on financial risk assessment. It explores how predictive models can identify companies or clients at elevated risk and how their predictions can support data-driven decisions.

The project combines machine learning, financial analysis, model evaluation, and risk-based decision-making in one reproducible workflow.

> Project status: In development. Model results, visualizations, and application features will be added as they are implemented and validated.

## Objectives

- Build a reproducible machine learning pipeline for financial risk prediction.
- Compare baseline and tree-based classification models.
- Evaluate models using metrics appropriate for imbalanced datasets.
- Analyze false positives, false negatives, and the consequences of incorrect predictions.
- Investigate probability calibration and decision thresholds.
- Explain model predictions and document the limitations of automated risk assessment.

## Problem Statement

Financial risk assessment is not simply a classification problem. In many applications, failing to identify a high-risk case may have a different cost from incorrectly flagging a low-risk case.

This project investigates three main questions:

1. How accurately can machine learning models identify high-risk cases?
2. How reliable are their predicted probabilities?
3. How do different decision thresholds affect the trade-off between missed risks and false alarms?

The exact target variable and interpretation of risk depend on the selected dataset.

## Dataset

The dataset will be documented before model training, including its source, target definition, feature descriptions, class distribution, and known limitations.

Potential dataset directions include:

- Company bankruptcy prediction: financial ratios and company-level indicators.
- Credit default prediction: borrower characteristics and repayment history.

These are different prediction tasks and will not be treated as interchangeable.

Dataset source: To be added after the final dataset is selected.

### Data Preparation

The pipeline may include:

- Missing-value analysis and imputation.
- Duplicate and data-quality checks.
- Target and feature validation.
- Stratified train/test splitting.
- Numerical feature scaling where appropriate.
- Categorical feature encoding where required.
- Leakage prevention by fitting preprocessing only on training data.

## Models

The planned baseline comparison includes:

| Model | Purpose |
|---|---|
| Logistic Regression | Interpretable baseline |
| Decision Tree | Simple nonlinear decision rules |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Boosting-based classification |

Model selection and hyperparameter tuning will be performed using cross-validation on the training data. The test set will be reserved for final evaluation.

## Evaluation Framework

Because financial risk datasets can be imbalanced, accuracy alone is insufficient.

| Metric | Purpose |
|---|---|
| ROC-AUC | Ranking ability across classification thresholds |
| PR-AUC / Average Precision | Performance focused on the positive class |
| Precision | Proportion of flagged cases that are positive |
| Recall | Proportion of positive cases identified |
| F1-score | Balance between precision and recall |
| Confusion Matrix | Counts of correct and incorrect classifications |
| Brier Score | Quality of probabilistic predictions |
| Log Loss | Penalty for inaccurate predicted probabilities |

Additional analyses, where implemented, will include:

- Cross-validation stability.
- Probability calibration curves.
- Decision-threshold comparison.
- False-positive and false-negative analysis.
- Training and inference time.
- Performance across relevant subgroups, when supported by the data.

## Risk-Based Decision Analysis

A model with the highest ROC-AUC is not necessarily the best model for every financial risk application.

The project will examine how the choice of threshold changes the number of missed risks and false alarms. If defensible cost assumptions are available, a cost-sensitive analysis may compare the consequences of different decisions.

Any assumed costs will be explicitly labelled as assumptions rather than observed financial losses.

## Explainability

The explainability component is intended to help users understand the factors associated with model predictions.

Potential methods include:

- Model coefficients for Logistic Regression.
- Feature importance for tree-based models.
- SHAP explanations, if appropriate for the selected models and dataset.
- Individual prediction explanations, where supported.

Feature importance and SHAP values describe model behaviour; they do not establish causation.

## Financial Analysis

A separate financial-analysis component may compare insurance brokerage companies such as WTW, Aon, Marsh McLennan, Arthur J. Gallagher, and Brown & Brown.

This analysis is intended to provide business context through financial metrics, ratios, peer comparisons, and clearly labelled stress scenarios.

Important: Company-level financial analysis will remain separate from the model-training dataset unless the data can be reliably linked and aligned with the prediction target and time period. Illustrative stress scenarios will not be presented as historical facts or model predictions.

## Technology Stack

- Language: Python
- Data analysis: pandas, NumPy
- Machine learning: scikit-learn
- Visualization: Matplotlib, Seaborn, Plotly
- Explainability: SHAP, if implemented
- Application: Streamlit, planned
- Version control: Git and GitHub
- Testing: pytest, planned

## Planned Project Structure

```text
AI-Risk-Analyzer/
├── app/
│   └── streamlit_app.py
├── configs/
│   └── config.yaml
├── data/
│   └── README.md
├── notebooks/
│   └── 01_eda.ipynb
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── calibration.py
│   ├── explain.py
│   └── decision_analysis.py
├── tests/
│   ├── test_data.py
│   └── test_pipeline.py
├── reports/
│   └── figures/
├── models/
├── requirements.txt
├── README.md
└── .gitignore
```

This is a planned structure. Files and features will be marked as implemented once they exist and have been tested.

## Getting Started

### 1. Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd AI-Risk-Analyzer
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset

Follow the instructions in `data/README.md`. Do not commit sensitive data, restricted datasets, or large raw files unless their licences permit redistribution.

### 5. Run the pipeline

Training and evaluation commands will be documented here after the corresponding scripts are implemented.

## Reproducibility

The project aims to:

- Fix random seeds where applicable.
- Separate training, validation, and test data.
- Keep preprocessing inside the training pipeline.
- Record model parameters and evaluation metrics.
- Save plots and reports generated by the experiment.
- Document dataset versions and assumptions.

## Limitations

- Model performance depends on the quality and representativeness of the selected dataset.
- Historical patterns may not generalize to future economic conditions.
- Class imbalance can make accuracy misleading.
- Predicted probabilities may require calibration.
- Financial risk predictions should support, not replace, professional judgment.
- The model's outputs should not be interpreted as guaranteed outcomes or personalized financial advice.

## Roadmap

- [ ] Select and document the final dataset.
- [ ] Perform exploratory data analysis.
- [ ] Build a reproducible preprocessing pipeline.
- [ ] Train and compare baseline models.
- [ ] Add cross-validation and hyperparameter tuning.
- [ ] Evaluate class imbalance and classification thresholds.
- [ ] Perform error analysis and stability checks.
- [ ] Evaluate probability calibration.
- [ ] Add model explainability.
- [ ] Develop a Streamlit dashboard.
- [ ] Add automated tests and usage documentation.

## Authors

Nientevvv-sketch

GitHub: [github.com/nientevvv-sketch](https://github.com/nientevvv-sketch)

airanplayhd-dotcom
GitHub: [github.com/airanplayhd-dotcom](https://github.com/airanplayhd-dotcom)

---

This project is developed for educational and portfolio purposes. Reported results will be added only after the corresponding experiments have been executed and validated.
