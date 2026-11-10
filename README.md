# AI Risk Analyzer

**Machine Learning · Financial Risk Analytics · Explainable Risk Assessment**

AI Risk Analyzer is a machine learning and financial analytics project focused on corporate bankruptcy prediction, financial risk assessment, and comparative financial analysis.

The project combines predictive modeling, model evaluation, financial benchmarking, and an interactive Streamlit application to explore how machine learning can support financial risk analysis.

> **Disclaimer:** This project is intended for research, educational purposes, and analytical decision support. It is not a certified credit rating, investment recommendation, or regulated underwriting system.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Machine Learning Methodology](#machine-learning-methodology)
- [Model Comparison](#model-comparison)
- [Evaluation Metrics](#evaluation-metrics)
- [Financial Analysis](#financial-analysis)
- [Interactive Application](#interactive-application)
- [Installation](#installation)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)
- [Model Metadata](#model-metadata)
- [Validation and Reliability Roadmap](#validation-and-reliability-roadmap)
- [Limitations](#limitations)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

AI Risk Analyzer explores the application of machine learning to corporate financial distress analysis.

The project consists of two complementary analytical components:

**1. Predictive Risk Modeling**

A classification pipeline trained on structured financial indicators to estimate bankruptcy-related risk.

**2. Financial Analysis and Benchmarking**

Exploratory financial analysis of Willis Towers Watson (WTW) and selected insurance brokerage companies, including financial ratio benchmarking and illustrative stress-testing scenarios.

These components serve different purposes: the bankruptcy dataset supports predictive modeling, while the financial datasets support descriptive analysis and scenario-based evaluation.

---

## Key Features

- Automated machine learning training pipeline
- Random Forest-based bankruptcy classification
- Missing-value imputation using median values
- Comparative evaluation of four machine learning algorithms
- Stratified train/test splitting
- ROC-AUC, PR-AUC, precision, recall, and F1-score evaluation
- Confusion matrices and error analysis
- Nested cross-validation and repeated validation experiments
- Model metadata generation and feature schema validation
- Interactive Streamlit risk assessment interface
- Adjustable probability threshold
- Financial ratio benchmarking and stress-testing analysis

---

## Project Architecture

```text
AI-Risk-Analyzer/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── data.csv
│   ├── Insurance_Brokers_Peer_Ultimate_Model_audit_v4.xlsx
│   └── WTW_Clean_Financial_Model.xlsx
│
├── models/
│   ├── model_metadata.json
│   ├── model_metadata.py
│   ├── risk_model.py
│   └── risk_model.joblib
│
├── notebooks/
│
├── results/
│
├── src/
│   └── train_risk_model.py
│
├── .gitignore
├── README.md
└── LICENSE
```

---

## Dataset

### Bankruptcy Prediction Dataset

The predictive model uses `data/data.csv`.

- **Target variable:** `Bankrupt?`
- **Input features:** Structured financial indicators
- **Task:** Binary classification
- **Objective:** Identify companies associated with the bankruptcy class

The dataset is used for model development, training, and evaluation.

### Financial Analysis Datasets

The repository also contains financial analysis workbooks:

| Dataset | Purpose |
|---|---|
| `WTW_Clean_Financial_Model.xlsx` | Financial statement analysis and historical performance |
| `Insurance_Brokers_Peer_Ultimate_Model_audit_v4.xlsx` | Comparative financial analysis of selected insurance brokerage companies |

These financial workbooks are used for exploratory analysis, financial ratio benchmarking, and illustrative stress-testing scenarios. They are not the bankruptcy model's labeled training dataset.

---

## Machine Learning Methodology

### Training Pipeline

The primary training pipeline follows these steps:

1. Load the bankruptcy dataset.
2. Validate the target variable.
3. Separate features and target.
4. Split the dataset into stratified training and test subsets.
5. Apply median imputation within a scikit-learn pipeline.
6. Train a Random Forest classifier.
7. Save the trained model.
8. Generate model metadata for application integration.

### Primary Model

The current training implementation uses:

- **Algorithm:** Random Forest Classifier
- **Number of estimators:** 200
- **Class weighting:** `balanced_subsample`
- **Random state:** 42
- **Missing-value handling:** Median imputation

The trained model is saved to:

```text
models/risk_model.joblib
```

---

## Model Comparison

The project evaluates four machine learning algorithms:

| Model | Description |
|---|---|
| Logistic Regression | Linear classification baseline |
| Decision Tree | Interpretable tree-based classifier |
| Random Forest | Ensemble of decision trees |
| Gradient Boosting | Sequential boosting-based classifier |

The comparison explores predictive performance, classification errors, and model stability.

Experimental outputs are available in the `results/` directory.

---

## Evaluation Metrics

The project evaluates bankruptcy classification using multiple metrics.

| Metric | Purpose |
|---|---|
| ROC-AUC | Measures discrimination between classes |
| PR-AUC | Evaluates precision-recall performance |
| Precision | Measures the proportion of predicted bankruptcy cases that are correct |
| Recall | Measures the proportion of actual bankruptcy cases identified |
| F1-score | Balances precision and recall |
| Confusion Matrix | Provides a breakdown of classification outcomes |
| Cross-validation scores | Evaluate performance across multiple validation folds |

Additional experiments investigate model stability, classification errors, and sensitivity to model configuration.

### Experimental Results

The repository contains evaluation artifacts such as:

- `test_metrics.csv`
- `final_comparison.csv`
- `classification_reports.csv`
- `nested_cv_scores.csv`
- `nested_cv_summary.csv`
- `repeated_cv_summary.csv`
- `misclassified_samples.csv`

These files provide detailed experimental results and supporting analysis.

---

## Financial Analysis

The financial analysis component focuses on corporate financial performance, historical trends, and comparative benchmarking across major insurance brokerage companies.

### WTW Financial Analysis

The WTW financial model supports analysis of:

- Revenue trends
- Profitability
- Operating performance
- Cash flow
- Financial ratios
- Historical financial performance

### Insurance Brokerage Peer Benchmarking

In addition to WTW, the project examines other major publicly traded insurance brokerage companies to provide a broader industry perspective.

Companies included in the comparative analysis:

- Willis Towers Watson (WTW)
- Aon (AON)
- Marsh McLennan (MMC)
- Arthur J. Gallagher (AJG)
- Brown & Brown (BRO)

The comparative analysis focuses on:

- Revenue and profitability comparisons
- Operating margins
- Financial performance trends
- Capital structure and leverage
- Cash flow indicators
- Financial ratios and peer benchmarking

### Stress Testing

The project also explores illustrative financial stress scenarios to examine how changes in selected financial assumptions may affect company-level financial indicators.

These scenarios are analytical exercises based on defined assumptions and should not be interpreted as forecasts or official credit-risk assessments.

---

## Interactive Application

The Streamlit application provides an interface for interacting with the trained risk model.

### Main Functionality

- Input financial indicators
- Validate input features against model metadata
- Generate a model-based risk assessment
- Display predicted probability
- Adjust the decision threshold
- Interpret the classification result
- Review model limitations

The application is designed to demonstrate how a trained classification model can be integrated into an interactive financial risk analysis workflow.

---

## Installation

### Requirements

- Python 3.11 or compatible version
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/nientevvv-sketch/AI-Risk-Analyzer.git
cd AI-Risk-Analyzer
```

If the repository folder already exists on your computer, skip the cloning step and navigate to the existing project directory.

### 2. Create a Virtual Environment

**Windows (PowerShell or CMD):**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install streamlit pandas scikit-learn joblib matplotlib openpyxl
```

---

## Usage

### Train the Model

```bash
python -m src.train_risk_model
```

The training script generates the trained model and metadata required by the application.

### Run the Application

```bash
python -m streamlit run app/streamlit_app.py
```

Open the local URL displayed by Streamlit in your browser.

### Using the Application

1. Enter the required financial indicators.
2. Submit the inputs for evaluation.
3. Review the predicted probability.
4. Compare the probability with the selected decision threshold.
5. Interpret the output in the context of the model's limitations.

---

## Troubleshooting

### Training Data Not Found

If you encounter the following error:

```text
FileNotFoundError: Training data not found:
C:\Users\hw\AI-Risk-Analyzer\data.csv
```

This means the training script is looking for the dataset in the project root directory instead of the `data/` folder.

**Step 1. Open the training script**

In the terminal, run:

```powershell
notepad src/train_risk_model.py
```

**Step 2. Find the DATA_PATH variable**

Locate this line:

```python
DATA_PATH = ROOT_DIR / "data.csv"
```

**Step 3. Correct the dataset path**

Add the `data` folder to the path.

Replace:

```python
DATA_PATH = ROOT_DIR / "data.csv"
```

With:

```python
DATA_PATH = ROOT_DIR / "data" / "data.csv"
```

**Step 4. Save the changes**

Press:

```text
Ctrl + S
```

Then close Notepad.

**Step 5. Verify the dataset location**

Run:

```powershell
dir data
```

Make sure `data.csv` exists inside the `data/` directory.

**Step 6. Run the training script again**

```powershell
python -m src.train_risk_model
```

The script should now find the dataset using the corrected path.

### Module Not Found

If you encounter an error such as:

```text
ModuleNotFoundError: No module named 'joblib'
```

Make sure your virtual environment is activated and install the dependencies:

```bash
python -m pip install streamlit pandas scikit-learn joblib matplotlib openpyxl
```

### Streamlit Command Not Found

If the `streamlit` command is not recognized, run:

```bash
python -m streamlit run app/streamlit_app.py
```

This runs Streamlit through the active Python environment.

---

## Model Metadata

The application uses metadata stored in:

```text
models/model_metadata.json
```

The metadata supports feature schema validation and alignment between application inputs and model predictors.

The application checks whether the feature order matches the expected model schema, helping reduce inconsistencies during inference.

---

## Validation and Reliability Roadmap

Future improvements focus on strengthening model reliability and analytical transparency.

| Area | Planned Improvement |
|---|---|
| Data Leakage Audit | Review feature availability, preprocessing, and evaluation integrity |
| Company-Level Validation | Investigate validation strategies for unseen companies |
| Probability Calibration | Evaluate probability reliability using calibration curves and Brier score |
| Threshold Optimization | Select decision thresholds using validation data and asymmetric error costs |
| Model Explainability | Implement SHAP or permutation importance |
| Robustness Analysis | Investigate sensitivity to input perturbations and distribution shifts |

These improvements aim to strengthen the reliability, interpretability, and practical applicability of the predictive model.

---

## Limitations

- Predictive performance depends on the quality and representativeness of the training dataset.
- Historical bankruptcy patterns may not generalize to other industries, markets, or economic conditions.
- Model outputs are statistical predictions and should not be interpreted as guarantees of future business performance.
- Probability estimates require appropriate calibration before being interpreted as reliable real-world probabilities.
- Decision thresholds involve trade-offs between false positives and false negatives.
- Financial stress-testing scenarios are illustrative and depend on the assumptions used.
- The project is not a substitute for professional financial analysis or regulated credit-risk systems.

---

## Contributors

**Nikita Nassyrov**  
Machine Learning Development, Data Science, Financial Risk Analytics

GitHub: [@nientevvv-sketch](https://github.com/nientevvv-sketch)

**Dyussembek Bauyrzhan**  
Financial Analysis, Excel Data Preparation, Industry Research

GitHub: [@airanplayhd-dotcom](https://github.com/airanplayhd-dotcom)

---

## License

The repository currently does not include a confirmed open-source license.

The licensing terms should be established before redistribution or commercial use. Dataset licensing and third-party financial data restrictions should also be respected.
