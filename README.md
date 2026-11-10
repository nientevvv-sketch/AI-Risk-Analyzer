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

The financial analysis component focuses on corporate financial performance and comparative industry analysis.

### WTW Financial Analysis

The WTW financial model supports analysis of:

- Revenue trends
- Profitability
- Operating performance
- Cash flow
- Financial ratios
- Historical financial performance

### Insurance Brokerage Peer Benchmarking

The peer benchmarking component examines selected insurance brokerage companies:

- Willis Towers Watson (WTW)
- Aon
- Marsh McLennan (MMC)
- Arthur J. Gallagher (AJG)
- Brown & Brown (BRO)

The analysis focuses on financial ratios, comparative performance, and illustrative financial stress scenarios.

### Stress Testing

The project includes scenario-based financial stress testing to explore how changes in financial assumptions may affect selected financial indicators.

Stress scenarios are illustrative analytical exercises and should not be interpreted as forecasts.

---

## Interactive Application

The Streamlit application provides an interface for interacting with the trained risk model
