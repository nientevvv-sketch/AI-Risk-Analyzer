# AI Risk Analyzer

AI Risk Analyzer is a machine learning project for assessing the likelihood of a company being in a high-risk financial state using structured financial indicators. The project includes a training pipeline, a web application, and supporting model metadata for evaluating corporate distress risk.

## Overview

This repository builds and serves a predictive risk model that evaluates company financial fundamentals such as leverage, profitability, liquidity, and balance-sheet indicators. It is designed for research-oriented financial analysis and decision support, not for regulated credit underwriting.

The project currently includes:

- A training script to build a Random Forest classification model
- A Streamlit-based app for entering company indicators and receiving a risk assessment
- Metadata describing model features, thresholds, and target class configuration
- Example data files and evaluation outputs

## Key features

- Automated model training with scikit-learn
- Median imputation for missing numeric values
- Feature schema validation between the model and app metadata
- Adjustable probability threshold in the UI
- Interactive dashboard for risk analysis using Streamlit

## Model Comparison

Four machine learning algorithms are evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Models are compared using a consistent evaluation pipeline and stratified data splitting.

## Evaluation Metrics

The project evaluates financial bankruptcy prediction using:

- ROC-AUC — ability to distinguish between bankrupt and non-bankrupt companies.
- PR-AUC — precision-recall performance, particularly important for imbalanced data.
- Precision — proportion of correctly identified bankruptcy cases among positive predictions.
- Recall — proportion of actual bankruptcy cases correctly identified.
- F1-score — harmonic mean of precision and recall.
- Confusion Matrix — breakdown of true positives, true negatives, false positives, and false negatives.

## Model Evaluation and Risk Analysis

The evaluation focuses on identifying financially distressed companies while accounting for class imbalance and the different consequences of classification errors.

Actual experimental results are reported in the project's results directory.

## Project structure

```text
AI-Risk-Analyzer/
├── app/
│   └── streamlit_app.py
├── data/
│   ├── data.csv
│   ├── Insurance_Brokers_Peer_Ultimate_Model_audit_v4.xlsx
│   └── WTW_Clean_Financial_Model.xlsx
├── models/
│   ├── model_metadata.json
│   ├── model_metadata.py
│   ├── risk_model.py
│   └── risk_model.joblib
├── notebooks/
├── results/
├── src/
│   └── train_risk_model.py
├── .gitignore
├── README.md
└── LICENSE
```

## Data and modeling

The model is trained from `data/data.csv`, where the target column is `Bankrupt?`.

Training flow:

1. Load the dataset
2. Validate the binary target label
3. Split into train/test data
4. Train a `RandomForestClassifier` inside a preprocessing pipeline
5. Save the trained model to `models/risk_model.joblib`
6. Save feature metadata to `models/model_metadata.json`

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/nientevvv-sketch/AI-Risk-Analyzer.git
cd AI-Risk-Analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit pandas scikit-learn joblib
```

## Train the model

```bash
python -m src.train_risk_model
```

This produces the model artifact and metadata used by the app.

## Run the app

```bash
streamlit run app/streamlit_app.py
```

Then open the local URL shown by Streamlit in your browser.

## Using the app

The app displays a form with the financial indicators used during training. Once values are entered, the model evaluates the probability of the configured positive class and compares it against the selected decision threshold.

The user can also inspect:

- submitted indicator values
- predicted probability
- decision threshold
- interpretation and model limitations

## Model metadata and feature validation

The app checks that the trained model feature order matches the schema saved in `models/model_metadata.json`. This helps ensure that UI inputs align with the model's expected predictors.

## Notes and limitations

- This project is a research-oriented financial risk model and is not a certified credit or underwriting system.
- Prediction quality depends on data quality, feature engineering, and the representativeness of the training data.
- The probability output should be interpreted carefully and is not a guarantee of future business performance.

## Repository responsibilities

- `app/` contains the Streamlit user interface
- `src/` contains the training code
- `models/` contains the trained model and model metadata
- `data/` contains datasets and documentation
- `results/` stores outputs such as metrics and charts
- `notebooks/` is available for exploratory analysis and experimentation

## License

This repository does not currently include a license file. If you plan to share or distribute the project publicly, consider adding an appropriate open-source license such as MIT or Apache 2.0.

## Contributing

Contributions and improvements are welcome. Common areas for enhancement include:

- stronger preprocessing and feature engineering
- validation and calibration analysis
- better error handling and deployment setup
- additional model comparisons and evaluation reports

## Authors

- **[Nikita Nassyrov]** — ML development, model evaluation, data analysis.
- **[Dyussembek Bauyrzhan]** — Financial analysis, Excel data preparation, industry research.

GitHub: [@nientevvv-sketch](https://github.com/nientevvv-sketch) [@airanplayhd-dotcom](https://github.com/airanplayhd-dotcom)
