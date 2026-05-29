# Insurance Fraud Detection

A production-grade machine learning pipeline that detects fraudulent insurance claims using XGBoost with 94.1% recall and 97.9% ROC-AUC. The project includes a full preprocessing pipeline, model training with explainability analysis, and an interactive Streamlit web application for real-time predictions.

---

## Project Structure

```
Insurance_fraud_detection/
├── apps/
│   └── streamlit_app.py          # Interactive web UI for real-time fraud prediction
├── data/
│   ├── raw/                      # Original dataset (51,000 records × 43 columns)
│   └── processed/                # Cleaned train/test splits (71 features)
├── models/
│   ├── final_model.pkl           # Trained XGBoost classifier
│   ├── model_metadata.pkl        # Hyperparameters, metrics, feature list
│   ├── scaler.pkl                # StandardScaler for continuous features
│   ├── scaling_cols.pkl          # Columns subject to scaling
│   ├── feature_names.pkl         # Ordered list of all 71 features
│   └── encoding_maps.pkl         # Binary, ordinal, and nominal encoding maps
├── Notebooks/
│   ├── data_understanding.ipynb  # EDA: shape, distributions, missing values, outliers
│   ├── preprocessing.ipynb       # Cleaning, feature engineering, encoding, scaling
│   └── model_training.ipynb      # Model training, tuning, evaluation, SHAP analysis
├── reports/
│   ├── data_quality_report.md    # Dataset quality assessment
│   ├── model_evaluation_report.md# Model comparison and final metrics
│   ├── feature_importance.png    # XGBoost native feature importance chart
│   ├── shap_summary.png          # SHAP dot plot (impact and direction)
│   ├── shap_bar.png              # SHAP mean absolute values
│   └── shap_force_plot.png       # SHAP force plot for a single prediction
├── src/                          # Modular source scripts
├── requirements.txt
└── .gitignore
```

---

## Dataset

| Property | Value |
|---|---|
| Raw records | 51,000 |
| Raw features | 43 |
| Clean records | 44,491 |
| Final features (engineered) | 71 |
| Target | `fraud_reported` (0 = Genuine, 1 = Fraud) |
| Class imbalance | ~11.5 : 1 (genuine : fraud) |

---

## Pipeline Summary

### 1. Data Preprocessing (`Notebooks/preprocessing.ipynb`)
- Removed 5 data-leakage columns and 1,000 duplicate records
- Cleaned currency (`$`), percentage (`%`), and multi-format date columns
- Imputed missing values (median for numeric, mode for categorical)
- Dropped 5,509 rows with physically impossible date logic
- Applied IQR-based outlier capping on 5 financial columns
- Engineered **14 new features**: date-derived, financial ratios, and binary risk flags
- One-hot encoded 5 nominal columns; ordinal-encoded severity; binary-encoded Yes/No columns
- Applied `StandardScaler` to 33 continuous features (train fit, test transform)

### 2. Model Training (`Notebooks/model_training.ipynb`)
Three classifiers evaluated:
- Logistic Regression (baseline)
- Random Forest
- XGBoost (default → hyperparameter tuned via GridSearchCV)

Class imbalance handled via `scale_pos_weight = 11.51` for XGBoost and `class_weight='balanced'` for others.

### 3. Explainability
SHAP `TreeExplainer` used on 500 test samples to validate that the model learned legitimate fraud patterns (claim ratios, missing documentation, behavioural scores).

---

## Final Model Performance

**Model:** XGBoost — `{n_estimators: 300, max_depth: 4, learning_rate: 0.05}`

| Metric | Value |
|---|---|
| Accuracy | 93.61% |
| Precision | 55.94% |
| **Recall** | **94.09%** |
| F1-Score | 0.7016 |
| **ROC-AUC** | **0.9789** |
| PR-AUC | 0.7358 |
| CV ROC-AUC (5-fold) | 0.9791 ± 0.0028 |

**Why XGBoost (Tuned)?** Missing fraud costs ~100× more than a false alarm. The tuned model catches 669 out of 711 fraud cases (94.1%) vs Random Forest's 528 (74.3%) — a difference of ₹66+ Lakh in estimated business cost per evaluation cycle.

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/insurance-fraud-detection.git
cd insurance-fraud-detection

# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Web Application

```bash
streamlit run apps/streamlit_app.py
```

The app will launch at `http://localhost:8501`. Enter claim details across 4 tabs and click **Analyze Claim** to get:
- Fraud probability (0–100%)
- Risk category (Low / Medium / High)
- Decision (Genuine / Suspect Fraud)
- Key risk factors identified
- Actionable recommendations

### Re-run the Full Pipeline

Open and run the notebooks in order:

```
1. Notebooks/data_understanding.ipynb   → EDA and quality checks
2. Notebooks/preprocessing.ipynb        → Cleaning and feature engineering
3. Notebooks/model_training.ipynb       → Training, tuning, and evaluation
```

All model artefacts are saved to `models/` automatically.

---

## Key Features Engineered

| Feature | Description |
|---|---|
| `claim_repair_ratio` | Claim amount / repair estimate |
| `claim_coverage_ratio` | Claim amount / coverage amount |
| `claim_delay_days` | Days from incident to claim filing |
| `policy_age_days` | Days from policy start to incident |
| `total_risk_flags` | Sum of 10 binary fraud risk indicators |
| `no_police_report` | 1 if no police report filed |
| `late_filing` | 1 if claim filed >30 days after incident |
| `odd_hour_incident` | 1 if incident reported between 00:00–05:00 |
| `frequent_claimer` | 1 if >2 claims in the past 12 months |
| `new_customer` | 1 if policy tenure < 1 year |

---

## Reports

- [Data Quality Report](reports/data_quality_report.md) — full dataset assessment including missing values, outliers, leakage, and cleaning steps
- [Model Evaluation Report](reports/model_evaluation_report.md) — comparative model metrics, confusion matrix, feature importance, and SHAP analysis

---

## Tech Stack

| Layer | Library |
|---|---|
| Data processing | pandas, numpy |
| Machine learning | scikit-learn, xgboost |
| Explainability | shap |
| Visualisation | matplotlib, seaborn |
| Web application | streamlit |
| Model persistence | joblib |
