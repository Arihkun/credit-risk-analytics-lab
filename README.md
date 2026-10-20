# Credit Risk Analytics Lab

> Independent public-data machine learning benchmark, probability of default (PD) estimation, and risk decile discrimination pipeline.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)

---

## 1. Overview & Problem Framing

This repository provides an end-to-end, reproducible credit default prediction pipeline built on standard quantitative finance methodologies.

In retail multi-finance and banking operations, credit scorecards evaluate the probability that a borrower fails to meet payment obligations over a predefined observation window (12-month default definition). This project benchmark trains, evaluates, and exports a production-grade Gradient Boosting credit scoring model using an independent public credit card cohort.

---

## 2. Benchmark Dataset

- **Source**: UCI Machine Learning Repository — *Default of Credit Card Clients Dataset*
- **Sample Size**: 30,000 borrower profiles
- **Holdout Evaluation**: Stratified 80/20 train/test split (24,000 train rows / 6,000 holdout rows; 1,328 default events in holdout)
- **Features Used**:
  - `LIMIT_BAL`: Credit limit extended to borrower
  - `PAY_0` to `PAY_6`: Historical monthly repayment statuses (-1 = pay duly, 1 = delay 1 month, 2 = delay 2 months, etc.)
  - `BILL_AMT1` to `BILL_AMT6`: Past bill statement amounts
  - `PAY_AMT1` to `PAY_AMT6`: Past repayment amounts

---

## 3. Benchmark Results & Discrimination Metrics

Evaluated on the 6,000-row holdout dataset without data leakage:

| Model Architecture | ROC-AUC | PR-AUC | Brier Score | Top Decile Capture |
| :--- | :---: | :---: | :---: | :---: |
| **Prevalence Baseline (Prior)** | 0.5000 | 0.2213 | 0.1723 | 10.00% |
| **Standardized Logistic Regression** | 0.7214 | 0.4682 | 0.1472 | 24.12% |
| **Histogram Gradient Boosting (LightGBM-style)** | **0.7773** | **0.5518** | **0.1354** | **31.48%** |

### Key Risk Insights:
- **ROC-AUC (0.7773)**: Strong ranking discrimination across borrower credit tiers.
- **Top 10% Decile Capture (31.48%)**: The top 10% highest-risk applicants identified by the model capture nearly one-third of all portfolio defaults, enabling aggressive risk cutoffs without rejecting prime applicants.
- **Brier Score (0.1354)**: Well-calibrated default probabilities suitable for pricing and expected loss (EL) calculations.

---

## 4. Quickstart: Installation & Training

### Prerequisites
- Python 3.10 or higher
- Git

### 1. Clone & Setup
```bash
git clone https://github.com/Arihkun/credit-risk-analytics-lab.git
cd credit-risk-analytics-lab
pip install -r requirements.txt
```

### 2. Train Model
```bash
python train.py
```
*The script automatically downloads the UCI benchmark dataset, runs feature preparation, trains the gradient boosting estimator, computes evaluation metrics, and saves the trained model to `models/credit_scoring_model.joblib`.*

### 3. Run Inference & Risk Tier Scoring
```bash
python evaluate.py
```

Sample output:
```text
[Case 1] Evaluating Prime Applicant Profile:
  * Default Probability : 8.40%
  * Risk Category       : Tier A (Very Low Risk)
  * System Decision     : AUTO_APPROVE
  * Policy Rationale    : Prime risk tier with low default probability.

[Case 2] Evaluating Delinquent / Subprime Applicant Profile:
  * Default Probability : 43.12%
  * Risk Category       : Tier D (High Risk)
  * System Decision     : DECLINE
  * Policy Rationale    : Default probability exceeds prudent risk limits.
```

---

## 5. Repository Structure

```text
credit-risk-analytics-lab/
├── train.py                  # End-to-end model training & evaluation pipeline
├── evaluate.py               # Inference script for scoring applicant profiles
├── requirements.txt          # Python dependencies
├── .gitignore
├── README.md                 # Technical specification and documentation
└── models/                   # Generated upon training
    ├── credit_scoring_model.joblib
    └── benchmark_metrics.json
```

---

## 6. Author

**Ari Hendrawan**  
Risk Development & Decision Automation  
GitHub: [@Arihkun](https://github.com/Arihkun)  
LinkedIn: [Ari Hendrawan](https://www.linkedin.com/in/arihendra2802)  
Email: [arihendra2802@gmail.com](mailto:arihendra2802@gmail.com)

---

## 7. License

This repository is distributed under the [MIT License](LICENSE). Educational benchmark based on public academic research data.
