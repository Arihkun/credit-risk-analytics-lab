"""Credit Risk Analytics Lab: Model Training & Benchmark Pipeline

Author: Ari Hendrawan (https://github.com/Arihkun)
Dataset: UCI Default of Credit Card Clients (30,000 observations)
Benchmark Task: 12-Month Credit Card Default Probability Prediction
"""

import json
import os
from io import BytesIO
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_URLS = [
    "https://azuremlexamples.blob.core.windows.net/datasets/credit_card/default_of_credit_card_clients.csv",
    "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip",
]
SEED = 42

FEATURES = [
    "LIMIT_BAL",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


def load_dataset():
    print("[1/5] Fetching UCI Credit Card Default benchmark dataset...")
    # Try Azure ML high-speed CDN first
    try:
        print("      Downloading from high-speed mirror (Azure CDN)...")
        frame = pd.read_csv(DATA_URLS[0], header=1)
        frame.columns = frame.columns.map(str)
        target_col = "default payment next month" if "default payment next month" in frame else "Y"
        x = frame[FEATURES].apply(pd.to_numeric)
        y = frame[target_col].astype(int)
        print(f"      Loaded {len(frame)} total rows ({y.sum()} positive defaults, default rate: {y.mean():.2%}).")
        return x, y
    except Exception as e:
        print(f"      Primary CDN fallback triggered: {e}")

    # Fallback to direct UCI zip
    print("      Connecting to secondary UCI archive...")
    response = requests.get(DATA_URLS[1], timeout=60)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as archive:
        workbook = next(name for name in archive.namelist() if name.lower().endswith(".xls"))
        frame = pd.read_excel(BytesIO(archive.read(workbook)), header=1)

    frame.columns = frame.columns.map(str)
    target_col = "Y" if "Y" in frame else "default payment next month"

    x = frame[FEATURES].apply(pd.to_numeric)
    y = frame[target_col].astype(int)

    print(f"      Loaded {len(frame)} total rows ({y.sum()} positive defaults, default rate: {y.mean():.2%}).")
    return x, y


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    print(f"      Training {name}...")
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]

    # Top decile default capture calculation
    top_decile_size = len(probabilities) // 10
    selected_indices = np.argsort(-probabilities, kind="stable")[:top_decile_size]
    captured_defaults = y_test.to_numpy()[selected_indices].sum()
    total_defaults = y_test.sum()
    top_decile_capture = float(captured_defaults / total_defaults)

    metrics = {
        "model": name,
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probabilities)), 4),
        "top_decile_capture_pct": round(top_decile_capture * 100, 2),
    }
    return metrics, model


def main():
    print("=" * 65)
    print("CREDIT RISK ANALYTICS LAB - MODEL TRAINING PIPELINE")
    print("Author: Ari Hendrawan | https://github.com/Arihkun")
    print("=" * 65)

    x, y = load_dataset()

    print("[2/5] Creating stratified 80/20 train-test split (seed=42)...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"      Train set: {len(x_train)} samples | Holdout test set: {len(x_test)} samples")

    print("[3/5] Evaluating benchmark models across risk metrics...")
    candidates = [
        ("Prevalence Baseline (Prior)", DummyClassifier(strategy="prior")),
        ("Standardized Logistic Regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("Histogram Gradient Boosting (LightGBM-style)", HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, learning_rate=0.05, random_state=SEED
        )),
    ]

    results = []
    best_model = None

    for name, candidate in candidates:
        metrics, trained_model = evaluate_model(name, candidate, x_train, x_test, y_train, y_test)
        results.append(metrics)
        if "Gradient Boosting" in name:
            best_model = trained_model

    print("\n[4/5] Benchmark Performance Summary:")
    print("-" * 65)
    for r in results:
        print(f"  * {r['model']:<36} | ROC-AUC: {r['roc_auc']:.4f} | PR-AUC: {r['pr_auc']:.4f} | Top 10% Capture: {r['top_decile_capture_pct']}%")
    print("-" * 65)

    print("\n[5/5] Exporting production artifacts...")
    os.makedirs("models", exist_ok=True)

    model_path = os.path.join("models", "credit_scoring_model.joblib")
    metrics_path = os.path.join("models", "benchmark_metrics.json")

    joblib.dump({
        "model": best_model,
        "features": FEATURES,
        "metrics": results[-1],
    }, model_path)
    print(f"      Saved model artifact -> {model_path}")

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"      Saved benchmark metrics -> {metrics_path}")

    print("\nPipeline completed successfully! Ready for inference with evaluate.py.")
    print("=" * 65)


if __name__ == "__main__":
    main()
