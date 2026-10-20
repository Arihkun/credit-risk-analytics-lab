"""Credit Risk Analytics Lab: Inference & Scoreband Evaluator

Author: Ari Hendrawan (https://github.com/Arihkun)
Loads the trained credit scoring model and evaluates applicant risk profiles.
"""

import os
import sys
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "credit_scoring_model.joblib")


def get_risk_tier(prob):
    """Categorizes credit applicant into standardized multi-finance score bands."""
    if prob < 0.10:
        return "Tier A (Very Low Risk)", "AUTO_APPROVE", "Prime risk tier with low default probability."
    elif prob < 0.22:
        return "Tier B (Moderate Risk)", "STANDARD_APPROVE", "Standard multi-finance threshold; acceptable risk profile."
    elif prob < 0.38:
        return "Tier C (Elevated Risk)", "MANUAL_REVIEW", "Elevated repayment risk; requires credit analyst review."
    else:
        return "Tier D (High Risk)", "DECLINE", "Default probability exceeds prudent risk limits."


def evaluate_applicant(data_dict):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model artifact not found at {MODEL_PATH}.")
        print("Please run `python train.py` first to train and export the model.")
        sys.exit(1)

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    features = artifact["features"]

    df = pd.DataFrame([data_dict])[features]
    prob = float(model.predict_proba(df)[0, 1])
    tier, decision, reason = get_risk_tier(prob)

    return {
        "default_probability": round(prob, 4),
        "score_band": tier,
        "recommendation": decision,
        "rationale": reason,
    }


def main():
    print("=" * 65)
    print("CREDIT RISK EVALUATION ENGINE (INFERENCE)")
    print("=" * 65)

    # Sample applicant profile: High limit, clean repayment history (no delays)
    sample_prime = {
        "LIMIT_BAL": 150000,
        "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 15000, "BILL_AMT2": 14000, "BILL_AMT3": 13000,
        "BILL_AMT4": 12000, "BILL_AMT5": 11000, "BILL_AMT6": 10000,
        "PAY_AMT1": 2000, "PAY_AMT2": 2000, "PAY_AMT3": 2000,
        "PAY_AMT4": 2000, "PAY_AMT5": 2000, "PAY_AMT6": 2000,
    }

    # Sample applicant profile: Low limit, repeated 2-month repayment delays
    sample_subprime = {
        "LIMIT_BAL": 20000,
        "PAY_0": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
        "BILL_AMT1": 19500, "BILL_AMT2": 19000, "BILL_AMT3": 18500,
        "BILL_AMT4": 18000, "BILL_AMT5": 17500, "BILL_AMT6": 17000,
        "PAY_AMT1": 500, "PAY_AMT2": 500, "PAY_AMT3": 0,
        "PAY_AMT4": 500, "PAY_AMT5": 500, "PAY_AMT6": 500,
    }

    print("\n[Case 1] Evaluating Prime Applicant Profile:")
    res1 = evaluate_applicant(sample_prime)
    print(f"  * Default Probability : {res1['default_probability']:.2%}")
    print(f"  * Risk Category       : {res1['score_band']}")
    print(f"  * System Decision     : {res1['recommendation']}")
    print(f"  * Policy Rationale    : {res1['rationale']}")

    print("\n[Case 2] Evaluating Delinquent / Subprime Applicant Profile:")
    res2 = evaluate_applicant(sample_subprime)
    print(f"  * Default Probability : {res2['default_probability']:.2%}")
    print(f"  * Risk Category       : {res2['score_band']}")
    print(f"  * System Decision     : {res2['recommendation']}")
    print(f"  * Policy Rationale    : {res2['rationale']}")
    print("=" * 65)


if __name__ == "__main__":
    main()
