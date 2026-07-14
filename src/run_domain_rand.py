"""Evaluate domain-randomized model on perturbed data."""
import os, sys
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import brier_score_loss, roc_auc_score
from compute_ece import compute_ece

SRC = os.path.dirname(os.path.abspath(__file__))
from composite_label import make_composite_fail_in_H

BASE = os.path.normpath(os.path.join(SRC, ".."))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
MODELS_DIR = os.path.join(BASE, "models")

FEATURES = ["cycle", "avg_voltage", "min_voltage", "avg_current", "avg_temp", "duration", "SOH"]
H_LIST = [10, 20, 30, 50]
SEEDS = [42, 123, 456, 789, 101112]


def safe_auc(y_true, p):
    return roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan


def main():
    bundle = joblib.load(os.path.join(MODELS_DIR, "domain_rand_model.joblib"))
    models, calibrators = bundle["models"], bundle["calibrators"]

    clean = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
    clean = clean.sort_values(["cell", "cycle"]).reset_index(drop=True)
    X_clean = clean[FEATURES].values

    labels = {}
    for H in H_LIST:
        labels[H] = make_composite_fail_in_H(clean, H)

    # Clean baseline
    print("=== DR model: Clean baseline ===")
    for H in H_LIST:
        model, calibrator = models[H], calibrators[H]
        p_raw = model.predict_proba(X_clean)[:, 1]
        p_cal = calibrator.transform(p_raw)
        ece = compute_ece(labels[H], p_cal)
        auc = safe_auc(labels[H], p_cal)
        print(f"  H={H}: ECE={ece:.4f} AUC={auc:.4f}")

    results = []
    for severity in [1, 2, 3, 4]:
        for seed in SEEDS:
            syn_path = os.path.join(DATA_DIR, "synthetic", f"nasa_perturbed_s{severity}_s{seed}.csv")
            df = pd.read_csv(syn_path)
            df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)
            X = df[FEATURES].values
            rng = np.random.default_rng(seed)

            for H in H_LIST:
                y = labels[H]
                row = {"severity": severity, "seed": seed, "H": H}
                model, calibrator = models[H], calibrators[H]
                p_raw = model.predict_proba(X)[:, 1]
                p_cal = calibrator.transform(p_raw)
                row["dr_ece_cal"] = compute_ece(y, p_cal)
                row["dr_auc_cal"] = safe_auc(y, p_cal)
                results.append(row)

    results_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "domain_rand_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(results_df)} rows)")


if __name__ == "__main__":
    main()
