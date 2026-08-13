"""Evaluate domain-randomized model on perturbed data.

FIXES APPLIED (vs. original repo):
  1. Restrict ALL perturbed metrics to VAL cells (same fix as run_robustness.py).
  2. All eval seeds [42, 123, 456, 789, 101112] are now out-of-sample
     because the DR model was trained on seed=2024 perturbations
     (see train_domain_rand.py).  No seed=42 contamination.
"""
import os, sys
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import brier_score_loss, roc_auc_score
from compute_ece import compute_ece
from composite_label import make_composite_fail_in_H

SRC = os.path.dirname(os.path.abspath(__file__))
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
    dr_train_seed = bundle.get("dr_train_seed", "unknown")
    print(f"DR model trained with augmentation seed = {dr_train_seed}")
    print(f"Eval seeds = {SEEDS}  (all distinct from training seed)")

    split = pd.read_csv(os.path.join(RESULTS_DIR, "train_val_split.csv"))
    held_out_label = "test" if (split["split"] == "test").any() else "val"
    val_row_idx = split.index[split["split"] == held_out_label].values

    clean = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
    clean = clean.sort_values(["cell", "cycle"]).reset_index(drop=True)
    labels = {H: make_composite_fail_in_H(clean, H) for H in H_LIST}

    # Clean baseline on VAL only
    print("\n=== DR model: Clean baseline (val cells only) ===")
    for H in H_LIST:
        X_val = clean.iloc[val_row_idx][FEATURES].values
        y_val = labels[H][val_row_idx]
        model, calibrator = models[H], calibrators[H]
        p_raw = model.predict_proba(X_val)[:, 1]
        p_cal = calibrator.transform(p_raw)
        ece = compute_ece(y_val, p_cal)
        auc = safe_auc(y_val, p_cal)
        print(f"  H={H}: ECE={ece:.4f} AUC={auc:.4f}")

    results = []
    print("\n=== DR model: Perturbed evaluation (val cells only) ===")
    for severity in [1, 2, 3, 4]:
        for seed in SEEDS:
            syn_path = os.path.join(DATA_DIR, "synthetic", f"nasa_perturbed_s{severity}_s{seed}.csv")
            df = pd.read_csv(syn_path)
            df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)
            X = df.iloc[val_row_idx][FEATURES].values

            for H in H_LIST:
                y = labels[H][val_row_idx]
                row = {"severity": severity, "seed": seed, "H": H,
                       "val_n": len(y), "val_pos": int(y.sum())}
                model, calibrator = models[H], calibrators[H]
                p_raw = model.predict_proba(X)[:, 1]
                p_cal = calibrator.transform(p_raw)
                row["dr_ece_raw"] = compute_ece(y, p_raw)
                row["dr_auc_raw"] = safe_auc(y, p_raw)
                row["dr_ece_cal"] = compute_ece(y, p_cal)
                row["dr_auc_cal"] = safe_auc(y, p_cal)
                results.append(row)
                print(f"  s={severity} seed={seed} H={H}: "
                      f"dr_ece_cal={row['dr_ece_cal']:.4f} dr_auc_cal={row['dr_auc_cal']:.4f}")

    results_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "domain_rand_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(results_df)} rows)")


if __name__ == "__main__":
    main()
