"""Vmin-only ablation: restore ONLY min_voltage to its clean value, keeping all
other features perturbed.  If the paper's root-cause claim (Vmin shift is the
primary driver of calibration collapse) is correct, this should substantially
recover calibration.

This script was ADDED to the original repo to test the paper's central
mechanistic claim causally rather than just correlationally.
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
    model_bundle = joblib.load(os.path.join(MODELS_DIR, "full_model.joblib"))
    models = model_bundle["models"]
    calibrators = model_bundle["calibrators"]

    split = pd.read_csv(os.path.join(RESULTS_DIR, "train_val_split.csv"))
    held_out_label = "test" if (split["split"] == "test").any() else "val"
    val_row_idx = split.index[split["split"] == held_out_label].values

    clean = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
    clean = clean.sort_values(["cell", "cycle"]).reset_index(drop=True)
    labels = {H: make_composite_fail_in_H(clean, H) for H in H_LIST}

    # Clean Vmin per (cell, cycle) — used to restore Vmin in the ablation
    clean_vmin = clean["min_voltage"].values  # row-aligned with clean df

    results = []
    print("=== Vmin-only ablation: restore Vmin, keep other features perturbed ===")
    print(f"{'s':>2} {'seed':>6} {'H':>3} | "
          f"{'ECE_pert':>9} {'ECE_ablat':>9} {'dECE':>7} | "
          f"{'AUC_pert':>8} {'AUC_ablat':>9}")
    print("-" * 75)

    for severity in [1, 2, 3, 4]:
        for seed in SEEDS:
            syn_path = os.path.join(DATA_DIR, "synthetic", f"nasa_perturbed_s{severity}_s{seed}.csv")
            df = pd.read_csv(syn_path)
            df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)

            # Original (fully perturbed) features on val cells
            X_pert = df.iloc[val_row_idx][FEATURES].values.copy()

            # Ablated features: restore ONLY min_voltage to clean value
            X_ablat = X_pert.copy()
            # clean_vmin[val_row_idx] gives the clean Vmin for each val row
            X_ablat[:, FEATURES.index("min_voltage")] = clean_vmin[val_row_idx]

            for H in H_LIST:
                y = labels[H][val_row_idx]
                model = models[H]
                calibrator = calibrators[H]

                # Perturbed (baseline)
                p_raw_pert = model.predict_proba(X_pert)[:, 1]
                p_cal_pert = calibrator.transform(p_raw_pert)
                ece_pert = compute_ece(y, p_cal_pert)
                auc_pert = safe_auc(y, p_cal_pert)

                # Ablated (Vmin restored)
                p_raw_ablat = model.predict_proba(X_ablat)[:, 1]
                p_cal_ablat = calibrator.transform(p_raw_ablat)
                ece_ablat = compute_ece(y, p_cal_ablat)
                auc_ablat = safe_auc(y, p_cal_ablat)

                results.append({
                    "severity": severity, "seed": seed, "H": H,
                    "ece_pert": ece_pert, "ece_ablat": ece_ablat,
                    "delta_ece": ece_ablat - ece_pert,
                    "frac_recovered": 1.0 - (ece_ablat / ece_pert) if ece_pert > 0 else np.nan,
                    "auc_pert": auc_pert, "auc_ablat": auc_ablat,
                    "val_n": len(y), "val_pos": int(y.sum()),
                })
                print(f"{severity:>2} {seed:>6} {H:>3} | "
                      f"{ece_pert:>9.4f} {ece_ablat:>9.4f} {ece_ablat-ece_pert:>+7.4f} | "
                      f"{auc_pert:>8.4f} {auc_ablat:>9.4f}")

    results_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "vmin_ablation_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(results_df)} rows)")

    # Summary
    print("\n=== Summary: mean across (severity, seed) per horizon ===")
    for H in H_LIST:
        sub = results_df[results_df.H == H]
        print(f"  H={H}: ECE_pert={sub.ece_pert.mean():.4f} -> ECE_ablat={sub.ece_ablat.mean():.4f}  "
              f"(mean recovered={sub.frac_recovered.mean()*100:.1f}%)")
    print("\nIf recovered% is high (>50%), Vmin shift is a major driver.")
    print("If recovered% is low (<20%), Vmin alone does NOT explain calibration collapse.")
    print("If recovered% is moderate (20-50%), Vmin contributes but other features also matter.")


if __name__ == "__main__":
    main()
