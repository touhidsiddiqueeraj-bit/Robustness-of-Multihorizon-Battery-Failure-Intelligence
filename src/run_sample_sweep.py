"""Sample-efficiency sweep: vary the calibration fraction (5/10/20/50%).

FIXES APPLIED (vs. original repo):
  1. Restrict to VAL cells (same fix as run_robustness.py).
  2. ece_cal AND ece_recal both computed on the held-out (1 - frac) subset,
     so the sweep is a fair apples-to-apples comparison at every fraction.
"""
import os, sys
import numpy as np
import pandas as pd
import joblib

from sklearn.isotonic import IsotonicRegression
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
CAL_SAMPLE_FRACS = [0.05, 0.10, 0.20, 0.50]


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

    results = []

    for severity in [1, 2, 3, 4]:
        for seed in SEEDS:
            syn_path = os.path.join(DATA_DIR, "synthetic", f"nasa_perturbed_s{severity}_s{seed}.csv")
            df = pd.read_csv(syn_path)
            df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)
            X = df.iloc[val_row_idx][FEATURES].values

            for H in H_LIST:
                y = labels[H][val_row_idx]
                model = models[H]
                calibrator = calibrators[H]
                p_raw = model.predict_proba(X)[:, 1]
                p_cal = calibrator.transform(p_raw)

                # Use a separate RNG stream for calibration sampling
                rng_cal = np.random.default_rng(seed + 100000 + int(H * 13))

                for frac in CAL_SAMPLE_FRACS:
                    n_cal = max(2, int(len(y) * frac))
                    cal_idx = rng_cal.choice(len(y), n_cal, replace=False)
                    mask = np.zeros(len(y), dtype=bool)
                    mask[cal_idx] = True
                    eval_idx = ~mask

                    # Both metrics on the same held-out subset
                    ece_cal_eval = compute_ece(y[eval_idx], p_cal[eval_idx])

                    if len(np.unique(y[mask])) >= 2:
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(p_raw[mask], y[mask])
                        p_recal = iso.transform(p_raw)
                    else:
                        p_recal = p_cal.copy()
                    ece_recal = compute_ece(y[eval_idx], p_recal[eval_idx])

                    results.append({
                        "severity": severity, "seed": seed, "H": H,
                        "cal_frac": frac, "ece_cal": ece_cal_eval,
                        "ece_recal": ece_recal,
                        "n_cal": int(mask.sum()),
                        "n_eval": int(eval_idx.sum()),
                    })

    results_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "robustness_results_sweep.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(results_df)} rows)")


if __name__ == "__main__":
    main()
