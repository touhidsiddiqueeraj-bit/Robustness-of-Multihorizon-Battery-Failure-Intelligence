import os, sys
import numpy as np
import pandas as pd
import joblib

from sklearn.isotonic import IsotonicRegression
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
CAL_SAMPLE_FRACS = [0.05, 0.10, 0.20, 0.50]


def main():
    model_bundle = joblib.load(os.path.join(MODELS_DIR, "full_model.joblib"))
    models = model_bundle["models"]
    calibrators = model_bundle["calibrators"]

    clean = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
    clean = clean.sort_values(["cell", "cycle"]).reset_index(drop=True)
    X_clean = clean[FEATURES].values

    labels = {}
    for H in H_LIST:
        labels[H] = make_composite_fail_in_H(clean, H)

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
                model = models[H]
                calibrator = calibrators[H]
                p_raw = model.predict_proba(X)[:, 1]
                p_cal = calibrator.transform(p_raw)
                base_ece_cal = compute_ece(y, p_cal)

                for frac in CAL_SAMPLE_FRACS:
                    n_cal = max(2, int(len(y) * frac))
                    cal_idx = rng.choice(len(y), n_cal, replace=False)
                    mask = np.zeros(len(y), dtype=bool)
                    mask[cal_idx] = True

                    if len(np.unique(y[mask])) >= 2:
                        iso = IsotonicRegression(out_of_bounds="clip")
                        iso.fit(p_raw[mask], y[mask])
                        p_recal = iso.transform(p_raw)
                    else:
                        p_recal = p_cal.copy()

                    ece_recal = compute_ece(y[~mask], p_recal[~mask])

                    results.append({
                        "severity": severity, "seed": seed, "H": H,
                        "cal_frac": frac, "ece_cal": base_ece_cal,
                        "ece_recal": ece_recal,
                    })

    results_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "robustness_results_sweep.csv")
    results_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(results_df)} rows)")


if __name__ == "__main__":
    main()
