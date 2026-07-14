import os, sys
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

from compute_ece import compute_ece

SRC = os.path.dirname(os.path.abspath(__file__))
from composite_label import make_composite_fail_in_H

BASE = os.path.normpath(os.path.join(SRC, ".."))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ["cycle", "avg_voltage", "min_voltage", "avg_current", "avg_temp", "duration", "SOH"]
H_LIST = [10, 20, 30, 50]
SEEDS = [42, 123, 456, 789, 101112]
CAL_SAMPLE_FRAC = 0.10


def safe_auc(y_true, p):
    return roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan


def main():
    model_bundle = joblib.load(os.path.join(MODELS_DIR, "full_model.joblib"))
    models = model_bundle["models"]
    calibrators = model_bundle["calibrators"]

    # Load clean data and compute fixed labels
    clean = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
    clean = clean.sort_values(["cell", "cycle"]).reset_index(drop=True)
    X_clean = clean[FEATURES].values

    labels = {}
    for H in H_LIST:
        labels[H] = make_composite_fail_in_H(clean, H)

    # Compute clean baseline metrics
    print("=== Clean baseline ===")
    for H in H_LIST:
        model = models[H]
        calibrator = calibrators[H]
        p_raw = model.predict_proba(X_clean)[:, 1]
        p_cal = calibrator.transform(p_raw)
        ece = compute_ece(labels[H], p_cal)
        auc = safe_auc(labels[H], p_cal)
        brier = brier_score_loss(labels[H], p_cal)
        print(f"  H={H}: ECE={ece:.4f} AUC={auc:.4f} Brier={brier:.4f}")

    results = []

    for severity in [1, 2, 3, 4]:
        for seed in SEEDS:
            syn_path = os.path.join(DATA_DIR, "synthetic", f"nasa_perturbed_s{severity}_s{seed}.csv")
            if not os.path.exists(syn_path):
                sys.path.insert(0, SRC)
                from synthetic_data import generate_synthetic_dataset
                generate_synthetic_dataset(severity, seed=seed)

            df = pd.read_csv(syn_path)
            df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)
            X = df[FEATURES].values
            rng = np.random.default_rng(seed)

            for H in H_LIST:
                y = labels[H]  # fixed clean labels

                row = {"severity": severity, "seed": seed, "H": H}

                model = models[H]
                calibrator = calibrators[H]

                p_raw = model.predict_proba(X)[:, 1]
                p_cal = calibrator.transform(p_raw)

                row["auc_raw"] = safe_auc(y, p_raw)
                row["brier_raw"] = brier_score_loss(y, p_raw)
                row["ece_raw"] = compute_ece(y, p_raw)

                row["auc_cal"] = safe_auc(y, p_cal)
                row["brier_cal"] = brier_score_loss(y, p_cal)
                row["ece_cal"] = compute_ece(y, p_cal)

                # Recalibration on small synthetic sample
                n_cal = max(2, int(len(y) * CAL_SAMPLE_FRAC))
                cal_idx = rng.choice(len(y), n_cal, replace=False)
                mask = np.zeros(len(y), dtype=bool)
                mask[cal_idx] = True

                if len(np.unique(y[mask])) >= 2:
                    iso_recal = IsotonicRegression(out_of_bounds="clip")
                    iso_recal.fit(p_raw[mask], y[mask])
                    p_recal = iso_recal.transform(p_raw)
                else:
                    p_recal = p_cal.copy()

                row["ece_recal"] = compute_ece(y[~mask], p_recal[~mask])

                # Platt (sigmoid) recalibration
                if len(np.unique(y[mask])) >= 2:
                    platt = LogisticRegression(C=9999)
                    platt.fit(p_raw[mask].reshape(-1, 1), y[mask])
                    p_recal_platt = platt.predict_proba(p_raw.reshape(-1, 1))[:, 1]
                else:
                    p_recal_platt = p_cal.copy()
                row["ece_recal_platt"] = compute_ece(y[~mask], p_recal_platt[~mask])

                # Temperature scaling
                if len(np.unique(y[mask])) >= 2:
                    eps = 1e-15
                    p_safe = np.clip(p_raw, eps, 1 - eps)
                    logits = np.log(p_safe / (1 - p_safe))
                    def nll(T):
                        p = 1 / (1 + np.exp(-logits / T))
                        p = np.clip(p, eps, 1 - eps)
                        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                    res = minimize(nll, x0=1.0, method='L-BFGS-B', bounds=[(1e-3, 10.0)])
                    T_opt = res.x[0]
                    p_temp = 1 / (1 + np.exp(-logits / T_opt))
                    p_recal_temp = np.clip(p_temp, 0, 1)
                else:
                    p_recal_temp = p_cal.copy()
                row["ece_recal_temp"] = compute_ece(y[~mask], p_recal_temp[~mask])

                results.append(row)
                print(f"s={severity} seed={seed} H={H}: "
                      f"ece_raw={row['ece_raw']:.4f} ece_cal={row['ece_cal']:.4f} "
                      f"ece_recal={row['ece_recal']:.4f} platt={row['ece_recal_platt']:.4f} temp={row['ece_recal_temp']:.4f}")

    results_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "robustness_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(results_df)} rows)")


if __name__ == "__main__":
    main()
