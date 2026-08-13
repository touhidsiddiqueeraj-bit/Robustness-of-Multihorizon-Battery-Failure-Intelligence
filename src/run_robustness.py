"""Evaluate the clean-trained model on perturbed data and recalibration strategies.

FIXES APPLIED (vs. original repo):
  1. Restrict ALL perturbed metrics (raw/cal/recal ECE and AUC) to VAL cells
     only.  The original code evaluated on all 37 cells, 80 % of which the
     model was trained on — making every perturbed metric in-sample.
  2. Compute ece_cal AND ece_recal on the SAME held-out 90 % subset.  The
     original code computed ece_cal on 100 % and ece_recal on 90 %, so the
     paired comparison (and the Wilcoxon test downstream) conflated
     "different calibrator" with "different evaluation sample".
  3. Platt scaling uses C=1.0 (sklearn default) instead of C=9999 (which is
     essentially no regularization and is unstable on small samples).
  4. Added AUC for recalibrated predictions, so reviewers can verify that
     recalibration does NOT meaningfully change AUC (it should be very close
     to raw AUC, since isotonic is rank-preserving on the in-range subset).
  5. Added per-row val_n and val_pos columns so downstream tests can report
     effective sample sizes.
"""
import os, sys
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

from compute_ece import compute_ece, compute_mce, compute_ace
from composite_label import make_composite_fail_in_H

SRC = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SRC, ".."))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")
MODELS_DIR = os.path.join(BASE, "models")

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

    # Load split to recover test cells (consistent with train_full_model.py)
    split = pd.read_csv(os.path.join(RESULTS_DIR, "train_val_split.csv"))
    # Accept either "test" (2-way split) or "val" (legacy 3-way) as the held-out set
    held_out_label = "test" if (split["split"] == "test").any() else "val"
    val_row_idx = split.index[split["split"] == held_out_label].values
    val_cells = sorted(split.loc[split["split"] == held_out_label, "cell"].unique())
    print(f"Held-out ({held_out_label}) cells ({len(val_cells)}): {val_cells}")

    # Load clean data and compute fixed labels
    clean = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
    clean = clean.sort_values(["cell", "cycle"]).reset_index(drop=True)
    labels = {H: make_composite_fail_in_H(clean, H) for H in H_LIST}

    # Compute clean baseline metrics on VAL only
    print("\n=== Clean baseline (val cells only) ===")
    baseline_rows = []
    for H in H_LIST:
        X_val = clean.iloc[val_row_idx][FEATURES].values
        y_val = labels[H][val_row_idx]
        model = models[H]
        calibrator = calibrators[H]
        p_raw = model.predict_proba(X_val)[:, 1]
        p_cal = calibrator.transform(p_raw)
        ece = compute_ece(y_val, p_cal)
        auc = safe_auc(y_val, p_cal)
        brier = brier_score_loss(y_val, p_cal) if len(np.unique(y_val)) > 1 else np.nan
        baseline_rows.append({
            "H": H,
            "ece_raw": compute_ece(y_val, p_raw),
            "auc_raw": safe_auc(y_val, p_raw),
            "ece_cal": ece,
            "mce_cal": compute_mce(y_val, p_cal),
            "ace_cal": compute_ace(y_val, p_cal),
            "auc_cal": auc,
            "brier_cal": brier,
            "n_test": len(y_val),
            "n_pos": int(y_val.sum()),
        })
        print(f"  H={H}: ECE={ece:.4f} MCE={baseline_rows[-1]['mce_cal']:.4f} "
              f"ACE={baseline_rows[-1]['ace_cal']:.4f} AUC={auc:.4f} Brier={brier:.4f} "
              f"(n={len(y_val)}, n_pos={int(y_val.sum())})")
    pd.DataFrame(baseline_rows).to_csv(
        os.path.join(RESULTS_DIR, "clean_baseline.csv"), index=False)
    print("Saved clean baseline (val only) -> results/clean_baseline.csv")

    results = []

    print("\n=== Perturbed evaluation (val cells only) ===")
    for severity in [1, 2, 3, 4]:
        for seed in SEEDS:
            syn_path = os.path.join(DATA_DIR, "synthetic", f"nasa_perturbed_s{severity}_s{seed}.csv")
            if not os.path.exists(syn_path):
                sys.path.insert(0, SRC)
                from synthetic_data import generate_synthetic_dataset
                generate_synthetic_dataset(severity, seed=seed)

            df = pd.read_csv(syn_path)
            df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)

            # FIX 1: Restrict to val cells
            X_full = df[FEATURES].values
            X = df.iloc[val_row_idx][FEATURES].values

            # Separate RNG for cal_idx (does NOT need to share state with perturbation RNG)
            rng_cal = np.random.default_rng(seed + 100000)

            for H in H_LIST:
                y_full = labels[H]
                y = labels[H][val_row_idx]   # paired with X

                row = {"severity": severity, "seed": seed, "H": H,
                       "val_n": len(y), "val_pos": int(y.sum())}

                model = models[H]
                calibrator = calibrators[H]

                # Raw + fixed-calibrator predictions on val
                p_raw = model.predict_proba(X)[:, 1]
                p_cal = calibrator.transform(p_raw)

                # --- FIX 2: hold out 10% for recalibrator fit, evaluate ALL metrics on remaining 90% ---
                n_cal = max(2, int(len(y) * CAL_SAMPLE_FRAC))
                cal_idx = rng_cal.choice(len(y), n_cal, replace=False)
                mask = np.zeros(len(y), dtype=bool)
                mask[cal_idx] = True
                eval_idx = ~mask

                # All reported metrics computed on eval_idx (the held-out 90 %)
                row["auc_raw"] = safe_auc(y[eval_idx], p_raw[eval_idx])
                row["brier_raw"] = brier_score_loss(y[eval_idx], p_raw[eval_idx]) if len(np.unique(y[eval_idx])) > 1 else np.nan
                row["ece_raw"] = compute_ece(y[eval_idx], p_raw[eval_idx])
                row["mce_raw"] = compute_mce(y[eval_idx], p_raw[eval_idx])
                row["ace_raw"] = compute_ace(y[eval_idx], p_raw[eval_idx])

                row["auc_cal"] = safe_auc(y[eval_idx], p_cal[eval_idx])
                row["brier_cal"] = brier_score_loss(y[eval_idx], p_cal[eval_idx]) if len(np.unique(y[eval_idx])) > 1 else np.nan
                row["ece_cal"] = compute_ece(y[eval_idx], p_cal[eval_idx])
                row["mce_cal"] = compute_mce(y[eval_idx], p_cal[eval_idx])
                row["ace_cal"] = compute_ace(y[eval_idx], p_cal[eval_idx])

                # --- Recalibration: isotonic ---
                if len(np.unique(y[mask])) >= 2:
                    iso_recal = IsotonicRegression(out_of_bounds="clip")
                    iso_recal.fit(p_raw[mask], y[mask])
                    p_recal_iso = iso_recal.transform(p_raw)
                else:
                    p_recal_iso = p_cal.copy()
                row["auc_recal"] = safe_auc(y[eval_idx], p_recal_iso[eval_idx])
                row["ece_recal"] = compute_ece(y[eval_idx], p_recal_iso[eval_idx])
                row["mce_recal"] = compute_mce(y[eval_idx], p_recal_iso[eval_idx])
                row["ace_recal"] = compute_ace(y[eval_idx], p_recal_iso[eval_idx])

                # --- Recalibration: Platt (sigmoid) ---
                if len(np.unique(y[mask])) >= 2:
                    # FIX 3: C=1.0 (default) instead of C=9999 (unregularized)
                    platt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
                    platt.fit(p_raw[mask].reshape(-1, 1), y[mask])
                    p_recal_platt = platt.predict_proba(p_raw.reshape(-1, 1))[:, 1]
                else:
                    p_recal_platt = p_cal.copy()
                row["ece_recal_platt"] = compute_ece(y[eval_idx], p_recal_platt[eval_idx])

                # --- Recalibration: temperature scaling ---
                if len(np.unique(y[mask])) >= 2:
                    eps = 1e-7
                    p_safe = np.clip(p_raw, eps, 1 - eps)
                    logits = np.log(p_safe / (1 - p_safe))

                    def nll(T):
                        # Use numerically stable sigmoid
                        z = logits[mask] / T[0]
                        # log(1+exp(-z)) and log(1+exp(z)) computed stably
                        log_p = -np.logaddexp(0, -z)
                        log_1mp = -np.logaddexp(0, z)
                        return -np.mean(y[mask] * log_p + (1 - y[mask]) * log_1mp)

                    # Multi-start to avoid local minima
                    best_res = None
                    for x0 in [0.5, 1.0, 2.0, 5.0]:
                        res = minimize(nll, x0=np.array([x0]), method="L-BFGS-B",
                                       bounds=[(1e-3, 10.0)])
                        if best_res is None or res.fun < best_res.fun:
                            best_res = res
                    T_opt = best_res.x[0]
                    # Stable sigmoid for full set
                    z_full = logits / T_opt
                    p_recal_temp = np.clip(1.0 / (1.0 + np.exp(-np.clip(z_full, -50, 50))), 0, 1)
                else:
                    p_recal_temp = p_cal.copy()
                row["ece_recal_temp"] = compute_ece(y[eval_idx], p_recal_temp[eval_idx])
                row["T_opt"] = float(T_opt) if len(np.unique(y[mask])) >= 2 else np.nan

                results.append(row)
                print(f"  s={severity} seed={seed} H={H}: "
                      f"ece_raw={row['ece_raw']:.4f} ece_cal={row['ece_cal']:.4f} "
                      f"ece_recal={row['ece_recal']:.4f} platt={row['ece_recal_platt']:.4f} "
                      f"temp={row['ece_recal_temp']:.4f} | "
                      f"auc_raw={row['auc_raw']:.4f} auc_cal={row['auc_cal']:.4f} auc_recal={row['auc_recal']:.4f}")

    results_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "robustness_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(results_df)} rows)")


if __name__ == "__main__":
    main()
