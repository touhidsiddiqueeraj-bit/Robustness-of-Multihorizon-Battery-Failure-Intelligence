"""Train the multihorizon XGBoost + isotonic calibrator on clean NASA data.

FIXES APPLIED (vs. original repo):
  1. Split by PHYSICAL BATTERY ID (extracted from cell string) instead of by
     the full cell string.  This eliminates cross-split leakage where the
     same physical battery (e.g. B0025, B0027) appeared in BOTH train and val
     because it was cycled under two NASA sub-campaigns.
  2. TWO-WAY split (train/test) by physical battery, with the calibrator fit
     on OUT-OF-FOLD predictions from 5-fold GroupKFold WITHIN train.  This:
       (a) gives the calibrator enough data (~700 OOF predictions) to be
           stable (a 3-way split leaves only ~7 cal batteries, which makes
           isotonic unstable);
       (b) avoids the in-sample optimism of fitting the calibrator on raw
           training predictions (which the original code did);
       (c) keeps a clean held-out test set for ALL reported metrics.
  3. Clean baseline metrics (ECE / AUC / Brier) are computed on TEST cells
     only, not on the full 37-cell dataset.
"""
import os, sys, re
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression

from compute_ece import compute_ece
from composite_label import make_composite_fail_in_H

SRC = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SRC, ".."))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
RESULTS_DIR = os.path.join(BASE, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ["cycle", "avg_voltage", "min_voltage", "avg_current", "avg_temp", "duration", "SOH"]
H_LIST = [10, 20, 30, 50]
TEST_FRAC = 0.20           # 20 % of batteries held out as test
RANDOM_STATE = 42
N_CV_FOLDS = 5             # for OOF calibrator fitting within train

XGB_PARAMS = {
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": RANDOM_STATE,
    "verbosity": 0,
    "use_label_encoder": False,
}


def safe_auc(y_true, p):
    return roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan


def physical_battery(cell_string):
    m = re.search(r"(B\d{4,5})", cell_string)
    return m.group(1) if m else cell_string


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
    df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)
    df["phys_battery"] = df["cell"].apply(physical_battery)
    print(f"Data: {len(df)} rows, {df['cell'].nunique()} cell strings, "
          f"{df['phys_battery'].nunique()} physical batteries")

    # 2-way split: train / test by physical battery
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=df["phys_battery"]))
    print(f"\n2-way split by physical battery:")
    print(f"  Train: {len(train_idx)} rows, {df.iloc[train_idx]['phys_battery'].nunique()} batteries")
    print(f"  Test:  {len(test_idx)} rows, {df.iloc[test_idx]['phys_battery'].nunique()} batteries")

    tr_phys = set(df.iloc[train_idx]["phys_battery"])
    te_phys = set(df.iloc[test_idx]["phys_battery"])
    overlap = tr_phys & te_phys
    assert not overlap, f"Cross-split contamination! overlap={overlap}"
    print(f"  Overlap check: PASS (no battery appears in both splits)")

    # Persist split (test rows are what all downstream metrics use)
    split_df = pd.DataFrame({
        "row_idx": np.arange(len(df)),
        "cell": df["cell"].values,
        "phys_battery": df["phys_battery"].values,
        "split": np.where(np.isin(np.arange(len(df)), train_idx), "train", "test"),
    })
    split_path = os.path.join(RESULTS_DIR, "train_val_split.csv")
    split_df.to_csv(split_path, index=False)
    print(f"Saved split: {split_path}")

    X = df[FEATURES].values
    models, calibrators = {}, {}

    print(f"\n--- Training per horizon (XGBoost on train; calibrator on {N_CV_FOLDS}-fold OOF train predictions) ---")
    for H in H_LIST:
        y = make_composite_fail_in_H(df, H)

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        groups_tr = df.iloc[train_idx]["phys_battery"].values

        # 1) Train XGBoost on ALL train cells (final classifier)
        clf = XGBClassifier(**XGB_PARAMS)
        clf.fit(X_tr, y_tr)

        # 2) Generate OUT-OF-FOLD predictions on train via GroupKFold.
        # These OOF predictions come from models that did NOT train on the
        # cell being predicted — so the calibrator sees honest predictions.
        oof_pred = np.zeros(len(y_tr))
        gkf = GroupKFold(n_splits=N_CV_FOLDS)
        for fold_tr, fold_va in gkf.split(X_tr, y_tr, groups=groups_tr):
            fold_clf = XGBClassifier(**XGB_PARAMS)
            fold_clf.fit(X_tr[fold_tr], y_tr[fold_tr])
            oof_pred[fold_va] = fold_clf.predict_proba(X_tr[fold_va])[:, 1]

        # 3) Fit isotonic calibrator on OOF predictions (no in-sample optimism)
        if len(np.unique(y_tr)) < 2:
            print(f"  [warn] H={H}: train has single class — using identity calibrator")
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(oof_pred, y_tr)

        # 4) Evaluate on TEST cells (truly held out from BOTH classifier & calibrator)
        p_raw_te = clf.predict_proba(X_te)[:, 1]
        p_cal_te = iso.transform(p_raw_te)

        b_te = brier_score_loss(y_te, p_cal_te) if len(np.unique(y_te)) > 1 else np.nan
        a_raw_te = safe_auc(y_te, p_raw_te)
        a_cal_te = safe_auc(y_te, p_cal_te)
        e_raw_te = compute_ece(y_te, p_raw_te)
        e_cal_te = compute_ece(y_te, p_cal_te)
        print(f"  H={H}: test n={len(y_te)}, n_pos={int(y_te.sum())}")
        print(f"        raw  AUC={a_raw_te:.4f} ECE={e_raw_te:.4f}")
        print(f"        cal  AUC={a_cal_te:.4f} ECE={e_cal_te:.4f} Brier={b_te:.4f}")

        models[H] = clf
        calibrators[H] = iso

    bundle = {"models": models, "calibrators": calibrators}
    joblib.dump(bundle, os.path.join(MODELS_DIR, "full_model.joblib"))
    print(f"\nSaved model bundle: {os.path.join(MODELS_DIR, 'full_model.joblib')}")

    # Clean baseline metrics on TEST cells only
    rows = []
    for H in H_LIST:
        y_te = make_composite_fail_in_H(df.iloc[test_idx], H)
        clf = models[H]
        iso = calibrators[H]
        p_raw = clf.predict_proba(X[test_idx])[:, 1]
        p_cal = iso.transform(p_raw)
        rows.append({
            "H": H,
            "ece_raw": compute_ece(y_te, p_raw),
            "auc_raw": safe_auc(y_te, p_raw),
            "ece_cal": compute_ece(y_te, p_cal),
            "auc_cal": safe_auc(y_te, p_cal),
            "brier_cal": brier_score_loss(y_te, p_cal) if len(np.unique(y_te)) > 1 else np.nan,
            "n_test": len(y_te),
            "n_pos": int(y_te.sum()),
        })
    baseline_path = os.path.join(RESULTS_DIR, "clean_baseline.csv")
    pd.DataFrame(rows).to_csv(baseline_path, index=False)
    print(f"\nSaved clean baseline (test only): {baseline_path}")
    print("\nClean baseline (test cells only):")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
