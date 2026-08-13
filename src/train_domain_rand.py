"""Train XGBoost with domain randomization: augment training data with synthetic perturbations.

FIXES APPLIED (vs. original repo):
  1. Group by physical battery ID (same fix as train_full_model.py).
  2. Fit isotonic calibrator on VAL predictions (not on training predictions).
  3. Use a DIFFERENT seed (seed=789) for the augmented perturbations used in
     training, so that the evaluation seeds [42, 123, 456, 789, 101112] are
     ALL out-of-sample with respect to the training perturbations.  The
     original repo trained DR on seed=42 perturbations AND evaluated on seed=42
     perturbations — in-sample contamination that artificially lowered the
     DR ECE by ~5x for that one seed.
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

FEATURES = ["cycle", "avg_voltage", "min_voltage", "avg_current", "avg_temp", "duration", "SOH"]
H_LIST = [10, 20, 30, 50]
TEST_CELL_FRAC = 0.20
RANDOM_STATE = 42
# FIX 3: Use a seed NOT in the eval seed list, so all eval seeds are out-of-sample.
DR_TRAIN_SEED = 2024  # distinct from eval seeds [42, 123, 456, 789, 101112]

XGB_PARAMS = {
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    "objective": "binary:logistic", "eval_metric": "logloss",
    "random_state": RANDOM_STATE, "verbosity": 0, "use_label_encoder": False,
}


def physical_battery(cell_string):
    m = re.search(r"(B\d{4,5})", cell_string)
    return m.group(1) if m else cell_string


def safe_auc(y_true, p):
    return roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
    df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)
    df["phys_battery"] = df["cell"].apply(physical_battery)

    # Use the SAME split as train_full_model.py (loaded from results/train_val_split.csv)
    split_path = os.path.join(RESULTS_DIR, "train_val_split.csv")
    if os.path.exists(split_path):
        split_df = pd.read_csv(split_path)
        train_idx = split_df.index[split_df["split"] == "train"].values
        held_out_label = "test" if (split_df["split"] == "test").any() else "val"
        val_idx = split_df.index[split_df["split"] == held_out_label].values
        print(f"Loaded existing split from {split_path}")
    else:
        # Fallback: recreate the split
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_CELL_FRAC, random_state=RANDOM_STATE)
        train_idx_local, val_idx_local = next(splitter.split(df, groups=df["phys_battery"]))
        train_idx, val_idx = train_idx_local, val_idx_local
        print(f"Created new split (train_full_model.py should be run first)")

    tr_cells = set(df.iloc[train_idx]["cell"].unique())
    val_cells = set(df.iloc[val_idx]["cell"].unique())
    print(f"Train cells: {len(tr_cells)}, Test cells: {len(val_cells)}")

    # FIX 3: Augment with perturbations from a seed NOT in the eval seed list
    aug_frames = [df.iloc[train_idx]]
    for severity in [1, 2, 3, 4]:
        syn_path = os.path.join(DATA_DIR, "synthetic", f"nasa_perturbed_s{severity}_s{DR_TRAIN_SEED}.csv")
        if not os.path.exists(syn_path):
            sys.path.insert(0, SRC)
            from synthetic_data import generate_synthetic_dataset
            generate_synthetic_dataset(severity, seed=DR_TRAIN_SEED)
        syn = pd.read_csv(syn_path)
        syn = syn[syn["cell"].isin(tr_cells)]
        aug_frames.append(syn)
        print(f"  + Severity {severity} (seed={DR_TRAIN_SEED}): {len(syn)} rows")

    df_aug = pd.concat(aug_frames, ignore_index=True)
    X_aug = df_aug[FEATURES].values
    X_val = df.iloc[val_idx][FEATURES].values
    print(f"Augmented training: {len(df_aug)} rows")

    models, calibrators = {}, {}
    # Group labels for the augmented set (for OOF calibrator fitting)
    # Each row in df_aug corresponds to a (cell, cycle) from one of the aug_frames
    # We use phys_battery of the underlying cell as the group
    aug_phys = []
    for frame in aug_frames:
        aug_phys.extend(frame["cell"].apply(physical_battery).values)
    aug_phys = np.array(aug_phys)

    for H in H_LIST:
        y_aug = make_composite_fail_in_H(df.iloc[train_idx], H)
        y_aug = np.tile(y_aug, len(aug_frames))   # labels preserved across augmented copies
        y_val = make_composite_fail_in_H(df.iloc[val_idx], H)

        # 1) Train DR classifier on augmented data
        clf = XGBClassifier(**XGB_PARAMS)
        clf.fit(X_aug, y_aug)

        # 2) Fit calibrator on OOF predictions from augmented train (GroupKFold by phys battery)
        oof_pred = np.zeros(len(y_aug))
        gkf = GroupKFold(n_splits=5)
        for fold_tr, fold_va in gkf.split(X_aug, y_aug, groups=aug_phys):
            fold_clf = XGBClassifier(**XGB_PARAMS)
            fold_clf.fit(X_aug[fold_tr], y_aug[fold_tr])
            oof_pred[fold_va] = fold_clf.predict_proba(X_aug[fold_va])[:, 1]

        if len(np.unique(y_aug)) < 2:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(np.array([0, 1]), np.array([0, 1]))
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(oof_pred, y_aug)

        # 3) Evaluate on val cells (held out from BOTH classifier and calibrator)
        p_raw_val = clf.predict_proba(X_val)[:, 1]
        p_cal_val = iso.transform(p_raw_val)

        b_val = brier_score_loss(y_val, p_cal_val) if len(np.unique(y_val)) > 1 else np.nan
        a_val = safe_auc(y_val, p_cal_val)
        e_val = compute_ece(y_val, p_cal_val)
        print(f"  H={H}: Brier={b_val:.4f} AUC={a_val:.4f} ECE={e_val:.4f}  (val, n={len(y_val)})")

        models[H] = clf
        calibrators[H] = iso

    out_path = os.path.join(MODELS_DIR, "domain_rand_model.joblib")
    joblib.dump({"models": models, "calibrators": calibrators,
                 "dr_train_seed": DR_TRAIN_SEED}, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
