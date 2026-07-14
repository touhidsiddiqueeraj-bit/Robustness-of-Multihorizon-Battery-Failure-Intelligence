"""Train XGBoost with domain randomization: augment training data with synthetic perturbations."""
import os, sys
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression

from compute_ece import compute_ece

SRC = os.path.dirname(os.path.abspath(__file__))
from composite_label import make_composite_fail_in_H

BASE = os.path.normpath(os.path.join(SRC, ".."))
DATA_DIR = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURES = ["cycle", "avg_voltage", "min_voltage", "avg_current", "avg_temp", "duration", "SOH"]
H_LIST = [10, 20, 30, 50]
TEST_CELL_FRAC = 0.20
RANDOM_STATE = 42

XGB_PARAMS = {
    "max_depth": 4, "learning_rate": 0.05, "n_estimators": 300,
    "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    "objective": "binary:logistic", "eval_metric": "logloss",
    "random_state": RANDOM_STATE, "verbosity": 0, "use_label_encoder": False,
}


def safe_auc(y_true, p):
    return roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
    df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)
    cells = df["cell"].unique()
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_CELL_FRAC, random_state=RANDOM_STATE)
    tr_idx, val_idx = next(splitter.split(df, groups=df["cell"]))
    tr_cells = set(df.iloc[tr_idx]["cell"].unique())
    val_cells = set(df.iloc[val_idx]["cell"].unique())
    print(f"Train cells: {len(tr_cells)}, Val cells: {len(val_cells)}")

    # Augment training set with perturbed data from all severity levels
    aug_frames = [df.iloc[tr_idx]]
    for severity in [1, 2, 3, 4]:
        syn_path = os.path.join(DATA_DIR, "synthetic", f"nasa_perturbed_s{severity}_s{RANDOM_STATE}.csv")
        syn = pd.read_csv(syn_path)
        syn = syn[syn["cell"].isin(tr_cells)]
        aug_frames.append(syn)
        print(f"  + Severity {severity}: {len(syn)} rows")

    # Labels computed on clean data only (augmented copies share same SOH → same labels)
    df_aug = pd.concat(aug_frames, ignore_index=True)
    X_aug = df_aug[FEATURES].values
    print(f"Augmented training: {len(df_aug)} rows")

    # Validation stays on clean data only
    X_val = df.iloc[val_idx][FEATURES].values

    models, calibrators = {}, {}
    for H in H_LIST:
        y_aug = make_composite_fail_in_H(df.iloc[tr_idx], H)
        y_aug = np.tile(y_aug, len(aug_frames))
        y_val = make_composite_fail_in_H(df.iloc[val_idx], H)

        clf = XGBClassifier(**XGB_PARAMS)
        clf.fit(X_aug, y_aug)
        p_raw_val = clf.predict_proba(X_val)[:, 1]

        if len(np.unique(y_val)) < 2 or len(np.unique(y_aug)) < 2:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(np.array([0, 1]), np.array([0, 1]))
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            p_raw_tr = clf.predict_proba(X_aug)[:, 1]
            iso.fit(p_raw_tr, y_aug)

        p_cal_val = iso.transform(p_raw_val)
        b_val = brier_score_loss(y_val, p_cal_val)
        a_val = safe_auc(y_val, p_cal_val)
        e_val = compute_ece(y_val, p_cal_val)
        print(f"  H={H}: Brier={b_val:.4f} AUC={a_val:.4f} ECE={e_val:.4f}")

        models[H] = clf
        calibrators[H] = iso

    out_path = os.path.join(MODELS_DIR, "domain_rand_model.joblib")
    joblib.dump({"models": models, "calibrators": calibrators}, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
