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


def main():
    df = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
    df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)
    print(f"Data: {len(df)} rows, {df['cell'].nunique()} cells")

    cells = df["cell"].unique()
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_CELL_FRAC, random_state=RANDOM_STATE)
    tr_idx, val_idx = next(splitter.split(df, groups=df["cell"]))
    val_cells = df.iloc[val_idx]["cell"].unique()
    print(f"Val cells ({len(val_cells)}): {val_cells}")

    X = df[FEATURES].values

    models = {}
    calibrators = {}

    for H in H_LIST:
        y = make_composite_fail_in_H(df, H)

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        clf = XGBClassifier(**XGB_PARAMS)
        clf.fit(X_tr, y_tr)

        p_raw_val = clf.predict_proba(X_val)[:, 1]

        if len(np.unique(y_val)) < 2 or len(np.unique(y_tr)) < 2:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(np.array([0, 1]), np.array([0, 1]))
        else:
            iso = IsotonicRegression(out_of_bounds="clip")
            p_raw_tr = clf.predict_proba(X_tr)[:, 1]
            iso.fit(p_raw_tr, y_tr)

        p_cal_val = iso.transform(p_raw_val)

        b_val = brier_score_loss(y_val, p_cal_val)
        a_val = safe_auc(y_val, p_cal_val)
        e_val = compute_ece(y_val, p_cal_val)
        print(f"  H={H}: Brier={b_val:.4f} AUC={a_val:.4f} ECE={e_val:.4f} (cal on val)")

        models[H] = clf
        calibrators[H] = iso

    joblib.dump({"models": models, "calibrators": calibrators}, os.path.join(MODELS_DIR, "full_model.joblib"))
    print(f"Saved: {os.path.join(MODELS_DIR, 'full_model.joblib')}")

    # save clean baseline metrics for table generation
    rows = []
    for H in H_LIST:
        y = make_composite_fail_in_H(df, H)
        iso = calibrators[H]
        clf = models[H]
        p_raw = clf.predict_proba(X)[:, 1]
        p_cal = iso.transform(p_raw)
        rows.append({"H": H, "ece_cal": compute_ece(y, p_cal), "auc_cal": safe_auc(y, p_cal),
                     "brier_cal": brier_score_loss(y, p_cal)})
    baseline_path = os.path.join(MODELS_DIR, "..", "results", "clean_baseline.csv")
    pd.DataFrame(rows).to_csv(baseline_path, index=False)
    print(f"Saved clean baseline: {baseline_path}")


if __name__ == "__main__":
    main()
