"""Generate robustness figures (reliability diagrams, ECE/AUC vs severity).

FIXES APPLIED (vs. original repo):
  1. All plots now use VAL cells only (consistent with the fixed
     train_full_model.py and run_robustness.py).
  2. Figure 2 (AUC vs severity) caption is corrected: AUC DOES decline
     visibly from S1 to S4 (was incorrectly captioned "AUC remains stable").
  3. Both raw and calibrated AUC are plotted so reviewers can see that
     recalibration does not meaningfully change AUC (rank-preserving).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
import joblib

SRC = os.path.dirname(os.path.abspath(__file__))
from composite_label import make_composite_fail_in_H
from plot_style import apply_style

apply_style()

BASE = os.path.normpath(os.path.join(SRC, ".."))
DATA_DIR = os.path.join(BASE, "data")
FIGS_DIR = os.path.join(BASE, "figs")
RESULTS_DIR = os.path.join(BASE, "results")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(FIGS_DIR, exist_ok=True)

FEATURES = ["cycle", "avg_voltage", "min_voltage", "avg_current", "avg_temp", "duration", "SOH"]
H_LIST = [10, 20, 30, 50]
SEEDS = [42, 123, 456, 789, 101112]
SEED_COLORS = plt.cm.Set2(np.linspace(0.05, 0.95, len(SEEDS)))
PRIMARY_H = 20
N_BINS = 10


def safe_auc(y_true, p):
    return roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan


def compute_ece(y_true, prob, bins=10):
    from compute_ece import compute_ece as _ece
    return _ece(y_true, prob, bins)


def main():
    model_bundle = joblib.load(os.path.join(MODELS_DIR, "full_model.joblib"))
    models = model_bundle["models"]
    calibrators = model_bundle["calibrators"]

    split = pd.read_csv(os.path.join(RESULTS_DIR, "train_val_split.csv"))
    held_out_label = "test" if (split["split"] == "test").any() else "val"
    val_row_idx = split.index[split["split"] == held_out_label].values

    clean = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
    clean = clean.sort_values(["cell", "cycle"]).reset_index(drop=True)
    X_clean_val = clean.iloc[val_row_idx][FEATURES].values
    y_clean_val = make_composite_fail_in_H(clean, PRIMARY_H)[val_row_idx]

    results = pd.read_csv(os.path.join(RESULTS_DIR, "robustness_results.csv"))
    res_h = results[results.H == PRIMARY_H]

    # ---- Fig 1: Reliability diagrams (4-panel) ----
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes = axes.flatten()
    seed_handles, seed_labels = [], []

    for idx, severity in enumerate([1, 2, 3, 4]):
        ax = axes[idx]
        all_fp, all_mpv = [], []

        for i, seed_val in enumerate(SEEDS):
            syn_path = os.path.join(DATA_DIR, "synthetic", f"nasa_perturbed_s{severity}_s{seed_val}.csv")
            df = pd.read_csv(syn_path)
            df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)
            X = df.iloc[val_row_idx][FEATURES].values
            model = models[PRIMARY_H]
            calibrator = calibrators[PRIMARY_H]
            p_cal = calibrator.transform(model.predict_proba(X)[:, 1])

            fp, mpv = calibration_curve(y_clean_val, p_cal, n_bins=N_BINS, strategy="uniform")
            all_fp.append(fp)
            all_mpv.append(mpv)

            h, = ax.plot(fp, mpv, color=SEED_COLORS[i], alpha=0.65, linewidth=1.4)
            if idx == 0:
                seed_handles.append(h)
                seed_labels.append(f"Seed {seed_val}")

        max_len = max(len(f) for f in all_fp)
        fp_interp = np.linspace(0, 1, max_len)
        mpv_interp = np.zeros(max_len)
        for mpv_arr in all_mpv:
            xp = np.linspace(0, 1, len(mpv_arr))
            mpv_interp += np.interp(fp_interp, xp, mpv_arr)
        mpv_interp /= len(all_mpv)

        ax.plot(fp_interp, mpv_interp, "steelblue", linewidth=2, label="Prediction")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")

        ece_mean = res_h[res_h.severity == severity]["ece_cal"].mean()
        ece_std = res_h[res_h.severity == severity]["ece_cal"].std()
        ax.set_title(f"S{severity}  (ECE={ece_mean:.3f}$\\pm${ece_std:.3f})", pad=10)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.legend(handles=seed_handles, labels=seed_labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.055), ncol=len(SEEDS), fontsize=13, frameon=False)
    fig.subplots_adjust(hspace=0.7, wspace=0.55, bottom=0.22, top=0.88)

    fig.savefig(os.path.join(FIGS_DIR, "F_Reliability_Diagrams.png"), bbox_inches="tight", pad_inches=0.15)
    print("Saved: F_Reliability_Diagrams.png")

    # ---- Fig 2: ECE vs Severity (raw, fixed-cal, recal-iso, recal-platt, recal-temp) ----
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    severities = [1, 2, 3, 4]
    methods = {
        "Raw (no cal.)": "ece_raw",
        "Fixed cal. (clean-fit)": "ece_cal",
        "Recal. isotonic": "ece_recal",
        "Recal. Platt": "ece_recal_platt",
        "Recal. temp. scaling": "ece_recal_temp",
    }
    colors = {"Raw (no cal.)": "#9ca3af", "Fixed cal. (clean-fit)": "crimson",
              "Recal. isotonic": "seagreen", "Recal. Platt": "purple",
              "Recal. temp. scaling": "royalblue"}
    markers = {"Raw (no cal.)": "x", "Fixed cal. (clean-fit)": "o",
               "Recal. isotonic": "s", "Recal. Platt": "D", "Recal. temp. scaling": "^"}

    for label, col in methods.items():
        means, stds = [], []
        for s in severities:
            vals = res_h[res_h.severity == s][col].values
            means.append(vals.mean())
            stds.append(vals.std())
        ax.errorbar(severities, means, yerr=stds, label=label,
                    color=colors[label], marker=markers[label],
                    linewidth=2, capsize=4, markersize=8)

    # Clean baseline (val only)
    model = models[PRIMARY_H]
    calibrator = calibrators[PRIMARY_H]
    p_clean_cal = calibrator.transform(model.predict_proba(X_clean_val)[:, 1])
    clean_ece = compute_ece(y_clean_val, p_clean_cal)
    ax.axhline(clean_ece, color="gray", linestyle="--", linewidth=1.5,
               label=f"Clean baseline (ECE={clean_ece:.3f})")

    ax.set_xlabel("Perturbation severity")
    ax.set_ylabel("Expected Calibration Error (ECE)", labelpad=8)
    ax.set_xticks(severities)
    ax.set_xticklabels(["Mild", "Moderate", "Severe", "Aggr."])
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "F_ECE_vs_Severity.png"),
                bbox_inches="tight", pad_inches=0.3)
    print("Saved: F_ECE_vs_Severity.png")

    # ---- Fig 3: AUC vs Severity (raw, cal, recal) — caption FIXED ----
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # Plot raw, calibrated, and recalibrated AUC — they should be nearly identical
    # because isotonic is rank-preserving (modulo out-of-range clipping)
    for label, col, color, marker in [
        ("AUC (raw)",       "auc_raw",   "#9ca3af", "x"),
        ("AUC (fixed cal.)", "auc_cal",   "darkorange", "^"),
        ("AUC (recal. iso)", "auc_recal", "seagreen", "s"),
    ]:
        means = [res_h[res_h.severity == s][col].mean() for s in severities]
        stds  = [res_h[res_h.severity == s][col].std() for s in severities]
        ax.errorbar(severities, means, yerr=stds, label=label,
                    color=color, marker=marker, linewidth=2, capsize=4, markersize=8)

    clean_auc = safe_auc(y_clean_val, p_clean_cal)
    ax.axhline(clean_auc, color="gray", linestyle="--", linewidth=1.5,
               label=f"Clean baseline (AUC={clean_auc:.3f})")

    ax.set_xlabel("Perturbation severity")
    ax.set_ylabel("AUC")
    ax.set_xticks(severities)
    ax.set_xticklabels(["Mild", "Moderate", "Severe", "Aggr."])
    ax.legend(loc="lower left", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    # Annotate the visible decline (was incorrectly captioned "remains stable")
    s1_mean = res_h[res_h.severity == 1]["auc_cal"].mean()
    s4_mean = res_h[res_h.severity == 4]["auc_cal"].mean()
    ax.annotate(f"Decline: {s1_mean:.2f}→{s4_mean:.2f}",
                xy=(4, s4_mean), xytext=(3.0, s4_mean - 0.10),
                fontsize=12, color="darkorange",
                arrowprops=dict(arrowstyle="->", color="darkorange", lw=1))
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "F_AUC_vs_Severity.png"), bbox_inches="tight")
    print("Saved: F_AUC_vs_Severity.png")

    # ---- Fig 4: Combined figure ----
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30,
                          left=0.08, right=0.95, bottom=0.08, top=0.95)

    # Panel A: Reliability diagram for severity 1 (saturation onset)
    ax = fig.add_subplot(gs[0, 0])
    all_fp, all_mpv = [], []
    for i, seed_val in enumerate(SEEDS):
        syn_path = os.path.join(DATA_DIR, "synthetic", f"nasa_perturbed_s1_s{seed_val}.csv")
        df = pd.read_csv(syn_path)
        df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)
        X = df.iloc[val_row_idx][FEATURES].values
        model = models[PRIMARY_H]
        calibrator = calibrators[PRIMARY_H]
        p_cal = calibrator.transform(model.predict_proba(X)[:, 1])
        fp, mpv = calibration_curve(y_clean_val, p_cal, n_bins=N_BINS, strategy="uniform")
        all_fp.append(fp)
        all_mpv.append(mpv)
        ax.plot(fp, mpv, color=SEED_COLORS[i], alpha=0.5, linewidth=0.8, label=f"Seed {seed_val}")
    max_len = max(len(f) for f in all_fp)
    fp_interp = np.linspace(0, 1, max_len)
    mpv_interp = np.zeros(max_len)
    for mpv_arr in all_mpv:
        xp = np.linspace(0, 1, len(mpv_arr))
        mpv_interp += np.interp(fp_interp, xp, mpv_arr)
    mpv_interp /= len(all_mpv)
    ax.plot(fp_interp, mpv_interp, "steelblue", linewidth=2, label="Prediction")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (Severity 1 = mildest)")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    # Panel B: ECE vs Severity
    ax = fig.add_subplot(gs[0, 1])
    for label, col in methods.items():
        means = [res_h[res_h.severity == s][col].mean() for s in severities]
        stds = [res_h[res_h.severity == s][col].std() for s in severities]
        ax.errorbar(severities, means, yerr=stds, label=label,
                    color=colors[label], marker=markers[label],
                    linewidth=2, capsize=4, markersize=8)
    ax.axhline(clean_ece, color="gray", linestyle="--", linewidth=1.5, label="Clean baseline")
    ax.set_xlabel("Perturbation severity")
    ax.set_ylabel("ECE")
    ax.set_xticks(severities)
    ax.set_xticklabels(["S1", "S2", "S3", "S4"])
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_title("Calibration Error vs Severity")

    # Panel C: AUC vs Severity
    ax = fig.add_subplot(gs[1, 0])
    for label, col, color, marker in [
        ("AUC (raw)", "auc_raw", "#9ca3af", "x"),
        ("AUC (fixed cal.)", "auc_cal", "darkorange", "^"),
        ("AUC (recal. iso)", "auc_recal", "seagreen", "s"),
    ]:
        means = [res_h[res_h.severity == s][col].mean() for s in severities]
        stds = [res_h[res_h.severity == s][col].std() for s in severities]
        ax.errorbar(severities, means, yerr=stds, label=label,
                    color=color, marker=marker, linewidth=2, capsize=4, markersize=8)
    ax.axhline(clean_auc, color="gray", linestyle="--", linewidth=1.5, label="Clean baseline")
    ax.set_xlabel("Perturbation severity")
    ax.set_ylabel("AUC")
    ax.set_xticks(severities)
    ax.set_xticklabels(["S1", "S2", "S3", "S4"])
    ax.legend(fontsize=11, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Discrimination vs Severity (declining, NOT stable)")

    # Panel D: Summary table
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    table_data = [["Sev.", "ECE raw", "ECE cal", "ECE rec.", "AUC cal"]]
    for s in severities:
        sub = res_h[res_h.severity == s]
        table_data.append([
            f"S{s}",
            f"{sub['ece_raw'].mean():.3f}",
            f"{sub['ece_cal'].mean():.3f}",
            f"{sub['ece_recal'].mean():.3f}",
            f"{sub['auc_cal'].mean():.3f}",
        ])
    table_data.append(["Clean", "—", f"{clean_ece:.3f}", "—", f"{clean_auc:.3f}"])

    col_widths = [0.15, 0.18, 0.18, 0.22, 0.18]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center", colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.0, 1.9)
    for j, key in enumerate(["Sev.", "ECE raw", "ECE cal", "ECE rec.", "AUC cal"]):
        table[0, j].set_facecolor("#40466e")
        table[0, j].set_text_props(color="w", fontweight="bold", fontsize=12)
    for i in range(1, len(table_data)):
        for j in range(5):
            table[i, j].set_facecolor("#f0f0f0" if i % 2 == 0 else "white")
    ax.set_title("Summary Table (H=20, val cells)", fontsize=16, pad=10)

    fig.savefig(os.path.join(FIGS_DIR, "F_Combined_Robustness.png"), bbox_inches="tight")
    print("Saved: F_Combined_Robustness.png")

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
