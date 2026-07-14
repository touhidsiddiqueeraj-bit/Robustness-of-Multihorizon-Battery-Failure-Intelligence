import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.isotonic import IsotonicRegression
import joblib

SRC = os.path.dirname(os.path.abspath(__file__))
from composite_label import make_composite_fail_in_H

BASE = os.path.normpath(os.path.join(SRC, ".."))
DATA_DIR = os.path.join(BASE, "data")
FIGS_DIR = os.path.join(BASE, "figs")
RESULTS_DIR = os.path.join(BASE, "results")
MODELS_DIR = os.path.join(BASE, "models")
os.makedirs(FIGS_DIR, exist_ok=True)

FEATURES = ["cycle", "avg_voltage", "min_voltage", "avg_current", "avg_temp", "duration", "SOH"]
H_LIST = [10, 20, 30, 50]
SEEDS = [42, 123, 456, 789, 101112]
PRIMARY_H = 20
N_BINS = 10

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def safe_auc(y_true, p):
    return roc_auc_score(y_true, p) if len(np.unique(y_true)) > 1 else np.nan


def compute_ece(y_true, prob, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (prob >= edges[i]) & (prob < edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += abs(prob[mask].mean() - y_true[mask].mean()) * mask.sum() / len(y_true)
    return ece


def main():
    model_bundle = joblib.load(os.path.join(MODELS_DIR, "full_model.joblib"))
    models = model_bundle["models"]
    calibrators = model_bundle["calibrators"]

    clean = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
    clean = clean.sort_values(["cell", "cycle"]).reset_index(drop=True)
    X_clean = clean[FEATURES].values
    y_clean = make_composite_fail_in_H(clean, PRIMARY_H)

    results = pd.read_csv(os.path.join(RESULTS_DIR, "robustness_results.csv"))
    res_h = results[results.H == PRIMARY_H]

    # ---- Fig 1: Reliability diagrams (4-panel) ----
    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes = axes.flatten()

    for idx, severity in enumerate([1, 2, 3, 4]):
        ax = axes[idx]

        # Plot each seed as thin line, plus mean
        all_fp = []
        all_mpv = []
        seed_data = res_h[res_h.severity == severity]

        for seed_val in SEEDS:
            syn_path = os.path.join(DATA_DIR, "synthetic", f"nasa_perturbed_s{severity}_s{seed_val}.csv")
            df = pd.read_csv(syn_path)
            df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)
            X = df[FEATURES].values
            model = models[PRIMARY_H]
            calibrator = calibrators[PRIMARY_H]
            p_cal = calibrator.transform(model.predict_proba(X)[:, 1])

            fp, mpv = calibration_curve(y_clean, p_cal, n_bins=N_BINS, strategy="uniform")
            all_fp.append(fp)
            all_mpv.append(mpv)

            ax.plot(fp, mpv, color="steelblue", alpha=0.2, linewidth=0.8)

        # Mean across seeds
        max_len = max(len(f) for f in all_fp)
        fp_interp = np.linspace(0, 1, max_len)
        mpv_interp = np.zeros(max_len)
        for mpv_arr in all_mpv:
            xp = np.linspace(0, 1, len(mpv_arr))
            mpv_interp += np.interp(fp_interp, xp, mpv_arr)
        mpv_interp /= len(all_mpv)

        ax.plot(fp_interp, mpv_interp, "steelblue", linewidth=2, label="Prediction")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")

        ece_mean = res_h[res_h.severity == severity]["ece_cal"].mean()
        ece_std = res_h[res_h.severity == severity]["ece_cal"].std()
        auc_mean = res_h[res_h.severity == severity]["auc_cal"].mean()
        ax.set_title(f"Severity {severity}  (ECE={ece_mean:.3f}±{ece_std:.3f})")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "F_Reliability_Diagrams.png"), dpi=150)
    print("Saved: F_Reliability_Diagrams.png")

    # ---- Fig 2: ECE vs Severity ----
    fig, ax = plt.subplots(figsize=(6, 4))

    severities = [1, 2, 3, 4]
    methods = {
        "Fixed calibrator": "ece_cal",
        "Recalibrated (isotonic)": "ece_recal",
        "Recalibrated (Platt)": "ece_recal_platt",
    }
    colors = {"Fixed calibrator": "crimson", "Recalibrated (isotonic)": "seagreen", "Recalibrated (Platt)": "purple"}
    markers = {"Fixed calibrator": "o", "Recalibrated (isotonic)": "s", "Recalibrated (Platt)": "D"}

    for label, col in methods.items():
        means = []
        stds = []
        for s in severities:
            vals = res_h[res_h.severity == s][col].values
            means.append(vals.mean())
            stds.append(vals.std())
        ax.errorbar(severities, means, yerr=stds, label=label,
                    color=colors[label], marker=markers[label],
                    linewidth=2, capsize=4, markersize=8)

    # Clean baseline
    model = models[PRIMARY_H]
    calibrator = calibrators[PRIMARY_H]
    p_clean_cal = calibrator.transform(model.predict_proba(X_clean)[:, 1])
    clean_ece = compute_ece(y_clean, p_clean_cal)
    ax.axhline(clean_ece, color="gray", linestyle="--", linewidth=1.5,
               label=f"Clean baseline (ECE={clean_ece:.3f})")

    ax.set_xlabel("Perturbation severity")
    ax.set_ylabel("Expected Calibration Error (ECE)")
    ax.set_xticks(severities)
    ax.set_xticklabels(["Mild", "Moderate", "Severe", "Aggressive"])
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "F_ECE_vs_Severity.png"), dpi=150)
    print("Saved: F_ECE_vs_Severity.png")

    # ---- Fig 3: AUC vs Severity ----
    fig, ax = plt.subplots(figsize=(6, 4))

    means = [res_h[res_h.severity == s]["auc_cal"].mean() for s in severities]
    stds = [res_h[res_h.severity == s]["auc_cal"].std() for s in severities]
    ax.errorbar(severities, means, yerr=stds, label="AUC (calibrated)",
                color="darkorange", marker="^", linewidth=2, capsize=4, markersize=8)

    clean_auc = safe_auc(y_clean, p_clean_cal)
    ax.axhline(clean_auc, color="gray", linestyle="--", linewidth=1.5,
               label=f"Clean baseline (AUC={clean_auc:.3f})")

    ax.set_xlabel("Perturbation severity")
    ax.set_ylabel("AUC")
    ax.set_xticks(severities)
    ax.set_xticklabels(["Mild", "Moderate", "Severe", "Aggressive"])
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGS_DIR, "F_AUC_vs_Severity.png"), dpi=150)
    print("Saved: F_AUC_vs_Severity.png")

    # ---- Fig 4: Combined figure (2x2: reliability + ECE + AUC + table) ----
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.30,
                          left=0.08, right=0.95, bottom=0.08, top=0.95)

    # Panel A: Reliability diagram for severity 3 (exemplary)
    ax = fig.add_subplot(gs[0, 0])
    all_fp, all_mpv = [], []
    for seed_val in SEEDS:
        syn_path = os.path.join(DATA_DIR, "synthetic", f"nasa_perturbed_s3_s{seed_val}.csv")
        df = pd.read_csv(syn_path)
        df = df.sort_values(["cell", "cycle"]).reset_index(drop=True)
        X = df[FEATURES].values
        model = models[PRIMARY_H]
        calibrator = calibrators[PRIMARY_H]
        p_cal = calibrator.transform(model.predict_proba(X)[:, 1])
        fp, mpv = calibration_curve(y_clean, p_cal, n_bins=N_BINS, strategy="uniform")
        all_fp.append(fp)
        all_mpv.append(mpv)
        ax.plot(fp, mpv, color="steelblue", alpha=0.2, linewidth=0.8)
    max_len = max(len(f) for f in all_fp)
    fp_interp = np.linspace(0, 1, max_len)
    mpv_interp = np.zeros(max_len)
    for mpv_arr in all_mpv:
        xp = np.linspace(0, 1, len(mpv_arr))
        mpv_interp += np.interp(fp_interp, xp, mpv_arr)
    mpv_interp /= len(all_mpv)
    ax.plot(fp_interp, mpv_interp, "steelblue", linewidth=2, label="Prediction")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (Severity 3)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel B: ECE vs Severity
    ax = fig.add_subplot(gs[0, 1])
    for label, col in methods.items():
        means = [res_h[res_h.severity == s][col].mean() for s in severities]
        stds = [res_h[res_h.severity == s][col].std() for s in severities]
        ax.errorbar(severities, means, yerr=stds, label=label,
                    color=colors[label], marker=markers[label],
                    linewidth=2, capsize=4, markersize=8)
    ax.axhline(clean_ece, color="gray", linestyle="--", linewidth=1.5,
               label=f"Clean baseline")
    ax.set_xlabel("Perturbation severity")
    ax.set_ylabel("ECE")
    ax.set_xticks(severities)
    ax.set_xticklabels(["Mild", "Moderate", "Severe", "Aggressive"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Calibration Error vs Severity")

    # Panel C: AUC vs Severity
    ax = fig.add_subplot(gs[1, 0])
    means = [res_h[res_h.severity == s]["auc_cal"].mean() for s in severities]
    stds = [res_h[res_h.severity == s]["auc_cal"].std() for s in severities]
    ax.errorbar(severities, means, yerr=stds, label="AUC (calibrated)",
                color="darkorange", marker="^", linewidth=2, capsize=4, markersize=8)
    ax.axhline(clean_auc, color="gray", linestyle="--", linewidth=1.5,
               label=f"Clean baseline")
    ax.set_xlabel("Perturbation severity")
    ax.set_ylabel("AUC")
    ax.set_xticks(severities)
    ax.set_xticklabels(["Mild", "Moderate", "Severe", "Aggressive"])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Discrimination vs Severity")

    # Panel D: Summary table
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    table_data = [["Severity", "ECE", "ECE (recal)", "AUC"]]
    for s in severities:
        sub = res_h[res_h.severity == s]
        table_data.append([
            f"S{s}", f"{sub['ece_cal'].mean():.3f}", f"{sub['ece_recal'].mean():.3f}", f"{sub['auc_cal'].mean():.3f}"
        ])
    table_data.append(["Clean", f"{clean_ece:.3f}", "—", f"{clean_auc:.3f}"])

    col_widths = [0.18, 0.22, 0.28, 0.22]
    table = ax.table(cellText=table_data, loc="center", cellLoc="center",
                     colWidths=col_widths)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    for j, key in enumerate(["Severity", "ECE", "ECE (recal)", "AUC"]):
        table[0, j].set_facecolor("#40466e")
        table[0, j].set_text_props(color="w", fontweight="bold")
    for i in range(1, len(table_data)):
        for j in range(4):
            table[i, j].set_facecolor("#f0f0f0" if i % 2 == 0 else "white")
    ax.set_title("Summary Table", fontsize=12, pad=8)

    fig.savefig(os.path.join(FIGS_DIR, "F_Combined_Robustness.png"), dpi=150)
    print("Saved: F_Combined_Robustness.png")

    print("\nAll figures saved.")


if __name__ == "__main__":
    main()
