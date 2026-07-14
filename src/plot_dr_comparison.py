"""Plot domain-randomization vs standard-model ECE across severities."""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.normpath(os.path.join(BASE, "..", "figs"))
RESULTS = os.path.normpath(os.path.join(BASE, "..", "results"))

std = pd.read_csv(os.path.join(RESULTS, "robustness_results.csv"))
dr = pd.read_csv(os.path.join(RESULTS, "domain_rand_results.csv"))

H_LIST = [10, 20, 30, 50]
SEVERITIES = [1, 2, 3, 4]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
h_centers = np.arange(len(H_LIST))
width = 0.35

# DR model ECE is isotonic-recalibrated, compare vs std isotonic (ece_recal)
for ax, (std_col, dr_col, label) in zip(axes, [
    ("ece_recal", "dr_ece_cal", "ECE"),
    ("auc_cal", "dr_auc_cal", "AUC"),
]):
    colors_s = plt.cm.Greys(np.linspace(0.3, 0.8, len(SEVERITIES)))

    for si, s in enumerate(SEVERITIES):
        means_std, errs_std, means_dr, errs_dr = [], [], [], []
        for H in H_LIST:
            srows = std[(std.severity == s) & (std.H == H)]
            d_rows = dr[(dr.severity == s) & (dr.H == H)]
            means_std.append(srows[std_col].mean())
            er = srows[std_col].std(ddof=0)
            errs_std.append(er if not np.isnan(er) else 0)
            means_dr.append(d_rows[dr_col].mean())
            er = d_rows[dr_col].std(ddof=0)
            errs_dr.append(er if not np.isnan(er) else 0)

        offset = (si - 1.5) * width / 4 + width / 8
        ax.bar(h_centers + offset - width / 4, means_std, width / 4,
               yerr=errs_std, capsize=2, color=colors_s[si], alpha=0.7,
               label=f"Std s={s}" if label == "ECE" else None)
        ax.bar(h_centers + offset + width / 4, means_dr, width / 4,
               yerr=errs_dr, capsize=2, color=colors_s[si], alpha=1.0,
               hatch="///", label=f"DR s={s}" if label == "ECE" else None)

    ax.set_xticks(h_centers)
    ax.set_xticklabels([f"H={h}" for h in H_LIST])
    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel(label)
    ax.set_title(f"{label}: Standard (isotonic) vs Domain-Randomized")
    if label == "ECE":
        ax.legend(fontsize=7, ncol=2)

plt.tight_layout()
out = os.path.join(FIGS, "F_DR_Comparison.png")
plt.savefig(out, dpi=200)
plt.close()
print(f"Saved: {out}")
