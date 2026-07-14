import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
FIGS_DIR = os.path.join(BASE, "figs")
RESULTS_DIR = os.path.join(BASE, "results")

PRIMARY_H = 20
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.labelsize": 12, "axes.titlesize": 13,
    "legend.fontsize": 10, "figure.dpi": 150,
})

def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "robustness_results_sweep.csv"))
    df = df[df.H == PRIMARY_H]

    clean_baseline = pd.read_csv(os.path.join(RESULTS_DIR, "clean_baseline.csv"))
    clean_ece = clean_baseline[clean_baseline.H == PRIMARY_H]["ece_cal"].values[0]

    fig, ax = plt.subplots(figsize=(6, 4))

    fracs = sorted(df.cal_frac.unique())
    severities = [1, 2, 3, 4]
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}

    for s in severities:
        means, stds = [], []
        for f in fracs:
            vals = df[(df.severity == s) & (df.cal_frac == f)].ece_recal.values
            means.append(vals.mean())
            stds.append(vals.std())
        ax.errorbar(fracs, means, yerr=stds, label=f"Severity {s}",
                    color=colors[s], marker=markers[s], linewidth=2, capsize=4, markersize=8)

    ax.axhline(clean_ece, color="gray", linestyle="--", linewidth=1.5,
               label=f"Clean baseline (ECE={clean_ece:.3f})")

    ax.set_xlabel("Calibration sample fraction")
    ax.set_ylabel("ECE (held-out)")
    ax.set_xticks(fracs)
    ax.set_xticklabels([f"{int(f*100)}%" for f in fracs])
    ax.set_xlim(0.02, 0.55)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "F_Sample_Sweep.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
