"""Recalibration data-requirement figure for the WIECON short paper.

Plots held-out ECE vs. calibration sample fraction (5/10/20/50%) for all
four horizons, with spread across severities x seeds, plus clean-baseline
and no-recalibration reference lines per horizon.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))
from plot_style import apply_style

apply_style()

RESULTS_DIR = os.path.join(BASE, "results")
FIGS_DIR = os.path.join(BASE, "figs")
HORIZONS = [10, 20, 30, 50]
COLORS = {10: "#1f77b4", 20: "#ff7f0e", 30: "#2ca02c", 50: "#d62728"}
MARKERS = {10: "o", 20: "s", 30: "^", 50: "D"}


def main():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "robustness_results_sweep.csv"))
    pert = pd.read_csv(os.path.join(RESULTS_DIR, "robustness_results.csv"))
    clean = pd.read_csv(os.path.join(RESULTS_DIR, "clean_baseline.csv"))

    fracs = sorted(df.cal_frac.unique())
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for h in HORIZONS:
        means, stds = [], []
        for f in fracs:
            v = df[(df.H == h) & (df.cal_frac == f)].ece_recal.values
            means.append(v.mean())
            stds.append(v.std())
        ax.errorbar(fracs, means, yerr=stds, label=f"H = {h}",
                    color=COLORS[h], marker=MARKERS[h], linewidth=2,
                    capsize=4, markersize=7)
        p = pert[pert.H == h].ece_cal.mean()
        ax.scatter([0.02], [p], color=COLORS[h], marker="x", s=60, zorder=5)
        c = clean[clean.H == h].ece_cal.values[0]
        ax.axhline(c, color=COLORS[h], linestyle=":", linewidth=1.2, alpha=0.6)

    ax.set_xlabel("Field calibration sample fraction")
    ax.set_ylabel("Held-out ECE after recalibration")
    ax.set_xticks(fracs)
    ax.set_xticklabels([f"{int(f * 100)}%" for f in fracs])
    ax.set_xlim(0.02, 0.55)
    ax.legend(ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(FIGS_DIR, "F_Recal_Sweep.png")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
