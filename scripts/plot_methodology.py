"""Methodology flow diagram for the WIECON paper (Fig. 1).

Pipeline: NASA raw curves -> feature extraction -> XGBoost + OOF isotonic
calibration (frozen) -> perturbation generator (4 severities x 5 seeds)
-> recalibration sweep on field sample -> held-out ECE evaluation.
Domain-randomization branch shown as dashed alternative.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(BASE, "src"))
from plot_style import apply_style

apply_style()

FIGS_DIR = os.path.join(BASE, "figs")
OUT = os.path.join(FIGS_DIR, "F_Methodology.png")


def box(ax, x, y, w, h, text, fc="#e8f0fe", ec="#1a3a6b", fs=8.5, dashed=False):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                       fc=fc, ec=ec, lw=1.3, linestyle="--" if dashed else "-")
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs)


def arrow(ax, x1, y1, x2, y2, text=None, fs=8):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=14, lw=1.2, color="#333333")
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.04, text, ha="center",
                va="bottom", fontsize=fs, color="#333333")


def main():
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    # ---- Row 1: training path (y = 7.0) ----
    box(ax, 0.3, 7.0, 2.9, 0.8,
        "NASA .mat raw cycling\ncurves (37 cells,\n1,028 cycles)", fs=8.5)
    box(ax, 3.9, 7.0, 2.9, 0.8,
        "Feature extraction\n7 per-cycle features\n(SOH, V, I, T, duration,\nt, Vmin)", fs=8)
    box(ax, 7.4, 7.0, 2.9, 0.8,
        "XGBoost + OOF isotonic\ncalibrator (clean train,\n865 rows, GroupKFold)", fs=8)
    arrow(ax, 3.2, 7.4, 3.9, 7.4)
    arrow(ax, 6.8, 7.4, 7.4, 7.4)

    # Frozen model below XGBoost
    box(ax, 7.4, 5.3, 2.9, 0.85,
        "Frozen hazard model\n(classifier + calibrator,\nnever retrained)",
        fc="#fdeee8", ec="#8a2f10", fs=8.5)
    arrow(ax, 8.85, 7.0, 8.85, 6.15)

    # ---- Row 2: perturbation generator (y = 5.0) ----
    box(ax, 0.3, 5.0, 6.1, 1.15,
        "Perturbation generator\npartial discharge (DoD 15\u2013100%) \u00b7 temp. noise \u03c3 0.5\u20133.0 \u00b0C\nrest irregularity \u03c1 0.01\u20130.05 \u00b7 4 severities \u00d7 5 seeds",
        fc="#eaf6ec", ec="#1e6b33", fs=8.5)
    arrow(ax, 2.5, 5.0, 2.5, 4.3, "perturbed curves")

    # ---- Row 3: datasets + sample (y = 3.4) ----
    box(ax, 0.3, 3.4, 2.9, 0.9,
        "20 perturbed datasets\n(20,560 records)", fs=9)
    box(ax, 3.9, 3.4, 2.5, 0.9,
        "Field sample\n5/10/20/50% of\nheld-out data", fs=8.5)
    arrow(ax, 3.2, 3.85, 3.9, 3.85)

    # ---- Row 4: recalibration + evaluation (y = 1.5) ----
    box(ax, 7.4, 3.4, 2.9, 0.9,
        "Refit isotonic map on\nfield sample (recalibra-\ntion; base model frozen)",
        fc="#f0e9fa", ec="#4a2a8a", fs=8.5)
    arrow(ax, 6.4, 3.85, 7.4, 3.85)
    ax.text(6.65, 3.89, "held-out inputs", ha="center", va="bottom",
            fontsize=8, color="#333333")
    box(ax, 7.4, 1.5, 2.9, 1.0,
        "Evaluation on held-out subset\nECE before/after recalibration\npaired bootstrap 95% CI",
        fs=8.5)
    arrow(ax, 8.85, 3.4, 8.85, 2.5)

    # ---- Bottom: comparison (y = 0.3) ----
    box(ax, 0.3, 0.3, 2.9, 0.85,
        "Domain randomization\n(retrain on 4,325\naugmented rows)",
        fc="#f7f7f7", ec="#666666", dashed=True, fs=8.5)
    box(ax, 3.9, 0.3, 2.5, 0.85,
        "Recalibration\n(lightweight, no\nretraining)", fs=8.5)
    arrow(ax, 3.2, 0.72, 3.9, 0.72, "vs.")
    # evaluation feeds the outcome banner (bottom center-right)
    arrow(ax, 8.85, 1.5, 6.9, 0.85)

    ax.text(8.0, 0.7, "Outcome: recalibration restores\ncalibration; domain randomization\ndoes not beat it at long horizons",
            ha="center", va="center", fontsize=8, color="#1a3a6b",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fffde7", ec="#b8a800"))

    plt.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
