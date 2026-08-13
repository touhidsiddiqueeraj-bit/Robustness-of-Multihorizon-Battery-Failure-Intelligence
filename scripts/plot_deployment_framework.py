"""Deployment framework flow diagram for the WIECON paper (Fig. 6).

Chain: lab model -> fixed hazard model + calibrator -> calibration
monitoring (ECE tracking); field data feeds the fixed model; monitoring
triggers recalibration of the isotonic map on a 20% field sample.
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
OUT = os.path.join(FIGS_DIR, "F_Deployment_Framework.png")


def box(ax, x, y, w, h, text, fc="#e8f0fe", ec="#1a3a6b", fs=8.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                       fc=fc, ec=ec, lw=1.3)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs)


def arrow(ax, x1, y1, x2, y2, text=None, fs=8):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=14, lw=1.2, color="#333333")
    ax.add_patch(a)
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, text, ha="center",
                va="bottom", fontsize=fs, color="#333333")


def main():
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.set_xlim(0, 11.2)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    # ---- Row 1: model + monitoring (y = 4.0) ----
    box(ax, 0.3, 4.0, 2.6, 1.05,
        "Lab model\ntraining\n(clean data)", fs=8)
    box(ax, 3.5, 4.0, 3.0, 0.85,
        "Fixed hazard model\n+ isotonic\ncalibrator", fc="#dbe7fb", fs=8.5)
    box(ax, 7.4, 4.0, 3.2, 0.85,
        "Calibration\nmonitoring\n(ECE tracking)",
        fc="#fde3e3", ec="#8a2f10", fs=8.5)
    arrow(ax, 2.9, 4.42, 3.5, 4.42)
    arrow(ax, 6.5, 4.42, 7.4, 4.42)

    # ---- Row 2: field data + recalibration (y = 1.6) ----
    box(ax, 3.5, 1.6, 3.0, 1.05,
        "Field data\n(partial cycles,\nnoise, rest\nirregularity)",
        fc="#fff3cc", ec="#8a6d00", fs=8.5)
    box(ax, 7.4, 1.6, 3.2, 1.05,
        "Recalibrate\nisotonic regression\n(20% field sample)",
        fc="#eaf6ec", ec="#1e6b33", fs=8.5)

    # field data feeds the fixed model (arrow routed right of the box text)
    arrow(ax, 6.05, 2.65, 6.05, 4.0)
    ax.text(6.32, 3.33, "inputs", fontsize=8, color="#333333",
            ha="left", va="center")
    # monitoring triggers recalibration when ECE drifts
    arrow(ax, 9.0, 4.0, 9.0, 2.65, "ECE drift \u2192 refresh")

    plt.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", dpi=300)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
