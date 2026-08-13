"""Shared matplotlib style for all paper figures.

Readability at 100% zoom in IEEE double-column format requires
generous fonts and high resolution.  All plot scripts import this.
"""
import matplotlib.pyplot as plt


def apply_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.5,
        "lines.markersize": 9,
        "errorbar.capsize": 5,
    })
