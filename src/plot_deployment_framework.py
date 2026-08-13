"""Deployment framework diagram (audit #9): lab training -> field operation
-> shift -> recalibration -> dispatch."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from plot_style import apply_style
apply_style()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS_DIR = os.path.join(BASE, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(9, 4.2))
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

BW, BH = 0.185, 0.40
boxes = [
    (0.005, 0.50, "Lab model\ntraining\n(clean data)", "#eef2ff"),
    (0.275, 0.50, "Fixed hazard model\n+ isotonic\ncalibrator", "#eef2ff"),
    (0.545, 0.50, "Calibration\nmonitoring\n(ECE tracking)", "#fee2e2"),
    (0.815, 0.50, "Recalibrate\nisotonic regression\n(10% sample)", "#dcfce7"),
    (0.42, 0.03, "Field data\n(partial cycles,\nnoise, rest\nirregularity)", "#fef3c7"),
]
for x, y, text, color in boxes:
    bbox = FancyBboxPatch((x, y), BW, BH, boxstyle="round,pad=0.006",
                          fc=color, ec="#334155", lw=1.5)
    ax.add_patch(bbox)
    ax.text(x + BW / 2, y + BH / 2, text, ha="center", va="center", fontsize=10.5)

arrows = [
    (0.190, 0.70, 0.275, 0.70),   # training -> fixed model
    (0.460, 0.70, 0.545, 0.70),   # fixed model -> monitoring
    (0.730, 0.70, 0.815, 0.70),   # monitoring -> recalibrate
    (0.815, 0.50, 0.815, 0.43),   # recalibrate tail down (endpoint outside box)
    (0.512, 0.43, 0.512, 0.50),   # field data -> monitoring (up)
]
for x1, y1, x2, y2 in arrows:
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, color="#334155", lw=1.8))

ax.text(0.23, 0.955, "Laboratory phase", ha="center", fontsize=13,
        fontstyle="italic", color="#475569")
ax.text(0.74, 0.955, "Field phase", ha="center", fontsize=13,
        fontstyle="italic", color="#475569")

plt.tight_layout(pad=0.3)
out = os.path.join(FIGS_DIR, "F_Deployment_Framework.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
