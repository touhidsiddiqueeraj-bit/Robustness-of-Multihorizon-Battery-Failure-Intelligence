import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from plot_style import apply_style
apply_style()

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
FIGS_DIR = os.path.join(BASE, "figs")
os.makedirs(FIGS_DIR, exist_ok=True)

FEATURES = ["cycle", "avg_voltage", "min_voltage", "avg_current", "avg_temp", "duration", "SOH"]
FEATURE_LABELS = {"min_voltage": "V_min (V)", "avg_voltage": "V_avg (V)", "duration": "Duration (s)", "avg_temp": "T_avg (°C)"}
COLORS = {"Clean": "#16a34a", "S1": "#f59e0b", "S2": "#d97706", "S3": "#dc2626", "S4": "#7f1d1d"}

clean = pd.read_csv(os.path.join(DATA_DIR, "nasa_clean_filtered.csv"))
clean["condition"] = "Clean"

dfs = [clean[["min_voltage", "condition"]]]
for s in [1, 2, 3, 4]:
    p = pd.read_csv(os.path.join(DATA_DIR, "synthetic", f"nasa_perturbed_s{s}_s42.csv"))
    p["condition"] = f"S{s}"
    dfs.append(p[["min_voltage", "condition"]])

df = pd.concat(dfs, ignore_index=True)

fig, ax = plt.subplots(figsize=(10, 5))
order = ["Clean", "S1", "S2", "S3", "S4"]
pal = [COLORS[c] for c in order]

parts = sns.violinplot(data=df, x="condition", y="min_voltage", order=order, palette=pal, ax=ax, cut=0, inner="quartile", linewidth=1.2)
ax.set_xlabel("")
ax.set_ylabel("Minimum voltage (V)", fontsize=16)
ax.tick_params(labelsize=13)

# annotate the clean and S1 means
means = df.groupby("condition")["min_voltage"].mean()
ax.annotate(f"{means['Clean']:.2f}V", xy=(0, means['Clean']), xytext=(0.44, means['Clean']-0.25),
            fontsize=13, fontweight="bold", color=COLORS["Clean"],
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5),
            arrowprops=dict(arrowstyle="->", color=COLORS["Clean"], lw=1.5))
ax.annotate(f"{means['S1']:.2f}V", xy=(1, means['S1']), xytext=(0.70, means['S1']+0.10),
            fontsize=13, fontweight="bold", color=COLORS["S1"],
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5),
            arrowprops=dict(arrowstyle="->", color=COLORS["S1"], lw=1.5))

# bracket showing the gap
ax.annotate("", xy=(0, means['Clean']), xytext=(1, means['S1']),
            arrowprops=dict(arrowstyle="<->", color="#64748b", lw=1.5))
ax.text(0.5, max(means['Clean'], means['S1']) + 0.55, f"+{(means['S1']/means['Clean'] - 1)*100:.0f}%", ha="center", fontsize=15, fontweight="bold", color="#334155", bbox=dict(fc="white", ec="#64748b", lw=1, pad=2))

ax.set_title("Minimum voltage shift across severity levels", fontsize=17, fontweight="bold", pad=14)
sns.despine()
plt.tight_layout()
out = os.path.join(FIGS_DIR, "F_Distribution_Vmin.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
