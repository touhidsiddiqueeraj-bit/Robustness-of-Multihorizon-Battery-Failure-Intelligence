"""Generate robustness tables and figure packages."""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(SRC, ".."))
FIGS = os.path.join(BASE, "figs")
TABLES = os.path.join(BASE, "tables")
RESULTS = os.path.join(BASE, "results")
os.makedirs(TABLES, exist_ok=True)

df = pd.read_csv(os.path.join(RESULTS, "robustness_results.csv"))
h20 = df[df.H == 20]

# Table 1: Primary results H=20
t1 = pd.DataFrame({
    "Severity": ["Clean", "S1 (mild)", "S2 (moderate)", "S3 (severe)", "S4 (aggressive)"],
    "ECE (raw)": ["—"] + [f"{h20[h20.severity==s].ece_raw.mean():.3f}" for s in [1,2,3,4]],
    "ECE (cal)": ["0.031"] + [f"{h20[h20.severity==s].ece_cal.mean():.3f}" for s in [1,2,3,4]],
    "ECE (recal)": ["—"] + [f"{h20[h20.severity==s].ece_recal.mean():.3f}" for s in [1,2,3,4]],
    "AUC": ["0.985"] + [f"{h20[h20.severity==s].auc_cal.mean():.3f}" for s in [1,2,3,4]],
})
t1.to_csv(os.path.join(TABLES, "Table1_Robustness_H20.csv"), index=False)

# Table 2: All horizons
rows = []
for H in [10, 20, 30, 50]:
    sub = df[df.H == H]
    for s in [1,2,3,4]:
        ss = sub[sub.severity == s]
        rows.append({"Horizon": f"H={H}", "Severity": s,
            "ECE (cal)": f"{ss.ece_cal.mean():.3f}", "ECE (recal)": f"{ss.ece_recal.mean():.3f}",
            "AUC": f"{ss.auc_cal.mean():.3f}"})
t2 = pd.DataFrame(rows)
t2.to_csv(os.path.join(TABLES, "Table2_Robustness_AllHorizons.csv"), index=False)

# Table 3: Recalibration recovery
clean_eces = {"10": "0.010", "20": "0.031", "30": "0.013", "50": "0.023"}
rows3 = []
for H in [10, 20, 30, 50]:
    sub = df[df.H == H]
    rows3.append({"Horizon": f"H={H}", "Clean ECE": clean_eces[str(H)],
        "Perturbed": f"{sub.ece_cal.min():.3f}--{sub.ece_cal.max():.3f}",
        "Recalibrated": f"{sub.ece_recal.min():.3f}--{sub.ece_recal.max():.3f}"})
t3 = pd.DataFrame(rows3)
t3.to_csv(os.path.join(TABLES, "Table3_Recalibration_Recovery.csv"), index=False)

# Render Table 1 as PNG
fig, ax = plt.subplots(figsize=(5, 2.2))
ax.axis("off")
tab = ax.table(cellText=t1.values, colLabels=t1.columns, loc="center", cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(9)
for j in range(len(t1.columns)):
    tab[0, j].set_facecolor("#2c3e50")
    tab[0, j].set_text_props(color="w", fontweight="bold")
nc = len(t1.columns)
for i in range(1, len(t1)):
    tab[i, nc - 1].set_facecolor("#e8f4f8")
fig.savefig(os.path.join(TABLES, "Table1_Robustness_H20.png"), dpi=200, bbox_inches="tight")
plt.close()

# Render Table 3 as PNG
fig, ax = plt.subplots(figsize=(5, 1.8))
ax.axis("off")
tab = ax.table(cellText=t3.values, colLabels=t3.columns, loc="center", cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(9)
for j in range(len(t3.columns)):
    tab[0, j].set_facecolor("#2c3e50")
    tab[0, j].set_text_props(color="w", fontweight="bold")
fig.savefig(os.path.join(TABLES, "Table3_Recalibration_Recovery.png"), dpi=200, bbox_inches="tight")
plt.close()

print(f"Tables saved to {TABLES}")
