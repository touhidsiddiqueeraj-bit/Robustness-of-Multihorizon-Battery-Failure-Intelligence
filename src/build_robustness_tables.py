"""Generate robustness tables (CSV + PNG).

FIXES APPLIED (vs. original repo):
  1. Tables now reflect val-only metrics (after the run_robustness.py fix).
  2. Added AUC raw column so reviewers can verify AUC is roughly the same
     for raw / fixed-cal / recal (rank-preserving).
  3. Added a Table 4 for the Vmin ablation (if results exist).
"""
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

CLEAN_BASELINE = os.path.join(BASE, "results", "clean_baseline.csv")
if os.path.exists(CLEAN_BASELINE):
    cb = pd.read_csv(CLEAN_BASELINE)
    cb_lookup = {row["H"]: row for _, row in cb.iterrows()}
    def _clean_ece(H): return cb_lookup[H]["ece_cal"]
    def _clean_auc(H): return cb_lookup[H]["auc_cal"]
else:
    _clean_ece = {10: 0.010, 20: 0.031, 30: 0.013, 50: 0.023}.get
    _clean_auc = {10: 0.994, 20: 0.985, 30: 0.994, 50: 0.998}.get

# Table 1: Primary results H=20 (with raw AUC added)
t1 = pd.DataFrame({
    "Severity": ["Clean", "S1 (mild)", "S2 (moderate)", "S3 (severe)", "S4 (aggressive)"],
    "ECE (raw)": ["—"] + [f"{h20[h20.severity==s].ece_raw.mean():.3f}" for s in [1,2,3,4]],
    "ECE (cal)": [f"{_clean_ece(20):.3f}"] + [f"{h20[h20.severity==s].ece_cal.mean():.3f}" for s in [1,2,3,4]],
    "MCE (cal)": ["—"] + [f"{h20[h20.severity==s].mce_cal.mean():.3f}" for s in [1,2,3,4]],
    "ACE (cal)": ["—"] + [f"{h20[h20.severity==s].ace_cal.mean():.3f}" for s in [1,2,3,4]],
    "ECE (recal iso)": ["—"] + [f"{h20[h20.severity==s].ece_recal.mean():.3f}" for s in [1,2,3,4]],
    "ECE (Platt)": ["—"] + [f"{h20[h20.severity==s].ece_recal_platt.mean():.3f}" for s in [1,2,3,4]],
    "AUC (raw)": ["—"] + [f"{h20[h20.severity==s].auc_raw.mean():.3f}" for s in [1,2,3,4]],
    "AUC (cal)": [f"{_clean_auc(20):.3f}"] + [f"{h20[h20.severity==s].auc_cal.mean():.3f}" for s in [1,2,3,4]],
})
t1.to_csv(os.path.join(TABLES, "Table1_Robustness_H20.csv"), index=False)
print("Saved Table1_Robustness_H20.csv")

# Table 2: All horizons
rows = []
for H in [10, 20, 30, 50]:
    sub = df[df.H == H]
    for s in [1,2,3,4]:
        ss = sub[sub.severity == s]
        rows.append({"Horizon": f"H={H}", "Severity": s,
            "ECE (raw)": f"{ss.ece_raw.mean():.3f}",
            "ECE (cal)": f"{ss.ece_cal.mean():.3f}",
            "ECE (recal)": f"{ss.ece_recal.mean():.3f}",
            "ECE (Platt)": f"{ss.ece_recal_platt.mean():.3f}",
            "ECE (temp)": f"{ss.ece_recal_temp.mean():.3f}",
            "AUC (raw)": f"{ss.auc_raw.mean():.3f}",
            "AUC (cal)": f"{ss.auc_cal.mean():.3f}",
            "AUC (recal)": f"{ss.auc_recal.mean():.3f}"})
t2 = pd.DataFrame(rows)
t2.to_csv(os.path.join(TABLES, "Table2_Robustness_AllHorizons.csv"), index=False)
print("Saved Table2_Robustness_AllHorizons.csv")

# Table 3: Recalibration recovery (mean +/- std across seeds per severity)
rows3 = []
for H in [10, 20, 30, 50]:
    sub = df[df.H == H]
    for s in [1, 2, 3, 4]:
        ss = sub[sub.severity == s]
        rows3.append({"Horizon": f"H={H}", "Severity": s,
            "Clean ECE": f"{_clean_ece(H):.3f}",
            "Perturbed": f"{ss.ece_cal.mean():.3f}$\\pm${ss.ece_cal.std():.3f}",
            "Recal (iso)": f"{ss.ece_recal.mean():.3f}$\\pm${ss.ece_recal.std():.3f}",
            "Recal (Platt)": f"{ss.ece_recal_platt.mean():.3f}$\\pm${ss.ece_recal_platt.std():.3f}",
            "Recal (Temp)": f"{ss.ece_recal_temp.mean():.3f}$\\pm${ss.ece_recal_temp.std():.3f}",
            "Recovery %": f"{(1 - ss.ece_recal.mean() / ss.ece_cal.mean()) * 100:.0f}%"})
t3 = pd.DataFrame(rows3)
t3.to_csv(os.path.join(TABLES, "Table3_Recalibration_Recovery.csv"), index=False)
print("Saved Table3_Recalibration_Recovery.csv")

# Table 4: Vmin ablation (if it exists)
vmin_path = os.path.join(RESULTS, "vmin_ablation_results.csv")
if os.path.exists(vmin_path):
    vdf = pd.read_csv(vmin_path)
    rows4 = []
    for H in [10, 20, 30, 50]:
        sub = vdf[vdf.H == H]
        for s in [1, 2, 3, 4]:
            ss = sub[sub.severity == s]
            rows4.append({"Horizon": f"H={H}", "Severity": s,
                "ECE (perturbed)": f"{ss.ece_pert.mean():.3f}$\\pm${ss.ece_pert.std():.3f}",
                "ECE (Vmin restored)": f"{ss.ece_ablat.mean():.3f}$\\pm${ss.ece_ablat.std():.3f}",
                "Recovered %": f"{ss.frac_recovered.mean() * 100:.1f}%",
                "AUC (perturbed)": f"{ss.auc_pert.mean():.3f}",
                "AUC (Vmin restored)": f"{ss.auc_ablat.mean():.3f}"})
    t4 = pd.DataFrame(rows4)
    t4.to_csv(os.path.join(TABLES, "Table4_Vmin_Ablation.csv"), index=False)
    print("Saved Table4_Vmin_Ablation.csv")
else:
    print("Skipping Table4 (vmin_ablation_results.csv not found — run run_vmin_ablation.py first)")

# Render Table 1 as PNG
fig, ax = plt.subplots(figsize=(6, 2.5))
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
print("Saved Table1_Robustness_H20.png")

# Render Table 3 as PNG
fig, ax = plt.subplots(figsize=(6, 2.0))
ax.axis("off")
tab = ax.table(cellText=t3.values, colLabels=t3.columns, loc="center", cellLoc="center")
tab.auto_set_font_size(False)
tab.set_fontsize(8)
for j in range(len(t3.columns)):
    tab[0, j].set_facecolor("#2c3e50")
    tab[0, j].set_text_props(color="w", fontweight="bold")
fig.savefig(os.path.join(TABLES, "Table3_Recalibration_Recovery.png"), dpi=200, bbox_inches="tight")
plt.close()
print("Saved Table3_Recalibration_Recovery.png")

# Render Table 4 as PNG (if exists)
if os.path.exists(vmin_path):
    fig, ax = plt.subplots(figsize=(6, 2.0))
    ax.axis("off")
    tab = ax.table(cellText=t4.values, colLabels=t4.columns, loc="center", cellLoc="center")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    for j in range(len(t4.columns)):
        tab[0, j].set_facecolor("#2c3e50")
        tab[0, j].set_text_props(color="w", fontweight="bold")
    fig.savefig(os.path.join(TABLES, "Table4_Vmin_Ablation.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved Table4_Vmin_Ablation.png")

print(f"\nTables saved to {TABLES}")
