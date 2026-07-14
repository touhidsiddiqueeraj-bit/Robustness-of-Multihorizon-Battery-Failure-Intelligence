"""Paired Wilcoxon signed-rank test comparing uncalibrated vs recalibrated ECE."""
import os
import pandas as pd
from scipy.stats import wilcoxon

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "results", "robustness_results.csv"))

H_LIST = [10, 20, 30, 50]
for H in H_LIST:
    sub = df[df.H == H]
    # paired: same (severity, seed) triple, compare ece_cal vs ece_recal
    diff = sub["ece_cal"].values - sub["ece_recal"].values
    stat, p = wilcoxon(diff, alternative="greater")
    print(f"H={H}:  statistic={stat:.0f}  p={p:.6e}  n_pairs={len(diff)}")
