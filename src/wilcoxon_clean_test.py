"""One-sample Wilcoxon test: recalibrated perturbed ECE vs clean baseline.

Tests whether the 20 recalibrated ECE values (4 severities x 5 seeds)
per horizon are significantly below the clean-baseline ECE scalar.

H0: median(ece_recal) >= clean_baseline  (recalibrated is no better)
H1: median(ece_recal) < clean_baseline   (recalibrated is better)
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rob = pd.read_csv(os.path.join(BASE, "results", "robustness_results.csv"))
clean = pd.read_csv(os.path.join(BASE, "results", "clean_baseline.csv"))

H_LIST = [10, 20, 30, 50]
ALPHA = 0.05

print("=" * 80)
print("One-sample Wilcoxon signed-rank test: recalibrated ECE vs clean baseline")
print("=" * 80)
print("Setup:")
print("  - One-sample test: 20 ece_recal values per horizon (4 sev x 5 seeds)")
print("  - Null hypothesis H0: median(ece_recal) >= clean_baseline")
print("  - Alternative H1: median(ece_recal) < clean_baseline (recal is BETTER)")
print("  - i.e. test(ece_recal - clean_baseline, alternative='less')")
print("  - No Bonferroni correction (exploratory, 4 tests)")
print()

for H in H_LIST:
    sub = rob[rob.H == H]["ece_recal"].values
    cb = clean[clean.H == H]["ece_cal"].values[0]
    diff = sub - cb
    n = len(sub)
    n_below = int((diff < 0).sum())
    median_recal = np.median(sub)
    print(f"  H={H:3d}: clean={cb:.4f}, median_recal={median_recal:.4f}, "
          f"below={n_below}/{n}, range=[{sub.min():.4f}, {sub.max():.4f}]")
    try:
        stat, p = wilcoxon(diff, alternative="less")
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < ALPHA else "ns"))
        print(f"         W={stat:.0f}, p={p:.3e} {sig}")
    except Exception as e:
        print(f"         test failed ({e})")
    print()

print("Significance: *** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant")
print()
print("Interpretation: p<0.05 means recalibrated ECE is significantly below clean.")
