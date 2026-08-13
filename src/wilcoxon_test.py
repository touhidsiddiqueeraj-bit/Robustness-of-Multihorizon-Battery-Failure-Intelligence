"""Paired Wilcoxon signed-rank test on (per-severity, per-seed) ECE pairs.

FIXES APPLIED (vs. original repo):
  1. The test now compares ece_cal vs ece_recal on the SAME held-out 90 %
     subset (after the fix in run_robustness.py).  The original code compared
     ece_cal (on 100 %) vs ece_recal (on 90 %), conflating "different
     calibrator" with "different evaluation sample".
  2. Documented test setup explicitly: paired by (severity, seed), n=20 pairs
     per horizon (4 severities x 5 seeds), one-sided Wilcoxon signed-rank
     (alternative='greater' = ece_cal > ece_recal).
  3. Apply Bonferroni correction across the 4 horizons (alpha = 0.05 / 4).
  4. Also runs a secondary test on ece_raw vs ece_recal for completeness,
     so the paper can describe both comparisons unambiguously.
"""
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "results", "robustness_results.csv"))

H_LIST = [10, 20, 30, 50]
N_HORIZONS = len(H_LIST)
ALPHA = 0.05
ALPHA_BONF = ALPHA / N_HORIZONS   # Bonferroni correction across 4 horizons

print("=" * 80)
print("Paired Wilcoxon signed-rank test")
print("=" * 80)
print(f"Setup:")
print(f"  - Pairing: per (severity, seed) -> n=20 pairs per horizon (4 sev x 5 seeds)")
print(f"  - Test: one-sided Wilcoxon signed-rank (alternative='greater')")
print(f"  - H0: ece_cal <= ece_recal  vs  H1: ece_cal > ece_recal")
print(f"  - Multiple-comparisons: Bonferroni across {N_HORIZONS} horizons")
print(f"  - Significance threshold: alpha = {ALPHA} / {N_HORIZONS} = {ALPHA_BONF:.4f}")
print(f"  - All metrics computed on the SAME held-out 90 % subset (post-fix)")
print()
print(f"Primary test: ece_cal (fixed clean-fit calibrator) vs ece_recal (recalibrated on 10 %)")
print("-" * 80)
for H in H_LIST:
    sub = df[df.H == H].sort_values(["severity", "seed"]).reset_index(drop=True)
    diff = sub["ece_cal"].values - sub["ece_recal"].values
    n_pairs = len(diff)
    n_positive = int((diff > 0).sum())
    try:
        stat, p = wilcoxon(diff, alternative="greater")
        sig = "***" if p < ALPHA_BONF else ("**" if p < 0.01 else ("*" if p < ALPHA else "ns"))
        print(f"  H={H:3d}: n={n_pairs}, positive={n_positive}/{n_pairs}, "
              f"mean diff={diff.mean():.4f}, W={stat:.0f}, p={p:.3e}  {sig}")
    except Exception as e:
        print(f"  H={H:3d}: test failed ({e})")

print()
print(f"Secondary test: ece_raw (no calibrator) vs ece_recal (recalibrated on 10 %)")
print("-" * 80)
for H in H_LIST:
    sub = df[df.H == H].sort_values(["severity", "seed"]).reset_index(drop=True)
    diff = sub["ece_raw"].values - sub["ece_recal"].values
    n_pairs = len(diff)
    n_positive = int((diff > 0).sum())
    try:
        stat, p = wilcoxon(diff, alternative="greater")
        sig = "***" if p < ALPHA_BONF else ("**" if p < 0.01 else ("*" if p < ALPHA else "ns"))
        print(f"  H={H:3d}: n={n_pairs}, positive={n_positive}/{n_pairs}, "
              f"mean diff={diff.mean():.4f}, W={stat:.0f}, p={p:.3e}  {sig}")
    except Exception as e:
        print(f"  H={H:3d}: test failed ({e})")
print()
print("Significance codes: *** p < Bonferroni-corrected alpha (0.0125), "
      "** p < 0.01, * p < 0.05, ns = not significant")
