"""Bootstrap confidence intervals + effect sizes for recalibration gains.

Audit response #5: report CIs and effect sizes alongside the Wilcoxon p-values.
Paired (severity, seed) comparisons of ece_cal vs ece_recal at a 10% field
sample, n=20 pairs per horizon (4 severities x 5 seeds), both metrics on the
same held-out subset. Source: the sample-sweep run (single canonical protocol
for Tables IV/V/VI), not the fixed-90%-holdout robustness run.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(BASE, "results", "robustness_results_sweep.csv"))
df = df[df["cal_frac"] == 0.10]

H_LIST = [10, 20, 30, 50]
N_BOOT = 10_000
ALPHA = 0.05
RNG = np.random.default_rng(42)


def rank_biserial(diff):
    """Paired rank-biserial correlation: (2*S+ - T) / T, T = n(n+1)/2."""
    n = len(diff)
    T = n * (n + 1) / 2
    r = rankdata(np.abs(diff))
    s_pos = float(r[diff > 0].sum())
    return (2 * s_pos - T) / T


rows = []
print(f"Paired ece_cal - ece_recal  (n=20 pairs/horizon, bootstrap n={N_BOOT})")
print("-" * 88)
for H in H_LIST:
    sub = df[df.H == H].sort_values(["severity", "seed"])
    d = sub["ece_cal"].to_numpy() - sub["ece_recal"].to_numpy()
    boot = RNG.choice(d, size=(N_BOOT, len(d))).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    mean_diff = float(d.mean())
    rb = rank_biserial(d)
    dz = float(d.mean() / d.std(ddof=1))
    rows.append({"H": H, "mean_diff": mean_diff, "ci_lo": lo, "ci_hi": hi,
                 "rank_biserial_r": rb, "cohen_dz": dz})
    print(f"  H={H:3d}: mean diff={mean_diff:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"rank-biserial r={rb:.3f}  Cohen's d_z={dz:.2f}")

out = pd.DataFrame(rows)
out_path = os.path.join(BASE, "results", "bootstrap_results.csv")
out.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
