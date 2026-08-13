"""Expected Calibration Error (ECE) with 10 equal-width bins.

FIXES APPLIED (vs. original repo):
  1. Last bin is CLOSED on the right edge (includes prob=1.0).  The original
     used strictly half-open bins `[lo, hi)` for ALL bins, which silently
     dropped any probability exactly equal to 1.0 from the ECE computation.
     With isotonic regression `out_of_bounds="clip"`, this edge case is rare
     but possible and would bias ECE downward.
"""
import numpy as np


def compute_ece(y_true, prob, bins=10):
    """Compute Expected Calibration Error.

    Args:
        y_true: 1-D array of binary labels (0/1).
        prob:   1-D array of predicted probabilities in [0, 1].
        bins:   number of equal-width bins (default 10).
    Returns:
        ECE (float in [0, 1]).
    """
    y_true = np.asarray(y_true).ravel()
    prob = np.asarray(prob).ravel()
    if len(y_true) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(bins):
        if i == bins - 1:
            # Last bin is closed on the right edge
            mask = (prob >= edges[i]) & (prob <= edges[i + 1])
        else:
            mask = (prob >= edges[i]) & (prob < edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += abs(prob[mask].mean() - y_true[mask].mean()) * mask.sum() / n
    return float(ece)


def compute_mce(y_true, prob, bins=10):
    """Maximum Calibration Error: worst-bin |confidence - accuracy| gap."""
    y_true = np.asarray(y_true).ravel()
    prob = np.asarray(prob).ravel()
    if len(y_true) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    mce = 0.0
    for i in range(bins):
        if i == bins - 1:
            mask = (prob >= edges[i]) & (prob <= edges[i + 1])
        else:
            mask = (prob >= edges[i]) & (prob < edges[i + 1])
        if mask.sum() == 0:
            continue
        mce = max(mce, abs(prob[mask].mean() - y_true[mask].mean()))
    return float(mce)


def compute_ace(y_true, prob, bins=10):
    """Adaptive Calibration Error: equal-mass bins (Naeini et al. 2015).

    ACE is the sample-weighted mean of per-bin gaps, NOT the max; a weighted
    mean over the same bins can never exceed the max.  This function was
    originally a copy of compute_mce and returned the max, which made ACE > MCE.
    """
    y_true = np.asarray(y_true).ravel()
    prob = np.asarray(prob).ravel()
    if len(y_true) == 0:
        return 0.0
    order = np.argsort(prob)
    ys, ps = y_true[order], prob[order]
    n = len(ys)
    edges = np.linspace(0, n, bins + 1).astype(int)
    ace = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if hi - lo < 1:
            continue
        ace += abs(ps[lo:hi].mean() - ys[lo:hi].mean()) * (hi - lo) / n
    return float(ace)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = rng.random(500)
        y = (rng.random(500) < p).astype(int)
        ece, mce, ace = compute_ece(y, p), compute_mce(y, p), compute_ace(y, p)
        assert 0.0 <= ace <= 1.0
        assert ace <= mce + 1e-12, "ACE (weighted mean) must not exceed MCE (max)"
        assert abs(ece - compute_ece(y, p)) < 1e-12
    assert abs(compute_ace(np.array([1, 1, 1]), np.array([1.0, 1.0, 1.0]))) < 1e-12
    assert abs(compute_ace(np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0]))) < 1e-12
    print("compute_ece / compute_mce / compute_ace self-check OK")
