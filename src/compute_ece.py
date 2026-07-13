import numpy as np


def compute_ece(y_true, prob, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (prob >= edges[i]) & (prob < edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += abs(prob[mask].mean() - y_true[mask].mean()) * mask.sum() / len(y_true)
    return ece
