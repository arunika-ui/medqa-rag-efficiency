"""95% bootstrap intervals (Efron & Tibshirani), exact McNemar tests, Holm correction."""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(correct, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    c = np.asarray(correct, dtype=float)
    rng = np.random.default_rng(seed)
    means = c[rng.integers(0, len(c), size=(n_boot, len(c)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def mcnemar_exact(a, b) -> float:
    """Two-sided exact McNemar test on paired correctness vectors."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out
