"""Calibration and uncertainty metrics for probabilistic structural predictions.

Inputs are 1-D arrays: ``p`` predicted probabilities in [0, 1], ``y`` binary
outcomes, ``uncertainty`` any score where larger means less confident.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def _check(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if p.shape != y.shape:
        raise ValueError(f"p and y must have the same length, got {p.size} and {y.size}")
    if p.size and (p.min() < 0 or p.max() > 1):
        raise ValueError("probabilities must lie in [0, 1]")
    return p, y


def reliability_table(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Equal-width bins on [0, 1]: columns bin, lo, hi, mean_p, frac_pos, count.

    Empty bins are kept with count 0 and NaN mean_p / frac_pos. A probability of
    exactly 1 falls into the last bin.
    """
    p, y = _check(p, y)
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    count = np.bincount(idx, minlength=n_bins).astype(np.int64)
    sum_p = np.bincount(idx, weights=p, minlength=n_bins)
    sum_y = np.bincount(idx, weights=y, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_p = np.where(count > 0, sum_p / np.maximum(count, 1), np.nan)
        frac_pos = np.where(count > 0, sum_y / np.maximum(count, 1), np.nan)
    return pd.DataFrame(
        {
            "bin": np.arange(n_bins),
            "lo": edges[:-1],
            "hi": edges[1:],
            "mean_p": mean_p,
            "frac_pos": frac_pos,
            "count": count,
        }
    )


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Count-weighted mean of |frac_pos - mean_p| over non-empty bins."""
    table = reliability_table(p, y, n_bins=n_bins)
    n = int(table["count"].sum())
    if n == 0:
        return 0.0
    t = table[table["count"] > 0]
    gap = np.abs(t["frac_pos"].to_numpy() - t["mean_p"].to_numpy())
    return float(np.sum(gap * t["count"].to_numpy()) / n)


def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    p, y = _check(p, y)
    if p.size == 0:
        return 0.0
    return float(np.mean((p - y) ** 2))


def log_score(p: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    """Mean negative log-likelihood with probabilities clipped to [eps, 1 - eps]."""
    p, y = _check(p, y)
    if p.size == 0:
        return 0.0
    q = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


def coverage(lo: np.ndarray, hi: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of truth values inside the closed intervals [lo, hi]."""
    lo = np.asarray(lo, dtype=np.float64).ravel()
    hi = np.asarray(hi, dtype=np.float64).ravel()
    truth = np.asarray(truth, dtype=np.float64).ravel()
    if not (lo.shape == hi.shape == truth.shape):
        raise ValueError("lo, hi and truth must have the same length")
    if truth.size == 0:
        return 0.0
    return float(np.mean((truth >= lo) & (truth <= hi)))


def risk_coverage_curve(p: np.ndarray, y: np.ndarray, uncertainty: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """Error rate of the retained predictions as low-uncertainty samples are kept first.

    Rows are sorted by ascending uncertainty; row k retains the k + 1 most confident
    samples. Columns: uncertainty, retained, retained_fraction, n_errors, error_rate,
    where an error is ``(p >= threshold) != y``.
    """
    p, y = _check(p, y)
    u = np.asarray(uncertainty, dtype=np.float64).ravel()
    if u.shape != p.shape:
        raise ValueError("uncertainty must have the same length as p")
    order = np.argsort(u, kind="stable")
    err = ((p[order] >= threshold).astype(np.float64) != y[order]).astype(np.float64)
    n = p.size
    retained = np.arange(1, n + 1)
    cum_err = np.cumsum(err)
    return pd.DataFrame(
        {
            "uncertainty": u[order],
            "retained": retained,
            "retained_fraction": retained / n if n else retained.astype(np.float64),
            "n_errors": cum_err.astype(np.int64),
            "error_rate": cum_err / retained if n else cum_err,
        }
    )


def plot_reliability(table: pd.DataFrame, path: Union[str, Path]) -> Path:
    """Reliability diagram (frac_pos vs mean_p with counts) saved as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t = table[table["count"] > 0]
    fig, (ax, axc) = plt.subplots(2, 1, figsize=(5, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1, label="perfect")
    ax.plot(t["mean_p"], t["frac_pos"], marker="o", color="C0", label="observed")
    ax.set_ylabel("fraction positive")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False)
    width = float(table["hi"].iloc[0] - table["lo"].iloc[0]) if len(table) else 0.1
    axc.bar(table["lo"] + width / 2, table["count"], width=width * 0.9, color="C0", alpha=0.6)
    axc.set_xlabel("predicted probability")
    axc.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


__all__ = [
    "brier_score",
    "coverage",
    "expected_calibration_error",
    "log_score",
    "plot_reliability",
    "reliability_table",
    "risk_coverage_curve",
]
