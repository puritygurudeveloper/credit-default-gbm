"""Discrimination and stability metrics.

Deliberately hand-rolled rather than imported.  Gini, KS and PSI are the three
numbers that appear in every credit model monitoring pack, and writing them out
fixes the conventions (which direction, which denominator, how ties are
handled) that otherwise differ between libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Discrimination:
    """Headline separation metrics for one model on one sample."""

    n: int
    n_events: int
    event_rate: float
    auc: float
    gini: float
    ks: float
    pr_auc: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """AUC via the Mann-Whitney U identity, with midpoint credit for ties."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.size != s.size:
        raise ValueError("y_true and scores must be the same length")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC is undefined when the sample has a single class")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    ranks[order] = np.arange(1, s.size + 1, dtype=float)
    # Average the ranks inside each tied block so ties score 0.5, not 0 or 1.
    sorted_scores = s[order]
    start = 0
    for i in range(1, s.size + 1):
        if i == s.size or sorted_scores[i] != sorted_scores[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def gini(y_true: np.ndarray, scores: np.ndarray) -> float:
    return 2.0 * roc_auc(y_true, scores) - 1.0


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Maximum vertical gap between the good and bad cumulative distributions."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("KS is undefined when the sample has a single class")
    cum_pos = np.cumsum(y) / n_pos
    cum_neg = np.cumsum(1 - y) / n_neg
    return float(np.max(np.abs(cum_pos - cum_neg)))


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision — the metric that survives an 8% base rate."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    precision = tp / np.arange(1, y.size + 1)
    n_pos = int(y.sum())
    if n_pos == 0:
        raise ValueError("average precision is undefined with no positives")
    return float((precision * y).sum() / n_pos)


def discrimination(y_true: np.ndarray, scores: np.ndarray) -> Discrimination:
    y = np.asarray(y_true, dtype=int)
    return Discrimination(
        n=int(y.size),
        n_events=int(y.sum()),
        event_rate=float(y.mean()),
        auc=roc_auc(y, scores),
        gini=gini(y, scores),
        ks=ks_statistic(y, scores),
        pr_auc=pr_auc(y, scores),
    )


def lift_table(y_true: np.ndarray, scores: np.ndarray, n_bands: int = 10) -> pd.DataFrame:
    """Risk-ranked bands, worst first — the standard scorecard monitoring view."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if n_bands < 2:
        raise ValueError("n_bands must be at least 2")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    bands = np.array_split(np.arange(y.size), n_bands)
    overall = y.mean()
    rows = []
    cumulative_events = 0
    for i, idx in enumerate(bands, start=1):
        events = int(y[idx].sum())
        cumulative_events += events
        rows.append(
            {
                "band": i,
                "n": len(idx),
                "min_score": float(s[idx].min()),
                "max_score": float(s[idx].max()),
                "events": events,
                "event_rate": events / len(idx),
                "lift": (events / len(idx)) / overall if overall > 0 else np.nan,
                "cumulative_event_capture": cumulative_events / max(1, int(y.sum())),
            }
        )
    return pd.DataFrame(rows)


def population_stability_index(expected: np.ndarray, actual: np.ndarray, n_bands: int = 10) -> float:
    """PSI of ``actual`` against the ``expected`` (development) distribution.

    The banding is fixed on the development sample, which is the whole point:
    re-banding on the new sample would report zero drift by construction.
    Convention: below 0.10 stable, 0.10-0.25 watch, above 0.25 investigate.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        raise ValueError("PSI needs non-empty samples")
    quantiles = np.linspace(0.0, 1.0, n_bands + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    exp_share = np.histogram(expected, bins=edges)[0] / expected.size
    act_share = np.histogram(actual, bins=edges)[0] / actual.size
    floor = 1e-6
    exp_share = np.clip(exp_share, floor, None)
    act_share = np.clip(act_share, floor, None)
    return float(np.sum((act_share - exp_share) * np.log(act_share / exp_share)))
