"""Metrics are checked against hand-computable cases, because a metric that is
only checked against itself will happily be wrong in the same way forever.
"""

import numpy as np
import pytest

from creditrisk.evaluate import (
    discrimination,
    gini,
    ks_statistic,
    lift_table,
    population_stability_index,
    pr_auc,
    roc_auc,
)


def test_auc_of_a_perfect_ranker_is_one():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_auc_of_a_reversed_ranker_is_zero():
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.0)


def test_auc_gives_ties_half_credit():
    # Every score identical: each (pos, neg) pair is a tie, so AUC is exactly 0.5.
    assert roc_auc(np.array([0, 1, 0, 1]), np.full(4, 0.5)) == pytest.approx(0.5)


def test_auc_matches_a_hand_counted_case():
    # Pairs (pos, neg): (0.6,0.5) win, (0.6,0.7) loss, (0.8,0.5) win, (0.8,0.7) win.
    assert roc_auc(np.array([0, 0, 1, 1]), np.array([0.5, 0.7, 0.6, 0.8])) == pytest.approx(0.75)


def test_auc_agrees_with_sklearn_on_random_data():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, size=2_000)
    s = rng.normal(size=2_000) + y * 0.8
    assert roc_auc(y, s) == pytest.approx(sklearn_metrics.roc_auc_score(y, s), abs=1e-9)


def test_gini_is_the_rescaled_auc():
    y, s = np.array([0, 0, 1, 1]), np.array([0.5, 0.7, 0.6, 0.8])
    assert gini(y, s) == pytest.approx(2.0 * roc_auc(y, s) - 1.0)


def test_ks_of_perfect_separation_is_one():
    assert ks_statistic(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_pr_auc_of_a_perfect_ranker_is_one():
    assert pr_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_single_class_samples_raise_rather_than_return_nan():
    for metric in (roc_auc, ks_statistic, pr_auc):
        with pytest.raises(ValueError):
            metric(np.zeros(10, dtype=int), np.linspace(0, 1, 10))


def test_lift_bands_capture_every_event_by_the_last_band():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, size=1_000)
    s = rng.normal(size=1_000) + y
    table = lift_table(y, s, n_bands=10)
    assert len(table) == 10
    assert table["cumulative_event_capture"].iloc[-1] == pytest.approx(1.0)
    assert table["n"].sum() == 1_000


def test_psi_is_zero_against_itself():
    rng = np.random.default_rng(6)
    sample = rng.normal(size=5_000)
    assert population_stability_index(sample, sample) == pytest.approx(0.0, abs=1e-9)


def test_psi_flags_a_real_shift():
    rng = np.random.default_rng(7)
    base = rng.normal(size=10_000)
    shifted = rng.normal(loc=1.0, size=10_000)
    assert population_stability_index(base, shifted) > 0.25


def test_discrimination_reports_the_sample_shape():
    rng = np.random.default_rng(8)
    y = rng.integers(0, 2, size=500)
    result = discrimination(y, rng.normal(size=500) + y)
    assert result.n == 500
    assert result.n_events == int(y.sum())
    assert result.gini == pytest.approx(2 * result.auc - 1)
