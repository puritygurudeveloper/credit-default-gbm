"""The economics decide who gets credit, so the algebra is pinned exactly."""

import numpy as np
import pandas as pd
import pytest

from creditrisk.economics import (
    LoanEconomics,
    _outcome_margins,
    evaluate_policy,
    optimal_threshold,
    profit_curve,
    risk_adjusted_policy,
    swap_set_analysis,
)


@pytest.fixture
def loans():
    return pd.DataFrame(
        {
            "loan_amount": [10_000.0, 10_000.0, 10_000.0, 10_000.0],
            "interest_rate": [0.07, 0.12, 0.18, 0.26],
            "term_months": [36, 36, 60, 60],
        }
    )


def test_breakeven_pd_rises_with_price(loans):
    breakeven = LoanEconomics().breakeven_pd(loans)
    assert np.all(np.diff(breakeven) > 0)
    assert np.all((breakeven > 0.0) & (breakeven < 1.0))


def test_breakeven_pd_is_the_zero_of_expected_profit(loans):
    """The two functions are separate code paths; they must agree exactly."""
    econ = LoanEconomics()
    breakeven = econ.breakeven_pd(loans)
    assert np.allclose(econ.expected_profit(loans, breakeven), 0.0, atol=1e-6)


def test_outcome_margins_bracket_the_expected_profit(loans):
    econ = LoanEconomics()
    good, bad = _outcome_margins(
        loans["loan_amount"].to_numpy(),
        loans["interest_rate"].to_numpy(),
        loans["term_months"].to_numpy(),
        econ,
    )
    assert np.all(bad < 0.0) and np.all(good > 0.0)
    # Expected profit at p is the probability-weighted mix of the two outcomes,
    # up to the interest term only accruing on surviving loans.
    assert econ.expected_profit(loans, np.zeros(4)) == pytest.approx(good, rel=1e-9)


def test_a_default_costs_less_than_the_full_principal(loans):
    """LGD and EAD are both below one, and the fee is earned regardless."""
    _, bad = _outcome_margins(
        loans["loan_amount"].to_numpy(),
        loans["interest_rate"].to_numpy(),
        loans["term_months"].to_numpy(),
        LoanEconomics(),
    )
    assert np.all(bad > -loans["loan_amount"].to_numpy())


def test_expected_life_is_shorter_than_the_contractual_term():
    econ = LoanEconomics()
    assert econ.expected_life_years(np.array([60.0]))[0] < 5.0
    assert econ.expected_life_years(np.array([1.0]))[0] >= 0.5


def test_lgd_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match="lgd"):
        LoanEconomics(lgd=1.4)


def _synthetic_decision_sample(n=4_000, seed=12):
    rng = np.random.default_rng(seed)
    pd_true = rng.beta(2.0, 20.0, size=n)
    X = pd.DataFrame(
        {
            "loan_amount": rng.choice([5_000.0, 12_000.0, 25_000.0], size=n),
            "interest_rate": np.clip(0.07 + 0.55 * pd_true + rng.normal(0, 0.01, n), 0.05, 0.30),
            "term_months": rng.choice([36, 60], size=n),
        }
    )
    y = (rng.random(n) < pd_true).astype(float)
    return X, y, pd_true


def test_profit_curve_is_ordered_and_covers_the_book():
    X, y, pd_hat = _synthetic_decision_sample()
    curve = profit_curve(X, y, pd_hat)
    assert curve["threshold"].is_monotonic_increasing
    assert curve["approval_rate"].is_monotonic_increasing
    assert curve["approval_rate"].iloc[-1] == pytest.approx(1.0)


def test_optimal_threshold_beats_approving_everyone():
    X, y, pd_hat = _synthetic_decision_sample()
    best = optimal_threshold(X, y, pd_hat)
    approve_all = evaluate_policy(X, y, np.ones(len(y), dtype=bool), policy="approve_all")
    assert best.total_profit >= approve_all.total_profit


def test_volume_floor_is_respected():
    X, y, pd_hat = _synthetic_decision_sample()
    best = optimal_threshold(X, y, pd_hat, min_approval_rate=0.90)
    assert best.approval_rate >= 0.90


def test_unreachable_volume_floor_raises():
    X, y, pd_hat = _synthetic_decision_sample()
    with pytest.raises(ValueError, match="no cut-off"):
        optimal_threshold(X, y, pd_hat, min_approval_rate=1.5)


def test_risk_adjusted_policy_beats_the_best_global_cutoff():
    """The headline claim of the project, asserted rather than asserted-in-prose.

    With price-dependent break-even PDs, no single threshold can dominate the
    per-loan rule when the PD estimate is good.
    """
    X, y, pd_hat = _synthetic_decision_sample(n=20_000)
    per_loan = risk_adjusted_policy(X, y, pd_hat)
    global_cut = optimal_threshold(X, y, pd_hat)
    assert per_loan.total_profit > global_cut.total_profit


def test_margin_of_safety_tightens_approvals():
    X, y, pd_hat = _synthetic_decision_sample()
    loose = risk_adjusted_policy(X, y, pd_hat, margin_of_safety=0.0)
    tight = risk_adjusted_policy(X, y, pd_hat, margin_of_safety=0.30)
    assert tight.approval_rate < loose.approval_rate
    assert tight.realised_bad_rate <= loose.realised_bad_rate


def test_policy_that_approves_nobody_raises():
    X, y, _ = _synthetic_decision_sample()
    with pytest.raises(ValueError, match="approves nothing"):
        evaluate_policy(X, y, np.zeros(len(y), dtype=bool), policy="approve_none")


def test_swap_sets_partition_the_book():
    X, y, pd_hat = _synthetic_decision_sample()
    rng = np.random.default_rng(3)
    challenger = np.clip(pd_hat + rng.normal(0, 0.02, len(pd_hat)), 0.001, 0.999)
    swap = swap_set_analysis(X, y, pd_hat, challenger)
    assert set(swap["group"]) == {"both_approve", "swap_in", "swap_out", "both_decline"}
    assert swap["n"].sum() == len(y)
    assert swap["share"].sum() == pytest.approx(1.0)
