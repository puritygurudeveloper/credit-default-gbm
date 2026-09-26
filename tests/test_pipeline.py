"""One end-to-end run, asserting the conclusions the README claims."""

import pytest

from creditrisk.data import LoanBookConfig
from creditrisk.pipeline import run_experiment

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def result():
    return run_experiment(config=LoanBookConfig(n_loans=25_000, seed=42), holdout_vintages=6)


def test_both_models_are_reported(result):
    assert {r.name for r in result.reports} == {"scorecard", "boosted_trees"}
    assert result.train_rows > result.test_rows > 0


def test_leaderboard_is_complete(result):
    board = result.leaderboard()
    assert len(board) == 2
    assert board.notna().all().all()


def test_out_of_time_discrimination_is_plausible(result):
    """Too low means the pipeline is broken; too high means something leaked."""
    for report in result.reports:
        assert 0.70 < report.discrimination.auc < 0.85


def test_predictions_stay_stable_out_of_time(result):
    # PSI above 0.25 is the industry trigger for a model investigation.
    for report in result.reports:
        assert report.psi < 0.25


def test_per_loan_policy_is_not_worse_than_the_global_cutoff(result):
    for report in result.reports:
        assert report.risk_adjusted.total_profit >= report.decision.total_profit


def test_per_loan_policy_reshapes_the_book_rather_than_shrinking_it(result):
    """More profit need not mean a lower bad rate.

    The per-loan rule is happy to approve a riskier loan when the price covers
    it, so the right assertion is that profit improves while the approved
    population genuinely changes — not that risk falls.
    """
    for report in result.reports:
        assert report.risk_adjusted.total_profit > report.decision.total_profit
        assert report.risk_adjusted.n_approved != report.decision.n_approved or (
            report.risk_adjusted.realised_bad_rate != report.decision.realised_bad_rate
        )


def test_swap_sets_are_riskier_than_the_common_book(result):
    """Whoever changes decision should be concentrated in the marginal tail."""
    swap = result.swap_sets.set_index("group")
    assert swap.loc["swap_in", "bad_rate"] > swap.loc["both_approve", "bad_rate"]
    assert swap.loc["both_decline", "bad_rate"] > swap.loc["both_approve", "bad_rate"]
