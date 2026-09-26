"""The generator is the substrate for every other test, so its promises are
pinned here: leakage columns stay out, missingness stays informative, and the
out-of-time split really is out of time.
"""

import numpy as np
import pytest

from creditrisk.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    LoanBookConfig,
    _amortising_payment,
    generate_loan_book,
    split_columns,
    time_split,
)


def test_base_rate_lands_near_the_configured_target():
    book = generate_loan_book(LoanBookConfig(n_loans=40_000, base_default_rate=0.12, seed=3))
    assert book[TARGET_COLUMN].mean() == pytest.approx(0.12, abs=0.006)


def test_feature_block_excludes_the_leakage_columns(book):
    X, y = split_columns(book)
    assert "pd_true" not in X.columns
    assert "quality" not in X.columns
    assert TARGET_COLUMN not in X.columns
    assert tuple(X.columns) == FEATURE_COLUMNS
    assert len(X) == len(y)


def test_missing_bureau_score_is_informative(book):
    thin = book["bureau_score"].isna()
    assert 0.02 < thin.mean() < 0.12
    # Informative, not missing-at-random: imputing the mean would erase this.
    assert book.loc[thin, TARGET_COLUMN].mean() > 2.0 * book.loc[~thin, TARGET_COLUMN].mean()


def test_grades_are_monotone_in_realised_risk_and_price(book):
    by_grade = book.groupby("credit_grade", observed=True).agg(
        rate=(TARGET_COLUMN, "mean"), apr=("interest_rate", "mean")
    )
    assert by_grade["rate"].is_monotonic_increasing
    assert by_grade["apr"].is_monotonic_increasing


def test_pricing_leaves_unpriced_risk_inside_each_grade(book):
    """If price fully captured risk there would be nothing for a model to add."""
    spread = book.groupby("credit_grade", observed=True)["pd_true"].std()
    assert (spread > 0.005).all()


def test_time_split_has_no_vintage_overlap(book):
    train, test = time_split(book, holdout_vintages=6)
    assert train["vintage"].max() < test["vintage"].min()
    assert len(train) + len(test) == len(book)


def test_time_split_rejects_an_impossible_holdout(book):
    with pytest.raises(ValueError, match="exceeds"):
        time_split(book, holdout_vintages=999)


@pytest.mark.parametrize("bad", [{"n_loans": 0}, {"base_default_rate": 1.4}, {"thin_file_rate": -0.1}])
def test_config_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        LoanBookConfig(**bad)


def test_amortising_payment_matches_the_closed_form():
    # $10,000 over 36 months at 12% nominal: textbook answer is $332.14.
    payment = _amortising_payment(np.array([10_000.0]), np.array([0.12]), np.array([36]))
    assert payment[0] == pytest.approx(332.14, abs=0.01)


def test_amortising_payment_handles_the_zero_rate_limit():
    payment = _amortising_payment(np.array([1_200.0]), np.array([0.0]), np.array([12]))
    assert payment[0] == pytest.approx(100.0)


def test_generation_is_reproducible():
    a = generate_loan_book(LoanBookConfig(n_loans=500, seed=99))
    b = generate_loan_book(LoanBookConfig(n_loans=500, seed=99))
    assert a[TARGET_COLUMN].equals(b[TARGET_COLUMN])
