import pytest

from creditrisk.data import LoanBookConfig, generate_loan_book, split_columns
from creditrisk.features import add_ratio_features


@pytest.fixture(scope="session")
def book():
    """A small book, generated once — the generator is the slow part."""
    return generate_loan_book(LoanBookConfig(n_loans=8_000, seed=11))


@pytest.fixture(scope="session")
def xy(book):
    X, y = split_columns(book)
    return add_ratio_features(X), y
