"""Synthetic consumer loan book.

Real bureau data cannot be redistributed, so the repository ships a generator
instead of a CSV.  The generator is not a toy: it reproduces the three
properties that make credit data awkward to model.

1. **A latent factor.**  Applicants differ along an unobserved creditworthiness
   axis that drives every observable at once, so the features are strongly
   correlated and a naive "drop collinear columns" step destroys signal.
2. **Non-linearity with a monotone core.**  The default hazard is monotone in
   most drivers but saturates, which is exactly the regime where a scorecard
   loses ground to gradient boosting without the loss being dramatic.
3. **Missingness that is itself informative.**  Thin-file applicants are both
   more likely to have a missing bureau score and more likely to default, so
   imputing the mean leaks the wrong prior.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Columns the downstream pipeline is allowed to see.
FEATURE_COLUMNS: tuple[str, ...] = (
    "bureau_score",
    "annual_income",
    "debt_to_income",
    "revolving_utilisation",
    "months_employed",
    "n_delinq_24m",
    "n_inquiries_6m",
    "loan_amount",
    "term_months",
    "credit_grade",
    "interest_rate",
    "age",
    "home_owner",
    "purpose",
)

TARGET_COLUMN = "defaulted"


@dataclass(frozen=True)
class LoanBookConfig:
    """Knobs for :func:`generate_loan_book`."""

    n_loans: int = 100_000
    #: Unconditional default rate the generator is calibrated towards.
    base_default_rate: float = 0.082
    #: Fraction of applicants with no usable bureau file.
    thin_file_rate: float = 0.06
    #: Vintages are spread over this many monthly cohorts.
    n_vintages: int = 24
    #: Peak-to-trough swing in the macro log-odds adjustment.
    macro_amplitude: float = 0.22
    #: Multiplier on the whole risk-driver block.  Tuned so that a perfectly
    #: specified model lands near AUC 0.78, which is where a production
    #: consumer scorecard actually sits; raise it to make the problem easy.
    signal_scale: float = 1.0
    seed: int = 20210614
    purposes: tuple[str, ...] = field(
        default=("debt_consolidation", "home_improvement", "major_purchase", "small_business", "other")
    )
    #: Board-approved rate card.  Pricing is set per grade, not per applicant.
    grade_rates: tuple[float, ...] = field(
        default=(0.068, 0.089, 0.112, 0.138, 0.168, 0.205, 0.256)
    )

    def __post_init__(self) -> None:
        if self.n_loans <= 0:
            raise ValueError("n_loans must be positive")
        if not 0.0 < self.base_default_rate < 1.0:
            raise ValueError("base_default_rate must lie in (0, 1)")
        if not 0.0 <= self.thin_file_rate < 1.0:
            raise ValueError("thin_file_rate must lie in [0, 1)")


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _amortising_payment(principal: np.ndarray, annual_rate: np.ndarray, term_months: np.ndarray) -> np.ndarray:
    """Level monthly payment on a fully amortising loan.

    Shared by the generator and :mod:`creditrisk.features` so the hazard and
    the model feature cannot silently drift apart.
    """
    monthly_rate = np.asarray(annual_rate, dtype=float) / 12.0
    n = np.asarray(term_months, dtype=float)
    principal = np.asarray(principal, dtype=float)
    interest_bearing = monthly_rate > 1e-12
    # np.where evaluates both branches, so the zero-rate entries are nudged
    # off zero before the division rather than after it.
    safe_rate = np.where(interest_bearing, monthly_rate, 1.0)
    growth = np.power(1.0 + safe_rate, n)
    return np.where(interest_bearing, principal * safe_rate * growth / (growth - 1.0), principal / n)


def _solve_intercept(linear: np.ndarray, target_rate: float) -> float:
    """Bisect for the intercept that yields ``target_rate`` on average.

    Cheaper and more robust than trying to reason analytically about the mean
    of a logistic transform, and it keeps the generated base rate stable when
    the coefficient block is edited.
    """
    lo, hi = -20.0, 20.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _logistic(linear + mid).mean() < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_loan_book(config: LoanBookConfig | None = None) -> pd.DataFrame:
    """Draw a loan book with an observable feature block and a realised target.

    Returns a frame with :data:`FEATURE_COLUMNS`, the ``vintage`` cohort index,
    the realised ``defaulted`` flag, and — for diagnostics only — the true
    ``pd_true`` and the latent ``quality`` factor.  Training code must never
    touch the last two; :func:`split_columns` enforces that.
    """
    cfg = config or LoanBookConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_loans

    # Latent creditworthiness: higher is better.
    quality = rng.normal(size=n)

    # Thin files are drawn first: the lender grades on the bureau score it can
    # actually see, so missingness has to exist before pricing happens.
    thin_propensity = _logistic(-2.9 - 1.1 * quality)
    thin_propensity *= cfg.thin_file_rate / thin_propensity.mean()
    is_thin = rng.random(n) < np.clip(thin_propensity, 0.0, 1.0)

    # Observables are noisy, non-linear reads on the latent factor.
    bureau_score = np.clip(690.0 + 62.0 * quality + rng.normal(scale=28.0, size=n), 300.0, 850.0)
    bureau_score_observed = np.where(is_thin, np.nan, bureau_score)
    revolving_utilisation = np.clip(0.52 - 0.17 * quality + rng.normal(scale=0.16, size=n), 0.0, 1.5)
    months_employed = np.clip(rng.gamma(2.4, 22.0 + 7.0 * (quality + 3.0), size=n), 0.0, 480.0)
    n_delinq_24m = rng.poisson(np.exp(-0.55 - 0.95 * quality), size=n)
    n_inquiries_6m = rng.poisson(np.exp(0.15 - 0.42 * quality), size=n)

    age = np.clip(rng.gamma(9.0, 4.3, size=n) + 18.0, 18.0, 92.0)
    home_owner = (rng.random(n) < _logistic(-0.4 + 0.55 * quality + 0.018 * (age - 40.0))).astype(np.int8)

    # Income follows a lognormal with a hump-shaped age profile; it is only
    # loosely tied to the latent factor, so income and creditworthiness stay
    # distinguishable rather than collapsing onto one axis.
    log_income = (
        10.72
        + 0.34 * quality
        + 0.020 * (age - 40.0)
        - 0.00048 * (age - 40.0) ** 2
        + rng.normal(scale=0.46, size=n)
    )
    annual_income = np.round(np.clip(np.exp(log_income), 14_000.0, 900_000.0), -2)
    monthly_income = annual_income / 12.0

    # Leverage is a ratio, so it is drawn directly rather than backed out of a
    # payment; this keeps debt_to_income and payment_to_income independent.
    debt_to_income = np.clip(0.30 - 0.055 * quality + rng.gamma(2.0, 0.035, size=n), 0.01, 0.95)

    # Requested amount scales with income but with a wide idiosyncratic band.
    loan_amount = np.round(
        np.clip(
            annual_income * np.exp(-1.72 + 0.10 * quality + rng.normal(scale=0.46, size=n)),
            1_000.0,
            45_000.0,
        ),
        -2,
    )
    term_months = np.where(rng.random(n) < _logistic(0.35 - 0.5 * quality), 60, 36).astype(np.int16)

    purpose_logits = np.column_stack(
        [
            0.9 + 0.10 * quality,
            0.2 + 0.45 * quality,
            0.1 + 0.20 * quality,
            -0.6 - 0.35 * quality,
            np.zeros(n),
        ]
    )
    purpose_probs = np.exp(purpose_logits)
    purpose_probs /= purpose_probs.sum(axis=1, keepdims=True)
    purpose_idx = np.array([rng.choice(len(cfg.purposes), p=row) for row in purpose_probs])
    purpose = np.asarray(cfg.purposes, dtype=object)[purpose_idx]

    # Pricing is grade-based, and the grade comes off the legacy underwriting
    # rule: bureau score alone, with thin files dropped two notches.  That is
    # the whole reason a model earns its keep — within a single priced grade
    # the residual risk dispersion is large and entirely unpriced.
    grade_labels = tuple("ABCDEFG"[: len(cfg.grade_rates)])
    score_for_grading = np.where(np.isnan(bureau_score_observed), 640.0, bureau_score_observed)
    grade_cuts = np.array([760.0, 730.0, 700.0, 672.0, 645.0, 610.0])
    grade_idx = np.searchsorted(-grade_cuts, -score_for_grading, side="left")
    grade_idx = np.clip(grade_idx + 2 * np.isnan(bureau_score_observed), 0, len(grade_labels) - 1)
    credit_grade = np.asarray(grade_labels, dtype=object)[grade_idx]

    interest_rate = np.clip(
        np.asarray(cfg.grade_rates)[grade_idx]
        + 0.011 * (term_months == 60)
        + rng.normal(scale=0.004, size=n),
        0.045,
        0.31,
    )

    vintage = rng.integers(0, cfg.n_vintages, size=n)
    macro = cfg.macro_amplitude * np.sin(2.0 * np.pi * vintage / cfg.n_vintages)

    # Default hazard: monotone in the drivers but saturating in utilisation and
    # delinquency, plus one genuine interaction (small-business x leverage).
    monthly_payment = _amortising_payment(loan_amount, interest_rate, term_months)
    payment_to_income = monthly_payment / monthly_income

    # Hazard: monotone in the drivers but saturating in utilisation and
    # delinquency, plus one genuine interaction (small business x leverage).
    drivers = (
        -0.58 * quality
        + 0.72 * np.tanh(2.2 * (revolving_utilisation - 0.55))
        + 0.40 * np.log1p(n_delinq_24m)
        + 0.11 * n_inquiries_6m
        + 1.15 * (debt_to_income - 0.30)
        + 2.10 * np.clip(payment_to_income - 0.08, 0.0, None)
        - 0.19 * home_owner
        - 0.0011 * months_employed
        + 0.24 * (term_months == 60)
        + 0.40 * (purpose == "small_business") * (debt_to_income > 0.40)
    )
    linear = cfg.signal_scale * drivers + macro
    intercept = _solve_intercept(linear, cfg.base_default_rate)
    pd_true = _logistic(linear + intercept)
    defaulted = (rng.random(n) < pd_true).astype(np.int8)

    frame = pd.DataFrame(
        {
            "loan_id": np.arange(1, n + 1, dtype=np.int64),
            "vintage": vintage.astype(np.int16),
            "bureau_score": bureau_score_observed,
            "annual_income": annual_income,
            "debt_to_income": debt_to_income,
            "revolving_utilisation": revolving_utilisation,
            "months_employed": months_employed,
            "n_delinq_24m": n_delinq_24m.astype(np.int16),
            "n_inquiries_6m": n_inquiries_6m.astype(np.int16),
            "loan_amount": loan_amount,
            "term_months": term_months,
            "credit_grade": pd.Categorical(credit_grade, categories=list(grade_labels)),
            "interest_rate": interest_rate,
            "age": age,
            "home_owner": home_owner,
            "purpose": pd.Categorical(purpose, categories=list(cfg.purposes)),
            TARGET_COLUMN: defaulted,
            "pd_true": pd_true,
            "quality": quality,
        }
    )

    # Employment history is patchy for the same thin-file population.
    frame.loc[is_thin & (rng.random(n) < 0.5), "months_employed"] = np.nan

    return frame


def split_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` with the diagnostic-only columns stripped from ``X``."""
    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"loan book is missing feature columns: {missing}")
    return frame.loc[:, list(FEATURE_COLUMNS)].copy(), frame[TARGET_COLUMN].astype(np.int8).copy()


def time_split(frame: pd.DataFrame, holdout_vintages: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split out-of-time, not at random.

    Credit models are always deployed on vintages the model has never seen, so
    a random split flatters every model in the comparison equally and hides
    the macro sensitivity that governance actually cares about.
    """
    if holdout_vintages <= 0:
        raise ValueError("holdout_vintages must be positive")
    cutoff = frame["vintage"].max() - holdout_vintages
    if cutoff < frame["vintage"].min():
        raise ValueError("holdout_vintages exceeds the number of available vintages")
    train = frame.loc[frame["vintage"] <= cutoff].reset_index(drop=True)
    test = frame.loc[frame["vintage"] > cutoff].reset_index(drop=True)
    return train, test
