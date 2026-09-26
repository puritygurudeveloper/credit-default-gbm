"""WOE has a closed form, so it is checked against arithmetic rather than
against itself.
"""

import numpy as np
import pandas as pd
import pytest

from creditrisk.features import MISSING_BIN, WoeEncoder, add_ratio_features, fit_woe


def test_woe_sign_convention_is_high_means_safe():
    # Low values default often, high values never: WOE must increase with x.
    x = pd.Series(np.repeat([1.0, 2.0, 3.0, 4.0], 250), name="x")
    y = pd.Series(np.repeat([1, 1, 0, 0], 250)).where(
        pd.Series(np.repeat([1, 1, 0, 0], 250)) == 0, pd.Series(np.repeat([1, 0, 0, 0], 250))
    )
    binning = fit_woe(x, y, max_bins=4, min_bin_fraction=0.1)
    woes = [b.woe for b in binning.bins if not b.is_missing]
    assert woes == sorted(woes)


def test_information_value_is_zero_for_a_useless_feature():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=4_000), name="noise")
    y = pd.Series(rng.integers(0, 2, size=4_000))
    assert fit_woe(x, y).information_value < 0.05


def test_missing_values_get_their_own_bin_not_an_imputation():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan] * 200, name="x")
    y = pd.Series(([0, 0, 0, 0, 1] * 200))
    binning = fit_woe(x, y, min_bin_fraction=0.05)
    labels = [b.label for b in binning.bins]
    assert MISSING_BIN in labels
    missing_bin = next(b for b in binning.bins if b.is_missing)
    assert missing_bin.event_rate == pytest.approx(1.0)


def test_transform_is_stable_on_unseen_categories():
    train = pd.Series(["a", "b"] * 200, name="cat", dtype="object")
    y = pd.Series([0, 1] * 200)
    binning = fit_woe(train, y, min_bin_fraction=0.05)
    out = binning.transform(pd.Series(["a", "zzz"], dtype="object"))
    assert np.isfinite(out).all()


def test_bins_respect_the_minimum_size_floor():
    rng = np.random.default_rng(1)
    # A spike at zero would otherwise produce a bin of a handful of rows.
    x = pd.Series(np.concatenate([np.zeros(3_800), rng.normal(5, 1, 200)]), name="skewed")
    y = pd.Series(rng.integers(0, 2, size=4_000))
    binning = fit_woe(x, y, max_bins=10, min_bin_fraction=0.05)
    assert all(b.n >= 0.04 * 4_000 for b in binning.bins)


def test_encoder_drops_features_below_the_iv_floor(xy):
    X, y = xy
    encoder = WoeEncoder.fit(X, y, min_information_value=0.30)
    assert 0 < len(encoder.binnings) < X.shape[1]
    assert (encoder.information_values()["iv"] >= 0.30).all()


def test_encoder_rejects_a_frame_missing_its_columns(xy):
    X, y = xy
    encoder = WoeEncoder.fit(X, y)
    with pytest.raises(KeyError, match="missing columns"):
        encoder.transform(X.drop(columns=list(encoder.binnings)[0:1]))


def test_ratio_features_are_not_algebraic_copies(xy):
    """payment_to_income once collapsed onto debt_to_income; never again."""
    X, _ = xy
    assert abs(X["payment_to_income"].corr(X["debt_to_income"])) < 0.5
    assert (X["payment_to_income"] > 0).all()


def test_ratio_features_survive_a_zero_interest_loan():
    frame = pd.DataFrame(
        {
            "annual_income": [60_000.0],
            "loan_amount": [12_000.0],
            "interest_rate": [0.0],
            "term_months": [12],
            "debt_to_income": [0.2],
            "n_delinq_24m": [0],
            "n_inquiries_6m": [0],
            "months_employed": [24.0],
            "revolving_utilisation": [0.3],
        }
    )
    out = add_ratio_features(frame)
    assert out["monthly_payment"].iloc[0] == pytest.approx(1_000.0)
