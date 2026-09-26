"""Model tests are about contracts and known failure modes, not about pinning
an accuracy number that will drift with every library upgrade.
"""

import numpy as np
import pytest

from creditrisk.calibration import PlattCalibrator, calibration_summary, reliability_table
from creditrisk.evaluate import roc_auc
from creditrisk.models import BoostedTrees, PdModel, Scorecard

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def fitted(xy):
    X, y = xy
    return X, y, Scorecard().fit(X, y), BoostedTrees(n_estimators=120).fit(X, y)


def test_both_models_satisfy_the_protocol():
    assert isinstance(Scorecard(), PdModel)
    assert isinstance(BoostedTrees(), PdModel)


def test_unfitted_models_refuse_to_predict(xy):
    X, _ = xy
    for model in (Scorecard(), BoostedTrees()):
        with pytest.raises(RuntimeError, match="fit must be called"):
            model.predict_proba(X)


def test_predictions_are_probabilities(fitted):
    X, _, scorecard, booster = fitted
    for model in (scorecard, booster):
        p = model.predict_proba(X)
        assert p.shape == (len(X),)
        assert np.all((p >= 0.0) & (p <= 1.0))


def test_both_models_rank_better_than_chance(fitted):
    X, y, scorecard, booster = fitted
    for model in (scorecard, booster):
        assert roc_auc(y.to_numpy(), model.predict_proba(X)) > 0.70


def test_every_retained_driver_points_the_right_way(fitted):
    """WOE is signed high-means-safe, so every coefficient must be negative.

    A positive one says safer applicants default more often, which is a
    collinearity artefact; ``enforce_direction`` is supposed to have cut it.
    """
    _, _, scorecard, _ = fitted
    assert (scorecard.coefficients()["coefficient"] < 0).all()


def test_direction_enforcement_actually_cuts_something(xy):
    """Guards the guard: if nothing is ever dropped the check is vacuous."""
    X, y = xy
    unconstrained = Scorecard(enforce_direction=False).fit(X, y)
    constrained = Scorecard(enforce_direction=True).fit(X, y)
    assert (unconstrained.coefficients()["coefficient"] > 0).any()
    assert constrained.dropped_for_direction
    assert set(constrained.dropped_for_direction) <= set(unconstrained.encoder.binnings)


def test_scorecard_score_is_monotone_decreasing_in_pd(fitted):
    X, _, scorecard, _ = fitted
    pd_hat = scorecard.predict_proba(X)
    score = scorecard.score(X)
    assert roc_auc((pd_hat > np.median(pd_hat)).astype(int), -score) > 0.99


def test_points_table_covers_every_retained_bin(fitted):
    _, _, scorecard, _ = fitted
    table = scorecard.points_table()
    assert set(table["feature"]) == set(scorecard.encoder.binnings)
    assert table["points"].notna().all()


def test_booster_tolerates_an_unseen_category(fitted):
    X, _, _, booster = fitted
    altered = X.copy()
    altered["purpose"] = altered["purpose"].cat.add_categories(["crypto"])
    altered.iloc[0, altered.columns.get_loc("purpose")] = "crypto"
    assert np.isfinite(booster.predict_proba(altered)).all()


def test_booster_reports_importance_for_every_encoded_column(fitted):
    _, _, _, booster = fitted
    importance = booster.feature_importance()
    assert len(importance) == len(booster._columns)
    assert importance["gain"].iloc[0] > 0


def test_platt_calibration_moves_a_biased_model_towards_truth(fitted):
    X, y, _, booster = fitted
    # Halve every predicted odds: a badly under-predicting model.
    p = booster.predict_proba(X)
    biased = p / (2.0 - p)
    before = calibration_summary(y.to_numpy(), biased)["observed_over_predicted"]
    after = calibration_summary(
        y.to_numpy(), PlattCalibrator().fit(biased, y.to_numpy()).transform(biased)
    )["observed_over_predicted"]
    assert abs(after - 1.0) < abs(before - 1.0)


def test_calibration_preserves_the_ranking(fitted):
    X, y, _, booster = fitted
    p = booster.predict_proba(X)
    calibrated = PlattCalibrator().fit(p, y.to_numpy()).transform(p)
    # Platt is monotone, so AUC must be identical, not merely close.
    assert roc_auc(y.to_numpy(), calibrated) == pytest.approx(roc_auc(y.to_numpy(), p), abs=1e-9)


def test_reliability_table_bands_sum_to_the_sample(fitted):
    X, y, _, booster = fitted
    table = reliability_table(y.to_numpy(), booster.predict_proba(X), n_bands=10)
    assert table["n"].sum() == len(X)
    assert len(table) == 10
