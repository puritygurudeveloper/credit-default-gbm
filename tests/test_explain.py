"""SHAP is an optional extra, so these tests skip cleanly without it."""

import pytest

from creditrisk.explain import (
    adverse_action_reasons,
    global_importance,
    shap_values,
    stability_of_importance,
)
from creditrisk.models import BoostedTrees, Scorecard

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def shap_frame(xy):
    pytest.importorskip("shap")
    X, y = xy
    model = BoostedTrees(n_estimators=60).fit(X, y)
    return shap_values(model, X, max_rows=400)


def test_shap_is_rejected_for_a_model_it_cannot_explain(xy):
    pytest.importorskip("shap")
    X, y = xy
    with pytest.raises(TypeError, match="BoostedTrees"):
        shap_values(Scorecard().fit(X, y), X)


def test_global_importance_is_ranked_and_complete(shap_frame):
    importance = global_importance(shap_frame)
    assert len(importance) == shap_frame.shape[1]
    assert importance["mean_abs_shap"].is_monotonic_decreasing
    assert (importance["mean_abs_shap"] >= 0).all()


def test_reason_codes_only_cite_things_that_hurt(shap_frame):
    reasons = adverse_action_reasons(shap_frame, top_k=3)
    assert (reasons["shap"] > 0).all()
    assert reasons.groupby("row_id")["rank"].max().max() <= 3
    # Every applicant with at least one adverse driver must get a reason.
    expected = int((shap_frame > 0).any(axis=1).sum())
    assert reasons["row_id"].nunique() == expected


def test_reason_codes_reject_a_nonsense_k(shap_frame):
    with pytest.raises(ValueError, match="top_k"):
        adverse_action_reasons(shap_frame, top_k=0)


def test_importance_stability_needs_more_than_one_draw(shap_frame):
    with pytest.raises(ValueError, match="at least two"):
        stability_of_importance([shap_frame])


def test_importance_ranking_is_stable_across_resamples(shap_frame):
    halves = [shap_frame.iloc[: len(shap_frame) // 2], shap_frame.iloc[len(shap_frame) // 2 :]]
    stability = stability_of_importance(halves)
    assert len(stability) == shap_frame.shape[1]
    # The single strongest driver should not move on a resample.
    assert stability["rank_sd"].iloc[0] < 2.0
