"""SHAP attributions and adverse action reasons.

Model risk review asks two questions of any credit model: what drives it
overall, and why was this particular applicant declined.  US lenders have to
answer the second one in writing under Regulation B, so reason codes are a
delivery requirement rather than an analysis nicety.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from creditrisk.models import BoostedTrees


def shap_values(model, X: pd.DataFrame, max_rows: int = 5_000, random_state: int = 7) -> pd.DataFrame:
    """Tree SHAP values for a fitted :class:`~creditrisk.models.BoostedTrees`.

    Subsampled by default — exact tree SHAP is linear in rows and the global
    picture is stable well before the full book is used.
    """
    import shap

    # Tree SHAP is exact for trees and simply wrong for anything else, so the
    # check is on the type rather than on whether an attribute happens to exist.
    if not isinstance(model, BoostedTrees):
        raise TypeError(
            f"tree SHAP needs a BoostedTrees model, got {type(model).__name__}; "
            "a Scorecard is already additive and explains itself via points_table()"
        )
    booster = model._model
    if booster is None:
        raise RuntimeError("BoostedTrees.fit must be called before use")

    frame = model._encode(X, fit=False)
    if len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=random_state)
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(frame)
    return pd.DataFrame(values, columns=frame.columns, index=frame.index)


def global_importance(shap_frame: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute SHAP per feature, with the sign of the average effect."""
    return (
        pd.DataFrame(
            {
                "feature": shap_frame.columns,
                "mean_abs_shap": shap_frame.abs().mean().to_numpy(),
                "mean_shap": shap_frame.mean().to_numpy(),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def adverse_action_reasons(shap_frame: pd.DataFrame, top_k: int = 4) -> pd.DataFrame:
    """The ``top_k`` features pushing each applicant towards decline.

    Only positive contributions qualify: a reason code has to name something
    that counted against the applicant, so features that helped are excluded
    even when their magnitude is larger.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    rows = []
    for row_id, row in shap_frame.iterrows():
        adverse = row[row > 0].sort_values(ascending=False).head(top_k)
        for rank, (feature, contribution) in enumerate(adverse.items(), start=1):
            rows.append(
                {"row_id": row_id, "rank": rank, "feature": feature, "shap": float(contribution)}
            )
    return pd.DataFrame(rows, columns=["row_id", "rank", "feature", "shap"])


def stability_of_importance(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Rank correlation of global importance across resamples.

    A feature ranking that reshuffles between bootstrap draws is not an
    explanation, and this is the cheapest way to catch that before it reaches
    a governance pack.
    """
    if len(frames) < 2:
        raise ValueError("need at least two frames to assess stability")
    ranks = pd.concat(
        [global_importance(f).set_index("feature")["mean_abs_shap"].rank(ascending=False) for f in frames],
        axis=1,
    )
    ranks.columns = [f"draw_{i}" for i in range(len(frames))]
    return (
        ranks.assign(mean_rank=ranks.mean(axis=1), rank_sd=ranks.std(axis=1))
        .sort_values("mean_rank")
        .reset_index()
    )
