"""The two candidate models.

The comparison is deliberately asymmetric.  The scorecard sees a WOE-encoded,
monotone, reviewer-legible representation and fits a penalised logistic
regression on it; the booster sees the raw block including native missing
values and categoricals.  That is how the choice is actually posed in a credit
shop — not "logistic versus trees" but "the representation a regulator will
sign off versus the one that squeezes out the last few Gini points".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from creditrisk.features import WoeEncoder


@runtime_checkable
class PdModel(Protocol):
    """Everything downstream code is allowed to assume about a model."""

    name: str

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PdModel": ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class Scorecard:
    """Penalised logistic regression on WOE features, plus points scaling.

    The points scaling is not decoration: it is what makes the model auditable.
    ``pdo`` is the number of points that doubles the good:bad odds, and every
    bin contributes a fixed, printable number of points.
    """

    name: str = "scorecard"
    c: float = 1.0
    max_bins: int = 8
    min_bin_fraction: float = 0.02
    min_information_value: float = 0.02
    pdo: float = 20.0
    base_score: float = 600.0
    base_odds: float = 50.0
    #: Drop features whose fitted coefficient has the wrong sign, one at a time.
    enforce_direction: bool = True
    random_state: int = 7

    encoder: WoeEncoder | None = None
    #: Features removed by direction enforcement, in the order they were cut.
    dropped_for_direction: list[str] = field(default_factory=list)
    _model: object | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Scorecard":
        from sklearn.linear_model import LogisticRegression

        self.encoder = WoeEncoder.fit(
            X,
            y,
            max_bins=self.max_bins,
            min_bin_fraction=self.min_bin_fraction,
            min_information_value=self.min_information_value,
        )
        self.dropped_for_direction = []
        while True:
            Z = self.encoder.transform(X)
            model = LogisticRegression(
                C=self.c, solver="lbfgs", max_iter=2_000, random_state=self.random_state
            )
            model.fit(Z.to_numpy(), y.to_numpy())
            betas = np.asarray(model.coef_).ravel()
            if not self.enforce_direction:
                break
            # WOE is signed so that high means safe; a positive coefficient
            # therefore says "safer applicants default more", which is a
            # suppression artefact rather than a finding.
            offenders = [(b, name) for b, name in zip(betas, self.encoder.binnings, strict=True) if b > 0]
            if not offenders:
                break
            if len(self.encoder.binnings) <= 2:
                raise ValueError(
                    "direction enforcement exhausted the feature set; "
                    "the WOE block is too collinear to fit a signed scorecard"
                )
            worst = max(offenders)[1]
            self.dropped_for_direction.append(worst)
            self.encoder = WoeEncoder(
                binnings={k: v for k, v in self.encoder.binnings.items() if k != worst}
            )
        self._model = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        model, encoder = self._require_fitted()
        Z = encoder.transform(X)
        return model.predict_proba(Z.to_numpy())[:, 1]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Map PD to an integer scorecard score (higher is safer)."""
        pd_hat = np.clip(self.predict_proba(X), 1e-9, 1.0 - 1e-9)
        good_bad_odds = (1.0 - pd_hat) / pd_hat
        factor = self.pdo / np.log(2.0)
        offset = self.base_score - factor * np.log(self.base_odds)
        return np.round(offset + factor * np.log(good_bad_odds)).astype(int)

    def coefficients(self) -> pd.DataFrame:
        model, encoder = self._require_fitted()
        return (
            pd.DataFrame(
                {
                    "feature": [n.removeprefix("woe_") for n in encoder.feature_names],
                    "coefficient": np.asarray(model.coef_).ravel(),
                }
            )
            .assign(abs_coefficient=lambda d: d["coefficient"].abs())
            .sort_values("abs_coefficient", ascending=False)
            .drop(columns="abs_coefficient")
            .reset_index(drop=True)
        )

    def points_table(self) -> pd.DataFrame:
        """Points awarded per bin — the artefact a credit committee signs off.

        A negative coefficient on a WOE feature means the fit has inverted the
        expected direction, which is a rejection reason in review, so the sign
        is surfaced here rather than buried in the coefficient vector.
        """
        model, encoder = self._require_fitted()
        coefs = dict(
            zip(
                (n.removeprefix("woe_") for n in encoder.feature_names),
                np.asarray(model.coef_).ravel(),
                strict=True,
            )
        )
        factor = self.pdo / np.log(2.0)
        rows = []
        for feature, binning in encoder.binnings.items():
            beta = coefs[feature]
            for b in binning.bins:
                rows.append(
                    {
                        "feature": feature,
                        "bin": b.label,
                        "n": b.n,
                        "event_rate": b.event_rate,
                        "woe": b.woe,
                        "points": round(-factor * beta * b.woe, 1),
                    }
                )
        return pd.DataFrame(rows)

    def _require_fitted(self):
        if self._model is None or self.encoder is None:
            raise RuntimeError("Scorecard.fit must be called before use")
        return self._model, self.encoder


@dataclass
class BoostedTrees:
    """XGBoost on the raw block, with early stopping on an out-of-time slice.

    Missing values are passed through rather than imputed: the generator makes
    missingness informative, and XGBoost's default direction learning picks
    that up for free.
    """

    name: str = "boosted_trees"
    n_estimators: int = 800
    learning_rate: float = 0.03
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: float = 20.0
    reg_lambda: float = 2.0
    early_stopping_rounds: int = 50
    validation_fraction: float = 0.15
    random_state: int = 7

    _model: object | None = None
    _columns: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BoostedTrees":
        from xgboost import XGBClassifier

        frame = self._encode(X, fit=True)
        n_valid = max(1, int(self.validation_fraction * len(frame)))
        # Tail slice, not a random slice: early stopping should be judged on
        # the most recent cohorts, the same way the model will be judged live.
        X_tr, X_va = frame.iloc[:-n_valid], frame.iloc[-n_valid:]
        y_tr, y_va = y.iloc[:-n_valid], y.iloc[-n_valid:]

        model = XGBClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_lambda=self.reg_lambda,
            objective="binary:logistic",
            eval_metric="logloss",
            early_stopping_rounds=self.early_stopping_rounds,
            random_state=self.random_state,
            n_jobs=4,
            tree_method="hist",
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        self._model = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("BoostedTrees.fit must be called before use")
        return self._model.predict_proba(self._encode(X, fit=False))[:, 1]

    def feature_importance(self) -> pd.DataFrame:
        if self._model is None or self._columns is None:
            raise RuntimeError("BoostedTrees.fit must be called before use")
        gains = self._model.get_booster().get_score(importance_type="gain")
        return (
            pd.DataFrame({"feature": self._columns})
            .assign(gain=lambda d: d["feature"].map(gains).fillna(0.0))
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )

    def _encode(self, X: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        frame = pd.get_dummies(X, columns=self._object_columns(X), dummy_na=False, dtype=float)
        if fit:
            self._columns = list(frame.columns)
            return frame
        if self._columns is None:
            raise RuntimeError("BoostedTrees.fit must be called before use")
        return frame.reindex(columns=self._columns, fill_value=0.0)

    @staticmethod
    def _object_columns(X: pd.DataFrame) -> list[str]:
        return [
            c
            for c in X.columns
            if isinstance(X[c].dtype, pd.CategoricalDtype) or X[c].dtype == object
        ]
