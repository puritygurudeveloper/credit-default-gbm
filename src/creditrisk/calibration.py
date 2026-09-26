"""Calibration: turning a ranking into a probability.

Discrimination is what gets a model picked; calibration is what makes it
usable.  Expected loss, pricing and capital all multiply the PD by something,
so a model that ranks perfectly but reports 3% when the truth is 8% produces
correctly ordered and uniformly wrong decisions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PlattCalibrator:
    """One-dimensional logistic recalibration on the log-odds scale.

    Two parameters only, so it is stable on small validation samples and
    cannot invent non-monotonicity — the usual choice when the recalibration
    slice is a handful of recent vintages.
    """

    slope: float = 1.0
    intercept: float = 0.0
    _fitted: bool = False

    def fit(self, p: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        from sklearn.linear_model import LogisticRegression

        z = _log_odds(p).reshape(-1, 1)
        # C is huge rather than penalty=None: two parameters do not need
        # shrinkage, and this spelling is stable across sklearn versions.
        model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1_000)
        model.fit(z, np.asarray(y, dtype=int))
        self.slope = float(model.coef_[0, 0])
        self.intercept = float(model.intercept_[0])
        self._fitted = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PlattCalibrator.fit must be called before use")
        return 1.0 / (1.0 + np.exp(-(self.intercept + self.slope * _log_odds(p))))


@dataclass
class IsotonicCalibrator:
    """Monotone non-parametric recalibration; flexible, and hungrier for data."""

    _model: object | None = None

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(np.asarray(p, dtype=float), np.asarray(y, dtype=float))
        self._model = model
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("IsotonicCalibrator.fit must be called before use")
        return np.clip(self._model.predict(np.asarray(p, dtype=float)), 1e-9, 1.0 - 1e-9)


def _log_odds(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0 - 1e-9)
    return np.log(q / (1.0 - q))


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, n_bands: int = 20) -> float:
    """Size-weighted mean absolute gap between predicted and realised rates."""
    table = reliability_table(y_true, p, n_bands=n_bands)
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["mean_predicted"] - table["observed_rate"]).abs()).sum())


def reliability_table(y_true: np.ndarray, p: np.ndarray, n_bands: int = 20) -> pd.DataFrame:
    """Predicted versus realised default rate, by equal-count band."""
    y = np.asarray(y_true, dtype=float)
    q = np.asarray(p, dtype=float)
    order = np.argsort(q, kind="mergesort")
    y, q = y[order], q[order]
    rows = []
    for i, idx in enumerate(np.array_split(np.arange(y.size), n_bands), start=1):
        if len(idx) == 0:
            continue
        rows.append(
            {
                "band": i,
                "n": len(idx),
                "mean_predicted": float(q[idx].mean()),
                "observed_rate": float(y[idx].mean()),
            }
        )
    table = pd.DataFrame(rows)
    table["gap"] = table["mean_predicted"] - table["observed_rate"]
    return table


def calibration_summary(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    q = np.asarray(p, dtype=float)
    return {
        "brier": brier_score(y, q),
        "ece": expected_calibration_error(y, q),
        "mean_predicted": float(q.mean()),
        "observed_rate": float(y.mean()),
        # Above 1.0 the model is under-predicting losses, which is the
        # direction the business cares about.
        "observed_over_predicted": float(y.mean() / q.mean()) if q.mean() > 0 else float("nan"),
    }
