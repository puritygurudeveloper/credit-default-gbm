"""The champion/challenger experiment, start to finish."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from creditrisk import calibration, economics, evaluate
from creditrisk.data import LoanBookConfig, generate_loan_book, split_columns, time_split
from creditrisk.features import add_ratio_features
from creditrisk.models import BoostedTrees, PdModel, Scorecard


@dataclass
class ModelReport:
    """Everything recorded about one model on the out-of-time sample."""

    name: str
    discrimination: evaluate.Discrimination
    calibration: dict[str, float]
    decision: economics.ThresholdDecision
    risk_adjusted: economics.ThresholdDecision
    psi: float
    pd_out_of_time: np.ndarray = field(repr=False)

    def as_row(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "auc": self.discrimination.auc,
            "gini": self.discrimination.gini,
            "ks": self.discrimination.ks,
            "pr_auc": self.discrimination.pr_auc,
            "brier": self.calibration["brier"],
            "ece": self.calibration["ece"],
            "obs_over_pred": self.calibration["observed_over_predicted"],
            "psi": self.psi,
            "cutoff": self.decision.threshold,
            "approval_rate": self.decision.approval_rate,
            "bad_rate": self.decision.realised_bad_rate,
            "profit": self.decision.total_profit,
            "ra_approval_rate": self.risk_adjusted.approval_rate,
            "ra_bad_rate": self.risk_adjusted.realised_bad_rate,
            "ra_profit": self.risk_adjusted.total_profit,
        }


@dataclass
class ExperimentResult:
    reports: list[ModelReport]
    swap_sets: pd.DataFrame
    train_rows: int
    test_rows: int

    def leaderboard(self) -> pd.DataFrame:
        return pd.DataFrame([r.as_row() for r in self.reports]).sort_values("ra_profit", ascending=False)


def _calibrated_predictions(
    model: PdModel,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_oot: pd.DataFrame,
    calibration_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on the early block, calibrate on the late block, predict out of time.

    Calibrating on the data the model was fitted to would report a slope of one
    by construction, so the tail of the development window is held back.
    """
    n_cal = max(1, int(calibration_fraction * len(X_fit)))
    X_core, y_core = X_fit.iloc[:-n_cal], y_fit.iloc[:-n_cal]
    X_cal, y_cal = X_fit.iloc[-n_cal:], y_fit.iloc[-n_cal:]

    model.fit(X_core, y_core)
    raw_cal = model.predict_proba(X_cal)
    calibrator = calibration.PlattCalibrator().fit(raw_cal, y_cal.to_numpy())
    return calibrator.transform(model.predict_proba(X_fit)), calibrator.transform(model.predict_proba(X_oot))


def _apply_development_cutoff(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    pd_dev: np.ndarray,
    X_oot: pd.DataFrame,
    y_oot: pd.Series,
    pd_oot: np.ndarray,
    *,
    econ: economics.LoanEconomics | None,
    min_approval_rate: float,
) -> economics.ThresholdDecision:
    """Pick the cut-off in development, then score it out of time.

    Tuning the threshold on the evaluation sample and then reporting its profit
    on that same sample is a look-ahead: the cut-off gets to see which loans
    actually defaulted.  It flatters the global-threshold policy by enough to
    reverse the comparison against the per-loan rule, which is exactly the kind
    of quiet leak that makes a backtest unreproducible in production.
    """
    chosen = economics.optimal_threshold(
        X_dev, y_dev.to_numpy(), pd_dev, economics=econ, min_approval_rate=min_approval_rate
    )
    return economics.evaluate_policy(
        X_oot,
        y_oot.to_numpy(),
        pd_oot <= chosen.threshold,
        economics=econ,
        policy="global_cutoff",
        threshold=chosen.threshold,
    )


def run_experiment(
    config: LoanBookConfig | None = None,
    econ: economics.LoanEconomics | None = None,
    holdout_vintages: int = 6,
    calibration_fraction: float = 0.15,
    min_approval_rate: float = 0.55,
) -> ExperimentResult:
    """Build the book, fit both models out of time, and price the decisions."""
    book = generate_loan_book(config)
    train_raw, test_raw = time_split(book, holdout_vintages=holdout_vintages)

    X_train, y_train = split_columns(train_raw)
    X_test, y_test = split_columns(test_raw)
    X_train, X_test = add_ratio_features(X_train), add_ratio_features(X_test)

    candidates: list[PdModel] = [Scorecard(), BoostedTrees()]
    reports: list[ModelReport] = []
    oot_predictions: dict[str, np.ndarray] = {}

    for model in candidates:
        in_sample, out_of_time = _calibrated_predictions(
            model, X_train, y_train, X_test, calibration_fraction
        )
        oot_predictions[model.name] = out_of_time
        reports.append(
            ModelReport(
                name=model.name,
                discrimination=evaluate.discrimination(y_test.to_numpy(), out_of_time),
                calibration=calibration.calibration_summary(y_test.to_numpy(), out_of_time),
                decision=_apply_development_cutoff(
                    X_train, y_train, in_sample, X_test, y_test, out_of_time,
                    econ=econ, min_approval_rate=min_approval_rate,
                ),
                risk_adjusted=economics.risk_adjusted_policy(
                    X_test, y_test.to_numpy(), out_of_time, economics=econ
                ),
                psi=evaluate.population_stability_index(in_sample, out_of_time),
                pd_out_of_time=out_of_time,
            )
        )

    swap = economics.swap_set_analysis(
        X_test,
        y_test.to_numpy(),
        pd_champion=oot_predictions["scorecard"],
        pd_challenger=oot_predictions["boosted_trees"],
        economics=econ,
    )
    return ExperimentResult(
        reports=reports, swap_sets=swap, train_rows=len(X_train), test_rows=len(X_test)
    )
