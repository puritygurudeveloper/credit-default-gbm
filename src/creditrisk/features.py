"""Weight-of-evidence binning.

WOE is the lingua franca of consumer credit: it linearises a feature against
the log-odds of default, handles missing values as a first-class bin rather
than an imputation problem, and produces an information value that model risk
reviewers already know how to read.  The gradient boosting model does not need
any of this, which is the point — the two models are given genuinely different
representations so the comparison measures something.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from creditrisk.data import _amortising_payment

#: Sentinel used for the missing-value bin so it survives a round trip to CSV.
MISSING_BIN = "__missing__"

_EPS = 0.5  # Haldane-Anscombe correction, keeps WOE finite in empty cells.


@dataclass(frozen=True)
class Bin:
    """One row of a WOE table."""

    label: str
    lower: float
    upper: float
    n: int
    n_events: int
    event_rate: float
    woe: float
    iv_contribution: float

    @property
    def is_missing(self) -> bool:
        return self.label == MISSING_BIN


@dataclass(frozen=True)
class WoeBinning:
    """Fitted WOE transform for a single feature."""

    feature: str
    bins: tuple[Bin, ...]
    edges: tuple[float, ...]
    categories: tuple[str, ...] | None

    @property
    def information_value(self) -> float:
        return float(sum(b.iv_contribution for b in self.bins))

    @property
    def strength(self) -> str:
        """Siddiqi's conventional IV bands, as used in scorecard reviews."""
        iv = self.information_value
        if iv < 0.02:
            return "unpredictive"
        if iv < 0.10:
            return "weak"
        if iv < 0.30:
            return "medium"
        if iv < 0.50:
            return "strong"
        return "suspicious"

    def transform(self, values: pd.Series) -> pd.Series:
        labels = self._assign(values)
        lookup = {b.label: b.woe for b in self.bins}
        neutral = 0.0
        return labels.map(lambda lab: lookup.get(lab, neutral)).astype(float)

    def _assign(self, values: pd.Series) -> pd.Series:
        if self.categories is not None:
            out = values.astype(object).where(values.notna(), MISSING_BIN)
            known = set(self.categories)
            return out.map(lambda v: v if v in known else MISSING_BIN)
        codes = pd.cut(values, bins=list(self.edges), labels=False, include_lowest=True, right=True)
        labels = pd.Series(np.where(np.isnan(codes), -1, codes), index=values.index).astype(int)
        numeric_labels = [b.label for b in self.bins if not b.is_missing]
        return labels.map(lambda i: MISSING_BIN if i < 0 else numeric_labels[min(i, len(numeric_labels) - 1)])

    def table(self) -> pd.DataFrame:
        """WOE table in the layout a credit risk reviewer expects."""
        return pd.DataFrame(
            [
                {
                    "feature": self.feature,
                    "bin": b.label,
                    "n": b.n,
                    "events": b.n_events,
                    "event_rate": b.event_rate,
                    "woe": b.woe,
                    "iv": b.iv_contribution,
                }
                for b in self.bins
            ]
        )


def _quantile_edges(values: pd.Series, max_bins: int) -> list[float]:
    finite = values.dropna().to_numpy(dtype=float)
    if finite.size == 0:
        return [-np.inf, np.inf]
    qs = np.linspace(0.0, 1.0, max_bins + 1)
    edges = np.unique(np.quantile(finite, qs))
    if edges.size < 2:
        return [-np.inf, np.inf]
    edges[0], edges[-1] = -np.inf, np.inf
    return [float(e) for e in edges]


def fit_woe(
    values: pd.Series,
    target: pd.Series,
    *,
    max_bins: int = 8,
    min_bin_fraction: float = 0.02,
) -> WoeBinning:
    """Fit a WOE transform, merging bins that fall below the size floor.

    Categorical columns get one bin per level; numeric columns are cut at
    quantiles and then merged left-to-right until every bin carries at least
    ``min_bin_fraction`` of the rows.  Small bins are the usual cause of
    scorecards that look excellent in development and collapse in production.
    """
    if len(values) != len(target):
        raise ValueError("values and target must be the same length")
    if not 0.0 < min_bin_fraction < 0.5:
        raise ValueError("min_bin_fraction must lie in (0, 0.5)")

    name = str(values.name or "feature")
    y = target.to_numpy(dtype=float)
    total_events = float(y.sum())
    total_non_events = float(len(y) - total_events)
    if total_events == 0 or total_non_events == 0:
        raise ValueError(f"feature {name!r}: target has a single class, WOE is undefined")

    is_categorical = isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object
    if is_categorical:
        categories = tuple(sorted({str(v) for v in values.dropna().unique()}))
        labels = values.astype(object).where(values.notna(), MISSING_BIN).map(str)
        edges: tuple[float, ...] = ()
        ordered = list(categories) + [MISSING_BIN]
    else:
        categories = None
        edge_list = _quantile_edges(values, max_bins)
        edge_list = _merge_small_bins(values, edge_list, min_bin_fraction)
        edges = tuple(edge_list)
        codes = pd.cut(values, bins=edge_list, labels=False, include_lowest=True, right=True)
        numeric_labels = [_edge_label(edge_list[i], edge_list[i + 1]) for i in range(len(edge_list) - 1)]
        labels = pd.Series(
            [MISSING_BIN if np.isnan(c) else numeric_labels[int(c)] for c in codes], index=values.index
        )
        ordered = numeric_labels + [MISSING_BIN]

    bins: list[Bin] = []
    for label in ordered:
        mask = (labels == label).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        events = float(y[mask].sum())
        non_events = float(n - events)
        dist_events = (events + _EPS) / (total_events + _EPS * len(ordered))
        dist_non_events = (non_events + _EPS) / (total_non_events + _EPS * len(ordered))
        woe = float(np.log(dist_non_events / dist_events))
        bins.append(
            Bin(
                label=label,
                lower=float("nan"),
                upper=float("nan"),
                n=n,
                n_events=int(events),
                event_rate=events / n,
                woe=woe,
                iv_contribution=float((dist_non_events - dist_events) * woe),
            )
        )

    return WoeBinning(feature=name, bins=tuple(bins), edges=edges, categories=categories)


def _edge_label(lo: float, hi: float) -> str:
    lo_s = "-inf" if np.isneginf(lo) else f"{lo:.4g}"
    hi_s = "inf" if np.isposinf(hi) else f"{hi:.4g}"
    return f"({lo_s}, {hi_s}]"


def _merge_small_bins(values: pd.Series, edges: list[float], min_fraction: float) -> list[float]:
    floor = max(1, int(round(min_fraction * len(values))))
    while len(edges) > 2:
        counts = pd.cut(values, bins=edges, include_lowest=True, right=True).value_counts(sort=False).to_numpy()
        smallest = int(np.argmin(counts))
        if counts[smallest] >= floor:
            break
        # Drop the interior edge that dissolves the offending bin.
        drop = smallest if smallest < len(edges) - 2 else smallest - 1
        edges = edges[: drop + 1] + edges[drop + 2 :]
    return edges


@dataclass(frozen=True)
class WoeEncoder:
    """A column-wise WOE transform over a whole feature block."""

    binnings: dict[str, WoeBinning]

    @classmethod
    def fit(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        max_bins: int = 8,
        min_bin_fraction: float = 0.02,
        min_information_value: float = 0.02,
        max_correlation: float = 0.80,
    ) -> "WoeEncoder":
        binnings: dict[str, WoeBinning] = {}
        for column in X.columns:
            binning = fit_woe(X[column], y, max_bins=max_bins, min_bin_fraction=min_bin_fraction)
            if binning.information_value >= min_information_value:
                binnings[column] = binning
        if not binnings:
            raise ValueError("no feature cleared the information value floor")
        binnings = _prune_correlated(binnings, X, max_correlation)
        return cls(binnings=binnings)

    @property
    def feature_names(self) -> list[str]:
        return [f"woe_{name}" for name in self.binnings]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.binnings if c not in X.columns]
        if missing:
            raise KeyError(f"frame is missing columns required by the encoder: {missing}")
        return pd.DataFrame(
            {f"woe_{name}": binning.transform(X[name]) for name, binning in self.binnings.items()},
            index=X.index,
        )

    def information_values(self) -> pd.DataFrame:
        rows = [
            {"feature": name, "iv": b.information_value, "strength": b.strength, "n_bins": len(b.bins)}
            for name, b in self.binnings.items()
        ]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)

    def woe_tables(self) -> pd.DataFrame:
        return pd.concat([b.table() for b in self.binnings.values()], ignore_index=True)


def _prune_correlated(
    binnings: dict[str, WoeBinning], X: pd.DataFrame, max_correlation: float
) -> dict[str, WoeBinning]:
    """Drop the weaker half of every strongly correlated pair.

    Scorecards are read as well as scored, and multicollinearity between WOE
    columns is the standard way a coefficient acquires the wrong sign: the fit
    is still optimal, but "higher utilisation lowers risk" is not a sentence
    that survives a credit committee.  Features are considered in descending
    information value, so the survivor of each pair is the more predictive one.
    """
    if not 0.0 < max_correlation <= 1.0:
        raise ValueError("max_correlation must lie in (0, 1]")
    ordered = sorted(binnings, key=lambda name: binnings[name].information_value, reverse=True)
    transformed = pd.DataFrame({name: binnings[name].transform(X[name]) for name in ordered})
    correlation = transformed.corr().abs()

    kept: list[str] = []
    for name in ordered:
        if all(correlation.loc[name, other] <= max_correlation for other in kept):
            kept.append(name)
    return {name: binnings[name] for name in kept}


def add_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
    """Domain ratios that neither model can construct from the raw block.

    Boosting can learn a split on two columns but not their quotient, and the
    payment-to-income burden is the single most reviewer-legible driver in a
    consumer book, so it is built explicitly rather than hoped for.
    """
    out = X.copy()
    monthly_income = X["annual_income"] / 12.0
    payment = _amortising_payment(X["loan_amount"], X["interest_rate"], X["term_months"])
    out["monthly_payment"] = payment
    out["payment_to_income"] = payment / monthly_income
    # Total burden after the new loan lands on top of existing obligations.
    out["total_obligation_ratio"] = out["payment_to_income"] + X["debt_to_income"]
    out["loan_to_income"] = X["loan_amount"] / X["annual_income"]
    out["delinq_per_inquiry"] = X["n_delinq_24m"] / (1.0 + X["n_inquiries_6m"])
    out["utilisation_x_dti"] = X["revolving_utilisation"] * X["debt_to_income"]
    return out
