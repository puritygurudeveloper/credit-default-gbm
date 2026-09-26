"""From a calibrated PD to an approve/decline decision.

A cut-off is an economic choice, not a statistical one.  Accuracy is silent on
it: at an 8% base rate, declining everyone scores 92% accurate and originates
nothing.  What matters is the expected profit of the marginal approval, which
is why the threshold search here maximises portfolio profit rather than F1 or
Youden's J.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from creditrisk.data import _amortising_payment

#: Average outstanding balance on a level-payment amortising loan, as a share
#: of the original principal.  Roughly half, and the usual desk shorthand for
#: interest income without simulating the amortisation schedule loan by loan.
AVERAGE_BALANCE_FRACTION = 0.52


@dataclass(frozen=True)
class LoanEconomics:
    """Unit economics of a single approved loan, in Basel-style components.

    ``lgd`` and ``ead_fraction`` are the usual credit parameters: loss given
    default as a share of the exposure, and exposure at default as a share of
    the original principal (below 1.0 because a defaulting borrower has
    typically amortised part of the balance first).
    """

    lgd: float = 0.82
    ead_fraction: float = 0.88
    #: Blended cost of funds, annualised.
    cost_of_funds: float = 0.055
    #: Fee deducted from the disbursement, so it is earned at origination and
    #: survives a subsequent default.  This is what makes the low-risk grades
    #: profitable at all: at a 6.8% APR the net interest margin alone does not
    #: cover the cost of putting the loan on the books.
    origination_fee_rate: float = 0.045
    #: Acquisition and servicing cost as a share of principal.
    origination_cost_rate: float = 0.022
    #: Unsecured consumer loans prepay heavily; expected life is well short
    #: of the contractual term, which is what keeps the break-even PD honest.
    prepayment_speed: float = 0.35

    def __post_init__(self) -> None:
        for field_name in ("lgd", "ead_fraction"):
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must lie in [0, 1]")

    def expected_life_years(self, term_months: np.ndarray) -> np.ndarray:
        """Contractual life haircut for prepayment, floored at six months."""
        contractual = np.asarray(term_months, dtype=float) / 12.0
        return np.maximum(contractual * (1.0 - self.prepayment_speed), 0.5)

    def expected_profit(self, X: pd.DataFrame, pd_hat: np.ndarray) -> np.ndarray:
        """Expected dollar profit per loan if approved.

        Revenue is the net interest margin earned over the expected life on the
        surviving balance; loss is ``PD x LGD x EAD``.  Both are discounted only
        implicitly — at these tenors the ranking is unchanged and the extra
        machinery would obscure the decision rule.
        """
        principal = X["loan_amount"].to_numpy(dtype=float)
        rate = X["interest_rate"].to_numpy(dtype=float)
        term = X["term_months"].to_numpy(dtype=float)
        pd_hat = np.clip(np.asarray(pd_hat, dtype=float), 0.0, 1.0)

        life = self.expected_life_years(term)
        average_balance = AVERAGE_BALANCE_FRACTION * principal
        net_margin = np.maximum(rate - self.cost_of_funds, 0.0)

        interest = average_balance * net_margin * life * (1.0 - pd_hat)
        fee = self.origination_fee_rate * principal
        loss = pd_hat * self.lgd * self.ead_fraction * principal
        costs = self.origination_cost_rate * principal
        return interest + fee - loss - costs

    def breakeven_pd(self, X: pd.DataFrame) -> np.ndarray:
        """PD at which a loan's expected profit crosses zero.

        Solves ``r(1-p) - p*L*E*P - c = 0`` for ``p``, so approvals can be
        explained per applicant instead of via a single portfolio cut-off.
        """
        principal = X["loan_amount"].to_numpy(dtype=float)
        rate = X["interest_rate"].to_numpy(dtype=float)
        term = X["term_months"].to_numpy(dtype=float)
        life = self.expected_life_years(term)
        margin_if_good, margin_if_bad = _outcome_margins(principal, rate, term, self)
        denominator = margin_if_good - margin_if_bad
        return np.clip(np.where(denominator > 0, margin_if_good / denominator, 0.0), 0.0, 1.0)


def _outcome_margins(
    principal: np.ndarray, rate: np.ndarray, term_months: np.ndarray, econ: LoanEconomics
) -> tuple[np.ndarray, np.ndarray]:
    """Realised dollar margin of one loan under each outcome.

    Returned as a pair rather than recomputed at each call site so that the
    profit curve, the break-even PD and the expected profit can never drift
    apart — the classic way a threshold analysis quietly stops being valid.
    """
    life = econ.expected_life_years(term_months)
    interest = AVERAGE_BALANCE_FRACTION * principal * np.maximum(rate - econ.cost_of_funds, 0.0) * life
    fee = econ.origination_fee_rate * principal
    costs = econ.origination_cost_rate * principal
    loss = econ.lgd * econ.ead_fraction * principal
    return interest + fee - costs, fee - costs - loss


@dataclass(frozen=True)
class ThresholdDecision:
    """A credit policy and what it buys on the out-of-time book."""

    policy: str
    threshold: float
    approval_rate: float
    n_approved: int
    realised_bad_rate: float
    total_profit: float
    profit_per_application: float

    def summary(self) -> str:
        cut = "per-loan break-even" if np.isnan(self.threshold) else f"PD <= {self.threshold:.4f}"
        return (
            f"{self.policy:24s} | {cut} | approve {self.approval_rate:6.2%} "
            f"| bad rate {self.realised_bad_rate:6.2%} | profit ${self.total_profit:,.0f}"
        )


def profit_curve(
    X: pd.DataFrame,
    y: np.ndarray,
    pd_hat: np.ndarray,
    economics: LoanEconomics | None = None,
    n_points: int = 200,
) -> pd.DataFrame:
    """Realised profit across the whole range of possible cut-offs.

    Profit is scored on the *realised* outcome, not the predicted one — an
    approved loan that defaults costs real money regardless of what the model
    said, which is what separates this from a plot of expected profit.
    """
    econ = economics or LoanEconomics()
    y = np.asarray(y, dtype=float)
    pd_hat = np.asarray(pd_hat, dtype=float)
    if not (len(X) == len(y) == len(pd_hat)):
        raise ValueError("X, y and pd_hat must all be the same length")

    margin_if_good, margin_if_bad = _outcome_margins(
        X["loan_amount"].to_numpy(dtype=float),
        X["interest_rate"].to_numpy(dtype=float),
        X["term_months"].to_numpy(dtype=float),
        econ,
    )
    realised = np.where(y > 0.5, margin_if_bad, margin_if_good)

    thresholds = np.unique(np.quantile(pd_hat, np.linspace(0.0, 1.0, n_points)))
    rows = []
    for t in thresholds:
        approved = pd_hat <= t
        n_approved = int(approved.sum())
        if n_approved == 0:
            continue
        rows.append(
            {
                "threshold": float(t),
                "n_approved": n_approved,
                "approval_rate": n_approved / len(y),
                "realised_bad_rate": float(y[approved].mean()),
                "total_profit": float(realised[approved].sum()),
                "profit_per_application": float(realised[approved].sum() / len(y)),
            }
        )
    return pd.DataFrame(rows)


def optimal_threshold(
    X: pd.DataFrame,
    y: np.ndarray,
    pd_hat: np.ndarray,
    economics: LoanEconomics | None = None,
    min_approval_rate: float = 0.0,
) -> ThresholdDecision:
    """Profit-maximising cut-off, optionally subject to a volume floor.

    The floor exists because origination targets are real: a model is rarely
    allowed to shrink the book by half even when that is the profit optimum.
    """
    curve = profit_curve(X, y, pd_hat, economics=economics)
    eligible = curve.loc[curve["approval_rate"] >= min_approval_rate]
    if eligible.empty:
        raise ValueError(f"no cut-off reaches an approval rate of {min_approval_rate:.2%}")
    best = eligible.loc[eligible["total_profit"].idxmax()]
    return ThresholdDecision(
        policy="global_cutoff",
        threshold=float(best["threshold"]),
        approval_rate=float(best["approval_rate"]),
        n_approved=int(best["n_approved"]),
        realised_bad_rate=float(best["realised_bad_rate"]),
        total_profit=float(best["total_profit"]),
        profit_per_application=float(best["profit_per_application"]),
    )


def evaluate_policy(
    X: pd.DataFrame,
    y: np.ndarray,
    approve: np.ndarray,
    economics: LoanEconomics | None = None,
    *,
    policy: str,
    threshold: float = float("nan"),
) -> ThresholdDecision:
    """Score an arbitrary approve/decline mask on realised outcomes."""
    econ = economics or LoanEconomics()
    y = np.asarray(y, dtype=float)
    approve = np.asarray(approve, dtype=bool)
    if not (len(X) == len(y) == len(approve)):
        raise ValueError("X, y and approve must all be the same length")
    if not approve.any():
        raise ValueError(f"policy {policy!r} approves nothing")

    margin_if_good, margin_if_bad = _outcome_margins(
        X["loan_amount"].to_numpy(dtype=float),
        X["interest_rate"].to_numpy(dtype=float),
        X["term_months"].to_numpy(dtype=float),
        econ,
    )
    realised = np.where(y > 0.5, margin_if_bad, margin_if_good)
    return ThresholdDecision(
        policy=policy,
        threshold=threshold,
        approval_rate=float(approve.mean()),
        n_approved=int(approve.sum()),
        realised_bad_rate=float(y[approve].mean()),
        total_profit=float(realised[approve].sum()),
        profit_per_application=float(realised[approve].sum() / len(y)),
    )


def risk_adjusted_policy(
    X: pd.DataFrame,
    y: np.ndarray,
    pd_hat: np.ndarray,
    economics: LoanEconomics | None = None,
    margin_of_safety: float = 0.0,
) -> ThresholdDecision:
    """Approve each loan on its own break-even, not a portfolio-wide cut-off.

    A single PD threshold is the wrong instrument once pricing varies: the
    break-even PD runs from roughly 6% in the best grade to over 30% in the
    worst, so any global cut-off simultaneously approves uneconomic prime
    loans and declines profitable subprime ones.  ``margin_of_safety`` shades
    the break-even downwards to buy protection against model error.
    """
    econ = economics or LoanEconomics()
    if not 0.0 <= margin_of_safety < 1.0:
        raise ValueError("margin_of_safety must lie in [0, 1)")
    hurdle = (1.0 - margin_of_safety) * econ.breakeven_pd(X)
    approve = np.asarray(pd_hat, dtype=float) <= hurdle
    return evaluate_policy(X, y, approve, economics=econ, policy="risk_adjusted")


def swap_set_analysis(
    X: pd.DataFrame,
    y: np.ndarray,
    pd_champion: np.ndarray,
    pd_challenger: np.ndarray,
    economics: LoanEconomics | None = None,
) -> pd.DataFrame:
    """Who changes decision when the challenger replaces the champion.

    Both models are held at their own profit-optimal cut-off and at a matched
    approval rate, because the swap-in/swap-out population is what a credit
    committee argues about — not the Gini delta.
    """
    econ = economics or LoanEconomics()
    y = np.asarray(y, dtype=float)
    champ = optimal_threshold(X, y, pd_champion, economics=econ)
    # Match volume so the comparison is not just "the challenger approves less".
    challenger_cut = float(np.quantile(pd_challenger, champ.approval_rate))

    approve_champ = pd_champion <= champ.threshold
    approve_chal = pd_challenger <= challenger_cut

    groups = {
        "both_approve": approve_champ & approve_chal,
        "swap_in": ~approve_champ & approve_chal,
        "swap_out": approve_champ & ~approve_chal,
        "both_decline": ~approve_champ & ~approve_chal,
    }
    rows = []
    for label, mask in groups.items():
        n = int(mask.sum())
        rows.append(
            {
                "group": label,
                "n": n,
                "share": n / len(y),
                "bad_rate": float(y[mask].mean()) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows)
