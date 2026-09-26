"""Command line entry point: ``python -m creditrisk.cli``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from creditrisk.data import LoanBookConfig, generate_loan_book
from creditrisk.economics import LoanEconomics
from creditrisk.pipeline import run_experiment


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="creditrisk", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="write a synthetic loan book to CSV")
    gen.add_argument("--rows", type=int, default=100_000)
    gen.add_argument("--seed", type=int, default=20210614)
    gen.add_argument("--out", type=Path, default=Path("data/loan_book.csv"))

    run = sub.add_parser("run", help="fit both models out of time and report")
    run.add_argument("--rows", type=int, default=100_000)
    run.add_argument("--seed", type=int, default=20210614)
    run.add_argument("--holdout-vintages", type=int, default=6)
    run.add_argument("--min-approval-rate", type=float, default=0.55)
    run.add_argument("--lgd", type=float, default=0.62)
    run.add_argument("--json", type=Path, default=None, help="also write the leaderboard as JSON")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 40)

    if args.command == "generate":
        book = generate_loan_book(LoanBookConfig(n_loans=args.rows, seed=args.seed))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        book.to_csv(args.out, index=False)
        print(f"wrote {len(book):,} loans to {args.out}")
        return 0

    result = run_experiment(
        config=LoanBookConfig(n_loans=args.rows, seed=args.seed),
        econ=LoanEconomics(lgd=args.lgd),
        holdout_vintages=args.holdout_vintages,
        min_approval_rate=args.min_approval_rate,
    )
    board = result.leaderboard()

    print(f"\ndevelopment rows {result.train_rows:,} | out-of-time rows {result.test_rows:,}\n")
    print(board.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print("\nswap-set analysis at matched approval rate\n")
    print(result.swap_sets.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    print("\ncredit policies\n")
    for report in result.reports:
        for decision in (report.decision, report.risk_adjusted):
            print(f"  {report.name:14s} {decision.summary()}")

    best = max(result.reports, key=lambda r: r.risk_adjusted.total_profit)
    baseline = min(r.decision.total_profit for r in result.reports)
    uplift = best.risk_adjusted.total_profit - baseline
    print(
        f"\nbest policy: {best.name} on a per-loan break-even rule, "
        f"${uplift:,.0f} ({uplift / abs(baseline):+.2%}) over the weakest global cut-off"
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(board.to_dict(orient="records"), indent=2))
        print(f"\nleaderboard written to {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
