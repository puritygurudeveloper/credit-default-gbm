"""The CLI is the only entry point CI actually invokes, so it gets tests."""

import json

import pandas as pd
import pytest

from creditrisk.cli import main

pytestmark = pytest.mark.slow


def test_generate_writes_a_readable_book(tmp_path):
    out = tmp_path / "nested" / "book.csv"
    assert main(["generate", "--rows", "500", "--out", str(out)]) == 0
    frame = pd.read_csv(out)
    assert len(frame) == 500
    assert "defaulted" in frame.columns


def test_run_reports_and_writes_json(tmp_path, capsys):
    target = tmp_path / "leaderboard.json"
    assert main(["run", "--rows", "12000", "--json", str(target)]) == 0

    out = capsys.readouterr().out
    assert "scorecard" in out and "boosted_trees" in out
    assert "swap-set analysis" in out

    payload = json.loads(target.read_text())
    assert {row["model"] for row in payload} == {"scorecard", "boosted_trees"}
    assert all(0.5 < row["auc"] < 1.0 for row in payload)


def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit) as excinfo:
        main(["frobnicate"])
    assert excinfo.value.code != 0
