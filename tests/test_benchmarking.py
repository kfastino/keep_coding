from __future__ import annotations

import json
from pathlib import Path

import pytest

from pioneer_adaptive.benchmarking import _dot_lookup, _parse_score, weighted_score
from pioneer_adaptive.config import ScoreParserConfig


def test_dot_lookup_nested_path() -> None:
    payload = {"metrics": {"pass_at_1": 0.72}}
    assert _dot_lookup(payload, "metrics.pass_at_1") == 0.72


def test_parse_json_file_score(tmp_path: Path) -> None:
    result_file = tmp_path / "result.json"
    result_file.write_text(json.dumps({"metrics": {"success_rate": 0.55}}), encoding="utf-8")
    parser = ScoreParserConfig(
        mode="json_file",
        json_path=str(result_file),
        key_path="metrics.success_rate",
    )
    assert _parse_score(parser, stdout="", stderr="", benchmark_cwd=tmp_path) == 0.55


def test_weighted_score() -> None:
    class Item:
        def __init__(self, name: str, score: float) -> None:
            self.name = name
            self.score = score

    score = weighted_score(
        [Item("a", 0.8), Item("b", 0.4)],  # type: ignore[arg-type]
        {"a": 0.75, "b": 0.25},
    )
    assert score == pytest.approx(0.7)

