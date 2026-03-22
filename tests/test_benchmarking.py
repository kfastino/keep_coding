from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from pioneer_adaptive.benchmarking import (
    _dot_lookup,
    _parse_score,
    run_benchmark,
    weighted_score,
)
from pioneer_adaptive.config import ScoreParserConfig
from pioneer_adaptive.config import BenchmarkConfig


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


def test_run_benchmark_rewrites_out_path_with_run_output_dir(tmp_path: Path) -> None:
    script = tmp_path / "emit.py"
    script.write_text(
        "\n".join(
            [
                "import argparse",
                "import json",
                "from pathlib import Path",
                "p = argparse.ArgumentParser()",
                "p.add_argument('--out', required=True)",
                "args = p.parse_args()",
                "out = Path(args.out)",
                "out.parent.mkdir(parents=True, exist_ok=True)",
                "out.write_text(json.dumps({'metrics': {'score': 0.5}}), encoding='utf-8')",
                "print(json.dumps({'ok': True}))",
            ]
        ),
        encoding="utf-8",
    )
    mode = script.stat().st_mode
    script.chmod(mode | stat.S_IXUSR)

    benchmark = BenchmarkConfig.model_validate(
        {
            "name": "emit",
            "cwd": str(tmp_path),
            "command": ["python3", str(script), "--out", "outputs/sample.json"],
            "parser": {
                "mode": "json_file",
                "json_path": "outputs/sample.json",
                "key_path": "metrics.score",
            },
        }
    )

    result = run_benchmark(
        benchmark,
        model_id="model-a",
        project_root=tmp_path,
        template_vars={"run_output_dir": "outputs/run-123"},
    )

    assert result.score == 0.5
    assert (tmp_path / "outputs" / "run-123" / "sample.json").exists()

