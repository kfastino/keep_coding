from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pioneer_adaptive.config import BenchmarkConfig, ScoreParserConfig


@dataclass
class BenchmarkResult:
    name: str
    model_id: str
    score: float
    duration_seconds: float
    return_code: int
    stdout: str
    stderr: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply_template(value: str, template_vars: dict[str, str]) -> str:
    rendered = value
    for key, replacement in template_vars.items():
        rendered = rendered.replace(f"{{{key}}}", replacement)
    return rendered


def _rewrite_output_path(path_value: str, run_output_dir: str | None) -> str:
    if not run_output_dir:
        return path_value
    if Path(path_value).is_absolute():
        return path_value
    return str(Path(run_output_dir) / Path(path_value).name)


def _dot_lookup(payload: dict[str, Any], key_path: str) -> Any:
    current: Any = payload
    for key in key_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing key '{key}' in path '{key_path}'")
        current = current[key]
    return current


def _parse_score(
    parser: ScoreParserConfig, *, stdout: str, stderr: str, benchmark_cwd: Path
) -> float:
    if parser.mode == "json_file":
        raw_json_path = parser.json_path or ""
        json_path = Path(raw_json_path)
        if not json_path.is_absolute():
            json_path = benchmark_cwd / json_path
        data = json.loads(json_path.read_text(encoding="utf-8"))
        value = _dot_lookup(data, parser.key_path or "")
        return float(value)

    if parser.mode == "stdout_json":
        candidate = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if not isinstance(candidate, dict):
            raise ValueError("Could not parse JSON object from benchmark stdout")
        value = _dot_lookup(candidate, parser.key_path or "")
        return float(value)

    if parser.mode == "regex":
        pattern = parser.pattern or ""
        match = re.search(pattern, f"{stdout}\n{stderr}", flags=re.MULTILINE)
        if not match:
            raise ValueError(f"Regex score pattern did not match: {pattern}")
        token = match.group(1) if match.groups() else match.group(0)
        return float(token)

    raise ValueError(f"Unsupported parser mode: {parser.mode}")


def run_benchmark(
    benchmark: BenchmarkConfig,
    *,
    model_id: str,
    project_root: Path,
    template_vars: dict[str, str] | None = None,
) -> BenchmarkResult:
    render_vars = {"model_id": model_id, **(template_vars or {})}

    formatted_command = [_apply_template(part, render_vars) for part in benchmark.command]
    run_output_dir = render_vars.get("run_output_dir")
    configured_out_path: str | None = None
    for idx, part in enumerate(formatted_command[:-1]):
        if part == "--out":
            configured_out_path = formatted_command[idx + 1]
            formatted_command[idx + 1] = _rewrite_output_path(formatted_command[idx + 1], run_output_dir)
            break

    benchmark_cwd = Path(_apply_template(benchmark.cwd, render_vars))
    if not benchmark_cwd.is_absolute():
        benchmark_cwd = project_root / benchmark_cwd

    env = {**benchmark.env}
    env = {key: _apply_template(value, render_vars) for key, value in env.items()}

    merged_env = None
    if env:
        merged_env = {**os.environ, **env}

    parser = benchmark.parser
    if parser.mode == "json_file" and parser.json_path:
        resolved_json_path = _apply_template(parser.json_path, render_vars)
        if configured_out_path and resolved_json_path == configured_out_path:
            resolved_json_path = _rewrite_output_path(resolved_json_path, run_output_dir)
        parser = parser.model_copy(update={"json_path": resolved_json_path})

    start = time.perf_counter()
    completed = subprocess.run(
        formatted_command,
        cwd=benchmark_cwd,
        env=merged_env,
        capture_output=True,
        text=True,
        check=False,
    )
    duration = time.perf_counter() - start

    if completed.returncode != 0:
        raise RuntimeError(
            f"Benchmark '{benchmark.name}' failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )

    score = _parse_score(
        parser,
        stdout=completed.stdout,
        stderr=completed.stderr,
        benchmark_cwd=benchmark_cwd,
    )
    return BenchmarkResult(
        name=benchmark.name,
        model_id=model_id,
        score=score,
        duration_seconds=duration,
        return_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def weighted_score(results: list[BenchmarkResult], weights: dict[str, float]) -> float:
    total_weight = 0.0
    weighted = 0.0
    for result in results:
        weight = weights.get(result.name, 1.0)
        weighted += result.score * weight
        total_weight += weight
    if total_weight == 0:
        raise ValueError("Cannot compute weighted score with total weight 0")
    return weighted / total_weight

