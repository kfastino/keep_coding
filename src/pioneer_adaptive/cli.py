from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich import print

from pioneer_adaptive.adaptive_loop import AdaptiveFinetuningLoop
from pioneer_adaptive.benchmarking import run_benchmark, weighted_score
from pioneer_adaptive.config import ExperimentConfig, load_config
from pioneer_adaptive.pioneer_client import PioneerClient

app = typer.Typer(help="Adaptive finetuning experiments for Pioneer models.")


def _load(path: str) -> tuple[ExperimentConfig, Path]:
    config_path = Path(path).resolve()
    config = load_config(config_path)
    return config, Path.cwd().resolve()


@app.command("validate-config")
def validate_config(path: str = typer.Argument("configs/experiment.yaml")) -> None:
    config, _ = _load(path)
    print(
        f"[green]Config OK[/green] seed={config.seed_model} benchmarks={len(config.benchmarks)}"
    )


@app.command("list-models")
def list_models(path: str = typer.Argument("configs/experiment.yaml")) -> None:
    config, _ = _load(path)
    key = os.getenv(config.api_key_env)
    if not key:
        raise typer.BadParameter(
            f"Environment variable {config.api_key_env} is not set."
        )
    client = PioneerClient(base_url=config.api_base_url, api_key=key)
    models = client.list_models()
    for model in models:
        print(f"- {model.get('id', model)}")


@app.command("run-benchmarks")
def run_benchmarks(
    model_id: str = typer.Argument(..., help="Model ID to evaluate"),
    path: str = typer.Argument("configs/experiment.yaml"),
) -> None:
    config, project_root = _load(path)
    enabled = [benchmark for benchmark in config.benchmarks if benchmark.enabled]
    results = [run_benchmark(benchmark, model_id=model_id, project_root=project_root) for benchmark in enabled]
    weights = {benchmark.name: benchmark.weight for benchmark in enabled}
    aggregate = weighted_score(results, weights)
    output = {
        "model_id": model_id,
        "aggregate_score": aggregate,
        "benchmarks": [result.to_dict() for result in results],
    }
    print(json.dumps(output, indent=2))


@app.command("run-cycle")
def run_cycle(path: str = typer.Argument("configs/experiment.yaml")) -> None:
    config, project_root = _load(path)
    loop = AdaptiveFinetuningLoop(config=config, project_root=project_root)
    history = loop.run()
    print(
        f"[green]Adaptive cycle complete[/green]: {len(history)} iteration(s), output in {config.output_dir}"
    )


if __name__ == "__main__":
    app()

