from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pioneer_adaptive.benchmarking import BenchmarkResult, run_benchmark, weighted_score
from pioneer_adaptive.config import ExperimentConfig
from pioneer_adaptive.pioneer_client import PioneerAPIError, PioneerClient


@dataclass
class IterationRecord:
    iteration: int
    base_model: str
    candidate_model: str
    benchmark_results: list[dict[str, Any]]
    aggregate_score: float
    finetune_job_id: str | None
    promoted_model: str | None


class AdaptiveFinetuningLoop:
    def __init__(self, config: ExperimentConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.output_dir = project_root / config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.output_dir / "history.json"

        api_key = self._load_api_key(config.api_key_env)
        self.client = PioneerClient(base_url=config.api_base_url, api_key=api_key)
        self.random = random.Random(config.policy.random_seed)

    @staticmethod
    def _load_api_key(env_var: str) -> str:
        import os

        key = os.getenv(env_var)
        if not key:
            raise PioneerAPIError(
                f"Environment variable {env_var} is not set. Export your Pioneer API key."
            )
        return key

    def _evaluate(self, model_id: str) -> tuple[list[BenchmarkResult], float]:
        enabled = [benchmark for benchmark in self.config.benchmarks if benchmark.enabled]
        results = [
            run_benchmark(benchmark, model_id=model_id, project_root=self.project_root)
            for benchmark in enabled
        ]
        weights = {benchmark.name: benchmark.weight for benchmark in enabled}
        score = weighted_score(results, weights=weights)
        return results, score

    def _existing_history(self) -> list[dict[str, Any]]:
        if not self.history_file.exists():
            return []
        return json.loads(self.history_file.read_text(encoding="utf-8"))

    def _save_history(self, history: list[IterationRecord]) -> None:
        payload = [asdict(item) for item in history]
        self.history_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _pick_base_model(self, scored_models: dict[str, float]) -> str:
        models = list(scored_models.keys())
        if not models:
            return self.config.seed_model
        if self.random.random() < self.config.policy.exploration_rate:
            return self.random.choice(models)
        return max(models, key=lambda model: scored_models[model])

    def _ensure_finetune_file_ids(self) -> tuple[str, str | None]:
        training_file_id = self.config.finetune.training_file_id
        validation_file_id = self.config.finetune.validation_file_id

        if not training_file_id:
            path = self.config.finetune.training_file_path
            if not path:
                raise PioneerAPIError("No training file path configured.")
            training_file_id = self.client.upload_file(self.project_root / path)
            self.config.finetune.training_file_id = training_file_id

        if not validation_file_id and self.config.finetune.validation_file_path:
            validation_file_id = self.client.upload_file(
                self.project_root / self.config.finetune.validation_file_path
            )
            self.config.finetune.validation_file_id = validation_file_id

        return training_file_id, validation_file_id

    def run(self) -> list[IterationRecord]:
        scored_models: dict[str, float] = {
            self.config.seed_model: -1.0,
            **{model_id: -1.0 for model_id in self.config.candidate_models},
        }

        history: list[IterationRecord] = []
        best_score = -1.0
        best_model = self.config.seed_model

        for iteration in range(1, self.config.policy.max_iterations + 1):
            base_model = self._pick_base_model(scored_models)
            benchmark_results, aggregate = self._evaluate(base_model)
            scored_models[base_model] = max(scored_models.get(base_model, -1.0), aggregate)

            if aggregate > best_score:
                best_score = aggregate
                best_model = base_model

            if aggregate >= self.config.policy.target_score:
                history.append(
                    IterationRecord(
                        iteration=iteration,
                        base_model=base_model,
                        candidate_model=base_model,
                        benchmark_results=[item.to_dict() for item in benchmark_results],
                        aggregate_score=aggregate,
                        finetune_job_id=None,
                        promoted_model=base_model,
                    )
                )
                break

            training_file_id, validation_file_id = self._ensure_finetune_file_ids()
            finetune_job = self.client.create_finetune_job(
                base_model=base_model,
                training_file_id=training_file_id,
                validation_file_id=validation_file_id,
                suffix=self.config.finetune.suffix,
                hyperparameters=self.config.finetune.hyperparameters,
            )
            job_id = str(finetune_job.get("id"))
            completed = self.client.wait_for_finetune_job(job_id)
            tuned_model = completed.get("fine_tuned_model")

            if not tuned_model:
                history.append(
                    IterationRecord(
                        iteration=iteration,
                        base_model=base_model,
                        candidate_model=base_model,
                        benchmark_results=[item.to_dict() for item in benchmark_results],
                        aggregate_score=aggregate,
                        finetune_job_id=job_id,
                        promoted_model=None,
                    )
                )
                break

            tuned_results, tuned_score = self._evaluate(str(tuned_model))
            scored_models[str(tuned_model)] = tuned_score
            gain = tuned_score - aggregate

            promoted_model = str(tuned_model) if gain >= self.config.policy.minimum_gain else None
            if promoted_model and tuned_score > best_score:
                best_score = tuned_score
                best_model = promoted_model

            history.append(
                IterationRecord(
                    iteration=iteration,
                    base_model=base_model,
                    candidate_model=str(tuned_model),
                    benchmark_results=[item.to_dict() for item in tuned_results],
                    aggregate_score=tuned_score,
                    finetune_job_id=job_id,
                    promoted_model=promoted_model,
                )
            )

        self._save_history(history)
        summary = self.output_dir / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "best_model": best_model,
                    "best_score": best_score,
                    "iterations_run": len(history),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return history

