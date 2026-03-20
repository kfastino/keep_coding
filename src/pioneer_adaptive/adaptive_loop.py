from __future__ import annotations

import json
import random
import time
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
    training_status: str | None = None
    training_loss: float | None = None


class AdaptiveFinetuningLoop:
    def __init__(self, config: ExperimentConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.output_dir = project_root / config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.output_dir / "history.json"
        self.baselines_file = self.output_dir / "baselines.json"

        api_key = self._load_api_key(config.api_key_env)
        self.client = PioneerClient(base_url=config.api_base_url, api_key=api_key)
        self.random = random.Random(config.policy.random_seed)
        self._training_dataset_ref: dict[str, str] | None = None

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

    def _save_baselines(self, model_scores: dict[str, Any]) -> None:
        self.baselines_file.write_text(
            json.dumps(model_scores, indent=2),
            encoding="utf-8",
        )

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

    def _wait_for_dataset_ready(self, dataset_id: str, max_wait_seconds: int = 180) -> dict[str, Any]:
        waited = 0
        while waited <= max_wait_seconds:
            datasets = self.client.list_datasets()
            matches = [item for item in datasets if str(item.get("id")) == dataset_id]
            if matches:
                dataset = matches[0]
                status = str(dataset.get("status", "")).lower()
                if status == "ready":
                    return dataset
                if status == "failed":
                    raise PioneerAPIError(
                        f"Dataset {dataset_id} failed processing: {dataset.get('processing_error')}"
                    )
            time.sleep(2)
            waited += 2
        raise PioneerAPIError(f"Dataset {dataset_id} was not ready within {max_wait_seconds}s")

    def _ensure_training_dataset_ref(self) -> dict[str, str]:
        if self._training_dataset_ref:
            return self._training_dataset_ref

        source_path = self.config.finetune.training_file_path
        if not source_path:
            raise PioneerAPIError("No training file path configured.")

        dataset_name = self.config.finetune.dataset_name or (
            f"{self.config.finetune.model_name_prefix}-dataset-{int(time.time())}"
        )
        uploaded = self.client.upload_dataset(
            self.project_root / source_path,
            dataset_name=dataset_name,
            dataset_type=self.config.finetune.dataset_type,
        )
        dataset_id = str(uploaded.get("id", ""))
        if not dataset_id:
            raise PioneerAPIError(f"Dataset upload did not return ID: {uploaded}")

        ready = self._wait_for_dataset_ready(dataset_id)
        version = str(ready.get("version_number") or "1")
        self._training_dataset_ref = {"name": str(ready["dataset_name"]), "version": version}
        return self._training_dataset_ref

    def _extract_training_loss(self, job_id: str) -> float | None:
        checkpoints = self.client.get_finetune_checkpoints(job_id)
        if not checkpoints:
            return None
        final = checkpoints[-1]
        value = final.get("training_loss")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _run_baselines(self, models: list[str]) -> tuple[dict[str, float], str, float]:
        scored_models: dict[str, float] = {}
        details: dict[str, Any] = {}
        best_model = models[0]
        best_score = -1.0
        for model_id in models:
            results, aggregate = self._evaluate(model_id)
            scored_models[model_id] = aggregate
            details[model_id] = {
                "aggregate_score": aggregate,
                "benchmarks": [item.to_dict() for item in results],
            }
            if aggregate > best_score:
                best_score = aggregate
                best_model = model_id
        self._save_baselines(details)
        return scored_models, best_model, best_score

    def run(self) -> list[IterationRecord]:
        baseline_models = [self.config.seed_model, *self.config.candidate_models]
        scored_models, best_model, best_score = self._run_baselines(baseline_models)
        history: list[IterationRecord] = []

        if best_score >= self.config.policy.target_score:
            history.append(
                IterationRecord(
                    iteration=0,
                    base_model=best_model,
                    candidate_model=best_model,
                    benchmark_results=[],
                    aggregate_score=best_score,
                    finetune_job_id=None,
                    promoted_model=best_model,
                )
            )
        else:
            for iteration in range(1, self.config.policy.max_iterations + 1):
                base_model = self._pick_base_model(scored_models)
                pre_score = scored_models[base_model]
                dataset_ref = self._ensure_training_dataset_ref()
                model_name = f"{self.config.finetune.model_name_prefix}-{int(time.time())}"
                finetune_job = self.client.create_finetune_job(
                    model_name=model_name,
                    datasets=[dataset_ref],
                    base_model=self.config.finetune.base_model,
                    training_type=self.config.finetune.training_type,
                    hyperparameters={**self.config.finetune.hyperparameters},
                )
                job_id = str(finetune_job.get("id"))
                completed = self.client.wait_for_finetune_job(job_id)
                status = str(completed.get("status", "")).lower()
                training_loss = self._extract_training_loss(job_id)
                tuned_model = job_id

                promoted_model = None
                aggregate_score = pre_score
                benchmark_results: list[dict[str, Any]] = []

                if self.client.model_available_for_inference(tuned_model):
                    tuned_results, tuned_score = self._evaluate(tuned_model)
                    benchmark_results = [item.to_dict() for item in tuned_results]
                    aggregate_score = tuned_score
                    gain = tuned_score - pre_score
                    if gain >= self.config.policy.minimum_gain:
                        promoted_model = tuned_model
                        scored_models[tuned_model] = tuned_score
                        if tuned_score > best_score:
                            best_score = tuned_score
                            best_model = tuned_model

                history.append(
                    IterationRecord(
                        iteration=iteration,
                        base_model=base_model,
                        candidate_model=tuned_model,
                        benchmark_results=benchmark_results,
                        aggregate_score=aggregate_score,
                        finetune_job_id=job_id,
                        promoted_model=promoted_model,
                        training_status=status,
                        training_loss=training_loss,
                    )
                )

                if status in {"failed", "cancelled", "error"}:
                    break
                if promoted_model and aggregate_score >= self.config.policy.target_score:
                    break

        self._save_history(history)
        summary = self.output_dir / "summary.json"
        summary.write_text(
            json.dumps(
                {
                    "best_model": best_model,
                    "best_score": best_score,
                    "baseline_models": baseline_models,
                    "iterations_run": len(history),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return history

