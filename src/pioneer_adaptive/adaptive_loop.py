from __future__ import annotations

import json
import random
import re
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
    agent_answer: str | None = None
    conversation_id: str | None = None
    tool_calls_made: int | None = None


class AdaptiveFinetuningLoop:
    UUID_PATTERN = re.compile(
        r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"
    )

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
        self._conversation_id: str | None = None

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

    @staticmethod
    def _extract_json_block(answer: str) -> dict[str, Any] | None:
        code_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", answer, flags=re.DOTALL)
        for block in code_blocks:
            try:
                parsed = json.loads(block)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return None

    def _extract_candidate_model_ids(self, answer: str) -> list[str]:
        candidates: list[str] = []

        parsed = self._extract_json_block(answer)
        if parsed:
            for key in ("recommended_model_id", "model_id", "training_job_id", "selected_model_id"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    candidates.append(value.strip())

        uuid_hits = re.findall(
            r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}",
            answer,
        )
        candidates.extend(uuid_hits)

        base_hits = re.findall(r"base:[A-Za-z0-9._/-]+", answer)
        candidates.extend(base_hits)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    def _query_adaptive_agent(
        self,
        *,
        base_model: str,
        baseline_scores: dict[str, float],
        target_score: float,
        iteration: int,
    ) -> tuple[str, int | None, str | None]:
        score_lines = "\n".join(f"- {model}: {score:.4f}" for model, score in baseline_scores.items())
        message = (
            "You are the adaptive fine-tuning system. Given benchmark scores, choose the next model to "
            "evaluate for coding performance improvement. Prefer a training job ID if you've created one. "
            "Return JSON only in a markdown code block with key `recommended_model_id`.\n\n"
            f"Iteration: {iteration}\n"
            f"Current base model: {base_model}\n"
            f"Target score: {target_score:.4f}\n"
            f"Observed model scores:\n{score_lines}\n"
        )
        payload = self.client.adaptive_finetuning_chat(
            message,
            conversation_id=self._conversation_id,
            filters={"model_id": base_model},
        )
        answer = str(payload.get("answer", ""))
        self._conversation_id = payload.get("conversation_id")
        tool_calls = payload.get("tool_calls_made")
        tool_count = int(tool_calls) if isinstance(tool_calls, int) else None
        return answer, tool_count, self._conversation_id

    def _normalize_candidate_model_id(self, model_id: str) -> str | None:
        if self.client.model_available_for_inference(model_id):
            return model_id
        probes = []
        if not model_id.startswith("base:"):
            probes.append(f"base:{model_id}")
        probes.append(model_id.lower())
        if not model_id.lower().startswith("base:"):
            probes.append(f"base:{model_id.lower()}")
        for probe in probes:
            if self.client.model_available_for_inference(probe):
                return probe
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
                answer, tool_calls, conversation_id = self._query_adaptive_agent(
                    base_model=base_model,
                    baseline_scores=scored_models,
                    target_score=self.config.policy.target_score,
                    iteration=iteration,
                )
                candidates = self._extract_candidate_model_ids(answer)
                tuned_model = next(
                    (
                        normalized
                        for model_id in candidates
                        for normalized in [self._normalize_candidate_model_id(model_id)]
                        if normalized
                    ),
                    base_model,
                )
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

                finetune_job_id = tuned_model if self.UUID_PATTERN.match(tuned_model) else None
                history.append(
                    IterationRecord(
                        iteration=iteration,
                        base_model=base_model,
                        candidate_model=tuned_model,
                        benchmark_results=benchmark_results,
                        aggregate_score=aggregate_score,
                        finetune_job_id=finetune_job_id,
                        promoted_model=promoted_model,
                        agent_answer=answer,
                        conversation_id=conversation_id,
                        tool_calls_made=tool_calls,
                    )
                )

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

