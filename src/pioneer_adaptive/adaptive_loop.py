from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
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
    adaptive_response: dict[str, Any] | None = None


class AdaptiveFinetuningLoop:
    UUID_REGEX = (
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
        r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}"
    )
    UUID_PATTERN = re.compile(f"^{UUID_REGEX}$")

    def __init__(self, config: ExperimentConfig, project_root: Path) -> None:
        self.config = config
        self.project_root = project_root
        self.run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.output_dir = project_root / config.output_dir / self.run_id
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
            run_benchmark(
                benchmark,
                model_id=model_id,
                project_root=self.project_root,
                template_vars={
                    "run_id": self.run_id,
                    "run_output_dir": str(self.output_dir.relative_to(self.project_root)),
                },
            )
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
    def _normalize_string(value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        return stripped or None

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

    @classmethod
    def _first_uuid(cls, text: str) -> str | None:
        match = re.search(cls.UUID_REGEX, text)
        return match.group(0) if match else None

    @classmethod
    def _resolve_agent_decision(
        cls,
        *,
        payload: dict[str, Any],
        base_model: str,
    ) -> tuple[str, str | None]:
        answer = str(payload.get("answer", ""))
        parsed_answer = cls._extract_json_block(answer) or {}
        sources: list[dict[str, Any]] = [payload, parsed_answer]

        recommended_model_id: str | None = None
        for source in sources:
            for key in ("recommended_model_id", "model_id", "selected_model_id"):
                value = cls._normalize_string(source.get(key))
                if value:
                    recommended_model_id = value
                    break
            if recommended_model_id:
                break

        training_job_id: str | None = None
        for source in sources:
            for key in ("training_job_id", "finetune_job_id", "job_id"):
                value = cls._normalize_string(source.get(key))
                if value:
                    training_job_id = value
                    break
            if training_job_id:
                break

        if training_job_id and not cls.UUID_PATTERN.match(training_job_id):
            training_job_id = cls._first_uuid(training_job_id)

        if not training_job_id:
            training_job_id = cls._first_uuid(answer)

        if (
            not training_job_id
            and recommended_model_id
            and cls.UUID_PATTERN.match(recommended_model_id)
        ):
            training_job_id = recommended_model_id

        if not recommended_model_id:
            base_match = re.search(r"base:[A-Za-z0-9._/-]+", answer)
            if base_match:
                recommended_model_id = base_match.group(0)

        resolved_model = recommended_model_id or training_job_id or base_model
        return resolved_model, training_job_id

    def _query_adaptive_agent(
        self,
        *,
        base_model: str,
        baseline_scores: dict[str, float],
        target_score: float,
        iteration: int,
    ) -> dict[str, Any]:
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
        self._conversation_id = (
            str(payload.get("conversation_id"))
            if payload.get("conversation_id") is not None
            else self._conversation_id
        )
        return payload

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
                adaptive_payload = self._query_adaptive_agent(
                    base_model=base_model,
                    baseline_scores=scored_models,
                    target_score=self.config.policy.target_score,
                    iteration=iteration,
                )
                answer = str(adaptive_payload.get("answer", ""))
                tool_calls = adaptive_payload.get("tool_calls_made")
                tool_count = int(tool_calls) if isinstance(tool_calls, int) else None
                conversation_id = (
                    str(adaptive_payload.get("conversation_id"))
                    if adaptive_payload.get("conversation_id") is not None
                    else self._conversation_id
                )
                recommended_model_id, finetune_job_id = self._resolve_agent_decision(
                    payload=adaptive_payload,
                    base_model=base_model,
                )
                decision_target = finetune_job_id or recommended_model_id
                tuned_model = self._normalize_candidate_model_id(decision_target)
                if tuned_model is None and decision_target != recommended_model_id:
                    tuned_model = self._normalize_candidate_model_id(recommended_model_id)
                if tuned_model is None:
                    tuned_model = base_model
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
                        finetune_job_id=finetune_job_id,
                        promoted_model=promoted_model,
                        agent_answer=answer,
                        conversation_id=conversation_id,
                        tool_calls_made=tool_count,
                        adaptive_response=adaptive_payload,
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
                    "run_id": self.run_id,
                    "output_dir": str(self.output_dir.relative_to(self.project_root)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return history

