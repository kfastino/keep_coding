from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ScoreParserConfig(BaseModel):
    mode: Literal["json_file", "stdout_json", "regex"] = "json_file"
    key_path: str | None = None
    json_path: str | None = None
    pattern: str | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "ScoreParserConfig":
        if self.mode == "json_file":
            if not self.json_path:
                raise ValueError("parser.json_path is required for json_file mode")
            if not self.key_path:
                raise ValueError("parser.key_path is required for json_file mode")
        elif self.mode == "stdout_json":
            if not self.key_path:
                raise ValueError("parser.key_path is required for stdout_json mode")
        elif self.mode == "regex" and not self.pattern:
            raise ValueError("parser.pattern is required for regex mode")
        return self


class BenchmarkConfig(BaseModel):
    name: str
    enabled: bool = True
    weight: float = 1.0
    cwd: str = "."
    command: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    parser: ScoreParserConfig

    @model_validator(mode="after")
    def validate_command(self) -> "BenchmarkConfig":
        if not self.command:
            raise ValueError(f"benchmark '{self.name}' needs a non-empty command list")
        return self


class FinetuneConfig(BaseModel):
    training_file_path: str | None = None
    training_file_id: str | None = None
    validation_file_path: str | None = None
    validation_file_id: str | None = None
    suffix: str = "adaptive"
    hyperparameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_training_source(self) -> "FinetuneConfig":
        if not self.training_file_path and not self.training_file_id:
            raise ValueError(
                "Provide finetune.training_file_path or finetune.training_file_id"
            )
        return self


class AdaptivePolicyConfig(BaseModel):
    max_iterations: int = 5
    target_score: float = 0.5
    exploration_rate: float = 0.2
    minimum_gain: float = 0.01
    random_seed: int = 7


class ExperimentConfig(BaseModel):
    api_base_url: str = "https://api.pioneer.ai/v1"
    api_key_env: str = "PIONEER_API_KEY"
    seed_model: str
    candidate_models: list[str] = Field(default_factory=list)
    output_dir: str = "outputs"
    benchmarks: list[BenchmarkConfig]
    finetune: FinetuneConfig
    policy: AdaptivePolicyConfig = Field(default_factory=AdaptivePolicyConfig)

    @model_validator(mode="after")
    def validate_benchmarks(self) -> "ExperimentConfig":
        enabled = [b for b in self.benchmarks if b.enabled]
        if not enabled:
            raise ValueError("At least one benchmark must be enabled")
        return self


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(raw)

