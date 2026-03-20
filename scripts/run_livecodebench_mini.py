#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from pioneer_adaptive.pioneer_client import PioneerAPIError, PioneerClient


def _extract_code(text: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()
    return text.strip()


def _load_lcb_modules(lcb_root: Path) -> tuple[object, object, object, object]:
    original_cwd = Path.cwd()
    os.chdir(lcb_root)
    try:
        sys.path.insert(0, str(lcb_root))
        from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
        from lcb_runner.evaluation import codegen_metrics
        from lcb_runner.lm_styles import LMStyle
        from lcb_runner.prompts.code_generation import format_prompt_generation
    finally:
        os.chdir(original_cwd)
    return load_code_generation_dataset, codegen_metrics, LMStyle, format_prompt_generation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--release-version", default="release_v1")
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-process-evaluate", type=int, default=4)
    parser.add_argument("--test-timeout", type=int, default=6)
    args = parser.parse_args()

    key = os.getenv("PIONEER_API_KEY")
    if not key:
        raise PioneerAPIError("PIONEER_API_KEY is not set")

    project_root = Path(__file__).resolve().parents[1]
    lcb_root = project_root / "benchmarks" / "LiveCodeBench"
    (
        load_code_generation_dataset,
        codegen_metrics,
        LMStyle,
        format_prompt_generation,
    ) = _load_lcb_modules(lcb_root)

    dataset = load_code_generation_dataset(args.release_version)
    dataset = sorted(dataset, key=lambda item: item.question_id)[: args.limit]

    client = PioneerClient(base_url="https://api.pioneer.ai", api_key=key, timeout=180)

    generated_codes: list[list[str]] = []
    per_problem: list[dict[str, object]] = []
    for problem in dataset:
        prompt = format_prompt_generation(problem, LMStyle.OpenAIChat)
        if not isinstance(prompt, list):
            raise PioneerAPIError("Expected chat message prompt format for OpenAIChat")
        error: str | None = None
        try:
            answer = client.chat_completion(
                args.model_id,
                prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            code = _extract_code(answer)
        except Exception as exc:  # Keep benchmark run alive on per-problem API errors.
            code = ""
            error = f"{type(exc).__name__}: {exc}"
        generated_codes.append([code])
        per_problem.append(
            {
                "question_id": problem.question_id,
                "question_title": problem.question_title,
                "code_chars": len(code),
                "error": error,
            }
        )

    eval_samples = [problem.get_evaluation_sample() for problem in dataset]
    metrics = codegen_metrics(
        eval_samples,
        generated_codes,
        k_list=[1],
        num_process_evaluate=args.num_process_evaluate,
        timeout=args.test_timeout,
    )
    pass_at_1 = float(metrics[0]["pass@1"])

    payload = {
        "benchmark": "livecodebench_mini",
        "model_id": args.model_id,
        "evaluated_questions": len(dataset),
        "metrics": {"pass_at_1": pass_at_1},
        "problems": per_problem,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

