#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pioneer_adaptive.pioneer_client import PioneerAPIError, PioneerClient


@dataclass(frozen=True)
class FunctionTask:
    name: str
    signature: str
    tests: list[tuple[tuple[Any, ...], Any]]


TASKS: list[FunctionTask] = [
    FunctionTask("double", "def double(x: int) -> int: ...", [((4,), 8)]),
    FunctionTask("triple", "def triple(x: int) -> int: ...", [((3,), 9)]),
    FunctionTask("square", "def square(x: int) -> int: ...", [((5,), 25)]),
    FunctionTask("cube", "def cube(x: int) -> int: ...", [((3,), 27)]),
    FunctionTask("is_even", "def is_even(x: int) -> bool: ...", [((4,), True)]),
    FunctionTask("is_odd", "def is_odd(x: int) -> bool: ...", [((5,), True)]),
    FunctionTask(
        "reverse_string",
        "def reverse_string(s: str) -> str: ...",
        [(("abc",), "cba")],
    ),
    FunctionTask(
        "is_palindrome",
        "def is_palindrome(s: str) -> bool: ...",
        [(("abba",), True)],
    ),
    FunctionTask("sum_list", "def sum_list(nums: list[int]) -> int: ...", [(([1, 2, 3],), 6)]),
    FunctionTask(
        "max_in_list",
        "def max_in_list(nums: list[int]) -> int: ...",
        [(([1, 5, 3],), 5)],
    ),
    FunctionTask(
        "min_in_list",
        "def min_in_list(nums: list[int]) -> int: ...",
        [(([1, 5, 3],), 1)],
    ),
    FunctionTask("count_vowels", "def count_vowels(s: str) -> int: ...", [(("hello",), 2)]),
    FunctionTask("factorial", "def factorial(n: int) -> int: ...", [((5,), 120)]),
    FunctionTask("fib", "def fib(n: int) -> int: ...", [((8,), 21)]),
    FunctionTask("gcd", "def gcd(a: int, b: int) -> int: ...", [((12, 18), 6)]),
    FunctionTask("lcm", "def lcm(a: int, b: int) -> int: ...", [((4, 6), 12)]),
    FunctionTask("clamp", "def clamp(x: int, lo: int, hi: int) -> int: ...", [((9, 0, 5), 5)]),
    FunctionTask("abs_diff", "def abs_diff(a: int, b: int) -> int: ...", [((3, 8), 5)]),
    FunctionTask(
        "starts_with_vowel",
        "def starts_with_vowel(s: str) -> bool: ...",
        [(("apple",), True)],
    ),
    FunctionTask(
        "unique_sorted",
        "def unique_sorted(nums: list[int]) -> list[int]: ...",
        [(([3, 1, 3, 2],), [1, 2, 3])],
    ),
]


def _extract_code(response_text: str) -> tuple[str, bool]:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", response_text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip(), True
    matches = list(re.finditer(r"(?m)^def\s+[a-zA-Z_][a-zA-Z0-9_]*\s*\(", response_text))
    if matches:
        return response_text[matches[-1].start() :].strip(), False
    return response_text.strip(), False


def _run_task(
    client: PioneerClient,
    model_id: str,
    task: FunctionTask,
    *,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    prompt = f"Write only Python code. Implement {task.signature}"
    error: str | None = None
    fenced = False
    code = ""
    try:
        response = client.chat_completion(
            model_id,
            [{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        code, fenced = _extract_code(response)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    runnable = False
    passed = False
    failure = None
    if not error and code:
        namespace: dict[str, Any] = {}
        try:
            exec(code, namespace, namespace)
            fn = namespace.get(task.name)
            if callable(fn):
                runnable = True
                passed = True
                for args, expected in task.tests:
                    got = fn(*args)
                    if got != expected:
                        passed = False
                        failure = f"expected={expected!r}, got={got!r}"
                        break
            else:
                failure = "function_missing"
        except Exception as exc:
            failure = f"exec_error:{type(exc).__name__}"

    return {
        "task": task.name,
        "fenced": fenced,
        "runnable": runnable,
        "passed": passed,
        "error": error,
        "failure": failure,
        "code_chars": len(code),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    key = os.getenv("PIONEER_API_KEY")
    if not key:
        raise PioneerAPIError("PIONEER_API_KEY is not set")
    base_url = os.getenv("PIONEER_API_BASE_URL", "https://api.pioneer.ai").rstrip("/")

    client = PioneerClient(base_url=base_url, api_key=key, timeout=180)

    repeats: list[dict[str, Any]] = []
    for rep in range(1, args.repeat + 1):
        details = [
            _run_task(
                client,
                args.model_id,
                task,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            for task in TASKS
        ]
        functional = sum(1 for row in details if row["passed"])
        runnable = sum(1 for row in details if row["runnable"])
        fenced = sum(1 for row in details if row["fenced"])
        repeats.append(
            {
                "repeat": rep,
                "functional_pass": functional,
                "runnable_count": runnable,
                "fenced_count": fenced,
                "details": details,
            }
        )

    total_tasks = len(TASKS)
    functional_scores = [row["functional_pass"] / total_tasks for row in repeats]
    runnable_scores = [row["runnable_count"] / total_tasks for row in repeats]
    fenced_scores = [row["fenced_count"] / total_tasks for row in repeats]

    payload = {
        "benchmark": "python_functions_20",
        "model_id": args.model_id,
        "task_count": total_tasks,
        "repeat": args.repeat,
        "metrics": {
            "functional_pass_rate_avg": sum(functional_scores) / len(functional_scores),
            "functional_pass_rate_min": min(functional_scores),
            "functional_pass_rate_max": max(functional_scores),
            "runnable_rate_avg": sum(runnable_scores) / len(runnable_scores),
            "fenced_rate_avg": sum(fenced_scores) / len(fenced_scores),
        },
        "runs": repeats,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

