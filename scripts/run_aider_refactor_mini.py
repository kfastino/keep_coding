#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any

from pioneer_adaptive.pioneer_client import PioneerAPIError, PioneerClient


def _extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def _count_nodes(node: ast.AST) -> int:
    return sum(1 for _ in ast.walk(node))


def _parse_task_metadata(test_file: Path) -> tuple[str, str]:
    content = test_file.read_text(encoding="utf-8")
    method_match = re.search(r'method\s*=\s*"([^"]+)"', content)
    class_match = re.search(r'class_name\s*=\s*"([^"]+)"', content)
    if not method_match or not class_match:
        raise ValueError(f"Could not parse metadata from {test_file}")
    return class_match.group(1), method_match.group(1)


def _verify_refactor(original_code: str, updated_code: str, class_name: str, method_name: str) -> tuple[bool, str]:
    try:
        original_tree = ast.parse(original_code)
    except SyntaxError as exc:
        return False, f"original_parse_error:{exc.msg}"

    try:
        updated_tree = ast.parse(updated_code)
    except SyntaxError as exc:
        return False, f"updated_parse_error:{exc.msg}"

    original_class = next(
        (node for node in original_tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    updated_class = next(
        (node for node in updated_tree.body if isinstance(node, ast.ClassDef) and node.name == class_name),
        None,
    )
    if original_class is None or updated_class is None:
        return False, "class_not_found"

    original_method = next(
        (node for node in original_class.body if isinstance(node, ast.FunctionDef) and node.name == method_name),
        None,
    )
    if original_method is None:
        return False, "original_method_not_found"

    updated_method = next(
        (node for node in updated_class.body if isinstance(node, ast.FunctionDef) and node.name == method_name),
        None,
    )
    if updated_method is not None:
        return False, "method_still_inside_class"

    top_level_func = next(
        (node for node in updated_tree.body if isinstance(node, ast.FunctionDef) and node.name == method_name),
        None,
    )
    if top_level_func is None:
        return False, "top_level_function_missing"

    original_nodes = _count_nodes(original_method)
    moved_nodes = _count_nodes(top_level_func)
    if moved_nodes < max(8, int(original_nodes * 0.8)):
        return False, "function_too_small"

    return True, "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=2400)
    args = parser.parse_args()

    key = os.getenv("PIONEER_API_KEY")
    if not key:
        raise PioneerAPIError("PIONEER_API_KEY is not set")

    project_root = Path(__file__).resolve().parents[1]
    bench_root = project_root / "benchmarks" / "refactor-benchmark" / "refactor-benchmark"
    task_dirs = sorted([path for path in bench_root.iterdir() if path.is_dir()])[: args.limit]

    client = PioneerClient(base_url="https://api.pioneer.ai", api_key=key, timeout=180)
    details: list[dict[str, Any]] = []
    success = 0

    for task_dir in task_dirs:
        test_candidates = sorted(task_dir.glob("*_test.py"))
        source_candidates = sorted(
            [path for path in task_dir.glob("*.py") if not path.name.endswith("_test.py")]
        )
        if not test_candidates or not source_candidates:
            continue

        test_file = test_candidates[0]
        source_file = source_candidates[0]
        class_name, method_name = _parse_task_metadata(test_file)
        source_code = source_file.read_text(encoding="utf-8")

        prompt = (
            "You are performing an automated code refactor benchmark task.\n"
            f"Refactor method `{method_name}` from class `{class_name}` into a top-level function "
            f"named `{method_name}`.\n"
            "Update call sites in the file so behavior is preserved.\n"
            "Return ONLY the full updated Python file in a single ```python fenced block.\n\n"
            f"File path: {source_file.name}\n\n"
            f"```python\n{source_code}\n```"
        )
        messages = [
            {"role": "system", "content": "You are an expert Python refactoring assistant."},
            {"role": "user", "content": prompt},
        ]

        api_error: str | None = None
        try:
            response = client.chat_completion(
                args.model_id,
                messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            updated = _extract_code(response)
            ok, reason = _verify_refactor(source_code, updated, class_name, method_name)
        except Exception as exc:  # Keep benchmark run alive on per-task API errors.
            updated = ""
            ok = False
            reason = "api_error"
            api_error = f"{type(exc).__name__}: {exc}"
        if ok:
            success += 1

        details.append(
            {
                "task": task_dir.name,
                "class_name": class_name,
                "method_name": method_name,
                "success": ok,
                "reason": reason,
                "updated_chars": len(updated),
                "error": api_error,
            }
        )

    total = len(details)
    success_rate = (success / total) if total else 0.0
    payload = {
        "benchmark": "aider_refactor_mini",
        "model_id": args.model_id,
        "evaluated_tasks": total,
        "metrics": {"success_rate": success_rate},
        "details": details,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

