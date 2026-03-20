#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _deterministic_score(model_id: str) -> float:
    digest = hashlib.md5(model_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    raw = int(digest[:8], 16)
    # 0.25 -> 0.90 range, useful for dry-runs.
    return round(0.25 + (raw % 6500) / 10000, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    score = _deterministic_score(args.model_id)
    payload = {
        "benchmark": "aider_editing",
        "model_id": args.model_id,
        "metrics": {"success_rate": score},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

