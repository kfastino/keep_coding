#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _deterministic_score(model_id: str) -> float:
    digest = hashlib.sha256(model_id.encode("utf-8")).hexdigest()
    raw = int(digest[:8], 16)
    # 0.30 -> 0.85 range, useful for dry-runs.
    return round(0.30 + (raw % 5500) / 10000, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    score = _deterministic_score(args.model_id)
    payload = {
        "benchmark": "livecodebench",
        "model_id": args.model_id,
        "metrics": {"pass_at_1": score},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()

