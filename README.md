# Pioneer Adaptive Finetuning Lab

Experiment harness for using `api.pioneer.ai` as an **adaptive finetuning system** for small coder models, with evaluation hooks for:

- **LiveCodeBench** style coding benchmarks
- **Aider** code editing/refactor benchmark style tasks

This repo is intentionally config-driven so you can iterate quickly on:

1. Which Pioneer base/candidate models to test
2. Which benchmark commands to run
3. Fine-tuning hyperparameters and stopping policy

---

## What this gives you

- A reusable Pioneer API client (`/v1/models`, `/v1/chat/completions`, `/felix/datasets/upload`, `/felix/training-jobs`)
- A benchmark runner that executes arbitrary commands and parses scores
- Weighted aggregation across benchmarks
- Adaptive loop:
  - establish inference baselines first (seed + candidates)
  - launch Felix fine-tune jobs when below target
  - track training status/loss per iteration
- JSON history + summary artifacts in `outputs/`

---

## Quick start

### 1) Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2) Configure API key

```bash
export PIONEER_API_KEY="your_key_here"
```

> Keep keys in environment variables; do not commit them into files.

### 3) Validate config

```bash
pioneer-adaptive validate-config configs/experiment.yaml
```

### 4) Run benchmark-only evaluation

```bash
pioneer-adaptive run-benchmarks "llama-3.1-8b" configs/experiment.yaml
```

### 5) Run adaptive finetuning cycle

```bash
pioneer-adaptive run-cycle configs/experiment.yaml
```

Results are written to:

- `outputs/history.json`
- `outputs/summary.json`

---

## Repository layout

```text
configs/experiment.yaml             # Main experiment config
data/train.jsonl                    # Example train file
data/valid.jsonl                    # Example validation file
scripts/setup_benchmarks.sh         # Clone external benchmark repos
scripts/run_livecodebench_mini.py   # LiveCodeBench mini baseline runner
scripts/run_aider_refactor_mini.py  # Aider refactor mini baseline runner
scripts/run_python_functions_benchmark.py # Deterministic 20-task function benchmark
scripts/run_*_stub.py               # Optional deterministic dry-run runners
src/pioneer_adaptive/
  config.py                         # Pydantic config schema
  pioneer_client.py                 # Pioneer API wrapper
  benchmarking.py                   # Command execution + score parsing
  adaptive_loop.py                  # Adaptive finetuning orchestration
  cli.py                            # Typer CLI
tests/
```

---

## Benchmarks used by default

Default config points to **real mini benchmark subsets**:

- `scripts/run_livecodebench_mini.py`: runs a subset of LiveCodeBench code-generation tasks via Pioneer inference and computes pass@1.
- `scripts/run_aider_refactor_mini.py`: runs a subset of Aider refactor tasks and checks refactor correctness via AST rules.

To prepare benchmark assets:

1. Clone benchmark repos:

   ```bash
   ./scripts/setup_benchmarks.sh
   ```

2. Ensure each benchmark emits a metric artifact (JSON file or JSON stdout), then point parser fields:

- `parser.mode`: `json_file`, `stdout_json`, or `regex`
- `parser.key_path`: dot path to metric field, e.g. `metrics.pass_at_1`

---

## CLI reference

```bash
pioneer-adaptive validate-config [CONFIG_PATH]
pioneer-adaptive list-models [CONFIG_PATH]
pioneer-adaptive run-benchmarks MODEL_ID [CONFIG_PATH]
pioneer-adaptive run-cycle [CONFIG_PATH]
```

## Additional benchmark for adaptive sweeps

For quicker and more stable coding experiments (less sensitive to huge prompt context),
use the included deterministic function benchmark:

```bash
python3 scripts/run_python_functions_benchmark.py \
  --model-id "base:Qwen/Qwen3-8B" \
  --out outputs/experiments/python_functions_base.json \
  --repeat 3 \
  --max-tokens 1800
```

You can run the same benchmark against a tuned decoder job ID:

```bash
python3 scripts/run_python_functions_benchmark.py \
  --model-id "<training-job-uuid>" \
  --out outputs/experiments/python_functions_tuned.json \
  --repeat 3 \
  --max-tokens 1800
```

An example config is provided at `configs/experiment_python_functions.yaml`.

---

## Notes

- Baseline inference uses:
  - `/v1/chat/completions` for standard inference model IDs
  - `/inference` for `base:*` decoder IDs
- Tuned decoder inference uses `/inference` with the **training job UUID** as `model_id`.
- Fine-tuning uses Felix endpoints (`/felix/datasets/upload`, `/felix/training-jobs`).
- If your Pioneer tenant uses custom endpoint conventions, adjust `src/pioneer_adaptive/pioneer_client.py`.
- Adaptive policy knobs live under `policy` in `configs/experiment.yaml`.

## Latest experiment learnings

- Mini LiveCodeBench/Aider runs are highly sensitive to long `<think>` traces and token budget.
- Session instability (`Too many active sessions`) can impact both inference and training jobs.
- On a deterministic executable benchmark (`python_functions_20`), we observed small but repeatable tuned improvements in some runs.
- Committed experiment artifacts are stored in `results/`.
