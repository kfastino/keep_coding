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

- A reusable Pioneer API client (`/v1/models`, `/v1/files`, `/v1/fine_tuning/jobs`)
- A benchmark runner that executes arbitrary commands and parses scores
- Weighted aggregation across benchmarks
- Adaptive loop:
  - evaluate a candidate model
  - launch finetune job when below target
  - re-evaluate tuned model
  - promote tuned model only if gain clears threshold
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
pioneer-adaptive run-benchmarks "pioneer/small-coder-base" configs/experiment.yaml
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
scripts/run_*_stub.py               # Local stub benchmark runners
src/pioneer_adaptive/
  config.py                         # Pydantic config schema
  pioneer_client.py                 # Pioneer API wrapper
  benchmarking.py                   # Command execution + score parsing
  adaptive_loop.py                  # Adaptive finetuning orchestration
  cli.py                            # Typer CLI
tests/
```

---

## Moving from stubs to real benchmarks

Default config points to stub runners so the system is runnable immediately.

To wire real benchmarks:

1. Clone benchmark repos:

   ```bash
   ./scripts/setup_benchmarks.sh
   ```

2. Update `configs/experiment.yaml` benchmark command blocks to real commands.

   Example LiveCodeBench invocation pattern:

   ```bash
   python -m lcb_runner.runner.main --model <MODEL> --scenario codegeneration --evaluate --release_version release_v2
   ```

3. Ensure each benchmark emits a metric artifact (JSON file or JSON stdout), then point parser fields:

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

---

## Notes

- The API workflow is implemented in an OpenAI-compatible shape (`/v1/files`, `/v1/fine_tuning/jobs`).
- If your Pioneer tenant uses custom endpoint conventions, adjust `src/pioneer_adaptive/pioneer_client.py`.
- Adaptive policy knobs live under `policy` in `configs/experiment.yaml`.
