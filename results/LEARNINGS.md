# Experiment learnings (Mar 2026)

This folder tracks committed benchmark outputs that are stable enough to compare across runs.

## Key files

- `python_functions_base_qwen3_latest.json`
- `python_functions_tuned_b3bee_latest.json`
- `python_functions_comparison_latest.json`

## Current takeaway

On `python_functions_20` with 3 repeats:

- Base `base:Qwen/Qwen3-8B` functional pass avg: `0.7667`
- Tuned `b3bee275-10f8-47ea-a6a8-34b62c95f8ac` functional pass avg: `0.8000`
- Delta: `+0.0333`

This is a modest gain; useful as a directional signal but not yet a breakthrough.

## Important caveats

1. **Reasoning spillover**
   - Decoder models often emit long `<think>` traces, causing truncation and parse failures on strict code benchmarks.
2. **Platform instability**
   - Recurrent API/training failures were observed with:
   - `Too many active sessions. Please ensure you are not creating extra ServiceClient objects...`
3. **Benchmark sensitivity**
   - Harder benchmarks (LiveCodeBench/Aider mini) remain noisy due to format + session issues.

## Recommended next steps

- Add two-pass decoding fallback in benchmark runners (first answer, then forced final-code only retry).
- Raise max token budgets selectively for tuned decoder UUIDs.
- Keep using repeat-based reporting (3+ repeats) to avoid over-reading single-run spikes.
