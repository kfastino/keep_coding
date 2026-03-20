#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmarks"

mkdir -p "${BENCH_DIR}"

if [ ! -d "${BENCH_DIR}/LiveCodeBench" ]; then
  git clone https://github.com/LiveCodeBench/LiveCodeBench.git "${BENCH_DIR}/LiveCodeBench"
fi

if [ ! -d "${BENCH_DIR}/refactor-benchmark" ]; then
  git clone https://github.com/Aider-AI/refactor-benchmark.git "${BENCH_DIR}/refactor-benchmark"
fi

echo "Benchmark repos ready under ${BENCH_DIR}"
echo "Install each benchmark's dependencies before running real evaluations."

