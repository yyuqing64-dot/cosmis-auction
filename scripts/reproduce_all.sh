#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON:-python}"
SEEDS="${SEEDS:-0-4}"
LOADS="0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6"
POLICIES="rr,greedy_latency,sqlf,cap_greedy,cosims_sinkhorn,cosims_auction_B"
CONFIG="configs/auction_B_recommended_sqlf_cloud.json"
TRACE_MANIFEST="data/trace_inputs/arrivals_h77_load_sweep_0.8_1.6_step0.1_manifest.csv"

mkdir -p results/fig5_1_synthetic_step01/main_results
mkdir -p results/fig5_3_trace_step01/main_results

if [[ "${RUN_EXPERIMENTS:-1}" == "1" ]]; then
  echo "[1/3] Running synthetic load sweep for Fig. 5.1"
  "$PYTHON_BIN" eval/runner.py \
    --config "$CONFIG" \
    --out results/fig5_1_synthetic_step01/main_results/summary_mean_std.csv \
    --loads "$LOADS" \
    --seeds "$SEEDS" \
    --policies "$POLICIES" \
    --sweep_topk 3

  echo "[2/3] Running trace replay load sweep for Fig. 5.3 and Fig. 5.4"
  "$PYTHON_BIN" eval/runner.py \
    --config "$CONFIG" \
    --out results/fig5_3_trace_step01/main_results/summary_mean_std.csv \
    --loads 1.0 \
    --seeds "$SEEDS" \
    --policies "$POLICIES" \
    --sweep_topk 3 \
    --trace_manifest "$TRACE_MANIFEST" \
    --trace_load_factors "$LOADS"
else
  echo "[1/3] RUN_EXPERIMENTS=0, copy bundled reference summaries."
  cp results/reference/summaries/fig5_1_synthetic_step01/summary_mean_std.csv     results/fig5_1_synthetic_step01/main_results/summary_mean_std.csv
  cp results/reference/summaries/fig5_3_trace_step01/summary_mean_std.csv     results/fig5_3_trace_step01/main_results/summary_mean_std.csv
fi

echo "[3/3] Generating figures"
"$PYTHON_BIN" scripts/plot_figures.py --results-dir results --config "$CONFIG" --seeds "$SEEDS"

echo "[OK] Reproduction finished. Figures are in results/figures"
