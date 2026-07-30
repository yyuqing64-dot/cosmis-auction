#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON:-python}"
mkdir -p results/smoke
"$PYTHON_BIN" eval/runner.py \
  --config configs/auction_B_recommended_sqlf_cloud.json \
  --out results/smoke/summary_mean_std.csv \
  --loads 0.8 \
  --seeds 0-1 \
  --policies rr,cosims_auction_B \
  --sweep_topk 3
"$PYTHON_BIN" - <<'PY2'
from pathlib import Path
import pandas as pd
p = Path('results/smoke/summary_mean_std.csv')
df = pd.read_csv(p)
print('[OK] smoke rows:', len(df))
print(df[['load','policy','avg_latency_mean','p95_latency_mean']].to_string(index=False))
PY2
