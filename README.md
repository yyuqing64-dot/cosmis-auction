# COSIMS-Auction Reproducible Experiments

This repository contains a cleaned, self-contained copy of the core simulation code and final experiment scripts for **COSIMS-Auction**. It is intended for artifact sharing and GitHub upload. The original working directory contained many intermediate trials; this folder keeps only the files needed to reproduce the main paper figures.

## Contents

```text
configs/                         Final experiment configuration
sim/                             Simulation environment, task and network model
policies/                        Baselines and COSIMS variants
eval/                            Experiment runner and metrics
scripts/reproduce_all.sh          One-command full reproduction
scripts/plot_figures.py           Paper figure generation
scripts/smoke_test.sh             Quick sanity check
data/trace_inputs/                Trace replay inputs for load factors 0.8--1.6
results/reference/                Bundled reference summaries and final figures
```

## Environment

Python 3.10+ is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

The code only requires `numpy`, `pandas`, and `matplotlib`.

## Quick Check

Run a small two-policy synthetic smoke test:

```bash
bash scripts/smoke_test.sh
```

This should create `results/smoke/summary_mean_std.csv`.

## Full Reproduction

Run all main experiments and regenerate the paper figures:

```bash
bash scripts/reproduce_all.sh
```

By default, this uses:

- load factors: `0.8, 0.9, ..., 1.6`
- seeds: `0-4`
- policies: `RR`, `Greedy-Latency`, `SQLF`, `Cap-Greedy`, `COSIMS-Sinkhorn`, `COSIMS-Auction`
- COSIMS top-K: `3`

The full trace replay is relatively slow because each trace-load file contains 233,271 task arrivals.

Generated figures will be written to:

```text
results/figures/
```

## Regenerate Figures From Bundled Reference Summaries

If you only want to regenerate plots without rerunning all simulations, use:

```bash
RUN_EXPERIMENTS=0 bash scripts/reproduce_all.sh
```

This copies bundled summaries from `results/reference/summaries/` and regenerates figures from them. Fig. 5.2 uses a task-level sample cache if available; otherwise it reruns the small load-1.2 task-level collection.

## Main Figures

The script generates:

- `fig5_1_synthetic_latency`: synthetic load average/P95 latency
- `fig5_2_task_delay_distribution`: task-level completion-delay distribution at load factor 1.2
- `fig5_3_trace_latency`: trace replay average/P95 latency
- `fig5_4_trace_tradeoff`: COSIMS-Auction relative to Cap-Greedy under trace replay

Reference copies are provided under:

```text
results/reference/fig/
```

## Trace Data

The trace replay inputs are stored under `data/trace_inputs/`. The manifest uses repository-relative paths, so the experiment can run after cloning without depending on the original local path.

## Notes

- This cleaned folder is copied from the original working directory; no original files were moved.
- Historical trial outputs, draft figures, and unused plotting scripts are intentionally excluded.
- `COSIMS-Auction` in figures corresponds to the final trigger-aware online correction method (`cosims_auction_B` in code).
- `COSIMS-Sinkhorn` corresponds to the OT-only variant.
