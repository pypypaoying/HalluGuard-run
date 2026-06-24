# Frozen Core Table Run Guide

This guide is the run contract for the HalluGuard core table. The goal is to
compare the frozen HalluGuard-SP method against same-position test-time,
adaptation, normalization, and smoothing baselines. It is not an autosearch
workflow.

## 1. Environment

Preferred:

```bash
bash scripts/setup_env.sh
conda activate halluguard-run
```

Manual conda route:

```bash
conda env create -f environment.yml
conda activate halluguard-run
```

Manual venv route:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA servers, install the matching PyTorch wheel according to the official
PyTorch instructions before or after installing `requirements.txt`. The repo
keeps a CPU-safe `torch>=2.1` default.

## 2. Download Datasets and Official Baseline Repos

Core datasets:

```bash
python scripts/fetch_core_datasets.py --datasets core
```

Core plus optional extended datasets:

```bash
python scripts/fetch_core_datasets.py --datasets extended
```

Official same-position baseline repositories:

```bash
bash scripts/fetch_plugin_repos.sh
```

The script fetches RevIN, Dish-TS, SAN, Non-stationary Transformer / NST, and
TAFAS under `external/plugin_baselines/` at the pinned commits listed in
`docs/core_table_manifest.yaml`.

## 3. Sanity Checks

```bash
python experiments/halluguard/run_mvp.py \
  --config experiments/halluguard/configs/halluguard_mvp.yaml \
  --quick

python external/halluguard_real_pipeline/export_predictions.py \
  --dataset ETTm1 \
  --model DLinear \
  --horizon 96 \
  --data-root external/ETDataset \
  --output experiments/halluguard/results/smoke_predictions/ETTm1_DLinear_96.jsonl \
  --epochs 1 \
  --max-train-windows 512 \
  --max-eval-windows 64 \
  --batch-size 128 \
  --device auto
```

Evaluate the smoke prediction file:

```bash
python experiments/halluguard/evaluate_predictions.py \
  --config experiments/halluguard/configs/halluguard_mvp.yaml \
  --input experiments/halluguard/results/smoke_predictions/ETTm1_DLinear_96.jsonl \
  --calibration-split val \
  --split test \
  --output-dir experiments/halluguard/results/smoke_eval/ETTm1_DLinear_96
```

## 4. Raw Backbone Prediction Export

This regenerates the original ETT clean prediction files for DLinear/PatchTST.

```bash
bash scripts/run_stage7_table.sh
```

Primary outputs:

```text
experiments/halluguard/results/stage7_big_table/predictions/*.jsonl
experiments/halluguard/results/stage7_big_table/combined_metrics.csv
experiments/halluguard/results/stage7_big_table/summary.md
```

## 5. Frozen HalluGuard Lines

Main frozen method:

```bash
python experiments/halluguard/run_stage14_autosearch.py \
  --scope clean_full \
  --config experiments/halluguard/configs/halluguard_core_table_sp_frozen.yaml
```

Stable-harm ablation:

```bash
python experiments/halluguard/run_stage14_autosearch.py \
  --scope clean_full \
  --config experiments/halluguard/configs/halluguard_core_table_stable_harm.yaml
```

The command name still says `run_stage14_autosearch.py` because it reuses the
existing evaluator, but these two configs are frozen and do not select new
candidates.

## 6. Official Baseline Prediction Export

Each official baseline should be run from its pinned source snapshot and export
the common schema:

```text
sample_id, dataset, model, split, context, prediction, target
```

Place exported files under:

```text
baseline_predictions/core_table/
```

Recommended labels:

```text
DLinear+raw_no_correction
DLinear+RevIN
DLinear+DishTS
DLinear+SAN
DLinear+NST
DLinear+TAFAS
PatchTST+raw_no_correction
PatchTST+RevIN
PatchTST+DishTS
PatchTST+SAN
PatchTST+NST
PatchTST+TAFAS
```

Then run:

```bash
python experiments/halluguard/run_stage12_external_batch.py \
  --config experiments/halluguard/configs/halluguard_stage12_external_batch.yaml \
  --input-dir baseline_predictions/core_table \
  --recursive \
  --continue-on-error \
  --output-dir experiments/halluguard/results/core_table/external_baselines
```

## 7. One-Command Local Harness

The helper below downloads data/repos, regenerates raw ETT predictions, runs the
two frozen HalluGuard configs, and evaluates external baseline predictions if
`baseline_predictions/core_table/` already exists:

```bash
bash scripts/run_core_table.sh
```

Environment overrides:

```bash
DATASET_SET=extended DEVICE=cuda EPOCHS=10 bash scripts/run_core_table.sh
```

## 8. Unified 12-Method Core Table

For the fair same-configuration table, use the unified core-12 runner:

```bash
bash scripts/run_core12_table.sh
```

This command uses one shared matrix:

```text
datasets: ETTm1, ETTh1
backbones: DLinear, PatchTST
horizons: 96, 192, 336, 720
methods: raw_no_correction, HalluGuard-SP frozen, stable-harm ablation,
         matched_sparse_smoothing, naive_smoothing, ema_smoothing,
         median_smoothing, RevIN, DishTS, SAN, NST, TAFAS
```

The official baseline entries are exported through
`scripts/run_core12_predictions.py` using a shared lightweight fair adapter
around the local DLinear/PatchTST exporter. This is intentionally stricter than
manually mixing different official scripts: every method shares the same data
split, context length, horizon, seed policy, training-window budget,
evaluation-window budget, and JSONL schema. It should be reported as
`adapter_mode=lightweight_fair_adapter`, not as a full reproduction of each
official repository's leaderboard setup.

Useful overrides:

```bash
DEVICE=cuda EPOCHS=10 MAX_TRAIN_WINDOWS=8192 MAX_EVAL_WINDOWS=1024 \
  bash scripts/run_core12_table.sh
```

Fast smoke:

```bash
DATASET_SET=ETTm1 MODELS=DLinear HORIZONS=96 METHODS=RevIN,DishTS \
EPOCHS=1 MAX_TRAIN_WINDOWS=128 MAX_EVAL_WINDOWS=16 \
  bash scripts/run_core12_table.sh
```

Final combined outputs:

```text
experiments/halluguard/results/core_table/core12_combined/core12_metrics.csv
experiments/halluguard/results/core_table/core12_combined/core12_summary.csv
experiments/halluguard/results/core_table/core12_combined/summary.md
```

## 9. Reversible Input-Layer HalluGuard-RDN

This optional follow-up tests the RevIN-like idea where the input and output
processing are symmetric, but the reversible transform is a HalluGuard-style
local dynamics baseline rather than ordinary mean/std normalization.

One-command run:

```bash
bash scripts/run_halluguard_rdn_table.sh
```

Fast smoke:

```bash
DATASET_SET=ETTm1 MODELS=DLinear HORIZONS=96 RDN_VARIANTS=level_slope_scale \
EPOCHS=1 MAX_TRAIN_WINDOWS=128 MAX_EVAL_WINDOWS=16 \
  bash scripts/run_halluguard_rdn_table.sh
```

Fuller ablation:

```bash
RDN_VARIANTS=level_only,level_scale,level_slope,level_slope_scale \
DEVICE=cuda EPOCHS=10 MAX_TRAIN_WINDOWS=8192 MAX_EVAL_WINDOWS=1024 \
  bash scripts/run_halluguard_rdn_table.sh
```

Outputs:

```text
baseline_predictions/halluguard_rdn/*.jsonl
baseline_predictions/halluguard_rdn_raw/*.jsonl
experiments/halluguard/results/halluguard_rdn/rdn_metrics.csv
experiments/halluguard/results/halluguard_rdn/rdn_summary.csv
experiments/halluguard/results/halluguard_rdn/summary.md
```

The method is comparable to RevIN/NST in placement because it wraps the
forecaster with reversible input/output processing. It should be reported as
`adapter_mode=reversible_dynamics_normalization`, not as a post-processing
smoothing baseline.

## 10. Required Reporting Fields

For each dataset/backbone/horizon/method row, keep:

- dataset
- backbone
- method
- horizon
- n validation samples
- n test samples
- MSE
- MAE
- delta vs raw backbone
- delta vs HalluGuard-SP frozen
- validation-only calibration flag
- output prediction path
- blocker reason, if any

Do not tune thresholds, routers, action choices, or baseline hyperparameters on
test targets.

## 11. Learnable Reversible Boundary Normalization Ablations

After `HalluGuard-RDN-level_only`, use this runner to test the four learnable
follow-up directions:

- robust anchor learning
- residual gate between raw and boundary-anchored forecasts
- horizon-wise learnable gate
- unified RevIN-RDN hybrid normalization

One-command run:

```bash
bash scripts/run_halluguard_lrbn_table.sh
```

Fast smoke:

```bash
DATASET_SET=ETTm1 MODELS=DLinear HORIZONS=96 \
LRBN_VARIANTS=fixed_level_only,learnable_robust_anchor \
EPOCHS=1 MAX_TRAIN_WINDOWS=128 MAX_EVAL_WINDOWS=16 \
  bash scripts/run_halluguard_lrbn_table.sh
```

Full server run:

```bash
DEVICE=cuda EPOCHS=10 MAX_TRAIN_WINDOWS=8192 MAX_EVAL_WINDOWS=1024 \
  bash scripts/run_halluguard_lrbn_table.sh
```

Suggested first focused run:

```bash
LRBN_VARIANTS=fixed_level_only,learnable_horizon_gate,unified_revin_rdn_hybrid \
DEVICE=cuda EPOCHS=10 MAX_TRAIN_WINDOWS=8192 MAX_EVAL_WINDOWS=1024 \
  bash scripts/run_halluguard_lrbn_table.sh
```

The default variant list is:

```text
fixed_level_only
learnable_robust_anchor
learnable_residual_gate
learnable_horizon_gate
unified_revin_rdn_hybrid
```

Outputs:

```text
baseline_predictions/halluguard_lrbn/*.jsonl
baseline_predictions/halluguard_lrbn_raw/*.jsonl
experiments/halluguard/results/halluguard_lrbn/lrbn_metrics.csv
experiments/halluguard/results/halluguard_lrbn/lrbn_summary.csv
experiments/halluguard/results/halluguard_lrbn/summary.md
```
