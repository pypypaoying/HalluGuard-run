#!/usr/bin/env bash
set -euo pipefail

DATASET_SET="${DATASET_SET:-ETTm1,ETTh1}"
MODELS="${MODELS:-DLinear,PatchTST}"
HORIZONS="${HORIZONS:-96,192,336,720}"
METHODS="${METHODS:-RevIN,DishTS,SAN,NST,TAFAS}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-2}"
MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-4096}"
MAX_EVAL_WINDOWS="${MAX_EVAL_WINDOWS:-512}"
BASELINE_PREDICTION_DIR="${BASELINE_PREDICTION_DIR:-baseline_predictions/core_table}"

echo "== Validate frozen HalluGuard configs =="
python scripts/validate_core_configs.py

echo "== Fetch data and official source snapshots =="
python scripts/fetch_core_datasets.py --datasets core
bash scripts/fetch_plugin_repos.sh

echo "== Ensure frozen HalluGuard local/control tables exist =="
python experiments/halluguard/run_real_table.py \
  --scope stage7 \
  --config experiments/halluguard/configs/halluguard_mvp.yaml \
  --data-root external/ETDataset \
  --epochs "${EPOCHS}" \
  --max-train-windows "${MAX_TRAIN_WINDOWS}" \
  --max-eval-windows "${MAX_EVAL_WINDOWS}" \
  --device "${DEVICE}"

python experiments/halluguard/run_stage14_autosearch.py \
  --scope clean_full \
  --config experiments/halluguard/configs/halluguard_core_table_sp_frozen.yaml

python experiments/halluguard/run_stage14_autosearch.py \
  --scope clean_full \
  --config experiments/halluguard/configs/halluguard_core_table_stable_harm.yaml

echo "== Export unified official baseline predictions =="
python scripts/run_core12_predictions.py \
  --datasets "${DATASET_SET}" \
  --models "${MODELS}" \
  --horizons "${HORIZONS}" \
  --methods "${METHODS}" \
  --data-root external/ETDataset \
  --output-dir "${BASELINE_PREDICTION_DIR}" \
  --epochs "${EPOCHS}" \
  --max-train-windows "${MAX_TRAIN_WINDOWS}" \
  --max-eval-windows "${MAX_EVAL_WINDOWS}" \
  --device "${DEVICE}" \
  --continue-on-error

echo "== Evaluate HalluGuard external batch on official baseline predictions =="
python experiments/halluguard/run_stage12_external_batch.py \
  --config experiments/halluguard/configs/halluguard_stage12_external_batch.yaml \
  --input-dir "${BASELINE_PREDICTION_DIR}" \
  --recursive \
  --continue-on-error \
  --output-dir experiments/halluguard/results/core_table/external_baselines

echo "== Build final 12-method core table =="
python scripts/build_core12_table.py \
  --baseline-dir "${BASELINE_PREDICTION_DIR}" \
  --output-dir experiments/halluguard/results/core_table/core12_combined
