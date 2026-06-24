#!/usr/bin/env bash
set -euo pipefail

DATASET_SET="${DATASET_SET:-core}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-2}"
MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-4096}"
MAX_EVAL_WINDOWS="${MAX_EVAL_WINDOWS:-512}"
BASELINE_PREDICTION_DIR="${BASELINE_PREDICTION_DIR:-baseline_predictions/core_table}"

echo "== Core table setup =="
python scripts/validate_core_configs.py
python scripts/fetch_core_datasets.py --datasets "${DATASET_SET}"
bash scripts/fetch_plugin_repos.sh

echo "== Raw backbone prediction export for HalluGuard frozen methods =="
python experiments/halluguard/run_real_table.py \
  --scope stage7 \
  --config experiments/halluguard/configs/halluguard_mvp.yaml \
  --data-root external/ETDataset \
  --epochs "${EPOCHS}" \
  --max-train-windows "${MAX_TRAIN_WINDOWS}" \
  --max-eval-windows "${MAX_EVAL_WINDOWS}" \
  --device "${DEVICE}"

echo "== Frozen HalluGuard-SP evaluation =="
python experiments/halluguard/run_stage14_autosearch.py \
  --scope clean_full \
  --config experiments/halluguard/configs/halluguard_core_table_sp_frozen.yaml

echo "== Stable-harm ablation evaluation =="
python experiments/halluguard/run_stage14_autosearch.py \
  --scope clean_full \
  --config experiments/halluguard/configs/halluguard_core_table_stable_harm.yaml

if [[ -d "${BASELINE_PREDICTION_DIR}" ]]; then
  echo "== External baseline prediction evaluation =="
  python experiments/halluguard/run_stage12_external_batch.py \
    --config experiments/halluguard/configs/halluguard_stage12_external_batch.yaml \
    --input-dir "${BASELINE_PREDICTION_DIR}" \
    --recursive \
    --continue-on-error \
    --output-dir experiments/halluguard/results/core_table/external_baselines
else
  echo "No ${BASELINE_PREDICTION_DIR} directory found."
  echo "Export official baseline predictions there using the schema in BASELINE_PLUGIN_PROTOCOL.md, then rerun this script."
fi
