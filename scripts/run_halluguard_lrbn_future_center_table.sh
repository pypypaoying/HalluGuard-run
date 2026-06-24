#!/usr/bin/env bash
set -euo pipefail

DATASET_SET="${DATASET_SET:-ETTm1,ETTh1}"
MODELS="${MODELS:-DLinear,PatchTST}"
HORIZONS="${HORIZONS:-96,192,336,720}"
FUTURE_CENTER_VARIANTS="${FUTURE_CENTER_VARIANTS:-unified_revin_rdn_hybrid,future_center_static,future_center_selector,future_center_selector_drift}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-2}"
MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-4096}"
MAX_EVAL_WINDOWS="${MAX_EVAL_WINDOWS:-512}"
PREDICTION_DIR="${PREDICTION_DIR:-baseline_predictions/halluguard_lrbn_future_center}"
RAW_PREDICTION_DIR="${RAW_PREDICTION_DIR:-baseline_predictions/halluguard_lrbn_future_center_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/halluguard/results/halluguard_lrbn_future_center}"

echo "== Fetch core ETT data =="
python scripts/fetch_core_datasets.py --datasets core

echo "== Run HalluGuard-LRBN future-center ablations =="
python scripts/run_halluguard_lrbn_future_center.py \
  --datasets "${DATASET_SET}" \
  --models "${MODELS}" \
  --horizons "${HORIZONS}" \
  --variants "${FUTURE_CENTER_VARIANTS}" \
  --data-root external/ETDataset \
  --prediction-dir "${PREDICTION_DIR}" \
  --raw-prediction-dir "${RAW_PREDICTION_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --max-train-windows "${MAX_TRAIN_WINDOWS}" \
  --max-eval-windows "${MAX_EVAL_WINDOWS}" \
  --device "${DEVICE}" \
  --continue-on-error

echo "== HalluGuard-LRBN future-center outputs =="
echo "${OUTPUT_DIR}/lrbn_metrics.csv"
echo "${OUTPUT_DIR}/lrbn_summary.csv"
echo "${OUTPUT_DIR}/summary.md"
