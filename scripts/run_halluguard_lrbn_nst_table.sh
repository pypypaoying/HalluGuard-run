#!/usr/bin/env bash
set -euo pipefail

DATASET_SET="${DATASET_SET:-ETTm1,ETTh1}"
MODELS="${MODELS:-DLinear,PatchTST}"
HORIZONS="${HORIZONS:-96,192,336,720}"
LRBN_NST_VARIANTS="${LRBN_NST_VARIANTS:-unified_revin_rdn_hybrid,nst_lightweight,lrbn_unified_nst_residual,lrbn_nst_output_blend}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-2}"
MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-4096}"
MAX_EVAL_WINDOWS="${MAX_EVAL_WINDOWS:-512}"
PREDICTION_DIR="${PREDICTION_DIR:-baseline_predictions/halluguard_lrbn_nst}"
RAW_PREDICTION_DIR="${RAW_PREDICTION_DIR:-baseline_predictions/halluguard_lrbn_nst_raw}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/halluguard/results/halluguard_lrbn_nst}"

echo "== Fetch core ETT data =="
python scripts/fetch_core_datasets.py --datasets core

echo "== Run HalluGuard-LRBN + NST complementarity ablations =="
python scripts/run_halluguard_lrbn_nst.py \
  --datasets "${DATASET_SET}" \
  --models "${MODELS}" \
  --horizons "${HORIZONS}" \
  --variants "${LRBN_NST_VARIANTS}" \
  --data-root external/ETDataset \
  --prediction-dir "${PREDICTION_DIR}" \
  --raw-prediction-dir "${RAW_PREDICTION_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --max-train-windows "${MAX_TRAIN_WINDOWS}" \
  --max-eval-windows "${MAX_EVAL_WINDOWS}" \
  --device "${DEVICE}" \
  --continue-on-error

echo "== HalluGuard-LRBN + NST outputs =="
echo "${OUTPUT_DIR}/lrbn_metrics.csv"
echo "${OUTPUT_DIR}/lrbn_summary.csv"
echo "${OUTPUT_DIR}/summary.md"
