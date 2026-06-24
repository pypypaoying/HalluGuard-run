#!/usr/bin/env bash
set -euo pipefail

# One-command HalluGuard-LRBN clean-claim big table.
#
# Default full matrix:
#   datasets: ETTm1, ETTm2, ETTh1, ETTh2, Weather, ECL, Traffic
#   backbones: DLinear, PatchTST, iTransformer, TimesNet, TimeMixer, Nonstationary_Transformer
#   horizons: 96,192,336,720
#   seeds: 2026,2027,2028
#
# Current lightweight in-repo runner completes DLinear/PatchTST on ETTm1/ETTh1
# and records unsupported rows as blocked with reproducible blocker reasons.

DATASETS="${DATASETS:-ETTm1,ETTm2,ETTh1,ETTh2,Weather,ECL,Traffic}"
BACKBONES="${BACKBONES:-DLinear,PatchTST,iTransformer,TimesNet,TimeMixer,Nonstationary_Transformer}"
HORIZONS="${HORIZONS:-96,192,336,720}"
SEEDS="${SEEDS:-2026,2027,2028}"
METHODS="${METHODS:-raw_no_correction,HalluGuard-LRBN,matched_sparse_smoothing,naive_smoothing,ema_smoothing,median_smoothing,RevIN,DishTS,SAN,NST,TAFAS}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-10}"
MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-8192}"
MAX_EVAL_WINDOWS="${MAX_EVAL_WINDOWS:-1024}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/halluguard/results/lrbn_clean_claim_bigtable_v1}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

python scripts/run_lrbn_clean_claim_bigtable.py \
  --datasets "${DATASETS}" \
  --backbones "${BACKBONES}" \
  --horizons "${HORIZONS}" \
  --seeds "${SEEDS}" \
  --methods "${METHODS}" \
  --data-root external/ETDataset \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --max-train-windows "${MAX_TRAIN_WINDOWS}" \
  --max-eval-windows "${MAX_EVAL_WINDOWS}" \
  --device "${DEVICE}" \
  --fetch-data \
  ${EXTRA_FLAGS}

echo "== BigTable outputs =="
echo "${OUTPUT_DIR}/combined_metrics.csv"
echo "${OUTPUT_DIR}/combined_metrics.json"
echo "${OUTPUT_DIR}/summary.md"
