#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

DATASETS="${DATASETS:-ETTm1,ETTm2,ETTh1,ETTh2,Weather,Exchange,ECL,Traffic}"
BACKBONES="${BACKBONES:-DLinear,PatchTST,iTransformer,TimesNet,TimeMixer,FreTS}"
HORIZONS="${HORIZONS:-96,192,336,720}"
SEEDS="${SEEDS:-2026,2027,2028}"
METHODS="${METHODS:-raw_no_correction,HalluGuard-LRBN,Safe-SRA,Balanced-SRA,RevIN,DishTS,SAN,NST,SoP-step-wise,SoP-variable-wise,SOLID-official-supported,matched_sparse_smoothing,naive_smoothing,ema_smoothing,median_smoothing}"

DATA_ROOT="${DATA_ROOT:-external/ETDataset}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/halluguard/results/tablea_full_v1}"
DEVICE="${DEVICE:-auto}"

SEQ_LEN="${SEQ_LEN:-96}"
TAIL_LEN="${TAIL_LEN:-48}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"

SAN_PERIOD_LEN="${SAN_PERIOD_LEN:-24}"
SAN_STATION_LR="${SAN_STATION_LR:-0.0001}"
SAN_PRETRAIN_EPOCHS="${SAN_PRETRAIN_EPOCHS:-5}"

SOP_PLUG_EPOCHS="${SOP_PLUG_EPOCHS:-10}"
SOP_PLUG_LR="${SOP_PLUG_LR:-0.001}"
SOP_STEP_CSEG_LEN="${SOP_STEP_CSEG_LEN:-1}"
SOP_VARIABLE_CSEG_LEN="${SOP_VARIABLE_CSEG_LEN:-1}"
SRA_POLICY_DIR="${SRA_POLICY_DIR:-experiments/halluguard/results/lrbn_sra_bp_stage5}"

# <=0 means all windows. This default is intentionally expensive for the final Table A.
MAX_TRAIN_WINDOWS="${MAX_TRAIN_WINDOWS:-0}"
MAX_EVAL_WINDOWS="${MAX_EVAL_WINDOWS:-0}"

FETCH_DATA="${FETCH_DATA:-1}"
FETCH_PLUGIN_REPOS="${FETCH_PLUGIN_REPOS:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
SMOKE="${SMOKE:-0}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

CMD=(
  "${PYTHON_BIN}" scripts/run_tablea_full.py
  --datasets "${DATASETS}"
  --backbones "${BACKBONES}"
  --horizons "${HORIZONS}"
  --seeds "${SEEDS}"
  --methods "${METHODS}"
  --data-root "${DATA_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --seq-len "${SEQ_LEN}"
  --tail-len "${TAIL_LEN}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --learning-rate "${LEARNING_RATE}"
  --san-period-len "${SAN_PERIOD_LEN}"
  --san-station-lr "${SAN_STATION_LR}"
  --san-pretrain-epochs "${SAN_PRETRAIN_EPOCHS}"
  --sop-plug-epochs "${SOP_PLUG_EPOCHS}"
  --sop-plug-lr "${SOP_PLUG_LR}"
  --sop-step-cseg-len "${SOP_STEP_CSEG_LEN}"
  --sop-variable-cseg-len "${SOP_VARIABLE_CSEG_LEN}"
  --sra-policy-dir "${SRA_POLICY_DIR}"
  --max-train-windows "${MAX_TRAIN_WINDOWS}"
  --max-eval-windows "${MAX_EVAL_WINDOWS}"
  --device "${DEVICE}"
)

if [[ "${FETCH_DATA}" == "1" ]]; then
  CMD+=(--fetch-data)
fi
if [[ "${FETCH_PLUGIN_REPOS}" == "1" ]]; then
  CMD+=(--fetch-plugin-repos)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  CMD+=(--skip-existing)
fi
if [[ "${SMOKE}" == "1" ]]; then
  CMD+=(--smoke)
fi
if [[ -n "${EXTRA_FLAGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARRAY=(${EXTRA_FLAGS})
  CMD+=("${EXTRA_ARRAY[@]}")
fi

echo "[TableA] command:"
printf ' %q' "${CMD[@]}"
echo
"${CMD[@]}"
