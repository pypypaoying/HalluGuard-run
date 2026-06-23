#!/usr/bin/env bash
set -euo pipefail

python experiments/halluguard/run_real_table.py \
  --scope stage7 \
  --config experiments/halluguard/configs/halluguard_mvp.yaml \
  --data-root external/ETDataset \
  --epochs "${EPOCHS:-2}" \
  --max-train-windows "${MAX_TRAIN_WINDOWS:-4096}" \
  --max-eval-windows "${MAX_EVAL_WINDOWS:-512}" \
  --device "${DEVICE:-auto}"
