#!/usr/bin/env bash
set -euo pipefail

python experiments/halluguard/run_stage14_autosearch.py \
  --scope clean_full \
  --config experiments/halluguard/configs/halluguard_core_table_sp_frozen.yaml

for stress_type in \
  boundary_discontinuity \
  trend_drift \
  slope_break \
  delayed_level_shift \
  high_frequency_perturbation \
  variance_shift
do
  python experiments/halluguard/run_stage14_stress_resume.py \
    --config experiments/halluguard/configs/halluguard_core_table_sp_frozen.yaml \
    --stress-types "${stress_type}"
done

python experiments/halluguard/run_stage14_autosearch.py \
  --scope external_batch \
  --config experiments/halluguard/configs/halluguard_core_table_sp_frozen.yaml
