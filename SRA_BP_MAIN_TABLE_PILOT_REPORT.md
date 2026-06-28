# SRA-BP Main-Table Pilot Report

## Question

The current core-table contract in `docs/core_table_manifest.yaml` compares HalluGuard against:

- local post-processing / controls: `raw_no_correction`, `HalluGuard-SP frozen`, `HalluGuard stable-harm ablation`, `matched_sparse_smoothing`, `naive_smoothing`, `ema_smoothing`, `median_smoothing`;
- official or near-official adaptation baselines: `RevIN`, `DishTS`, `SAN`, `NST`, `TAFAS`.

Previous SRA-BP validation compared Safe-SRA and Balanced-SRA against LRBN, raw predictions, smoothing controls, and internal BP/SRA ablations on the compact 8-config validation setup. It did **not** provide a fair same-protocol direct comparison against the official core-table baselines `RevIN`, `DishTS`, `SAN`, `NST`, or `TAFAS`.

## Pilot Setup

- Runner: `experiments/halluguard/run_sra_bp_main_table_pilot.py`
- Output: `experiments/halluguard/results/sra_bp_main_table_pilot/`
- Input predictions: `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`
- SRA parameters: `experiments/halluguard/results/lrbn_sra_bp_stage5/stage5_selected_safe_params.json` and `stage5_selected_balanced_params.json`
- Scope: 8 compact test configs, 768 test samples.
- Strictly aligned methods: raw, HalluGuard-LRBN, matched sparse smoothing, naive smoothing, EMA smoothing, median smoothing, Safe-SRA, Balanced-SRA.
- Leakage: `False`.

## Aligned Results

| method | configs | mean MSE | mean MAE | MSE delta vs raw | MSE delta vs LRBN | improved configs vs LRBN | harmed configs vs LRBN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LRBN-SRA-BP-balanced | 8 | 4.766983 | 1.645627 | -22.302684% | -2.975935% | 8 | 0 |
| LRBN-SRA-BP-safe | 8 | 4.813149 | 1.660950 | -21.562981% | -1.996233% | 8 | 0 |
| HalluGuard-LRBN | 8 | 4.894158 | 1.682162 | -20.028831% | 0.000000% | 0 | 0 |
| ema_smoothing | 8 | 6.060487 | 1.848221 | -5.922068% | +24.694504% | 4 | 4 |
| naive_smoothing | 8 | 6.072069 | 1.849256 | -5.763867% | +25.002055% | 4 | 4 |
| median_smoothing | 8 | 6.129929 | 1.860337 | -4.849369% | +26.440553% | 4 | 4 |
| matched_sparse_smoothing | 8 | 6.222475 | 1.877730 | -2.971133% | +28.340476% | 3 | 5 |
| raw_no_correction | 8 | 6.427221 | 1.914908 | 0.000000% | +32.849732% | 0 | 8 |

## Official Baseline Status

`SAN` and `DishTS` have local reference outputs on overlapping config keys, but their raw baseline on the overlap is about `+47.914700%` MSE higher than the pilot raw baseline. Therefore those rows are not folded into the aligned mean. `RevIN`, `NST`, and `TAFAS` are not locally available for this compact pilot.

Detailed availability is saved in:

- `experiments/halluguard/results/sra_bp_main_table_pilot/official_baseline_availability.csv`

## Verdict

Balanced-SRA and Safe-SRA both pass this pre-table pilot against the aligned local controls. Balanced-SRA is stronger on mean MSE, while Safe-SRA is the lower-harm variant. This supports adding both variants to the next real core-table run.

The pilot does **not** prove superiority over RevIN/DishTS/SAN/NST/TAFAS. That claim requires a fresh TableA/Core run where every method uses the same dataset, backbone, horizon, seed, split, raw backbone predictions, and prediction schema.

