# HalluGuard-LRBN + NST Complementarity Report

Status: implementation in progress.

## Parent

Main claim-clean parent: `unified_revin_rdn_hybrid`.

## Candidate Variants

- `unified_revin_rdn_hybrid`: parent baseline.
- `nst_lightweight`: shared-runner NST-style stationarization baseline.
- `lrbn_unified_nst_residual`: claim-clean residual stationarization on top of
  LRBN normalized coordinates.
- `lrbn_nst_output_blend`: diagnostic train-split blend of LRBN and NST
  branches.
- `lrbn_nst_feature_gate`: train-split context-feature gate between LRBN and NST
  branches.
- `lrbn_nst_conservative_gate`: feature gate initialized near the LRBN parent
  (`0.95` LRBN weight) to test whether NST can be a safe complement.

## Results

### Smoke 1

Command:

```powershell
python scripts\run_halluguard_lrbn_nst.py --datasets ETTm1 --models DLinear,PatchTST --horizons 96 --variants unified_revin_rdn_hybrid,nst_lightweight,lrbn_unified_nst_residual,lrbn_nst_output_blend --data-root external\ETDataset --prediction-dir baseline_predictions\halluguard_lrbn_nst_smoke --raw-prediction-dir baseline_predictions\halluguard_lrbn_nst_smoke_raw --output-dir experiments\halluguard\results\halluguard_lrbn_nst_smoke --epochs 1 --max-train-windows 128 --max-eval-windows 32 --device cpu --continue-on-error
```

Observed:

- DLinear: parent `unified_revin_rdn_hybrid` MSE `3.089931`; residual NST
  candidate MSE `3.139437`; diagnostic output blend MSE `3.113672`.
- PatchTST: parent MSE `3.286239`; residual NST candidate MSE `3.291171`;
  diagnostic output blend MSE `3.163492`.

Interpretation: residual stationarization does not improve the LRBN parent in
this first smoke, suggesting redundancy between NST-style stationarization and
LRBN center/scale. The diagnostic output blend improves PatchTST, so there may
be branch complementarity worth testing with a deployable context gate.

### Low-Budget L1: Feature Gate

Command:

```powershell
python scripts\run_halluguard_lrbn_nst.py --datasets ETTm1,ETTh1 --models DLinear,PatchTST --horizons 96,192,336,720 --variants unified_revin_rdn_hybrid,nst_lightweight,lrbn_nst_feature_gate,lrbn_nst_output_blend --data-root external\ETDataset --prediction-dir baseline_predictions\halluguard_lrbn_nst_l1 --raw-prediction-dir baseline_predictions\halluguard_lrbn_nst_l1_raw --output-dir experiments\halluguard\results\halluguard_lrbn_nst_l1 --epochs 2 --max-train-windows 1024 --max-eval-windows 128 --device cpu --continue-on-error
```

Summary versus `unified_revin_rdn_hybrid` parent:

- `nst_lightweight`: wins `2/16`, mean MSE delta vs parent `+2.4951%`.
- `lrbn_nst_feature_gate`: wins `7/16`, mean MSE delta vs parent `-0.2047%`;
  DLinear `+0.5178%`, PatchTST `-0.9272%`.
- `lrbn_nst_output_blend`: wins `9/16`, mean MSE delta vs parent `+0.0429%`;
  DLinear `+1.3208%`, PatchTST `-1.2349%`.

Interpretation: NST is not a standalone improvement here, but it is useful on
PatchTST when blended with LRBN. The next candidate is a conservative gate that
starts near the LRBN parent and only borrows NST when training supports it.

### Main Local 16-Config Table

Command:

```powershell
python scripts\run_halluguard_lrbn_nst.py --datasets ETTm1,ETTh1 --models DLinear,PatchTST --horizons 96,192,336,720 --variants unified_revin_rdn_hybrid,lrbn_nst_feature_gate,lrbn_nst_conservative_gate --data-root external\ETDataset --prediction-dir baseline_predictions\halluguard_lrbn_nst_main --raw-prediction-dir baseline_predictions\halluguard_lrbn_nst_main_raw --output-dir experiments\halluguard\results\halluguard_lrbn_nst_main --epochs 4 --max-train-windows 4096 --max-eval-windows 512 --device cpu --continue-on-error
```

Output:

- `experiments/halluguard/results/halluguard_lrbn_nst_main/lrbn_metrics.csv`
- `experiments/halluguard/results/halluguard_lrbn_nst_main/lrbn_summary.csv`
- `experiments/halluguard/results/halluguard_lrbn_nst_main/parent_comparison.json`

Summary:

| variant | completed | mean MSE | mean MAE | mean MSE delta vs raw |
| --- | ---: | ---: | ---: | ---: |
| `raw_no_correction` | 16/16 | 6.506251 | 1.909880 | 0.000000% |
| `unified_revin_rdn_hybrid` | 16/16 | 5.524497 | 1.780036 | -10.492039% |
| `lrbn_nst_feature_gate` | 16/16 | 5.495285 | 1.775948 | -10.963676% |
| `lrbn_nst_conservative_gate` | 16/16 | 5.531665 | 1.780901 | -10.467990% |

Relative to `unified_revin_rdn_hybrid`:

- `lrbn_nst_feature_gate`: wins `9/16`; mean MSE delta `-0.458720%`;
  mean MAE delta `-0.208078%`; max MSE harm `0.601473%`; max MSE gain
  `-4.162905%`.
- `lrbn_nst_feature_gate` by backbone: DLinear mean MSE delta `+0.120518%`
  with `1/8` wins; PatchTST mean MSE delta `-1.037958%` with `8/8` wins.
- `lrbn_nst_conservative_gate`: wins `10/16`; mean MSE delta `+0.026373%`;
  mean MAE delta `+0.022057%`; max MSE harm `2.191027%`.

Verdict: `lrbn_nst_feature_gate` is the best current LRBN+NST complementarity
candidate. It improves the local 16-config table mainly by repairing PatchTST,
while causing only small DLinear harm. It should be retained as a candidate
hybrid line, but the claim-clean parent remains `unified_revin_rdn_hybrid`
until the same-budget server run confirms this signal.

## Claim Boundary

This experiment does not claim to reproduce the full Non-stationary Transformer
architecture unless the official attention de-stationary factors are integrated
and evaluated. The first pass tests whether NST-style stationarization is
complementary to HalluGuard-LRBN under the existing fair adapter protocol.

## Next Experiment

Run the same candidate on the server budget used for the LRBN combo table:

```bash
LRBN_NST_VARIANTS=unified_revin_rdn_hybrid,lrbn_nst_feature_gate,lrbn_nst_conservative_gate \
DEVICE=cuda EPOCHS=10 MAX_TRAIN_WINDOWS=8192 MAX_EVAL_WINDOWS=1024 \
PREDICTION_DIR=baseline_predictions/halluguard_lrbn_nst_server \
RAW_PREDICTION_DIR=baseline_predictions/halluguard_lrbn_nst_server_raw \
OUTPUT_DIR=experiments/halluguard/results/halluguard_lrbn_nst_server \
  bash scripts/run_halluguard_lrbn_nst_table.sh
```

Promotion rule: if `lrbn_nst_feature_gate` keeps PatchTST wins near `8/8` and
keeps DLinear max harm below `1%`, promote it as the LRBN+NST hybrid candidate.
If the gain disappears, conclude that NST-style stationarization is not robustly
complementary to LRBN under this adapter.
