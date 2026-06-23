# Plug-in Baseline Protocol

HalluGuard must be compared against plug-in modules attached to the same
forecasting backbone, not only against the raw backbone.

## Main Baselines

Run these first when implementation time is limited:

| Priority | Module | Venue | Baseline role |
| --- | --- | --- | --- |
| P0 | RevIN | ICLR 2022 | Reversible instance normalization for distribution shift. |
| P0 | Dish-TS | AAAI 2023 | Model-agnostic distribution-shift plugin. |
| P0 | SAN | NeurIPS 2023 | Adaptive normalization for non-stationary TSF. |
| P0 | SIN | ICML 2024 | Selective/interpretable normalization. |
| P0 | FAN | NeurIPS 2024 | Frequency adaptive normalization. |
| P0 | DDN | NeurIPS 2024 | Dual-domain dynamic normalization. |
| P1 | CCM | NeurIPS 2024 | Channel clustering plug-in for multivariate TSF. |
| P1 | LIFT | ICLR 2024 | Leading-indicator plug-in for arbitrary TSF methods. |

Conditional baseline:

| Module | Venue | Use only when |
| --- | --- | --- |
| TAFAS | AAAI 2025 | The experiment allows streaming partial-label/test-time adaptation feedback. |

Transformer-specific baseline:

| Module | Venue | Scope |
| --- | --- | --- |
| Non-stationary Transformer | NeurIPS 2022 | Compare only on Transformer/PatchTST-style backbones if integrated fairly. |

## Fairness Rules

1. Use the same dataset, horizon, split, context length, target variable, and
   sample export format across original backbone, HalluGuard, and plug-in
   baselines.
2. Fit plug-in parameters and any HalluGuard policy using validation data only.
3. Use test split only for final metrics.
4. Keep smoothing controls visible: naive smoothing, EMA smoothing, median
   smoothing, matched sparse smoothing, and random action/router controls.
5. Report baseline failures as blocked rows rather than deleting them.

## Prediction Export Contract

Every plug-in baseline should export JSONL or CSV rows:

```text
sample_id, dataset, model, split, context, prediction, target
```

Where `model` should include the backbone and plug-in name, for example:

```text
DLinear+RevIN
PatchTST+DDN
```

The HalluGuard external batch runner can then evaluate all files in a directory:

```bash
python experiments/halluguard/run_stage12_external_batch.py \
  --config experiments/halluguard/configs/halluguard_stage12_external_batch.yaml \
  --input-dir baseline_predictions \
  --recursive \
  --continue-on-error \
  --output-dir experiments/halluguard/results/plugin_baseline_external_batch
```

## Recommended Table Axes

- Datasets: ETTm1, ETTh1 first; then ETTm2, ETTh2, Weather, ECL, Traffic.
- Backbones: DLinear and PatchTST first.
- Horizons: 96, 192, 336, 720.
- Modules: raw backbone, HalluGuard router, RevIN, Dish-TS, SAN, SIN, FAN, DDN,
  CCM, LIFT, optional TAFAS.

## Minimum Server Run

For the first full baseline table, use:

```text
2 datasets x 2 backbones x 4 horizons x 10 methods
```

Methods:

```text
raw backbone
HalluGuard Stage 14 router
RevIN
Dish-TS
SAN
SIN
FAN
DDN
CCM
LIFT
```

TAFAS should be reported separately if it uses extra streaming feedback that the
other methods do not use.
