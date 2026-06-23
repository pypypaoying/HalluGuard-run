# Core Table Baseline Protocol

This repo is now a frozen-method core-table harness. It should compare the
selected HalluGuard line against same-position test-time correction,
normalization, smoothing, non-stationary, and adaptation baselines. It should
not keep searching for new Stage 14 variants on the server.

## Fixed 12 Methods

| ID | Method | Role | Source |
| --- | --- | --- | --- |
| 1 | raw_no_correction | Unmodified backbone prediction export. | local |
| 2 | HalluGuard-SP frozen | Frozen Stage 14 `s14_smoothing_cap_selective_router`. | local |
| 3 | HalluGuard stable-harm ablation | Frozen Stage 14 `s14_stable_smoothing_cap_router`. | local |
| 4 | matched_sparse_smoothing | Sparse smoothing control matched to HalluGuard action rate. | local |
| 5 | naive_smoothing | Full-horizon moving-average smoothing control. | local |
| 6 | ema_smoothing | Full-horizon exponential moving-average smoothing control. | local |
| 7 | median_smoothing | Full-horizon median smoothing control. | local |
| 8 | RevIN | Reversible instance normalization. | official repo |
| 9 | Dish-TS | Distribution-shift plugin. | official repo |
| 10 | SAN | Adaptive normalization. | official repo |
| 11 | Non-stationary Transformer / NST | Non-stationary Transformer baseline. | official repo |
| 12 | TAFAS | Test-time adaptation baseline. | official repo |

Machine-readable details live in `docs/core_table_manifest.yaml`.

## Official Repo Setup

Fetch official baseline snapshots with:

```bash
bash scripts/fetch_plugin_repos.sh
```

Pinned repos:

| Method | Repo | Commit |
| --- | --- | --- |
| RevIN | `https://github.com/ts-kim/RevIN.git` | `fee40bc6c87cb536d048bcf1c14c4ed644b875e1` |
| Dish-TS | `https://github.com/weifantt/Dish-TS.git` | `e674d3b94b832491f63a533d60e40a75031d2c75` |
| SAN | `https://github.com/icantnamemyself/SAN.git` | `7e1ca66251a91a89290846b310145c5f5db3ffc3` |
| NST | `https://github.com/thuml/Nonstationary_Transformers.git` | `c4ec40675d11d50b3d9923657f408d0db6f90f56` |
| TAFAS | `https://github.com/kimanki/TAFAS.git` | `139bf980671da4daad728a0fc21d8df508b9203d` |

The cloned repos are intentionally ignored by git under
`external/plugin_baselines/`.

## Fairness Rules

1. Use the same dataset, horizon, split, context length, target variable, and
   sample export format across raw backbones, HalluGuard, smoothing controls,
   and official baselines.
2. Fit HalluGuard policies and any baseline calibration using validation data
   only.
3. Use test split only for final metrics.
4. Keep smoothing controls visible. Do not hide naive, EMA, median, or matched
   sparse smoothing when HalluGuard is reported.
5. Report baseline failures as blocked rows with a reproducible reason.
6. If TAFAS uses feedback or adaptation signals unavailable to black-box
   post-processing methods, mark that row as conditional rather than directly
   equivalent.

## Prediction Export Contract

Every method should export JSONL or CSV rows:

```text
sample_id, dataset, model, split, context, prediction, target
```

The `model` field should include the backbone and method name, for example:

```text
PatchTST+raw_no_correction
PatchTST+HalluGuard-SP
PatchTST+RevIN
PatchTST+NST
DLinear+DishTS
DLinear+TAFAS
```

`split=val` is for calibration or policy selection. `split=test` is for final
metrics only.

## Core Table Axes

Default core datasets:

```text
ETTm1, ETTh1, Weather, Electricity
```

Optional extended datasets:

```text
ETTm2, ETTh2, Traffic
```

Backbones:

```text
DLinear, PatchTST
```

Horizons:

```text
96, 192, 336, 720
```

## External Batch Evaluation

After exporting official baseline predictions into `baseline_predictions/`,
evaluate them with:

```bash
python experiments/halluguard/run_stage12_external_batch.py \
  --config experiments/halluguard/configs/halluguard_stage12_external_batch.yaml \
  --input-dir baseline_predictions \
  --recursive \
  --continue-on-error \
  --output-dir experiments/halluguard/results/core_table/external_baselines
```

The output is an aggregate CSV/markdown report with validation-only calibration
flags and blocker reasons.
