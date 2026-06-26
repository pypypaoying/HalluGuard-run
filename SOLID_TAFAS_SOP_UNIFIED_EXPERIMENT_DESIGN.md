# Unified Experiment Design for HalluGuard-LRBN, SOLID, TAFAS, and SoP

This document defines the next broad comparison protocol for HalluGuard-LRBN
against recent calibration/adaptation competitors:

- SOLID / Calibration-CDS
- TAFAS
- SoP
- existing normalization and smoothing baselines from the HalluGuard-run tables

The goal is not to reproduce every official leaderboard recipe independently.
The goal is to create one fair, auditable setting that covers the largest
intersection of these methods, while preserving separate official-protocol
appendix runs where a method needs extra information or a different setting.

## 1. Main Comparison Principle

Use two tables, not one overloaded table.

### Table A: Unified-Budget Point Forecast Table

This is the paper-facing MSE/MAE table for HalluGuard-LRBN versus point-forecast
calibration/adaptation methods under the same data split, backbone, horizon,
seed, and training budget.

Rules:

- Same train/val/test split for every method.
- Same forecasting backbone per row.
- Same history length and prediction horizon per row.
- Same random seeds.
- Validation split may tune thresholds, early stopping, plug selection, and
  adaptation hyperparameters.
- Test split is final evaluation only.
- No method may use future test targets unless it is explicitly moved to
  Table B as an online/partially-observed method.

### Table B: Protocol-Specific / Online Adaptation Appendix

This is for methods whose official setting is not purely offline point
post-processing:

- TAFAS with partially observed ground truth.
- SOLID official PatchTST protocol with `seq_len=336`, if we want to reproduce
  the KDD-style setting exactly.

Table B should be interpreted as mechanism validation and official-repo
alignment, not as the primary fair clean-MSE table.

SoP is not appendix-only. It is an offline frozen-backbone calibration method,
so it belongs in Table A under the unified setting. Its official Exchange
script can be kept only as a reproducibility sanity check, not as a separate
scientific table.

## 2. Dataset Matrix

### Core datasets

These are the main datasets because they are covered by the largest overlap of
HalluGuard-run, SOLID, TAFAS, and SoP-style LTSF code:

```text
ETTm1
ETTm2
ETTh1
ETTh2
Weather
Exchange
```

Rationale:

- ETT datasets are shared by SOLID, TAFAS, and the existing HalluGuard tables.
- Weather is supported by SOLID, TAFAS, and the existing table.
- Exchange is important because SoP's public scripts are Exchange-centered.

### Expanded datasets

These should be included in the broad generalization table, with unsupported
method rows marked as `blocked` rather than silently omitted:

```text
ECL / Electricity
Traffic
ILI / Illness
```

Rationale:

- ECL and Traffic are important hard datasets from the existing HalluGuard
  big tables and SOLID scripts.
- ILI uses different official horizons, so it should be reported in an appendix
  rather than mixed into the standard 96/192/336/720 table.

### Dataset split contract

Use Time-Series-Library style chronological splits:

- ETT: official 12/4/4 month train/val/test split.
- Weather, ECL, Traffic, Exchange: chronological 70/10/20 split unless the
  imported official framework provides an explicit standard split.
- ILI: official long-term forecasting split and horizons only in appendix.

No random split is allowed.

## 3. Forecasting Task Contract

Primary task:

```text
multivariate-to-multivariate long-term forecasting
features = M
```

Metrics are averaged over:

- test windows
- forecast horizon steps
- target variables/channels

Legacy single-target results from previous HalluGuard-run experiments may be
kept as continuity checks, but the new cross-method comparison should use M->M
because SOLID, TAFAS, SoP, RevIN, SAN, Dish-TS, and most LTSF backbones are
designed and reported this way.

## 4. Horizon and Context Length

### Primary unified setting

```text
seq_len = 96
label_len = 48, only for decoder-style models that require it
pred_len in {96, 192, 336, 720}
```

Rationale:

- Matches the previous HalluGuard-run clean-claim tables.
- Matches SoP's public script convention.
- Matches common TSL long-term forecasting settings.
- Avoids giving SOLID/PatchTST a longer context than other methods in the
  unified table.

### SOLID official appendix

SOLID's official PatchTST scripts often use:

```text
seq_len = 336
pred_len in {96, 192, 336, 720}
```

Therefore we should optionally run:

```text
Table B-SOLID-official:
datasets = ETTm1, ETTm2, ETTh1, ETTh2, Weather, Electricity, Traffic
backbone = PatchTST
seq_len = 336
pred_len = 96, 192, 336, 720
```

This appendix should not be averaged into the primary `seq_len=96` table.

## 5. Backbone Matrix

### Tier 1: Core intersection backbones

These are the main paper-facing backbones:

```text
DLinear
PatchTST
iTransformer
```

Rationale:

- DLinear: simple linear baseline, strong for normalization methods.
- PatchTST: official SOLID and TAFAS support, widely used.
- iTransformer: modern strong transformer family, TAFAS scripts support it,
  and SoP has public script support.

### Tier 2: Generalization backbones

```text
TimesNet
TimeMixer
FreTS
```

Rationale:

- TimesNet and TimeMixer appeared in the existing broader HalluGuard-run plan.
- FreTS is supported by TAFAS checkpoints/scripts and is frequency-oriented,
  useful for checking whether LRBN helps beyond a frequency backbone.

Tier 2 rows should be included when adapters are available. Unsupported
official rows are recorded as `blocked` with a concrete reason.

## 6. Method Matrix

### Offline / fair point-forecast methods

These belong in Table A:

```text
raw_no_correction
HalluGuard-LRBN unified_revin_rdn_hybrid
RevIN
Dish-TS
SAN
NST
SoP-step-wise
SoP-variable-wise
SOLID-PatchTST-only
matched_sparse_smoothing
naive_smoothing
ema_smoothing
median_smoothing
```

SOLID is listed as a PatchTST-only strong baseline. Do not average SOLID across
non-PatchTST backbones unless a faithful official-compatible adapter is
implemented for those models. For non-PatchTST rows, use
`status=not_applicable` or omit SOLID from that backbone-specific mean rather
than pretending it is a model-agnostic competitor.

### Online / partially observed methods

These belong in Table B unless we explicitly define a no-POGT variant:

```text
TAFAS-online
TAFAS-online + HalluGuard-LRBN source model
```

TAFAS uses partially observed ground truth at test time. It is valid and
important, but it is not the same information budget as offline post-processing.

### Optional combined methods

Only after single-method baselines are stable:

```text
HalluGuard-LRBN + SoP
HalluGuard-LRBN + SOLID, PatchTST only
HalluGuard-LRBN + TAFAS-online
```

These are not baseline rows; they are candidate next-method rows.

## 7. Method-Specific Fair Settings

### HalluGuard-LRBN

Main variant:

```text
unified_revin_rdn_hybrid
```

Training:

- Fit all LRBN parameters on train split only.
- Use validation only for early stopping and model selection.
- Report learned center/scale gates when available.

### SoP

Use the source backbone as a frozen Socket.

Variants:

```text
SoP-step-wise: cfintune=0
SoP-variable-wise: cfintune=1
```

Settings:

- Train source backbone on train split.
- Freeze source backbone.
- Train Plug modules on train split.
- Validation split selects early stopping and plug grouping hyperparameters.
- Test split is final only.

Recommended plug grouping grid, validation-only:

```text
cseg_len in {1, 3, 6}
```

Report both:

- best validation-selected SoP variant
- raw step-wise and variable-wise ablations

### SOLID

Unified table variant:

```text
SOLID-PatchTST-only
```

Settings:

- Run only where the source forecaster is PatchTST, unless a faithful
  official-compatible implementation for another backbone is added later.
- Train PatchTST source forecaster on train split.
- Build residual/context library from train + val windows only.
- Use validation split to choose:
  - `selected_data_num`
  - `adapted_lr_times`
  - adaptation learning-rate multiplier
- At test time, for each test window:
  - retrieve similar windows only from the train/val library
  - adapt prediction layer only
  - do not use test target or later test windows as labels

Suggested validation grid:

```text
selected_data_num in {5, 10, 20}
adapted_lr_times in {5, 10, 20}
adapted_lr_multiplier in {1, 10, 100}
```

Default if validation budget must be small:

```text
selected_data_num = 10
adapted_lr_times = 10
adapted_lr_multiplier = 10
```

Interpretation:

- SOLID is a strong PatchTST-specific adaptation baseline.
- It should compete directly against `raw_no_correction`, `HalluGuard-LRBN`,
  SoP, RevIN/Dish-TS/SAN/NST, and smoothing on PatchTST rows.
- It should not be used to claim all-backbone average superiority unless the
  non-PatchTST rows are explicitly supported.

### TAFAS

TAFAS should be evaluated in an online appendix.

Settings:

- Train source forecaster on train split.
- Validation split selects:
  - base adaptation learning rate
  - gating initialization
  - waiting/partial-observation policy if configurable
- Test evaluation is causal:
  - at forecast issue time, no future target is visible;
  - partial ground truth may be used only when it would have been revealed
    under the TAFAS protocol.

Recommended grid:

```text
BASE_LR in {0.0001, 0.0005, 0.001}
WEIGHT_DECAY in {0.0, 0.0001}
GATING_INIT in {0.01, 0.1, 0.3}
```

Report TAFAS separately as:

```text
TAFAS-online
TAFAS-online + LRBN-source
```

Do not merge these rows into the offline HalluGuard-vs-SoP-vs-SOLID mean.

### NST

NST is architecture-sensitive.

Rules:

- For Transformer-like backbones, use official or faithful de-stationary
  attention implementation.
- For non-attention backbones, either:
  - report `blocked: architecture_incompatible`, or
  - use the existing HalluGuard-run approximation and label it explicitly as
    `NST-wrapper`, not official NST.

### Smoothing controls

Smoothing controls remain mandatory:

```text
naive_smoothing
ema_smoothing
median_smoothing
matched_sparse_smoothing
```

`matched_sparse_smoothing` must match the correction rate of the compared
method using validation-only calibration.

## 8. Training Budget and Optimizer Contract

Current HalluGuard-run implementation status:

```text
scripts/run_clean_claim_bigtable.sh default:
EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
SAN_PRETRAIN_EPOCHS = 5
MAX_TRAIN_WINDOWS = 8192
MAX_EVAL_WINDOWS = 1024
early_stopping = not currently implemented in the lightweight runner
```

The current runner aligns LRBN, RevIN/Dish-TS/SAN/NST/TAFAS-wrapper, and
smoothing controls by giving them the same epoch count, batch size, learning
rate, seed, window cap, and train/val/test split. It does not yet align SoP and
SOLID because those adapters are not implemented in the same runner.

Recommended final unified setting after adding SoP/SOLID adapters:

```text
seeds = 2026, 2027, 2028
optimizer = Adam
learning_rate = validation-selected from {1e-4, 5e-4, 1e-3}, with the same grid for all trainable wrappers on a given backbone
batch_size = 256 by default, reduced only for OOM and recorded per row
max_epochs = 10 for the broad resource-controlled table
early_stopping_patience = none in the current runner; use fixed epochs for strict budget alignment
loss = MSE
```

If we implement uniform early stopping, use the same rule for every trainable
method:

```text
validation_metric = val MSE
max_epochs = 20
early_stopping_patience = 5
restore_best_val_checkpoint = true
```

Do not give SoP or SOLID a longer hidden tuning budget than LRBN/RevIN/Dish-TS.
If SoP plug training needs a second stage, the base forecaster checkpoint must
be shared and frozen, and the plug stage budget must be reported separately:

```text
base_forecaster_epochs = same as raw/LRBN source
SoP_plug_max_epochs = 10 broad table, 20 confirmation table
SoP_plug_patience = none if fixed-budget; 5 only if all methods use early stopping
SOLID_adaptation_steps = validation-selected from the declared grid
```

If server limits require caps, use the same caps for every method and report
them in the table metadata:

```text
MAX_TRAIN_WINDOWS = 8192
MAX_VAL_WINDOWS = all or 2048
MAX_TEST_WINDOWS = all or 2048
```

Do not mix capped and uncapped rows in one mean.

## 9. Output Schema

All methods should emit or be converted into the same prediction schema:

```text
sample_id
dataset
model
method
seed
horizon
split
context
prediction
target
```

For online methods, add:

```text
protocol = online_partial_gt
available_truth_steps
issue_time
adaptation_time
```

## 10. Required Metrics

Per row:

```text
dataset
model
method
seed
horizon
seq_len
protocol
status
mse
mae
mse_delta_pct_vs_raw
mae_delta_pct_vs_raw
rank_by_mse
training_time_sec
inference_latency_ms
adaptation_latency_ms
test_threshold_leakage
blocker_reason
```

Aggregate:

- mean/std across seeds
- mean delta vs raw
- win count vs raw
- win count vs HalluGuard-LRBN
- win count vs RevIN
- win count vs best smoothing control
- dataset-level summary
- backbone-level summary
- horizon-level summary

For online appendix:

- causal availability rule
- mean adaptation latency
- whether partial ground truth was used

## 11. Leakage Rules

Allowed:

- train split for fitting models and learnable modules
- validation split for early stopping, hyperparameter selection, plug grouping,
  SOLID retrieval/adaptation settings, and threshold/correction-rate matching
- test context and prediction at evaluation time
- TAFAS partial ground truth only in the online appendix and only when causally
  available

Forbidden:

- choosing hyperparameters by test MSE/MAE
- building SOLID residual libraries from labeled test windows
- using full future target for TAFAS in the offline table
- reporting online TAFAS rows as if they used the same information budget as
  offline HalluGuard-LRBN
- selecting the final method by a single lucky seed

## 12. Recommended Reporting Layout

### Main table

```text
Table A1: Unified offline point forecast table
datasets = ETTm1, ETTm2, ETTh1, ETTh2, Weather, Exchange
models = DLinear, PatchTST, iTransformer
horizons = 96, 192, 336, 720
seeds = 2026, 2027, 2028
methods = raw, HalluGuard-LRBN, RevIN, Dish-TS, SAN, NST-compatible,
          SoP-step, SoP-variable, smoothing controls
extra PatchTST-only baseline = SOLID-PatchTST-only
```

### Expanded generalization table

```text
Table A2: Expanded offline table
add datasets = ECL, Traffic
add models = TimesNet, TimeMixer, FreTS
unsupported rows = blocked with reason
```

### Online appendix

```text
Table B1: TAFAS online partial-ground-truth table
datasets = ETTm1, ETTm2, ETTh1, ETTh2, Weather, Exchange
models = DLinear, PatchTST, iTransformer, FreTS if available
horizons = 96, 192, 336, 720
methods = source raw, TAFAS-online, LRBN source, LRBN + TAFAS-online
```

### Official protocol appendix

```text
Table B2: SOLID official-style PatchTST table
seq_len = 336
models = PatchTST
datasets = official SOLID supported datasets
horizons = 96, 192, 336, 720
```

## 13. Claim Boundary

The new experiment should support one of three conclusions:

1. HalluGuard-LRBN is a strong offline normalization/calibration method under a
   unified budget.
2. HalluGuard-LRBN is complementary to SoP/SOLID/TAFAS when used as a source
   model or front-end normalizer.
3. Some competitors win under their native online or official protocol, in
   which case the claim should narrow to offline black-box-safe calibration.

Do not claim:

- HalluGuard-LRBN beats TAFAS if TAFAS is using partial ground truth and LRBN is
  not.
- HalluGuard-LRBN beats SoP unless both use the same backbone and split.
- HalluGuard-LRBN beats SOLID globally unless the comparison is restricted to
  PatchTST rows or SOLID is faithfully implemented for the other backbones.
- NST results on non-attention backbones are official NST unless the attention
  mechanism is actually implemented.
- A capped-window broad table is identical to an official full-data leaderboard.

## 14. Immediate Implementation Plan

1. Add a unified method registry with method metadata:
   - offline vs online
   - requires frozen source
   - requires partial ground truth
   - official-compatible or wrapper approximation
2. Convert every method output to the unified prediction schema.
3. Implement Table A1 first.
4. Add Table A2 only after A1 smoke passes.
5. Add Table B1 TAFAS online appendix.
6. Add Table B2/B3 official-protocol appendices only after the unified table is
   stable.
