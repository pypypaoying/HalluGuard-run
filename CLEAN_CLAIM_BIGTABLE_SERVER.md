# HalluGuard-LRBN Clean Claim BigTable Server Run

This is the one-command server entry for the clean-claim method:

```text
HalluGuard-LRBN unified_revin_rdn_hybrid
```

The exploratory NST/future-center lines are not part of the main claim table.

## One Command

```bash
bash scripts/run_clean_claim_bigtable.sh
```

Outputs:

```text
experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/combined_metrics.csv
experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/combined_metrics.json
experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/summary.md
experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/summary_by_method.csv
experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/summary_by_backbone.csv
experiments/halluguard/results/lrbn_clean_claim_bigtable_v1/summary_by_dataset.csv
```

## Default Matrix

- Datasets: `ETTm1, ETTm2, ETTh1, ETTh2, Weather, ECL, Traffic`
- Backbones: `DLinear, PatchTST, iTransformer, TimesNet, TimeMixer, Nonstationary_Transformer`
- Horizons: `96,192,336,720`
- Seeds: `2026,2027,2028`
- Methods:
  - `raw_no_correction`
  - `HalluGuard-LRBN`
  - `matched_sparse_smoothing`
  - `naive_smoothing`
  - `ema_smoothing`
  - `median_smoothing`
  - `RevIN`
  - `DishTS`
  - `SAN`
  - `NST`
  - `TAFAS`

Every requested row is written as `completed` or `blocked`. The current
lightweight in-repo exporter completes `DLinear/PatchTST` on `ETTm1/ETTh1` and
records the wider dataset/backbone rows as blocked until the corresponding
official adapters are connected. Smoothing controls are post-hoc baselines
generated from the same raw prediction files; `matched_sparse_smoothing`
calibrates its trigger only on `split="val"`.

## Smoke

```bash
EXTRA_FLAGS="--smoke" EPOCHS=1 MAX_TRAIN_WINDOWS=128 MAX_EVAL_WINDOWS=32 \
  bash scripts/run_clean_claim_bigtable.sh
```

## Full Server Knobs

```bash
DEVICE=cuda \
EPOCHS=10 \
MAX_TRAIN_WINDOWS=8192 \
MAX_EVAL_WINDOWS=1024 \
OUTPUT_DIR=experiments/halluguard/results/lrbn_clean_claim_bigtable_v1 \
  bash scripts/run_clean_claim_bigtable.sh
```

To restrict the run:

```bash
DATASETS=ETTm1,ETTh1 \
BACKBONES=DLinear,PatchTST \
SEEDS=2026,2027,2028 \
  bash scripts/run_clean_claim_bigtable.sh
```

## Fairness Contract

- Same `(dataset, backbone, horizon, seed)` across methods.
- Same `seq_len=96`, train/val/test split, optimizer, batch size, learning rate,
  epoch budget, and train/eval window caps.
- Train split fits all model and normalization parameters.
- Test split is final evaluation only.
- `test_threshold_leakage=False` for all completed rows.

## Claim Interpretation

The clean claim is not that LRBN beats every possible official model. The claim
under this table is:

> HalluGuard-LRBN is a simple learnable reversible boundary-normalization layer
> that improves raw forecasters and is competitive with normalization/adaptation
> baselines under a shared evaluation contract.

Unsupported rows are blockers for framework coverage, not hidden failures.
