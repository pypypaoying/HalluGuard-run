# HalluGuard TableA Full Server Run

This document is the server contract for the complete offline TableA run.

## Main Claim Method

- `HalluGuard-LRBN`
- Variant: `unified_revin_rdn_hybrid`
- Protocol: offline point forecasting post-processing / normalization baseline table
- Leakage contract: validation may calibrate policies; test is evaluation-only.

## Default TableA Matrix

- Datasets: `ETTm1, ETTm2, ETTh1, ETTh2, Weather, Exchange, ECL, Traffic`
- Backbones: `DLinear, PatchTST, iTransformer, TimesNet, TimeMixer, FreTS`
- Horizons: `96, 192, 336, 720`
- Seeds: `2026, 2027, 2028`
- Methods:
  - `raw_no_correction`
  - `HalluGuard-LRBN`
  - `RevIN`
  - `DishTS`
  - `SAN`
  - `NST`
  - `SoP-step-wise`
  - `SoP-variable-wise`
  - `SOLID-official-supported`
  - `matched_sparse_smoothing`
  - `naive_smoothing`
  - `ema_smoothing`
  - `median_smoothing`

`TAFAS` is intentionally not in TableA by default because its target-online protocol uses partially observed future ground truth. It can be run as a separate appendix protocol by passing it explicitly in `METHODS`.

`SOLID-official-supported` is kept in the table matrix. Cells whose official prediction-head adaptation is not yet fairly wired are recorded as explicit `blocked` rows with a reproducible reason rather than silently omitted.

## Server Environment

Create a dedicated environment so official baseline dependencies do not collide with other projects:

```bash
cd /dev_data/jack/workspace/HalluGuard-run
git pull

conda create -n halluguard-tablea python=3.10 -y
conda activate halluguard-tablea

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the server needs a specific CUDA wheel, install PyTorch first using the official command for that machine, then run `pip install -r requirements.txt`.

Before launching the full table, check the installed PyTorch wheel against the
server driver:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

If the run fails with `The NVIDIA driver on your system is too old (found
version 12020)`, the installed PyTorch wheel likely targets a newer CUDA runtime
than the server driver supports. On a CUDA 12.2 driver, reinstall a compatible
wheel before rerunning:

```bash
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
python -m pip install -r requirements.txt
```

`DEVICE=cpu` is acceptable for a wiring smoke test, but not recommended for the
full TableA.

## Full One-Command Run

```bash
cd /dev_data/jack/workspace/HalluGuard-run
conda activate halluguard-tablea

PYTHON_BIN=python \
DEVICE=cuda \
OUTPUT_DIR=experiments/halluguard/results/tablea_full_v1 \
bash scripts/run_tablea_full.sh
```

The default command is intentionally expensive:

- `MAX_TRAIN_WINDOWS=0` means all train windows.
- `MAX_EVAL_WINDOWS=0` means all validation/test windows.
- `EPOCHS=10`
- `SOP_PLUG_EPOCHS=10`
- `BATCH_SIZE=256`
- `LEARNING_RATE=0.001`
- `SAN_PRETRAIN_EPOCHS=5`

## Resume / Skip Existing

If a long run is interrupted:

```bash
cd /dev_data/jack/workspace/HalluGuard-run
conda activate halluguard-tablea

PYTHON_BIN=python \
DEVICE=cuda \
OUTPUT_DIR=experiments/halluguard/results/tablea_full_v1 \
SKIP_EXISTING=1 \
bash scripts/run_tablea_full.sh
```

## Smoke Test

Use this only to verify wiring, not as a scientific result:

```bash
cd /dev_data/jack/workspace/HalluGuard-run
conda activate halluguard-tablea

PYTHON_BIN=python \
DEVICE=cuda \
SMOKE=1 \
OUTPUT_DIR=experiments/halluguard/results/tablea_full_smoke \
bash scripts/run_tablea_full.sh
```

## Logs

Progress prints to stdout. Per-config logs are under:

```text
experiments/halluguard/results/tablea_full_v1/logs/
```

Useful live checks:

```bash
tail -f experiments/halluguard/results/tablea_full_v1/logs/ETTm1_DLinear_96_seed2026_lrbn.log
tail -f experiments/halluguard/results/tablea_full_v1/logs/ETTm1_DLinear_96_seed2026_tablea_adapters.log
```

## Outputs

```text
experiments/halluguard/results/tablea_full_v1/
  run_contract.json
  combined_metrics.csv
  combined_metrics.json
  summary.md
  summary_by_method.csv
  summary_by_backbone.csv
  summary_by_dataset.csv
  logs/
  predictions/
```

`combined_metrics.csv` is the source table for analysis. It contains completed and blocked rows. Blocked rows must be inspected through `blocker_reason`.

## Expected Full Size

Default matrix size:

```text
8 datasets x 6 backbones x 4 horizons x 3 seeds x 13 methods = 7488 rows
```

Rows can be `completed` or `blocked`; there should be no silently missing matrix cells.
