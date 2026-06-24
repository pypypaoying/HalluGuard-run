# HalluGuard Run Repository

This repository is a portable run package for the frozen HalluGuard core table.
It is intended to be cloned on a remote server and used to regenerate
real-data prediction tables, fixed HalluGuard-SP evaluations, and same-position
test-time/adaptation baseline comparisons.

## Included

- HalluGuard core code under `experiments/halluguard/`.
- Stage 7 to Stage 14 runners and frozen core-table configs.
- ETT-small data under `external/ETDataset/ETT-small/`.
- Lightweight DLinear/PatchTST prediction exporter under
  `external/halluguard_real_pipeline/`.
- A lightweight snapshot of Time-Series-Library under
  `external/Time-Series-Library/` for reference and baseline integration.
- Dataset and official baseline repo fetch scripts under `scripts/`.
- Core-table method manifest under `docs/core_table_manifest.yaml`.
- Research reports and candidate ledger through Stage 14.

Generated outputs are intentionally not committed. They are large and are
recreated by the commands in `RUN_BASELINE_TABLE.md`.

## Quick Start

```bash
git clone https://github.com/pypypaoying/HalluGuard-run.git
cd HalluGuard-run

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Or use the prepared setup script:

```bash
bash scripts/setup_env.sh
conda activate halluguard-run  # when conda is available
source .venv/bin/activate      # when the script falls back to venv
```

Then run:

```bash
python experiments/halluguard/run_mvp.py \
  --config experiments/halluguard/configs/halluguard_mvp.yaml \
  --quick
```

For the full server workflow, see `RUN_BASELINE_TABLE.md`.

To run the unified 12-method same-configuration table:

```bash
bash scripts/run_core12_table.sh
```

The final table is written to
`experiments/halluguard/results/core_table/core12_combined/`.

## Frozen HalluGuard Lines

- Main method: `HalluGuard-SP frozen`
  (`s14_smoothing_cap_selective_router`).
- Safety ablation: `HalluGuard stable-harm ablation`
  (`s14_stable_smoothing_cap_router`).
- Historical Stage 12/13/14 reports are kept for provenance, but the server run
  path should use the frozen configs in
  `experiments/halluguard/configs/halluguard_core_table_*.yaml`.

Use `BASELINE_PLUGIN_PROTOCOL.md` when comparing HalluGuard against the fixed
12-method core table:

```text
raw_no_correction
HalluGuard-SP frozen
HalluGuard stable-harm ablation
matched_sparse_smoothing
naive_smoothing
ema_smoothing
median_smoothing
RevIN
Dish-TS
SAN
Non-stationary Transformer / NST
TAFAS
```

The official repos are downloaded with:

```bash
bash scripts/fetch_plugin_repos.sh
```

Core datasets are downloaded with:

```bash
python scripts/fetch_core_datasets.py --datasets core
```
