# HalluGuard Run Repository

This repository is a portable run package for HalluGuard experiments. It is
intended to be cloned on a remote server and used to regenerate real-data
prediction tables and HalluGuard evaluation tables.

## Included

- HalluGuard core code under `experiments/halluguard/`.
- Stage 7 to Stage 14 runners and configs.
- ETT-small data under `external/ETDataset/ETT-small/`.
- Lightweight DLinear/PatchTST prediction exporter under
  `external/halluguard_real_pipeline/`.
- A lightweight snapshot of Time-Series-Library under
  `external/Time-Series-Library/` for reference and baseline integration.
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

python experiments/halluguard/run_mvp.py \
  --config experiments/halluguard/configs/halluguard_mvp.yaml \
  --quick
```

For the full server workflow, see `RUN_BASELINE_TABLE.md`.

## Current HalluGuard Parent Lines

- Stage 12 selected `boundary_only` as the conservative external-ready
  HalluGuard-Dynamics variant.
- Stage 13 selected the adaptive rule router as a stronger router over boundary,
  smoothing, and no-correction actions.
- Stage 14 found `s14_smoothing_cap_selective_router` as the best clean/stress
  parent and `s14_stable_smoothing_cap_router` as the best external PatchTST
  harm diagnostic.

Use `BASELINE_PLUGIN_PROTOCOL.md` when comparing HalluGuard against other
top-conference plug-in modules such as RevIN, Dish-TS, SAN, SIN, FAN, DDN, CCM,
LIFT, and TAFAS.
