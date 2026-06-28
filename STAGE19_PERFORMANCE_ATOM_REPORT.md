# Stage19 Residual Quantile / Non-Boundary Shape Atom Validation Report

## Verdict

Stage19 completed the compact validation requested in `halluguard_stage19_performance_atom_validation_doc.md`.

Result: **compact failed; do not promote RQA/NBSA continuous atom adapters to mini-extension or TableA.**

The Stage18 oracle diagnosis remains valid: residual-distribution and smoothing-teacher families still contain large post-SRA oracle targets. However, the Stage19 deployable continuous adapters could not convert those targets into a safe test-time improvement over `SRA-BP-balanced`.

## Protocol

- Parent: `SRA-BP-balanced`
- References: `LRBN`, `SRA-BP-safe`
- Datasets: `ETTm1`, `ETTh1`
- Backbones: `DLinear`, `PatchTST`
- Horizons: `96`, `192`
- Seed: `2026`
- Test rows: `768`
- Split contract:
  - inner-train fits atom bases and coefficient models
  - inner-calib selects shrink/cap/coverage/segment policy
  - test only evaluates
- Bootstrap: `2000`
- Test threshold leakage: `False`

## Family Oracle Targets

The residual atom targets are still present:

| split | family | non-parent selection | mean target gain vs SRA | mean target norm |
| --- | --- | ---: | ---: | ---: |
| inner_train | residual_distribution | 0.888060 | -1.171984 | 3.395458 |
| inner_train | smoothing_teacher | 0.787313 | -0.674076 | 3.974518 |
| inner_calib | residual_distribution | 0.905172 | -1.240111 | 3.704988 |
| inner_calib | smoothing_teacher | 0.784483 | -0.703242 | 4.114774 |
| test | residual_distribution | 0.845052 | -0.685877 | 3.680088 |
| test | smoothing_teacher | 0.729167 | -0.512017 | 5.564195 |

This means Stage19 did not fail because the oracle atom source disappeared. It failed because the deployable continuous adapter could not safely recover the oracle action.

## Compact Variant Results

| variant | MSE delta vs SRA | harm vs SRA | max config harm | coverage | gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `RQA-PCA-Coef` | 0.000000% | 0.000000 | 0.000000 | 1.000000 | fail |
| `RQA-DCT-Coef` | 0.000000% | 0.000000 | 0.000000 | 1.000000 | fail |
| `RQA-HarmAwareCoef` | 0.000000% | 0.000000 | 0.000000 | 1.000000 | fail |
| `NBSA-DCT-Shape` | 0.000000% | 0.000000 | 0.000000 | 1.000000 | fail |
| `NBSA-RoughnessAdapter` | 0.000000% | 0.000000 | 0.000000 | 1.000000 | fail |
| `NBSA-NonBoundaryOnly` | 0.000000% | 0.000000 | 0.000000 | 0.747396 | fail |
| `NBSA-LocalShapeEnvelope` | 0.000000% | 0.000000 | 0.000000 | 1.000000 | fail |
| `RQA-QuantileHead` | +0.324914% | 0.149740 | 0.281250 | 0.313802 | fail |
| `RQA+NBSA` | 0.000000% | 0.000000 | 0.000000 | 1.000000 | fail |

Most coefficient variants were pushed to zero update by the validation-only harm-aware policy. This is a negative result, not a bug: nonzero settings existed in the calibration grid, but the validation objective found them too harmful or unstable to promote.

`RQA-QuantileHead` was the only nonzero deployed test variant. It looked useful on calibration (`-0.551496%`) but failed on test (`+0.324914%`) with high harm (`0.149740`) and max config harm (`0.281250`).

## Where RQA-QuantileHead Failed

Per-config results show that the nonzero residual quantile atom helped PatchTST on ETTh1 but hurt most DLinear settings and ETTm1 PatchTST:

| dataset | backbone | horizon | MSE delta vs SRA |
| --- | --- | ---: | ---: |
| ETTh1 | DLinear | 96 | +0.719587% |
| ETTh1 | DLinear | 192 | +1.518007% |
| ETTh1 | PatchTST | 96 | -1.037026% |
| ETTh1 | PatchTST | 192 | -0.864334% |
| ETTm1 | DLinear | 96 | +1.418029% |
| ETTm1 | DLinear | 192 | +1.170570% |
| ETTm1 | PatchTST | 96 | +0.386104% |
| ETTm1 | PatchTST | 192 | +0.305760% |

Slice diagnostics explain the gate failure:

- `non_boundary`: `-0.451150%`
- `amplitude_mismatch`: `-0.663721%`
- `q4_boundary`: `+3.010081%`
- `high_gap_low_repair`: `+2.724328%`
- `known_harmed_config`: `+1.170570%`

So the residual quantile atom contains a real non-boundary benefit, but the current deployable rule leaks harm into boundary-like and known-harmed regions.

## NBSA Result

NBSA coefficient fit looked more learnable than RQA:

- `NBSA-DCT-Shape` coefficient R2: `0.202890`
- coefficient sign accuracy: `0.637946`

But the validation policy selected zero shrink for all NBSA variants. This means target-free NBSA coefficients were not reliable enough under the compact harm gate, despite the Stage18 smoothing-teacher oracle signal.

## Combination Result

The combined `RQA+NBSA` adapter also selected zero effective update.

Calibration did contain higher-gain nonzero combinations, for example `-1.047517%`, but they had unacceptable harm:

- harm rate: `0.413793`
- max config harm: `0.517241`

Therefore the combination did not pass the structural-complementarity gate.

## Decision

Stage19 answer:

- RQA deployable adapter: **fail**
- NBSA deployable adapter: **fail**
- RQA+NBSA deployable combination: **fail**
- Mini-extension: **not run**, because compact gates did not pass
- TableA promotion: **no**

The important scientific update is narrow but useful:

1. Stage18 oracle atoms are real.
2. Fixed prototype application failed.
3. Stage19 continuous coefficient application also failed under low-harm compact gates.
4. The obstacle is not basis compression; it is harm-localization and slice-specific application.

## Recommended Next Route

Do not continue by adding another generic selector or larger coefficient head.

If this line is revisited, the next plausible direction should explicitly isolate the failure slices:

- residual quantile atom only in non-boundary / amplitude-mismatch slices;
- hard veto on q4-boundary and known-harmed configs;
- per-backbone or per-regime calibration only if justified as validation-only and not a dataset shortcut;
- or abandon deployable performance atoms and keep Stage18 as mechanism diagnosis.

Given this compact failure, the current main line should remain `SRA-BP-balanced` / LRBN-derived clean-claim work, not Stage19 atoms.

## Artifacts

- Results directory: `experiments/halluguard/results/stage19_performance_atom_validation/`
- Verdict: `experiments/halluguard/results/stage19_performance_atom_validation/stage19_verdict.json`
- Variant metrics: `experiments/halluguard/results/stage19_performance_atom_validation/compact_variant_metrics.csv`
- Calibration grid: `experiments/halluguard/results/stage19_performance_atom_validation/calibration_grid.csv`
- Slice metrics: `experiments/halluguard/results/stage19_performance_atom_validation/compact_slice_metrics.csv`
- Segment metrics: `experiments/halluguard/results/stage19_performance_atom_validation/compact_segment_metrics.csv`
