# Stage17 Sequence Teacher Projection Plan

## Selected Idea

Stage17 tests whether the Stage16 learned-teacher line improves when patch-local editing is replaced by a sequence-level SSL teacher, minimal-norm sequence projection, uncertainty-aware residual envelopes, and global/local structural consistency. LRBN remains the frozen parent forecast; all training/calibration uses validation-only data and test is evaluation-only.

## Run Contract

- Protocol: compact validation only, not TableA.
- Parent baseline: frozen HalluGuard-LRBN predictions from the existing compact assets.
- Datasets: `ETTm1`, `ETTh1`.
- Backbones: `DLinear`, `PatchTST`.
- Horizons: `96`, `192`.
- Seed: `2026`.
- Teacher training source: validation inner-train proxy, because the local compact asset package does not include original model training windows. This deviation is recorded and uses no test labels.
- Corrector fitting: validation inner-train only.
- Policy calibration: validation inner-calib only.
- Evaluation: test only.
- Bootstrap: `2000` for the formal compact run.
- Test threshold leakage: must remain `False`.

## Candidates

- `SL-TMP Sequence Teacher Minimal-Norm Projection`
- `UTRE Uncertainty Teacher Residual Envelope`
- `SSP Structured Sequence Projector`
- `TRAP Teacher-Residual Agreement Projector`
- `IMDR Iterative Minimal-Norm Denoising Refiner`

## Baseline References

- `LRBN`
- `SRA-BP-safe`
- `SRA-BP-balanced`
- `SafeTAE-safe`
- `Stage14 FamilyMix Selector`
- `Stage15 H1/H2`
- `Stage16 H6-safe / H2H6`

## Success Gates

Safe pass:

- MSE delta vs LRBN `<= -1.8%`
- harm `<= 0.025`
- max config harm `<= 0.08`
- bootstrap CI high `< 0`
- q4 boundary and non-boundary slice deltas `<= 0`
- known harmed config `<= +0.5%`
- `lrbn_equiv_rate < 0.80`

Tradeoff pass:

- MSE delta vs LRBN `<= -2.6%`
- harm `<= 0.10`
- max config harm `<= 0.18`
- bootstrap CI high `< 0`
- known harmed config `<= +1.0%`

Mechanism pass:

- teacher energy delta / true MSE delta Spearman `>= 0.20`
- residual alignment `A > 1` rate `>= 0.60`
- active patch ratio `>= 0.08`
- edit energy ratio is not near zero

## Outputs

All formal outputs go under `experiments/halluguard/results/stage17_sequence_teacher/`:

- `stage17_config.json`
- `stage17_training_log_teacher.csv`
- `stage17_training_log_correctors.csv`
- `stage17_overall.csv`
- `stage17_per_config.csv`
- `stage17_slice_metrics.csv`
- `stage17_mechanism_metrics.csv`
- `stage17_alignment_metrics.csv`
- `stage17_uncertainty_metrics.csv`
- `stage17_bootstrap_ci.json`
- `stage17_gate_table.csv`
- `stage17_verdict.json`
- `summary.md`

## Revision Log

- 2026-06-28: Created Stage17 compact validation contract from the user-provided validation document and skeleton.
