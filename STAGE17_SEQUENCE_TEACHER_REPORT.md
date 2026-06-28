# Stage17 Sequence Teacher Projection Report

## Verdict

Stage17 implemented and validated the proposed sequence-level teacher projection line:

- `SL-TMP Sequence Teacher Minimal-Norm Projection`
- `UTRE Uncertainty Teacher Residual Envelope`
- `SSP Structured Sequence Projector`
- reserve `TRAP Teacher-Residual Agreement Projector`
- reserve `IMDR Iterative Minimal-Norm Denoising Refiner`

Result: **failed compact gate; stop before mini-extension / TableA**.

The strongest deployable candidate was `TRAP`, but its gain was far below the Stage16 frontier and the compact gates:

- MSE `4.890483`
- MAE `1.681058`
- MSE delta vs LRBN `-0.075090%`
- harm `0.087240`
- max config harm `0.218750`
- oracle gain fraction `0.002020`
- bootstrap CI high raw delta `-0.001371`

No Stage17 candidate passed safe, tradeoff, or mechanism gates.

## Protocol

- parent: frozen HalluGuard-LRBN
- datasets: `ETTm1`, `ETTh1`
- backbones: `DLinear`, `PatchTST`
- horizons: `96`, `192`
- seed: `2026`
- samples: `768` test rows
- teacher training: validation inner-train proxy
- corrector fitting: validation inner-train only
- calibration: validation inner-calib only
- evaluation: test only
- bootstrap: `2000`
- test threshold leakage: `False`

The local compact package does not include original forecast-model training windows. Teacher training therefore used validation inner-train and this deviation is recorded in `stage17_config.json`; no test labels were used for teacher training, policy selection, thresholds, caps, or shrink.

## Overall Results

| variant | MSE | MAE | MSE delta vs LRBN | harm | max config harm | oracle gain | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LRBN | 4.894158 | 1.682162 | 0.000000% | 0.000000 | 0.000000 | NA | 0.000000 |
| SRA-BP-safe | 4.813149 | 1.660950 | -1.655217% | 0.035156 | 0.114583 | 0.044525 | 0.000000 |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508% | 0.104167 | 0.197917 | 0.069900 | 0.000000 |
| SafeTAE-safe (Stage7) | 4.804843 | 1.660837 | -1.824928% | 0.018229 | NA | 0.100424 | 0.519531 |
| Stage16 H6-safe | 4.844570 | 1.672768 | -1.013201% | 0.016927 | 0.041667 | 0.027255 | 0.102865 |
| Stage16 H2H6 hybrid | 4.806725 | 1.665870 | -1.786478% | 0.105469 | 0.187500 | 0.048056 | 0.378906 |
| SL-TMP | 4.892852 | 1.681732 | -0.026667% | 0.108073 | 0.239583 | 0.000717 | 0.235677 |
| UTRE | 4.992662 | 1.680256 | +2.012688% | 0.316406 | 0.437500 | -0.054141 | 0.645833 |
| SSP | 4.895090 | 1.682190 | +0.019053% | 0.075521 | 0.354167 | -0.000513 | 0.141927 |
| TRAP | 4.890483 | 1.681058 | -0.075090% | 0.087240 | 0.218750 | 0.002020 | 0.225260 |
| IMDR | 4.891470 | 1.681722 | -0.054922% | 0.005208 | 0.031250 | 0.001477 | 0.028646 |

## Mechanism Findings

The teacher signal was not completely random, but it was not useful enough as a correction mechanism:

- `TRAP` had teacher-energy / MSE-delta Spearman `0.283821`, which is the best non-trivial mechanism signal among active candidates.
- `IMDR` had Spearman `0.630225`, but it achieved this by barely editing: `lrbn_equiv_rate 0.971354`, `active_patch_ratio 0.034722`.
- `TRAP` residual alignment `A > 1` rate was only `0.138021`, far below the required `0.60`.
- `SL-TMP` and `SSP` produced weak or harmful directions; `UTRE` widened coverage but caused large harm.
- Teacher-energy improvement magnitudes were tiny: TRAP mean teacher-energy delta was `+0.000005`, not a real projection improvement.

This means the teacher manifold learned in compact validation can rank a few easy no-harm refinements, but cannot supply a sufficiently aligned residual direction.

## Slice Results

`TRAP` was the best balanced Stage17 candidate:

- q4 boundary delta: `-0.164906%`
- non-boundary delta: `-0.046897%`
- known harmed config `ETTm1/DLinear/192`: `-0.060871%`
- low-gap/high-repair slice: `+0.016393%`

`IMDR` was safer but almost inactive:

- harm `0.005208`
- max config harm `0.031250`
- q4 boundary delta `+0.015107%`
- non-boundary delta `-0.076904%`
- active patch ratio `0.034722`

`UTRE` failed the uncertainty hypothesis directly:

- overall MSE worsened `+2.012688%`
- q4 boundary worsened `+2.869220%`
- known harmed config worsened `+0.478881%`
- harm `0.316406`

## Gate Results

No candidate passed:

- safe gate: none
- tradeoff gate: none
- mechanism gate: none

The closest safe-looking candidate was `IMDR`, but it is rejected because it is essentially an LRBN-equivalent tiny edit and misses the required active patch ratio. The closest active candidate was `TRAP`, but it misses MSE, max-harm, oracle-capture, and residual-alignment gates.

## Interpretation

Stage17 answers the Stage16 follow-up cleanly:

1. Sequence-level structure alone did not recover the Stage16 strong residual gains.
   - Stage16 `H2H6` reached `-1.786478%`.
   - Stage17 best reached only `-0.075090%`.

2. Uncertainty-aware quantile residual editing did not control harm.
   - `UTRE` increased coverage and edit energy, but moved in wrong directions on many configs.

3. Agreement projection helped safety relative to raw residual editing, but collapsed most gain.
   - `TRAP` is the only useful Stage17 direction, yet its gain is too small and max config harm is too high.

4. The current compact teacher is still too weak.
   - Teacher energy has some rank signal, but the projection vector is not aligned enough with true residuals.

## Decision

Do **not** promote Stage17 to mini-extension or TableA.

Keep the artifacts as negative evidence. The teacher-manifold line should be paused unless a stronger teacher can be trained on true original training windows or a much richer external sequence corpus. The near-term main claim should remain outside this line: LRBN clean-claim / Stage14 selective router lines remain stronger for performance, and Stage16 remains the better teacher checkpoint.

## Artifacts

- Results directory: `experiments/halluguard/results/stage17_sequence_teacher/`
- Overall metrics: `experiments/halluguard/results/stage17_sequence_teacher/stage17_overall.csv`
- Gate table: `experiments/halluguard/results/stage17_sequence_teacher/stage17_gate_table.csv`
- Mechanism metrics: `experiments/halluguard/results/stage17_sequence_teacher/stage17_mechanism_metrics.csv`
- Slice metrics: `experiments/halluguard/results/stage17_sequence_teacher/stage17_slice_metrics.csv`
- Generated summary: `experiments/halluguard/results/stage17_sequence_teacher/summary.md`
