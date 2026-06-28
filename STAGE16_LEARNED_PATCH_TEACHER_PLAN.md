# Stage 16 Learned Patch / Teacher Projector Plan

## Objective

Continue from Stage15's failure mode: static patch prototypes were safe but inactive, and the untrained teacher-manifold direction was blocked. Stage16 tests whether a **learned patch representation** and a **denoising teacher manifold projector** can turn H2/H6 from static ideas into measurable low-harm editors.

## Hypothesis

Frozen LRBN predictions can be improved by patch-local learned editors trained only on validation inner-train data, with calibration selected only on validation inner-calib:

1. H2L Learned Patch Residual Editor
   - Train a patch-level MLP to predict bounded residual corrections from LRBN patch, context-tail patch, and target-free patch features.
2. H6 Denoising Teacher Manifold Projector
   - Train a denoising autoencoder on validation inner-train target patches as a lightweight teacher manifold.
   - At inference, project LRBN prediction patches through the teacher and apply only validation-calibrated small corrections.
3. H2H6 Learned Patch Teacher Hybrid
   - Combine the supervised residual patch direction with the teacher manifold direction under a validation-calibrated mix/shrink/cap/gate.

Bounded safety fix if the first run is mean-positive but high-harm:

4. H6-Safe Sparse Teacher Projector
   - Same teacher direction, but with smaller shrink/cap and high-score sparse patch gates.
5. H2H6-Safe Sparse Learned Teacher Hybrid
   - Same hybrid direction, but with conservative sparse deployment grid only.

## Compact Protocol

- Scope: `ETTm1`, `ETTh1` x `DLinear`, `PatchTST` x horizons `96`, `192` x seed `2026`.
- Parent: frozen HalluGuard-LRBN.
- Fit split: validation inner-train only.
- Calibration split: validation inner-calib only.
- Test split: final evaluation only.
- Bootstrap: `2000`.
- Leakage: no test threshold/lambda/gate tuning.

## Baselines and References

- LRBN
- SRA-BP-safe
- SRA-BP-balanced
- Stage14 FamilyMix Selector
- Stage15 H1 Residual Atom Simplex Editor
- Stage15 static H2 Prototype Codebook Local Editor
- SafeTAE-safe from Stage7 reference table when available

## Metrics and Gates

Primary metrics:

- MSE / MAE
- MSE delta vs LRBN
- harm rate
- max config harm
- bootstrap CI
- oracle gain fraction
- q4 boundary and known harmed config slices
- non-degeneration: `lrbn_equiv_rate`, `active_patch_ratio`, `edit_energy_ratio`, `mean_delta_norm`

Compact safe gate:

- MSE delta <= `-1.8%`
- harm <= `0.02`
- max config harm <= `0.08`
- CI high < `0`
- lrbn_equiv_rate < `0.80`
- active_patch_ratio >= `0.08`
- q4_boundary <= `0`
- known harmed config <= `+0.5%`

Tradeoff gate:

- MSE delta <= `-2.6%`
- harm <= `0.10`
- max config harm <= `0.18`
- CI high < `0`
- lrbn_equiv_rate < `0.70`
- active_patch_ratio >= `0.12`
- q4_boundary <= `0`
- known harmed config <= `+0.5%`

Mechanism gate:

- oracle gain fraction >= `0.08`
- q4_boundary <= `0`
- known harmed config <= `+0.5%`

## Stop Rule

If no learned patch/teacher variant passes compact safe or tradeoff gates, do not run mini-extension. Report whether the failure is because the learned representation is too weak, too harmful, or validation-calibration cannot stabilize it.

## Outputs

- `STAGE16_LEARNED_PATCH_TEACHER_REPORT.md`
- `STAGE16_LEARNED_PATCH_TEACHER_CHECKLIST.md`
- `experiments/halluguard/halluguard_stage16_learned_patch_teacher.py`
- `experiments/halluguard/run_stage16_learned_patch_teacher.py`
- `experiments/halluguard/results/stage16_learned_patch_teacher/`
