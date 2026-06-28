# Stage 15 Endogenous Low-Harm Editors Plan

## Objective

Validate the priority plan from `deep-research-report (5).md`: stop treating CGA as a discrete selector problem, keep candidate families as knowledge sources, and test whether bounded residual editors can internalize safety.

## Compact Protocol

- Scope: `ETTm1`, `ETTh1` x `DLinear`, `PatchTST` x horizons `96`, `192` x seed `2026`.
- Parent: frozen HalluGuard-LRBN.
- Source families: Stage 10 CGA source candidates.
- Fit split: validation inner-train.
- Calibration split: validation inner-calib only.
- Test split: final evaluation only.
- Bootstrap: `2000`.
- Leakage: no test threshold tuning.

## First-Round Hypotheses

1. H1 Residual Atom Simplex Editor
   - Convert source family outputs to residual atoms.
   - Use nonnegative simplex mixing with bounded total edit budget.
2. H3 Any-Quantile Residual Envelope
   - Convert residual-distribution signal to config/horizon residual quantile envelope.
   - Use width-aware shrink and bounded center correction.
3. H5 Local-Global Decoupled Sparse Editor
   - Separate smooth global residual correction from sparse local/boundary correction.

## Reserve Compact Hypotheses

The report also names H2/H4/H6 as follow-on architecture directions. Because the first-round hypotheses produced only partial support, Stage 15 adds the two lightweight reserve hypotheses to the same compact run before stopping:

4. H2 Prototype Codebook Local Editor
   - Learn residual patch prototypes from validation inner-train candidate repairs.
   - Apply overlap-add local edits only where target-free patch similarity and boundary support are high.
5. H4 Retrieval-Conditioned Residual Adapter
   - Use validation inner-train nearest-neighbor residuals as bounded conditional guidance.
   - Decay edits by feature-space retrieval confidence.

H6 Self-Supervised Teacher Manifold Projector is not executed in this compact pass because it requires a new masked/Siamese teacher pretraining stack. It is recorded as blocked unless H1/H2/H3/H4/H5 evidence justifies a heavier mini-extension.

## Baselines

- LRBN
- SRA-BP-safe
- SRA-BP-balanced
- SafeTAE-safe from Stage 7 result table
- Stage14 FamilyMix Selector from Stage 14 result table

## Metrics

- MSE / MAE
- MSE delta vs LRBN
- harm rate
- max config harm
- bootstrap CI
- oracle gain fraction
- slice metrics: `q4_boundary`, `high_gap_low_repair`, `non_boundary`, `low_gap_high_repair`, known harmed config
- non-degeneration: `lrbn_equiv_rate`, `active_patch_ratio`, `edit_energy_ratio`, `mean_delta_norm`

## Gates

Safe pass:

- MSE delta <= `-1.8%`
- harm <= `0.02`
- max config harm <= `0.08`
- CI high < `0`
- lrbn_equiv_rate < `0.80`
- active_patch_ratio >= `0.08`
- q4_boundary <= `0`
- known harmed config <= `+0.5%`

Tradeoff pass:

- MSE delta <= `-2.6%`
- harm <= `0.10`
- max config harm <= `0.18`
- CI high < `0`
- lrbn_equiv_rate < `0.70`
- active_patch_ratio >= `0.12`
- q4_boundary <= `0`
- known harmed config <= `+0.5%`

Mechanism pass:

- oracle gain fraction >= `0.08`
- q4_boundary <= `0`
- known harmed config <= `+0.5%`

## Stop Rule

If no first-round or reserve compact hypothesis passes compact gates, do not run mini-extension. Record whether any result is partial support and whether the next line should pivot away from selector-like correction toward a heavier learned representation/teacher experiment.
