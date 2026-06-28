# Stage18 Performance Atom Extraction Plan

## Selected Idea

Stage18 treats the strong TAE/CGA oracle as a performance microscope rather than a deployable selector. The goal is to diagnose whether the oracle gain that remains after `SRA-BP-balanced` can be compressed into a few stable, interpretable, low-harm performance atoms that complement SRA-BP.

## Run Contract

- Tier: compact mechanism diagnosis, not TableA.
- Parent methods: `LRBN`, `SRA-BP-safe`, `SRA-BP-balanced`.
- Main parent for atom extraction: `SRA-BP-balanced`.
- Candidate pools:
  - `tae_old_pool`: deployable old/TAE candidates from the existing compact assets.
  - `cga_new_family_pool`: Stage10 new families only (`residual_distribution`, `smoothing_teacher`, `retrieval_memory`), plus the current parent.
  - `union_full_pool`: all deployable old + Stage10 CGA candidates, plus the current parent.
- Dataset scope: `ETTm1`, `ETTh1` x `DLinear`, `PatchTST` x horizons `96`, `192`.
- Seed: `2026`.
- Calibration and distillation: validation inner-train / inner-calib only.
- Evaluation: test only.
- Bootstrap: `2000`.
- Test threshold leakage: must remain `False`.

## Required Experiments

1. Incremental oracle over LRBN/SRA parents.
2. Family leave-one-out and only-one-family oracle.
3. Correction vector decomposition with PCA and KMeans atoms.
4. Atom residual alignment and harm profile.
5. SRA complementarity matrix by slices and early/mid/late regions.
6. Atom distillability diagnostics using target-free features.
7. Prototype atom composition only as a compact diagnostic if earlier gates justify it.

## Success / Stop Rules

Enter prototype / next-stage consideration if any of these hold:

- Union oracle over `SRA-BP-balanced` improves MSE by at least `5%`.
- A family leave-one-out degradation is at least `2%`.
- Top-5 atom PCA explained variance is at least `60%`.
- At least one atom has `A>1 rate >= 0.60` and slice harm `<= 0.08`.

Stop the atom route if any hard negative dominates:

- Oracle over `SRA-BP-balanced` is below `2%`.
- Correction vectors have no stable low-dimensional structure.
- All atoms have `A>1 rate < 0.55`.
- Distilled atoms collapse to LRBN/SRA-equivalent tiny edits.
- Known harmed config repeatedly exceeds `10%` harm.

Mini-extension consideration requires a deployable prototype atom relative to `SRA-BP-balanced` with:

- MSE improvement at least `0.8%`.
- harm `<= 0.05`.
- max config harm `<= 0.10`.
- bootstrap CI high `< 0`.
- at least one non-SRA-main slice improvement.

## Outputs

All formal outputs go under `experiments/halluguard/results/stage18_performance_atom_diagnosis/`:

- `stage18_config.json`
- `parent_oracle_table.csv`
- `family_leave_one_out.csv`
- `only_family_oracle.csv`
- `oracle_selected_candidates.csv`
- `correction_vectors.parquet`
- `atom_pca_report.csv`
- `atom_cluster_report.csv`
- `atom_alignment_report.csv`
- `atom_slice_profile.csv`
- `sra_complementarity_matrix.csv`
- `atom_distillability_report.csv`
- `prototype_atom_metrics.csv`
- `bootstrap_ci.json`
- `stage18_verdict.json`
- `summary.md`

## Revision Log

- 2026-06-28: Created compact mechanism-diagnosis contract from the Stage18 validation document.
