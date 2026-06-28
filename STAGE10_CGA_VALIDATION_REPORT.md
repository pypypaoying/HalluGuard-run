# Stage 10 CGA Validation Report

## Objective

Stage 10 validated **HalluGuard-CGA: Candidate Generation + Hierarchical Arbitration** after the Stage 9 finding that the ceiling is candidate-pool and arbitration limited.

This was a compact mechanism validation, not a TableA submission.

## Protocol

- Parent: frozen `HalluGuard-LRBN`.
- Scope: `ETTm1`, `ETTh1` x `DLinear`, `PatchTST` x horizons `96`, `192`, seed `2026`.
- Calibration: validation-only inner train/calib.
- Evaluation: test-only.
- Bootstrap: `2000`.
- Output directory: `experiments/halluguard/results/stage10_cga`.
- Test threshold leakage: `False`.

Run command:

```powershell
python experiments\halluguard\run_stage10_cga.py --output-dir experiments\halluguard\results\stage10_cga --n-bootstrap 2000
```

## Verdict

Status: `mechanism_pass_selector_still_insufficient`.

Stage 10 strongly validates the **candidate generation** hypothesis, but does not yet produce a deployable CGA selector. The three-family candidate pool opens much larger oracle space, and family-level suitability is much easier than exact candidate selection. However, the selected Safe-CGA/Balanced-CGA policy still has too much worst-config harm and captures too little of the oracle gain.

## Mechanism Evidence

Restricted oracle:

| pool | MSE | MSE delta vs LRBN |
| --- | ---: | ---: |
| old deployable oracle | 4.217668 | -13.822391% |
| Stage 9 expanded oracle | 3.908520 | -20.139072% |
| Stage 10 CGA full oracle | 3.074767 | -37.174740% |

Stage 10 oracle improves the old deployable oracle by `-27.097931%`, well beyond the `5%` mechanism gate.

New-family oracle share is `60.15625%`:

- `residual_distribution`: `35.2865%`
- `smoothing_teacher`: `19.9219%`
- `retrieval_memory`: `4.9479%`

Top oracle candidates include:

- `residual_q25`: `24.3490%`
- `teacher_naive_smoothing`: `10.4167%`
- `residual_q75`: `8.2031%`
- `teacher_ema_smoothing`: `5.0781%`
- `teacher_median_smoothing`: `3.3854%`
- `residual_memory_knn_median`: `2.6042%`
- `residual_memory_knn_weighted`: `2.3438%`

Family selector evidence:

- family top-2 hit: `0.619792`
- candidate top-2 hit: `0.093750`
- family minus candidate top-2: `52.6042pp`
- candidate gain AUROC on test: `0.782210`
- candidate harm AUROC on test: `0.745531`

Interpretation: representation is good enough to identify useful **families**, but not the exact trajectory candidate. This supports Stage 10 H2 and points toward family-level mixture/blending rather than hard candidate selection.

## Deployable CGA Results

Safe-CGA and Balanced-CGA selected the same policy in this compact grid:

- `tau_leave=0.55`
- `tau_family_gain=0.55`
- `tau_family_harm=0.15`
- `tau_candidate_gain=0.55`
- `tau_candidate_harm=0.15`
- `lambda_existing=1.0`
- `lambda_smoothing=1.0`
- `lambda_residual=0.75`
- `lambda_memory=0.75`

Test result:

| method | MSE | MSE delta vs LRBN | harm | max config harm | improved configs | oracle gain fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LRBN | 4.894158 | 0.000000% | 0.000000 | 0.000000 | 0/8 | NA |
| SRA-BP-balanced | 4.766983 | -2.598508% | 0.104167 | 0.197917 | 8/8 | NA |
| Safe-CGA | 4.831768 | -1.274774% | 0.052083 | 0.260417 | 7/8 | 0.034291 |
| Balanced-CGA | 4.831768 | -1.274774% | 0.052083 | 0.260417 | 7/8 | 0.034291 |
| Stage 10 full oracle | 3.074767 | -37.174740% | 0.000000 | 0.000000 | 8/8 | 1.000000 |

Bootstrap for Safe/Balanced CGA mean raw MSE delta: `[-0.178399, 0.050802]`; the upper bound is positive, so the deployable CI gate fails.

## Slice and Failure Analysis

Safe/Balanced CGA improves overall and non-boundary slices but harms the boundary-heavy slices:

- overall: `-1.274774%`
- non-boundary: `-2.052579%`
- low-gap/high-repair: `-5.647936%`
- high-gap/low-repair: `+0.946762%`
- q4 boundary: `+1.203108%`
- known harmed config: `+13.482354%`

The main failure is concentrated in `ETTm1 / DLinear / 192`:

- per-config MSE delta: `+13.482354%`
- harm rate: `0.260417`
- coverage: `0.500000`

This means the current selector leaves too much high-risk smoothing/residual action active in a config where LRBN/SRA-style boundary behavior is safer.

## Hypothesis Outcomes

| ID | Outcome | Evidence |
| --- | --- | --- |
| H1 candidate families expand oracle space | pass | Stage 10 oracle MSE `3.074767`, `-27.097931%` vs old deployable oracle; new families `60.15625%` oracle share. |
| H2 family suitability easier than exact candidate selection | pass | family top-2 `0.619792` vs candidate top-2 `0.093750`. |
| H3 smoothing-teacher candidates useful | pass as oracle, unsafe as deployed | smoothing teacher `19.9219%` oracle share, but deployed policy harms boundary/q4 slices. |
| H4 residual-distribution candidates useful | pass as oracle | residual_distribution `35.2865%` oracle share; `residual_q25` is the largest oracle candidate. |
| H5 retrieval-memory candidates useful | weak/partial | retrieval_memory `4.9479%` oracle share; present but not dominant. |
| H6 hierarchical arbitration deployable | fail | MSE improves only `-1.274774%`, max config harm `0.260417`, oracle gain fraction `0.034291`. |

## Recommendation

Do not promote current Safe-CGA/Balanced-CGA to TableA.

Promote **CGA candidate generation** as the next architecture direction. The next experiment should not add more candidates blindly; it should redesign arbitration:

1. Use family-level routing as the primary decision because family top-2 is much stronger than candidate top-2.
2. Replace hard candidate selection with family-level mixtures or convex blending.
3. Add config/slice harm guards learned on validation, especially for the `ETTm1 / DLinear / 192` failure pattern.
4. Introduce a boundary-slice veto before allowing smoothing-teacher candidates.
5. Treat residual q25/q75 as high-value candidate generators, but require per-family harm calibration before deployment.

The likely ceiling-breaking direction is **family-level CGA with harm-aware mixture weights**, not the current exact-candidate selector.

## Artifacts

All required Stage 10 outputs were generated under `experiments/halluguard/results/stage10_cga/`; `stage10_output_completeness.csv` reports 19/19 artifacts present.

