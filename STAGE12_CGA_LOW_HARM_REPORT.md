# Stage 12 CGA Low-Harm Priority Validation Report

## Scope

This stage implements and validates the priority plan from `deep-research-report (1).md`.

The compact protocol was:

- Datasets: `ETTm1`, `ETTh1`
- Backbones: `DLinear`, `PatchTST`
- Horizons: `96`, `192`
- Seed: `2026`
- Baseline: frozen `HalluGuard-LRBN`
- Candidate pool: Stage 10 CGA deployable candidates
- Calibration: validation-only inner calibration
- Evaluation: test-only
- Bootstrap: `2000`
- Test threshold leakage: `False`

Outputs are in `experiments/halluguard/results/stage12_cga_low_harm/`.

## Implemented Priority Plan

The runner validates four mechanisms in the order recommended by the report:

1. `Sparse-Family-CGA`: sparse top-k family admission with median family representatives.
2. `Sparse-Residual-Simplex-CGA`: adds a score-weighted residual-family quantile simplex over residual candidates.
3. `NoHarm-Selective-CGA`: adds validation-selected expected-harm gating.
4. `LambdaVeto-CGA`: adds boundary smoothing veto plus uncertainty-conditioned lambda shrink.

Baselines included in the same output:

- `LRBN`
- `SRA-BP-balanced`
- `oracle_stage12_cga_full`

## Overall Results

| Variant | MSE | MAE | MSE delta vs LRBN | Harm rate | Max config harm | Coverage | Oracle gain fraction | Bootstrap high raw delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LRBN | 4.894158 | 1.682162 | 0.000000% | 0.000000 | 0.000000 | 0.000000 | NA | 0.000000 |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508% | 0.104167 | 0.197917 | 0.000000 | 0.069900 | -0.099202 |
| oracle_stage12_cga_full | 3.074767 | 1.322182 | -37.174740% | 0.000000 | 0.000000 | 0.000000 | 1.000000 | -1.564427 |
| Sparse-Family-CGA | 4.823736 | 1.665719 | -1.438899% | 0.251302 | 0.489583 | 0.998698 | 0.038706 | -0.037877 |
| Sparse-Residual-Simplex-CGA | 4.849834 | 1.671598 | -0.905652% | 0.226562 | 0.479167 | 0.998698 | 0.024362 | -0.025583 |
| NoHarm-Selective-CGA | 4.848960 | 1.671547 | -0.923492% | 0.235677 | 0.479167 | 0.998698 | 0.024842 | -0.026707 |
| LambdaVeto-CGA | 4.845183 | 1.670993 | -1.000674% | 0.208333 | 0.458333 | 0.990885 | 0.026918 | -0.031733 |

## Gate Verdict

Status: **compact failed; stop before mini-extension/TableA**.

The best MSE variant was `Sparse-Family-CGA`, but it failed the compact gate:

- MSE delta vs LRBN: `-1.438899%`
- oracle gain fraction: `0.038706`, below the required `0.08`
- max config harm: `0.489583`, far above the allowed `0.18`
- family top-2 hit: `0.619792`, below the target `0.65`
- coverage: `0.998698`, meaning sparse admission still behaved like near-all-sample correction

The final safety-oriented `LambdaVeto-CGA` also failed:

- MSE delta vs LRBN: `-1.000674%`
- oracle gain fraction: `0.026918`
- harm rate: `0.208333`
- max config harm: `0.458333`
- known harmed config delta: `-0.059859%`
- boundary-like worst slice delta: `-0.617803%`

It did repair the specific known harmed config mean and the boundary-like slice mean, but it did not control per-sample or per-config harm enough to pass.

## Mechanism Findings

### 1. Sparse family mixture is directionally useful but still diffuse

`Sparse-Family-CGA` improved average MSE by `-1.438899%`, but coverage remained `99.8698%`. The selected family distribution was still dominated by `residual_distribution,smoothing_teacher`, so the sparse top-k rule did not become a true abstaining safety controller.

### 2. Residual quantile simplex did not improve over median family representative

`Sparse-Residual-Simplex-CGA` was worse than sparse median family by `+0.533247` percentage points in MSE delta. This rejects the first simple score-weighted residual simplex implementation. The failure does not reject residual family itself; it rejects this particular candidate-score softmax simplex.

### 3. No-harm gating failed because harm probabilities are badly calibrated

`NoHarm-Selective-CGA` selected almost the same samples as sparse CGA, with coverage `99.8698%`. Its mean expected harm was only about `0.001471`, while observed test harm was `23.5677%`. This is the clearest diagnostic from the run: the existing harm head has useful ranking signal in Stage 10, but its probability scale is not calibrated enough to drive a safety gate.

### 4. Boundary veto and dynamic lambda fixed the targeted slice but not global safety

`LambdaVeto-CGA` reduced the known harmed config `ETTm1/DLinear/192` from the sparse variant's `+1.133398%` MSE harm to `-0.059859%`, and boundary-like slices were negative. However, per-sample harm remained high and max config harm was still `45.8333%`.

This means slice-aware veto is a useful local safety valve, but it is not sufficient as the whole deployable arbitration mechanism.

## Decision

Do not run the mini-extension.

Do not promote this Stage 12 low-harm CGA line to TableA.

The compact evidence says the next useful work is not another threshold sweep. The blocker is more fundamental: the family/risk heads need calibrated no-harm probabilities or a different selective-risk training target before they can safely control CGA's large oracle space.

