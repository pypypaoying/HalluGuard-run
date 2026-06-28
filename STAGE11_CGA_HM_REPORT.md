# Stage 11 CGA-HM Validation Report

## Scope

This report validates the first stage requested by `deep-research-report (2).md`: a compact mechanism test for CGA-HM, a family-level soft-mixture and harm-aware admission layer over the Stage 10 CGA candidate pool.

The validation followed the compact protocol:

- Datasets: `ETTm1`, `ETTh1`
- Backbones: `DLinear`, `PatchTST`
- Horizons: `96`, `192`
- Seed: `2026`
- Baseline: frozen `HalluGuard-LRBN`
- Calibration: validation-only inner calibration
- Evaluation: test-only
- Bootstrap: `2000`
- Test threshold leakage: `False`

Outputs are in `experiments/halluguard/results/stage11_cga_hm/`.

## Implemented Mechanism

CGA-HM was implemented as:

- shared safe base expert: LRBN remains in the mixture with a validation-selected safe weight;
- family-level admission: smoothing-teacher, residual-distribution, and retrieval-memory families are admitted by validation-fitted CGA gain/harm heads;
- family mixture: selected families are softly mixed with the safe base;
- residual-quantile bank: Stage 10 residual-distribution candidates remain available as a family;
- optional boundary veto: validation-calibrated boundary veto variants suppress smoothing-teacher mass on boundary-risk samples.

For this compact run, the family representative was `median` within each family. This preserves the family-level mixture test while avoiding an expensive per-sample candidate-soft calibration loop.

## Overall Results

| Variant | MSE | MAE | MSE delta vs LRBN | Harm rate | Max config harm | Oracle gain fraction | Coverage | Leakage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| LRBN | 4.894158 | 1.682162 | 0.000000% | 0.000000 | 0.000000 | NA | 0.000000 | False |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508% | 0.104167 | 0.197917 | NA | 0.000000 | False |
| oracle_stage11_cga_full | 3.074767 | 1.322182 | -37.174740% | 0.000000 | 0.000000 | 1.000000 | 0.000000 | False |
| CGA-HM-safe | 4.841243 | 1.670342 | -1.081187% | 0.204427 | 0.479167 | 0.029084 | 0.998698 | False |
| CGA-HM-balanced | 4.841243 | 1.670342 | -1.081187% | 0.204427 | 0.479167 | 0.029084 | 0.998698 | False |
| CGA-HM-veto-safe | 4.839807 | 1.670458 | -1.110515% | 0.199219 | 0.468750 | 0.029873 | 0.990885 | False |
| CGA-HM-veto-balanced | 4.839807 | 1.670458 | -1.110515% | 0.199219 | 0.468750 | 0.029873 | 0.990885 | False |

The best deployable policy is `CGA-HM-veto-safe`:

- MSE: `4.839807`
- MAE: `1.670458`
- MSE delta vs LRBN: `-1.110515%`
- harm rate: `0.199219`
- max config harm: `0.468750`
- oracle gain fraction: `0.029873`
- test threshold leakage: `False`

## Stage 1 Gate

The required go/no-go gate was:

- oracle gain fraction >= `0.08`
- max config harm <= `0.18`
- MSE delta vs LRBN negative
- no test threshold leakage

Result: **failed**.

Failure reason from `stage11_verdict.json`:

```text
best deployable policy failed gate: oracle_gain_fraction=0.029873, max_config_harm=0.468750, mse_delta_pct_vs_lrbn=-1.110515
```

The MSE direction is positive, but the mechanism does not capture enough of the Stage 10 oracle gap and the worst-config harm is far above the allowed limit.

## Failure Analysis

The main problem is not candidate generation. The full CGA oracle remains strong at MSE `3.074767`, or `-37.174740%` vs LRBN. The failure is deployable arbitration.

The best CGA-HM policy selects almost every sample:

- coverage: `0.990885`
- selected count: `761 / 768`
- mean safe weight: `0.801823`
- mean smoothing-teacher weight: `0.061220`
- mean residual-distribution weight: `0.095167`
- mean retrieval-memory weight: `0.041790`

This broad coverage gives a small average MSE improvement but does not behave like a precise harm-aware selector. The worst harmed config is `ETTm1 / DLinear / 192`, where MSE is harmed by `+0.496038%` and per-sample harm rate reaches `0.468750`.

## Decision

Per the research plan, Stage 1 failure means:

- do **not** run the mini-extension;
- do **not** run the TableA candidate stage;
- do **not** promote CGA-HM as a clean-claim method.

CGA-HM remains useful as a negative mechanism result: family-level mixture is directionally helpful on mean MSE, but this version is too diffuse and captures only `2.9873%` of the oracle gap. A future attempt would need a substantially sharper admission mechanism, not just another safe-floor or threshold sweep.

