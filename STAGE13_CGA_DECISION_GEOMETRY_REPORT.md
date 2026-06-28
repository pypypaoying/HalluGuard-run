# Stage 13 CGA Decision-Geometry Validation Report

## Scope

This stage implements the validation plan from `deep-research-report (3).md`. The report argued that the failure mode is not residual correction or safety itself, but a poor decision geometry: exact candidate hard choice, marginally calibrated safety, and sample-level gates. Stage 13 therefore tested five architecture-level alternatives under the same compact protocol.

Compact protocol:

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

Outputs are in `experiments/halluguard/results/stage13_cga_decision_geometry/`.

## Implemented Candidates

1. `Residual-Prior Convex Mixer`
   - Family residual priors are mixed inside a bounded convex hull around LRBN.
2. `Time-Step Gated Hybrid Editor`
   - Smoothing and local/boundary edits are mixed at time-step level.
3. `Selection-Conditional Conformal Family Editor`
   - Per-family selected-subset bias centering and dead-zone editing.
4. `Retrieval-Augmented Local Residual Editor`
   - Retrieval/memory is used as a local residual prior with agreement gating.
5. `Conservative Challenger Comparator`
   - Family recall proposes a challenger; a conservative pairwise rule partially accepts it.

## Overall Results

| Variant | MSE | MAE | MSE delta vs LRBN | Harm rate | Max config harm | Coverage | Oracle capture | Accept precision | Selected non-harm | Bootstrap high raw delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LRBN | 4.894158 | 1.682162 | 0.000000% | 0.000000 | 0.000000 | 0.000000 | NA | NA | NA | 0.000000 |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508% | 0.104167 | 0.197917 | 0.000000 | 0.069900 | NA | NA | -0.099202 |
| oracle_stage13_cga_full | 3.074767 | 1.322182 | -37.174740% | 0.000000 | 0.000000 | 0.000000 | 1.000000 | NA | NA | -1.564427 |
| Residual-Prior Convex Mixer | 4.880528 | 1.679483 | -0.278490% | 0.268229 | 0.479167 | 0.998698 | 0.007491 | 0.731421 | 0.731421 | -0.010835 |
| Time-Step Gated Hybrid Editor | 4.893841 | 1.679466 | -0.006465% | 0.277344 | 0.572917 | 0.500000 | 0.000174 | 0.708333 | 0.708333 | 0.012587 |
| Selection-Conditional Conformal Family Editor | 4.844880 | 1.680459 | -1.006873% | 0.470052 | 0.531250 | 0.994792 | 0.027085 | 0.527487 | 0.527487 | -0.024156 |
| Retrieval-Augmented Local Residual Editor | 4.906838 | 1.680344 | +0.259102% | 0.472656 | 0.593750 | 1.000000 | -0.006970 | 0.527344 | 0.527344 | 0.029028 |
| Conservative Challenger Comparator | 4.875935 | 1.680378 | -0.372335% | 0.436198 | 0.552083 | 0.998698 | 0.010016 | 0.563233 | 0.563233 | -0.011526 |

## Gate Verdict

Status: **compact failed; stop before mini-extension**.

No candidate passed its compact gate.

The best average-MSE candidate was `Selection-Conditional Conformal Family Editor`, with `-1.006873%` MSE delta vs LRBN. However, it is not a valid safety/calibration success:

- selected non-harm rate: `0.527487`, far from the required 90% target;
- harm rate: `0.470052`;
- max config harm: `0.531250`;
- oracle capture: only `0.027085`.

An initial gate table bug treated its small expected-vs-observed harm gap as sufficient. This was corrected before final reporting: calibration agreement is meaningless if the calibrated target itself is unsafe. The final gate requires selected non-harm coverage near 90%, so SCCFE fails.

## Candidate-Level Findings

### Residual-Prior Convex Mixer

Directionally positive but too weak:

- MSE delta: `-0.278490%`
- oracle capture: `0.007491`
- harm rate: `0.268229`
- max config harm: `0.479167`

It confirms that bounded convex residual mixing is safer than hard retrieval replacement, but it still behaves like near-full-coverage correction (`0.998698`) and does not capture enough oracle gain.

### Time-Step Gated Hybrid Editor

This did not validate the key time-step hypothesis:

- MSE delta: `-0.006465%`
- q4 boundary delta: `+0.026578%`
- non-boundary delta: `-0.016837%`
- max config harm: `0.572917`

The candidate did not make non-boundary smoothing strong enough, and it failed to make q4 boundary negative. The current temporal mask is not reliable enough.

### Selection-Conditional Conformal Family Editor

This was the strongest on mean MSE but unsafe:

- MSE delta: `-1.006873%`
- q4 boundary delta: `-1.332105%`
- known harmed config delta: `-0.756869%`
- selected non-harm: `0.527487`
- max config harm: `0.531250`

It suggests bias-centering can improve mean and boundary slices, but this implementation does not meet selection-conditional safety. It mainly calibrates to an unsafe selected subset, not to a safe 90% selected coverage target.

### Retrieval-Augmented Local Residual Editor

This failed outright:

- MSE delta: `+0.259102%`
- non-boundary delta: `+0.371538%`
- low-gap/high-repair delta: `+0.544797%`
- oracle capture: `-0.006970`

Retrieval as a direct local residual prior is not reliable in this compact implementation.

### Conservative Challenger Comparator

This produced small mean gains but remained unsafe:

- MSE delta: `-0.372335%`
- accept precision: `0.563233`, below the required `0.60`
- oracle capture: `0.010016`, below `0.10`
- max config harm: `0.552083`

The conservative comparator does not yet separate true wins from false challengers.

## Decision

Do not run mini-extension.

Do not promote any Stage 13 decision-geometry candidate to TableA.

The strongest insight is negative but useful: simply changing the final decision geometry is not enough unless the selected subset itself becomes safe. The most promising sub-signal is the SCCFE bias-centering effect on MSE and boundary slices, but it must be redesigned around a real selection-conditional safety target rather than expected-vs-observed calibration agreement alone.

## Recommended Next Direction

The next attempt should not be another broad five-method sweep. It should isolate one failure:

> Can we construct a selected subset with observed non-harm near 90% on validation and preserve that property on test?

That means the next candidate should explicitly optimize for selected non-harm coverage, not MSE first. If it cannot select a safe subset, the CGA oracle space remains non-deployable regardless of how attractive the full oracle is.

