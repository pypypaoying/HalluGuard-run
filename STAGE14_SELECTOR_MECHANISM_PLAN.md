# Stage 14 Selector Mechanism Plan

## Selected Hypothesis

The next selector line should keep the Stage 10 CGA candidate families fixed and test whether a family-level, selected-subset-safe selector can turn the strong oracle space into deployable low-harm gains. This stage does not add new candidate families and does not tune on test.

## Protocol

- Parent line: Stage 10 CGA candidate-family mechanism, after Stage 11-13 selector failures.
- Fixed candidate families: smoothing-teacher, residual-distribution, retrieval-memory, and existing SRA/local variants.
- Compact scope: ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192 x seed 2026.
- Baseline: frozen HalluGuard-LRBN.
- Controls: SRA-BP-balanced, Stage10 hard selector, full CGA oracle.
- Trainable selector parameters: fit only on validation inner-train.
- Thresholds, residual strength, top-k and abstain gates: selected only on validation inner-calib.
- Test split: evaluation only.
- Leakage rule: no feature used at test time may depend on target; test target is used only in metric computation.

## Candidate Selectors

1. FamilyMix Selector: learn safe family routing, then do convex residual mixture within selected families.
2. Two-stage Cost-Sensitive Router: learn family routing first, then candidate utility minus harm.
3. ListSafe Top-k Selector: optimize deploy decision around family/candidate top-k ranking signal.
4. Retrieval-Prior Selector: use validation-memory nearest-neighbor family outcome as a selector prior, not an editor.
5. Bayes-Abstain Selector: estimate utility uncertainty and abstain unless lower-confidence utility is positive.

## Success Gates

Safe gate:

- MSE delta vs LRBN <= -1.8%.
- Harm <= 0.03.
- Max config harm <= 0.10.
- Selected non-harm >= 0.90.
- Family top-2 >= 0.70.
- Oracle capture >= 0.08.
- Bootstrap 95% CI high for raw MSE delta < 0.

Balanced gate:

- MSE delta vs LRBN <= -2.7%.
- Harm <= 0.10.
- Max config harm <= 0.18.
- Selected non-harm >= 0.85.
- Family top-2 >= 0.75.
- Candidate top-2 >= 0.20.
- Oracle capture >= 0.12.
- Bootstrap 95% CI high for raw MSE delta < 0.

## Stop Rule

If no selector passes compact safe or balanced gate, stop before mini-extension and report that family-level selected-subset safety remains unsolved. If one or more selectors pass, promote the top 2-3 variants to the mini-extension protocol in the research report.

## Outputs

- `experiments/halluguard/results/stage14_selector_mechanism/`
- `STAGE14_SELECTOR_MECHANISM_REPORT.md`
- `STAGE14_SELECTOR_MECHANISM_CHECKLIST.md`
- `CANDIDATE_BOARD.md`
- `results_halluguard.tsv`

