# Stage18 Performance Atom Extraction Diagnosis Report

## Verdict

Stage18 completed the requested mechanism diagnosis: TAE/CGA oracle was used as a performance microscope rather than as a deployable selector.

Result: **atom route mechanism pass, but distillation/prototype not ready**.

The oracle evidence is strong:

- `union_full_pool` oracle over `SRA-BP-balanced`: `-35.498668%` MSE.
- SRA-complement atom pool oracle over `SRA-BP-balanced`: `-21.280744%` MSE.
- `residual_distribution` leave-one-out degradation: `7.768633%`.
- `smoothing_teacher` leave-one-out degradation: `5.545641%`.
- top-5 PCA atoms explain `80.611193%` of oracle correction-vector variance.

But deployability is not established:

- The best validation-calibrated prototype selected `0` test rows, so it was SRA-equivalent.
- The nonzero prototypes all worsened MSE versus `SRA-BP-balanced`.
- Therefore Stage18 supports **mechanism diagnosis**, not a new deployable module.

## Protocol

- datasets: `ETTm1`, `ETTh1`
- backbones: `DLinear`, `PatchTST`
- horizons: `96`, `192`
- seed: `2026`
- test rows: `768`
- parent methods: `LRBN`, `SRA-BP-safe`, `SRA-BP-balanced`
- main parent for atoms: `SRA-BP-balanced`
- oracle pools:
  - `tae_old_pool`
  - `cga_new_family_pool`
  - `union_full_pool`
  - `sra_complement_atom_pool`
- bootstrap: `2000`
- distillation/prototype calibration: validation inner-train / inner-calib only
- test threshold leakage: `False`

## Experiment 1: Incremental Oracle over SRA

`SRA-BP-balanced` remains far from the oracle envelope:

| oracle pool | MSE | delta vs SRA-BP-balanced | oracle gain fraction vs LRBN oracle | non-parent selection |
| --- | ---: | ---: | ---: | ---: |
| `tae_old_pool` | 4.217668 | -11.523317% | 0.301922 | 0.899740 |
| `cga_new_family_pool` | 3.795198 | -20.385746% | 0.534127 | 0.914062 |
| `union_full_pool` | 3.074767 | -35.498668% | 0.930100 | 0.959635 |
| `sra_complement_atom_pool` | 3.752533 | -21.280744% | 0.557577 | 0.934896 |

This passes the Stage18 mechanism gate by a wide margin. The gain is not merely a correction of LRBN defects before SRA; even after SRA, non-boundary/residual/smoothing-style candidates contain large residual performance space.

## Experiment 2: Family Leave-One-Out

The main atom sources are clear:

| family group | leave-one-out degradation | oracle share | only-family gain |
| --- | ---: | ---: | ---: |
| `residual_distribution` | 7.768633% | 0.450521 | -14.388075% |
| `smoothing_teacher` | 5.545641% | 0.201823 | -10.740900% |
| `old_residual` | 1.037140% | 0.167969 | -7.988098% |
| `retrieval_memory` | 0.203281% | 0.053385 | -5.156478% |
| `volatility_amplitude_level` | 0.095745% | 0.059896 | -2.958937% |
| `ensemble` | near zero | 0.001302 | -0.514091% |

Interpretation:

- `residual_distribution` is the strongest performance atom family.
- `smoothing_teacher` is the second strongest, especially for larger corrections.
- `retrieval_memory` has oracle cases but low marginal contribution; keep it diagnostic-only for now.
- Volatility/amplitude/level atoms are weaker and not a priority.

## Experiment 3: Correction Vector Decomposition

PCA shows the oracle corrections are compressible:

- PC1 EVR: `0.521031`
- PC1-2 cumulative: `0.689852`
- PC1-5 cumulative: `0.806112`

This passes the low-dimensional structure gate. The oracle corrections are not arbitrary per-sample chaos; they concentrate into a small number of trajectory-shape directions.

KMeans atom composition on test:

- Atom 0: mostly `old_residual`, 220 selected rows, mean delta `-0.747266`.
- Atom 1: mostly `residual_distribution`, 165 rows, mean delta `-1.260361`.
- Atom 3: mostly `smoothing_teacher`, 92 rows, mean delta `-2.758750`.
- Atom 4: mostly `residual_distribution`, 241 rows, mean delta `-0.634581`.

Cluster stability is moderate rather than perfect: test-vs-train JS distance is `0.133920`. This is acceptable for mechanism diagnosis, but not enough alone for deployment.

## Experiment 4: Residual Alignment

Every nonempty test atom has `A>1 rate = 1.0`, but this must be interpreted carefully: these are oracle-selected corrections, so positive MSE alignment is partly guaranteed by construction.

The useful distinction is magnitude and family:

- Atom 3, `smoothing_teacher`, has the largest mean gain: `-2.758750`.
- Atom 1, `residual_distribution`, has strong and broad gain: `-1.260361`.
- Atom 4, `residual_distribution`, is broader but shallower: `-0.634581`.
- Atom 0, old residual, is also useful: `-0.747266`.

This means the oracle microscope finds valid residual directions, but not yet a deployable rule for when to apply them.

## Experiment 5: SRA Complementarity

The atoms are genuinely complementary to SRA in oracle mode:

- Atom 3 improves `non_boundary` by `-57.417372%`.
- Atom 3 improves `low_gap_high_repair` by `-61.903052%`.
- Atom 1 improves `non_boundary` by `-16.493542%`.
- Atom 1 improves `low_gap_high_repair` by `-14.972269%`.
- Known harmed config is also oracle-improved for all nonempty atoms.

This supports the claim that SRA-BP is not exhausting the correction space. The complementarity direction is strongest for:

1. `smoothing_teacher` non-boundary shape atom.
2. `residual_distribution` residual quantile atom.

## Experiment 6: Distillability

Target-free feature prediction is mixed:

| atom | chosen model | activation AUROC test | PR-AUC test | sign accuracy | R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | random forest | 0.565411 | 0.327318 | 0.417969 | -0.211325 |
| 1 | random forest | 0.699040 | 0.350102 | 0.740885 | 0.078126 |
| 3 | logistic | 0.840574 | 0.444039 | 0.666667 | 0.155127 |
| 4 | random forest | 0.715299 | 0.565637 | 0.742188 | 0.035587 |

Distillation signal exists, especially atom 3 and atom 4. However, activation ranking alone did not produce a useful correction when translated into a fixed atom-center edit.

## Experiment 7: Prototype Atom Composition

No deployable prototype passed.

The validation-selected safe prototype selected no test rows and therefore exactly matched `SRA-BP-balanced`:

- best prototype: `atom_1_prototype`
- coverage: `0.0`
- MSE delta: `0.0%`

The nonzero prototype diagnostics worsened MSE:

- atom 0: `+0.445520%`, harm `0.040365`, max config harm `0.125000`
- atom 2: `+0.658307%`, harm `0.048177`, max config harm `0.166667`
- atom 3: `+0.828973%`, harm `0.151042`, max config harm `0.531250`

So the atom shapes are real in oracle space, but the current distilled application rule is not safe or useful.

## Decision

Stage18 supports a **family-specific mechanism pass**, not a deployable module pass.

Keep for next-stage innovation:

1. Residual Quantile Atom over SRA-BP-balanced.
2. Non-Boundary Shape Atom over SRA-BP-balanced.

Do not promote to mini-extension yet. The next stage should not train a generic selector over full candidates. It should design a safer atom-specific application mechanism, probably with:

- per-family residual quantile atom coefficients rather than fixed cluster centers;
- non-boundary / low-gap-high-repair slice-specific gating;
- explicit “do not edit unless predicted atom benefit is calibrated” abstention;
- harm-aware coefficient prediction rather than binary activation only.

## Artifacts

- Results directory: `experiments/halluguard/results/stage18_performance_atom_diagnosis/`
- Verdict: `experiments/halluguard/results/stage18_performance_atom_diagnosis/stage18_verdict.json`
- Parent oracle table: `experiments/halluguard/results/stage18_performance_atom_diagnosis/parent_oracle_table.csv`
- Family LOO: `experiments/halluguard/results/stage18_performance_atom_diagnosis/family_leave_one_out.csv`
- Correction vectors: `experiments/halluguard/results/stage18_performance_atom_diagnosis/correction_vectors.parquet`
- Atom alignment: `experiments/halluguard/results/stage18_performance_atom_diagnosis/atom_alignment_report.csv`
- Distillability: `experiments/halluguard/results/stage18_performance_atom_diagnosis/atom_distillability_report.csv`
- Prototype diagnostics: `experiments/halluguard/results/stage18_performance_atom_diagnosis/prototype_atom_metrics.csv`
