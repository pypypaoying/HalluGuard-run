# Stage 15 Endogenous Low-Harm Editors Report

## Verdict

Stage 15 completed the compact validation plan from `deep-research-report (5).md`.

Result: **failed compact gate; stop before mini-extension**.

The strongest candidate is **H1 Residual Atom Simplex Editor**. It is safer and slightly stronger than Stage14 FamilyMix, but it does not clear the safe/tradeoff/mechanism gates:

- MSE: `4.825331`
- MAE: `1.669162`
- MSE delta vs frozen LRBN: `-1.406308%`
- harm rate: `0.007812`
- max config harm: `0.020833`
- oracle gain fraction: `0.037830`
- q4 boundary delta: `-3.368323%`
- known harmed config delta: `-0.392384%`
- bootstrap CI high raw delta: `-0.055759`
- test threshold leakage: `False`

This is partial support for bounded residual atom editing, but not enough to justify mini-extension or TableA promotion.

## What Was Run

Compact protocol:

- datasets: `ETTm1`, `ETTh1`
- backbones: `DLinear`, `PatchTST`
- horizons: `96`, `192`
- seed: `2026`
- samples: `768`
- bootstrap: `2000`
- fit: validation inner-train
- calibration: validation inner-calib
- test: final evaluation only

Implemented and evaluated:

- H1 Residual Atom Simplex Editor
- H3 Any-Quantile Residual Envelope
- H5 Local-Global Decoupled Sparse Editor
- H2 Prototype Codebook Local Editor
- H4 Retrieval-Conditioned Residual Adapter

H6 Self-Supervised Teacher Manifold Projector was **not executed** in this compact pass. It requires a separate SSL teacher pretraining/evaluation stack, and no compact candidate passed enough evidence to unlock that heavier mini-extension. This is recorded as a methodological blocker, not as a negative result for H6.

## Overall Table

| variant | MSE | MAE | MSE delta vs LRBN | harm | max config harm | oracle gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LRBN | 4.894158 | 1.682162 | 0.000000% | 0.000000 | 0.000000 | NA |
| SRA-BP-safe | 4.813149 | 1.660950 | -1.655217% | 0.035156 | 0.114583 | 0.044525 |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508% | 0.104167 | 0.197917 | 0.069900 |
| Stage14 FamilyMix Selector | 4.827022 | 1.669096 | -1.371750% | 0.002604 | 0.010417 | 0.036900 |
| H1 Residual Atom Simplex Editor | 4.825331 | 1.669162 | -1.406308% | 0.007812 | 0.020833 | 0.037830 |
| H2 Prototype Codebook Local Editor | 4.893026 | 1.682025 | -0.023125% | 0.018229 | 0.041667 | 0.000622 |
| H3 Any-Quantile Residual Envelope | 4.890125 | 1.681795 | -0.082403% | 0.451823 | 0.500000 | 0.002217 |
| H4 Retrieval-Conditioned Residual Adapter | 4.906913 | 1.683740 | +0.260622% | 0.076823 | 0.218750 | -0.007011 |
| H5 Local-Global Decoupled Sparse Editor | 4.870041 | 1.679941 | -0.492753% | 0.436198 | 0.520833 | 0.013255 |
| SafeTAE-safe (Stage7 table) | 4.804843 | 1.660837 | -1.824928% | 0.018229 | NA | 0.100424 |

## Gate Results

No Stage15 endogenous editor passed compact gates.

| variant | safe | tradeoff | mechanism | main failure |
| --- | --- | --- | --- | --- |
| H1 Residual Atom Simplex Editor | false | false | false | MSE delta `-1.406%` is below the safe target `-1.8%`; oracle gain `0.0378` is below `0.08`. |
| H2 Prototype Codebook Local Editor | false | false | false | Very safe, but nearly no useful movement: MSE delta `-0.023%`, active patch ratio `0.0078`, oracle gain `0.0006`. |
| H3 Any-Quantile Residual Envelope | false | false | false | Mean gain is tiny and harm is high: harm `0.4518`, max config harm `0.5000`. |
| H4 Retrieval-Conditioned Residual Adapter | false | false | false | Test MSE worsens `+0.2606%`; known harmed config worsens `+1.8485%`. |
| H5 Local-Global Decoupled Sparse Editor | false | false | false | Some mean gain but high dense harm: harm `0.4362`, max config harm `0.5208`. |

## Candidate Interpretation

H1 is the only useful Stage15 signal. Compared with Stage14 FamilyMix, it improves MSE slightly (`4.825331` vs `4.827022`) and keeps harm low, while maintaining non-degenerate edits (`lrbn_equiv_rate=0.766927`, `active_patch_ratio=0.237703`). Its best slice is `q4_boundary`, where it reaches `-3.368323%`. The problem is gain capture: oracle utilization remains `3.783%`, barely above Stage14 and far below the `8%` compact mechanism target.

H2 validates the safety intuition but not the performance intuition. Its local prototype matching avoided catastrophic harm, but the selected policy mostly abstained (`lrbn_equiv_rate=0.976562`) and active patches were too sparse to matter.

H3 and H5 are rejected as currently implemented. Both edit too broadly and reproduce the old failure mode: small mean improvement paired with high sample/config harm.

H4 is rejected for split instability. Calibration showed some strong MSE configurations, but they were high-harm; the selected conservative retrieval policy still worsened test MSE and harmed the known fragile config.

## Decision

Do **not** promote Stage15 to mini-extension or TableA.

The next promising direction is not another selector/threshold pass over the same candidate families. The evidence says:

- bounded residual atoms are safer than hard candidate selection, but too weak;
- local codebooks are safe but underactive;
- retrieval as direct residual guidance is unstable;
- distribution/envelope and local-global residual corrections still need learned structure before they are safe.

If this line continues, the next credible experiment should be a heavier learned representation route: either a true patch-level representation/codebook trained with reconstruction/contrastive objectives, or the H6 teacher-manifold projector with an explicit SSL teacher. It should not be reported as a continuation of the current lightweight compact editor unless that training stack is implemented and validated.

## Artifacts

- Results directory: `experiments/halluguard/results/stage15_endogenous_editors/`
- Overall metrics: `experiments/halluguard/results/stage15_endogenous_editors/stage15_overall.csv`
- Gate table: `experiments/halluguard/results/stage15_endogenous_editors/stage15_gate_table.csv`
- Calibration grid: `experiments/halluguard/results/stage15_endogenous_editors/stage15_calibration_grid.csv`
- Policies: `experiments/halluguard/results/stage15_endogenous_editors/stage15_policies.json`
- Generated summary: `experiments/halluguard/results/stage15_endogenous_editors/summary.md`
