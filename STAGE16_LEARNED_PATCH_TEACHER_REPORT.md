# Stage 16 Learned Patch / Teacher Projector Report

## Verdict

Stage16 implemented and validated a heavier learned patch representation / teacher-manifold line.

Result: **failed compact gate; stop before mini-extension / TableA**.

The line produced a real mechanism signal, but not a deployable improvement over the current compact frontier:

- The strongest MSE candidate, `H2H6 Learned Patch Teacher Hybrid`, reaches `-1.786478%` MSE delta vs LRBN, close to the `-1.8%` safe threshold and better than Stage15 H1.
- However, its harm is too high: harm `0.105469`, max config harm `0.187500`.
- The conservative teacher variant, `H6-Safe Sparse Teacher Projector`, is genuinely safer: harm `0.016927`, max config harm `0.041667`.
- But that safe variant is too weak: MSE delta `-1.013201%`, below Stage15 H1 and SRA-BP-safe.

Therefore Stage16 is **partial mechanism support**, not a new main method.

## What Changed From Stage15

Stage15 tested static endogenous editors:

- static residual patch prototypes were safe but inactive;
- retrieval residual guidance was split-unstable;
- the teacher manifold route was blocked because no learned teacher existed.

Stage16 added actual learned modules:

1. `H2L Learned Patch Residual Editor`
   - patch-level MLP trained on validation inner-train residual patches;
   - target-free inputs: LRBN patch, context-tail patch, LRBN/context difference patch, position/horizon/stats.
2. `H6 Denoising Teacher Manifold Projector`
   - denoising patch autoencoder trained on validation inner-train target patches;
   - inference projects LRBN forecast patches toward the learned teacher manifold.
3. `H2H6 Learned Patch Teacher Hybrid`
   - validation-calibrated mixture of learned residual patch direction and teacher projection direction.
4. Conservative sparse variants after the first formal run showed high harm:
   - `H6-Safe Sparse Teacher Projector`
   - `H2H6-Safe Sparse Learned Teacher Hybrid`

All thresholds, shrink, caps, gates, and mix weights were selected on validation inner-calib only.

## Protocol

- datasets: `ETTm1`, `ETTh1`
- backbones: `DLinear`, `PatchTST`
- horizons: `96`, `192`
- seed: `2026`
- samples: `768`
- bootstrap: `2000`
- fit: validation inner-train only
- calibration: validation inner-calib only
- test: final evaluation only
- test threshold leakage: `False`

## Overall Results

| variant | MSE | MAE | MSE delta vs LRBN | harm | max config harm | oracle gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LRBN | 4.894158 | 1.682162 | 0.000000% | 0.000000 | 0.000000 | NA |
| SRA-BP-safe | 4.813149 | 1.660950 | -1.655217% | 0.035156 | 0.114583 | 0.044525 |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508% | 0.104167 | 0.197917 | 0.069900 |
| Stage14 FamilyMix Selector | 4.827022 | 1.669096 | -1.371750% | 0.002604 | 0.010417 | 0.036900 |
| Stage15 H1 Residual Atom Simplex | 4.825331 | 1.669162 | -1.406308% | 0.007812 | 0.020833 | 0.037830 |
| H2L Learned Patch Residual Editor | 4.899526 | 1.682689 | +0.109683% | 0.072917 | 0.177083 | -0.002950 |
| H6 Denoising Teacher Manifold Projector | 4.843571 | 1.672442 | -1.033610% | 0.024740 | 0.083333 | 0.027804 |
| H2H6 Learned Patch Teacher Hybrid | 4.806725 | 1.665870 | -1.786478% | 0.105469 | 0.187500 | 0.048056 |
| H6-Safe Sparse Teacher Projector | 4.844570 | 1.672768 | -1.013201% | 0.016927 | 0.041667 | 0.027255 |
| H2H6-Safe Sparse Learned Teacher Hybrid | 4.878476 | 1.678922 | -0.320413% | 0.059896 | 0.156250 | 0.008619 |
| SafeTAE-safe (Stage7 table) | 4.804843 | 1.660837 | -1.824928% | 0.018229 | NA | 0.100424 |

## Gate Results

No Stage16 variant passed compact safe or tradeoff gates.

| variant | safe | tradeoff | mechanism | failure |
| --- | --- | --- | --- | --- |
| H2L Learned Patch Residual Editor | false | false | false | Test MSE worsened `+0.109683%`; known harmed config worsened `+0.403311%`. |
| H6 Denoising Teacher Manifold Projector | false | false | false | Safe-ish but too weak: MSE delta `-1.033610%`, oracle gain `0.027804`. |
| H2H6 Learned Patch Teacher Hybrid | false | false | false | Good MSE `-1.786478%`, but harm `0.105469` and max config harm `0.187500`. |
| H6-Safe Sparse Teacher Projector | false | false | false | Low harm `0.016927`, max config harm `0.041667`, but MSE delta only `-1.013201%`. |
| H2H6-Safe Sparse Learned Teacher Hybrid | false | false | false | Conservative hybrid loses most gain: MSE delta `-0.320413%`, harm `0.059896`. |

## Slice Findings

The best mean-MSE hybrid improved all eight configs, but with broad harm:

- `H2H6 Learned Patch Teacher Hybrid`
  - q4 boundary: `-1.597931%`
  - non-boundary: `-1.845663%`
  - low-gap/high-repair: `-2.212056%`
  - known harmed config: `-1.500589%`
  - harm rate: `0.105469`

The safe teacher variant is more defensible mechanistically:

- `H6-Safe Sparse Teacher Projector`
  - q4 boundary: `-0.439223%`
  - non-boundary: `-1.193372%`
  - known harmed config: `-0.799539%`
  - harm rate: `0.016927`
  - max config harm: `0.041667`

But it does not clear the required MSE/oracle-capture gates.

## Interpretation

Stage16 gives a sharper answer than Stage15:

1. A learned teacher manifold is better than static H2 codebooks.
   - Static H2 from Stage15: MSE delta `-0.023125%`.
   - H6-safe learned teacher: MSE delta `-1.013201%` with low harm.

2. Learned residual directions are powerful but still unsafe.
   - Hybrid gets close to SafeTAE-safe MSE, but with SRA-balanced-like harm.
   - This means the representation can find useful directions, but validation-only gating still cannot make them safe enough.

3. The current compact teacher is too shallow for a breakthrough.
   - It is only a patch MLP / denoising autoencoder trained on the compact validation split.
   - The safe version under-edits; the strong version over-harms.

## Decision

Do **not** promote Stage16 to mini-extension or TableA.

Keep as a mechanism checkpoint:

- learned teacher manifolds are worth revisiting;
- static patch codebooks should be retired;
- residual MLP without stronger uncertainty/safety modeling is too risky.

The next credible route is not more shrink/gate tuning. It should be one of:

1. Train a stronger SSL teacher on the full training split or additional public series, then use minimal-norm projection.
2. Add explicit uncertainty/calibration to the patch residual model, e.g. ensemble / quantile residual head / conformalized patch risk.
3. Move from independent patch edits to structured sequence-level teacher projection so local patches cannot create incoherent global harm.

## Artifacts

- Results directory: `experiments/halluguard/results/stage16_learned_patch_teacher/`
- Overall metrics: `experiments/halluguard/results/stage16_learned_patch_teacher/stage16_overall.csv`
- Gate table: `experiments/halluguard/results/stage16_learned_patch_teacher/stage16_gate_table.csv`
- Calibration grid: `experiments/halluguard/results/stage16_learned_patch_teacher/stage16_calibration_grid.csv`
- Policies: `experiments/halluguard/results/stage16_learned_patch_teacher/stage16_policies.json`
- Training log: `experiments/halluguard/results/stage16_learned_patch_teacher/stage16_training_log.csv`
- Generated summary: `experiments/halluguard/results/stage16_learned_patch_teacher/summary.md`
