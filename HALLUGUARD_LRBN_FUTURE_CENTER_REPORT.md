# HalluGuard-LRBN Future-Center Restoration Report

Status: initial validation in progress.

## Hypothesis

The LRBN+NST gain decomposition showed that most improvement came from
level/center restoration. A targeted center-selection mechanism should capture
that signal more cleanly than a generic NST branch.

## Variants

- `unified_revin_rdn_hybrid`: parent baseline.
- `future_center_static`: global trainable mixture of instance mean, boundary
  anchor, tail median, and short trend extrapolated anchor.
- `future_center_static_drift`: static mixture plus a small horizon drift.
- `future_center_selector`: feature-conditioned softmax over the same anchors.
- `future_center_selector_drift`: selector plus learned horizon drift.
- `future_center_residual_shift`: parent LRBN center plus a bounded learned
  shift toward boundary/tail/trend anchors.
- `future_center_residual_shift_cap015`: lower-shift-cap safety ablation.
- `future_center_horizon_selector`: fully learnable selector conditioned on
  context features and horizon embedding.
- `future_center_horizon_conservative`: learnable horizon selector with an
  additional learned parent-blend gate initialized conservatively.

## Initial Results

### Smoke

Command:

```powershell
python scripts\run_halluguard_lrbn_future_center.py --datasets ETTm1 --models DLinear,PatchTST --horizons 96 --variants unified_revin_rdn_hybrid,future_center_static,future_center_selector,future_center_selector_drift --data-root external\ETDataset --prediction-dir baseline_predictions\halluguard_lrbn_future_center_smoke --raw-prediction-dir baseline_predictions\halluguard_lrbn_future_center_smoke_raw --output-dir experiments\halluguard\results\halluguard_lrbn_future_center_smoke --epochs 1 --max-train-windows 128 --max-eval-windows 32 --device cpu --continue-on-error
```

Signal:

- DLinear: `future_center_selector` MSE `3.073134` vs parent `3.089931`.
- PatchTST: `future_center_selector` MSE `3.260146` vs parent `3.286239`.
- Drift was weaker on PatchTST, so it is not the first promotion target.

### L1 16-Config Low-Budget Table

Command:

```powershell
python scripts\run_halluguard_lrbn_future_center.py --datasets ETTm1,ETTh1 --models DLinear,PatchTST --horizons 96,192,336,720 --variants unified_revin_rdn_hybrid,future_center_static,future_center_selector,future_center_selector_drift --data-root external\ETDataset --prediction-dir baseline_predictions\halluguard_lrbn_future_center_l1 --raw-prediction-dir baseline_predictions\halluguard_lrbn_future_center_l1_raw --output-dir experiments\halluguard\results\halluguard_lrbn_future_center_l1 --epochs 2 --max-train-windows 1024 --max-eval-windows 128 --device cpu --continue-on-error
```

Summary versus `unified_revin_rdn_hybrid`:

| variant | wins | mean MSE delta vs parent | mean MAE delta vs parent | max MSE harm |
| --- | ---: | ---: | ---: | ---: |
| `future_center_selector` | 8/16 | -0.215354% | -0.157812% | 1.329137% |
| `future_center_selector_drift` | 9/16 | -0.083454% | -0.060665% | 1.861692% |
| `future_center_static` | 8/16 | -0.055984% | -0.083228% | 1.464503% |

Backbone split for `future_center_selector`:

- DLinear: mean MSE delta `-0.250554%`, wins `4/8`, max harm `0.341847%`.
- PatchTST: mean MSE delta `-0.180154%`, wins `4/8`, max harm `1.329137%`.

### Bounded Residual Shift L1

Command:

```powershell
python scripts\run_halluguard_lrbn_future_center.py --datasets ETTm1,ETTh1 --models DLinear,PatchTST --horizons 96,192,336,720 --variants unified_revin_rdn_hybrid,future_center_selector,future_center_residual_shift,future_center_residual_shift_cap015 --data-root external\ETDataset --prediction-dir baseline_predictions\halluguard_lrbn_future_center_shift_l1 --raw-prediction-dir baseline_predictions\halluguard_lrbn_future_center_shift_l1_raw --output-dir experiments\halluguard\results\halluguard_lrbn_future_center_shift_l1 --epochs 2 --max-train-windows 1024 --max-eval-windows 128 --device cpu --continue-on-error
```

Summary versus `unified_revin_rdn_hybrid`:

| variant | wins | mean MSE delta vs parent | mean MAE delta vs parent | max MSE harm |
| --- | ---: | ---: | ---: | ---: |
| `future_center_selector` | 8/16 | -0.215354% | -0.157812% | 1.329137% |
| `future_center_residual_shift` | 9/16 | -0.045515% | -0.041223% | 1.191168% |
| `future_center_residual_shift_cap015` | 7/16 | +0.122315% | +0.024871% | 3.509343% |

## Verdict

The future-center hypothesis is partially validated:

- A center-only selector improves over the LRBN parent on mean MSE/MAE.
- It is much more mechanism-clean than the NST feature gate because it changes
  only the reversible center restoration.
- It is not yet stronger than `lrbn_nst_feature_gate`, whose earlier local table
  achieved about `-0.458720%` MSE versus parent.
- The main weakness is stability across PatchTST horizons: `future_center_selector`
  helps some PatchTST configs but hurts ETTm1 long horizons.

Recommended next variant:

- Keep `future_center_selector` as the center-only parent candidate.
- Do not promote drift or residual-shift cap yet.
- Next optimization should use a learnable horizon-conditioned center selector,
  rather than larger unconstrained center shifts. Any harm-aware behavior should
  be learned through train-split differentiable gates/regularization, not a
  hand-coded validation rule.

### Learnable Horizon-Aware Selector L1

Command:

```powershell
python scripts\run_halluguard_lrbn_future_center.py --datasets ETTm1,ETTh1 --models DLinear,PatchTST --horizons 96,192,336,720 --variants unified_revin_rdn_hybrid,future_center_selector,future_center_horizon_selector,future_center_horizon_conservative --data-root external\ETDataset --prediction-dir baseline_predictions\halluguard_lrbn_future_center_horizon_l1 --raw-prediction-dir baseline_predictions\halluguard_lrbn_future_center_horizon_l1_raw --output-dir experiments\halluguard\results\halluguard_lrbn_future_center_horizon_l1 --epochs 2 --max-train-windows 1024 --max-eval-windows 128 --device cpu --continue-on-error
```

Important design note: this selector is fully learnable. It uses context
features plus a horizon embedding, and the network learns anchor weights by
backpropagation on the training split. The conservative variant uses a learned
parent-blend gate; no validation/test rule is used to choose anchors.

Summary versus `unified_revin_rdn_hybrid`:

| variant | wins | mean MSE delta vs parent | mean MAE delta vs parent | max MSE harm |
| --- | ---: | ---: | ---: | ---: |
| `future_center_selector` | 8/16 | -0.215354% | -0.157812% | 1.329137% |
| `future_center_horizon_selector` | 9/16 | -0.181016% | -0.150137% | 0.672366% |
| `future_center_horizon_conservative` | 6/16 | +0.148383% | +0.122615% | 1.040169% |

Backbone split for `future_center_horizon_selector`:

- DLinear: mean MSE delta `-0.388041%`, wins `5/8`, max harm `0.094585%`.
- PatchTST: mean MSE delta `+0.026008%`, wins `4/8`, max harm `0.672366%`.

Interpretation:

- Learnable horizon conditioning does reduce harm: max harm drops from
  `1.329137%` to `0.672366%`.
- It is slightly weaker on mean MSE than the non-horizon selector.
- The conservative learned parent gate is too conservative/underfit and should
  not be promoted.

Updated recommendation: keep `future_center_horizon_selector` as the safer
learnable center selector, and keep `future_center_selector` as the more
aggressive center-only candidate. The next improvement should make the
horizon-aware selector regain PatchTST benefit without losing its lower harm.
