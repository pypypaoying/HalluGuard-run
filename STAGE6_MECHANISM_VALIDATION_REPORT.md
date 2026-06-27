# Stage 6 Mechanism Validation Report

## Goal

Stage 6 tested three next-line mechanism hypotheses on the compact real-forecast assets:

1. MRC: whether post-LRBN residuals are learnable and can support point correction, uncertainty intervals, and abstention.
2. TAE: whether multiple interpretable trajectory experts have a useful upper bound and can be routed without target leakage.
3. FOMC: whether matured-label frequency residual drift supports leakage-safe online calibration.

This is a compact mechanism validation stage, not a full TableA result.

## Setup

- Input: `experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv`
- Output: `experiments/halluguard/results/stage6_mechanism/`
- Datasets: `ETTm1`, `ETTh1`
- Backbones: `DLinear`, `PatchTST`
- Horizons: `96`, `192`
- Seed: `2026`
- Validation samples: `768`
- Test samples: `768`
- Test configs: `8`
- Calibration: validation-only
- Test threshold leakage: `False`

Command:

```bash
python experiments/halluguard/run_stage6_mechanism.py --metrics-csv experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv --stage5-dir experiments/halluguard/results/lrbn_sra_bp_stage5 --output-dir experiments/halluguard/results/stage6_mechanism --n-bootstrap 2000
```

## Overall Verdict

From `stage6_verdict.json`:

```json
{
  "mrc_go": false,
  "tae_go": false,
  "fomc_go": false,
  "test_threshold_leakage": false
}
```

Stage 6 does not promote any of the three lines directly. The strongest partial signal is MRC point correction; TAE has a large oracle space but poor deployable routing; FOMC has a real spectral signal but unsafe harm.

## MRC Results

MRC used validation-only ridge residual heads by horizon plus a validation-only shrink/cap safety grid. The selected cap was:

```json
{
  "shrink": 0.4,
  "cap_mult": 2.0,
  "risk_threshold": 0.35,
  "alpha_by_horizon": {
    "96": 10.0,
    "192": 10.0
  }
}
```

Test point results vs HalluGuard-LRBN:

| Method | MSE | MAE | MSE Delta % | Harm |
|---|---:|---:|---:|---:|
| LRBN | 4.894158 | 1.682162 | 0.000000 | 0.000000 |
| MRC-mean-residual | 4.863475 | 1.700407 | -0.626920 | 0.540365 |
| MRC-ridge-residual | 4.786189 | 1.644985 | -2.206079 | 0.411458 |
| MRC-ridge-abstain | 4.835280 | 1.668420 | -1.203025 | 0.026042 |
| SRA-BP-safe | 4.813149 | 1.660950 | -1.655217 | 0.035156 |
| SRA-BP-balanced | 4.766983 | 1.645627 | -2.598508 | 0.104167 |

Bootstrap CI:

- `MRC-ridge-residual`: mean delta `-0.107969`, 95% CI `[-0.191756, -0.026203]`
- `MRC-ridge-abstain`: mean delta `-0.058878`, 95% CI `[-0.087876, -0.031990]`

MRC verdict:

- `point_pass`: true
- `harm_pass`: true
- `coverage_pass`: false
- `abstention_pass`: true
- `non_sra_slice_pass`: true
- `safe_go`: false

Interpretation: residual point correction is real and safer with abstention, but interval calibration over-covers by `5.71pp` to `10.12pp`, so the full MRC mechanism is not ready. MRC should continue only as a point-correction/abstention subline, not as a calibrated residual-distribution claim yet.

## TAE Results

TAE generated interpretable candidates including `keep_lrbn`, `raw`, `sra_safe`, `sra_balanced`, `bp_always`, level bias, phase shifts, amplitude scaling, volatility shrink, and ensemble median.

Decision-level results vs LRBN:

| Method | MSE | MSE Delta % | Harm |
|---|---:|---:|---:|
| TAE-oracle-best | 4.106885 | -16.085977 | 0.000000 |
| bp_always | 4.643981 | -5.111746 | 0.423177 |
| volatility_shrink | 4.761939 | -2.701555 | 0.160156 |
| level_bias | 4.765325 | -2.632373 | 0.298177 |
| sra_balanced | 4.766983 | -2.598508 | 0.104167 |
| sra_safe | 4.813149 | -1.655217 | 0.035156 |
| TAE-router | 5.329668 | +8.898583 | 0.305990 |
| TAE-ranker | 5.913439 | +20.826486 | 0.394531 |

TAE verdict:

- Oracle gain vs LRBN: `-16.085977%`
- Oracle extra vs SRA-balanced: `-13.847292%`
- Router top-1 accuracy: `0.341146`
- Router top-2 hit: `0.522135`
- Ranker score/gain Spearman: `0.494597`
- Router gain fraction: `-0.553189`
- Ranker gain fraction: `-1.294698`
- Failure-mode separability: accuracy `0.623698`, macro-F1 `0.647751`
- `compact_go`: false

Interpretation: the candidate expert set contains real complementary upper-bound value, and several non-boundary experts are individually useful. However the target-free router/ranker is not deployable in this compact setup: it fails the top-2 gate and produces worse-than-LRBN decisions. This argues for a better arbitration objective or a safer oracle-distillation setup before TAE can become a method.

## FOMC Results

FOMC used validation as historical matured buffer and test as a chronological replay. The protocol guard passed with no future-label usage.

Online adapter results:

| Method | MSE | MSE Delta % | MAE Delta % | Harm | Coverage90 |
|---|---:|---:|---:|---:|---:|
| spectral_adapter | 4.837441 | -1.158871 | -0.091480 | 0.467448 | 0.966587 |
| rolling_mean_residual | 4.864903 | -0.597737 | +1.025934 | 0.531250 | 0.968438 |
| no_update | 4.894158 | 0.000000 | 0.000000 | 0.000000 | 0.964959 |
| time_ema_residual | 4.901755 | +0.155240 | +1.456221 | 0.541667 | 0.968051 |

FOMC verdict:

- Spectral delta vs LRBN: `-1.158871%`
- Spectral advantage vs rolling mean residual: `-0.561135%`
- Mean spectral autocorrelation: `0.362599`
- Spectral harm: `0.467448`
- Coverage gap: `6.658664pp`
- Protocol guard pass: true
- `compact_go`: false

Interpretation: frequency residual drift is not random; spectral correction beats rolling mean residual on point MSE. But harm is far above the allowed `0.10` and interval coverage is too wide/over-covered. FOMC should not be promoted as a deployable correction yet. It may be worth revisiting only with a harm-aware online shrink/conformal controller.

## Claim Update

Stage 6 narrows the next-stage direction:

- Supported: post-LRBN residuals contain learnable signal; sparse/risk-aware correction can reduce harm.
- Supported: multi-expert trajectory candidates have a large oracle upper bound.
- Partially supported: spectral residual statistics have online signal under a leakage-safe protocol.
- Not supported: any of MRC, TAE, or FOMC is currently ready as a full deployable method.
- Not supported: current TAE target-free routing can capture the oracle upper bound.
- Not supported: current FOMC can control harm.

## Recommendation

Do not add Stage 6 mechanisms directly to TableA.

Recommended next work:

1. Continue MRC as a bounded residual point-correction/abstention mini-line, not as calibrated interval modeling yet.
2. Treat TAE oracle results as evidence for expert complementarity, but redesign routing around safe pairwise/no-harm distillation before another full run.
3. Treat FOMC as a diagnostic signal; only revisit after adding validation-only harm-aware shrink and coverage calibration.
4. Keep SRA-BP-safe/balanced as the current BP mini-extension parent; Stage 6 did not replace it.
