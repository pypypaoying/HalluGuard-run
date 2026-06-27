# Stage 4E Learnable Alpha Adapter Summary

- Secondary experiment; does not replace Stage 4A-C decision.
- Validation split learns alpha labels/model; test split only evaluates.

## Results

| Method | Split | MSE | MAE | Delta % vs LRBN | Harm | Mean alpha | Nonzero alpha |
|---|---|---:|---:|---:|---:|---:|---:|
| adaptive-alpha-safe-loss | val | 8.435078 | 2.241743 | -4.116097 | 0.429688 | 0.267660 | 1.000000 |
| adaptive-alpha-safe-loss | test | 4.704777 | 1.638125 | -3.869534 | 0.385417 | 0.257851 | 1.000000 |
| global-alpha-safe-loss | val | 8.419233 | 2.230162 | -4.296210 | 0.453125 | 0.500000 | 1.000000 |
| global-alpha-safe-loss | test | 4.643981 | 1.621852 | -5.111746 | 0.423177 | 0.500000 | 1.000000 |

## Bootstrap CI

```json
{
  "mean_delta": -0.18938108906915316,
  "ci95_low": -0.2357970649436767,
  "ci95_high": -0.14407536604801296,
  "p_improve_bootstrap": 1.0
}
```
