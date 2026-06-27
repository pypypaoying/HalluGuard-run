# Stage 6 Mechanism Validation Summary

## Setup

- Input metrics: `experiments\halluguard\results\research_direction_validation\forecast_inputs\combined_metrics.csv`
- Output directory: `experiments\halluguard\results\stage6_mechanism`
- Validation samples: `768`
- Test samples: `768`
- Test configs: `8`
- Test threshold leakage: `False`

## Go / No-Go

- MRC: `False`
- TAE: `False`
- FOMC: `False`

## Headline Metrics

- MRC ridge-abstain MSE delta vs LRBN: `-1.203025%`
- MRC ridge-abstain harm: `0.026042`
- TAE oracle gain vs LRBN: `-16.085977%`
- TAE router/ranker best gain fraction: `-0.553189`
- FOMC spectral delta vs LRBN: `-1.158871%`
- FOMC protocol guard pass: `True`

## Interpretation

Stage 6 is mechanism validation only. Promote only lines with `go=True`; failed lines remain useful diagnostics.