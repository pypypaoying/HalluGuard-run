# HalluGuard Experiment Directory

This directory contains the first-stage MVP for **Idea 9: HalluGuard Trend-Frequency Test-Time Correction**.

The method is an inference-time / offline post-processing layer. It never changes the forecasting model, training loss, or hidden states. It reads a prediction sample with:

```text
sample_id, dataset, model, context, prediction, target
```

The synthetic MVP adds metadata such as `split`, `stress_type`, and `is_turning_point`, but evaluation code treats the six fields above as the common prediction contract.

## Files

```text
correction.py
metrics.py
stress.py
evaluate_predictions.py
run_mvp.py
configs/halluguard_mvp.yaml
results/
```

## Run

Smoke test:

```bash
python experiments/halluguard/run_mvp.py --config experiments/halluguard/configs/halluguard_mvp.yaml --quick
```

Full MVP:

```bash
python experiments/halluguard/run_mvp.py --config experiments/halluguard/configs/halluguard_mvp.yaml
```

External prediction file smoke test:

```bash
python experiments/halluguard/evaluate_predictions.py --config experiments/halluguard/configs/halluguard_mvp.yaml --input experiments/halluguard/fixtures/external_predictions.jsonl --calibration-split val --split test --output-dir experiments/halluguard/results/external_smoke
```

## Leakage Rule

`tau_trend` and `tau_freq` are calibrated only from the configured validation/calibration split. The default synthetic calibration split contains clean baseline forecasts only; the test split contains clean, trend drift, high-frequency noise, local oscillation, and real turning-point cases.

The default correction is intentionally conservative: it uses the 99th percentile calibration threshold, caps trend adjustment magnitude, and skips HalluGuard correction when the forecast itself looks like an internal turning-point case.

## Metrics

- `MSE`, `MAE`: point forecast error against `target`.
- `HallucinationRate`: fraction of corrected predictions still violating the calibrated trend or frequency threshold.
- `TrendViolationRate`, `FreqViolationRate`: post-correction rule violation rates.
- `SpectralConsistency`: `1 / (1 + spectral_distance)`, higher is better.
- `TurningPointFalseCorrectionRate`: among real turning-point samples, fraction where correction changed the forecast and increased sample MSE versus the original prediction.
- `InferenceLatency`: average correction time per sample in milliseconds.

## External Framework Adapter

External frameworks only need to export predictions in the schema documented in `EXTERNAL_PREDICTION_SCHEMA.md`. HalluGuard does not need the original model, dataloader, trainer, or hidden states; it only reads `context`, `prediction`, and `target` rows from JSONL or CSV.
