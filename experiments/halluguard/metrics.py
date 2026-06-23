"""Metrics for HalluGuard MVP evaluation."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np

from correction import Thresholds, score_sample, trigger_flags


def _arr(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def mse(prediction: Iterable[float], target: Iterable[float]) -> float:
    pred = _arr(prediction)
    tgt = _arr(target)
    return float(np.mean((pred - tgt) ** 2))


def mae(prediction: Iterable[float], target: Iterable[float]) -> float:
    pred = _arr(prediction)
    tgt = _arr(target)
    return float(np.mean(np.abs(pred - tgt)))


def aggregate_metrics(
    samples: List[dict],
    corrected_predictions: List[np.ndarray],
    thresholds: Thresholds,
    high_freq_cutoff_ratio: float,
    latencies_ms: List[float],
    variant: str,
    lambda_trend: float,
    lambda_freq: float,
    freq_score_mode: str = "excess_plus_spectral",
) -> Dict[str, float]:
    if len(samples) != len(corrected_predictions):
        raise ValueError("samples and corrected_predictions must have the same length.")

    sample_mse = []
    sample_mae = []
    baseline_mse = []
    baseline_mae = []
    trend_flags = []
    freq_flags = []
    spectral_consistency = []
    changed_flags = []
    tp_false_flags = []
    tp_changed_flags = []

    for sample, corrected in zip(samples, corrected_predictions):
        target = sample["target"]
        original = sample["prediction"]
        sample_mse.append(mse(corrected, target))
        sample_mae.append(mae(corrected, target))
        baseline_mse.append(mse(original, target))
        baseline_mae.append(mae(original, target))
        score = score_sample(sample["context"], corrected, high_freq_cutoff_ratio, freq_score_mode=freq_score_mode)
        trend, freq = trigger_flags(score, thresholds)
        trend_flags.append(trend)
        freq_flags.append(freq)
        spectral_consistency.append(1.0 / (1.0 + score["spectral_distance"]))
        changed = bool(np.max(np.abs(_arr(corrected) - _arr(original))) > 1e-10)
        changed_flags.append(changed)
        if sample.get("is_turning_point", False):
            tp_changed_flags.append(changed)
            tp_false_flags.append(changed and (sample_mse[-1] > baseline_mse[-1] + 1e-12))

    trend_arr = np.asarray(trend_flags, dtype=bool)
    freq_arr = np.asarray(freq_flags, dtype=bool)
    hallucination = np.logical_or(trend_arr, freq_arr)
    sample_mse_arr = np.asarray(sample_mse, dtype=np.float64)
    sample_mae_arr = np.asarray(sample_mae, dtype=np.float64)
    baseline_mse_arr = np.asarray(baseline_mse, dtype=np.float64)
    baseline_mae_arr = np.asarray(baseline_mae, dtype=np.float64)

    tp_false_rate = float(np.mean(tp_false_flags)) if tp_false_flags else 0.0
    tp_correction_rate = float(np.mean(tp_changed_flags)) if tp_changed_flags else 0.0
    baseline_mse_mean = float(baseline_mse_arr.mean()) if baseline_mse_arr.size else 0.0
    baseline_mae_mean = float(baseline_mae_arr.mean()) if baseline_mae_arr.size else 0.0

    return {
        "variant": variant,
        "n_samples": len(samples),
        "lambda_trend": float(lambda_trend),
        "lambda_freq": float(lambda_freq),
        "threshold_quantile": float(thresholds.quantile),
        "mse": float(sample_mse_arr.mean()) if sample_mse_arr.size else 0.0,
        "mae": float(sample_mae_arr.mean()) if sample_mae_arr.size else 0.0,
        "baseline_mse": baseline_mse_mean,
        "baseline_mae": baseline_mae_mean,
        "mse_delta_pct_vs_original": _pct_delta(float(sample_mse_arr.mean()), baseline_mse_mean),
        "mae_delta_pct_vs_original": _pct_delta(float(sample_mae_arr.mean()), baseline_mae_mean),
        "hallucination_rate": float(hallucination.mean()) if hallucination.size else 0.0,
        "trend_violation_rate": float(trend_arr.mean()) if trend_arr.size else 0.0,
        "freq_violation_rate": float(freq_arr.mean()) if freq_arr.size else 0.0,
        "spectral_consistency": float(np.mean(spectral_consistency)) if spectral_consistency else 0.0,
        "turning_point_false_correction_rate": tp_false_rate,
        "turning_point_correction_rate": tp_correction_rate,
        "correction_rate": float(np.mean(changed_flags)) if changed_flags else 0.0,
        "inference_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
    }


def stress_slice_metrics(
    samples: List[dict],
    corrected_predictions: List[np.ndarray],
) -> Dict[str, Dict[str, float]]:
    by_type: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        by_type.setdefault(sample.get("stress_type", "unknown"), []).append(idx)

    out: Dict[str, Dict[str, float]] = {}
    for stress_type, indices in sorted(by_type.items()):
        corrected_mse = [mse(corrected_predictions[i], samples[i]["target"]) for i in indices]
        corrected_mae = [mae(corrected_predictions[i], samples[i]["target"]) for i in indices]
        original_mse = [mse(samples[i]["prediction"], samples[i]["target"]) for i in indices]
        original_mae = [mae(samples[i]["prediction"], samples[i]["target"]) for i in indices]
        out[stress_type] = {
            "n_samples": len(indices),
            "mse": float(np.mean(corrected_mse)),
            "mae": float(np.mean(corrected_mae)),
            "baseline_mse": float(np.mean(original_mse)),
            "baseline_mae": float(np.mean(original_mae)),
            "mse_delta_pct_vs_original": _pct_delta(float(np.mean(corrected_mse)), float(np.mean(original_mse))),
            "mae_delta_pct_vs_original": _pct_delta(float(np.mean(corrected_mae)), float(np.mean(original_mae))),
        }
    return out


def _pct_delta(value: float, baseline: float) -> float:
    if abs(baseline) <= 1e-12:
        return 0.0
    return 100.0 * (value - baseline) / baseline
