"""Trend/frequency guarded correction for HalluGuard MVP."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class Thresholds:
    quantile: float
    trend: float
    freq: float
    source_split: str
    trend_val_rate: float
    freq_val_rate: float
    combined_val_rate: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class CorrectionResult:
    prediction: np.ndarray
    info: Dict[str, float]


def as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def ols_slope(values: Iterable[float]) -> float:
    series = as_array(values)
    if series.size < 2:
        return 0.0
    t = np.arange(series.size, dtype=np.float64)
    t = t - t.mean()
    y = series - series.mean()
    denom = float(np.dot(t, t))
    if denom <= EPS:
        return 0.0
    return float(np.dot(t, y) / denom)


def normalized_power_spectrum(values: Iterable[float], n_points: int = 32) -> np.ndarray:
    series = as_array(values)
    if series.size < 2:
        return np.zeros(n_points, dtype=np.float64)
    centered = series - series.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    if power.size:
        power[0] = 0.0
    total = float(power.sum())
    if total <= EPS:
        return np.zeros(n_points, dtype=np.float64)
    power = power / total
    freqs = np.fft.rfftfreq(series.size)
    grid = np.linspace(0.0, float(freqs.max()) if freqs.size else 0.5, n_points)
    return np.interp(grid, freqs, power, left=0.0, right=0.0)


def spectral_distance(context: Iterable[float], prediction: Iterable[float]) -> float:
    ctx_power = normalized_power_spectrum(context)
    pred_power = normalized_power_spectrum(prediction)
    return float(0.5 * np.abs(ctx_power - pred_power).sum())


def high_frequency_energy_ratio(
    values: Iterable[float],
    high_freq_cutoff_ratio: float = 0.5,
) -> float:
    series = as_array(values)
    if series.size < 4:
        return 0.0
    centered = series - series.mean()
    power = np.abs(np.fft.rfft(centered)) ** 2
    if power.size:
        power[0] = 0.0
    total = float(power.sum())
    if total <= EPS:
        return 0.0
    freqs = np.fft.rfftfreq(series.size)
    cutoff = float(high_freq_cutoff_ratio) * float(freqs.max())
    mask = freqs >= cutoff
    if mask.size:
        mask[0] = False
    return float(power[mask].sum() / total)


def curvature_roughness(values: Iterable[float]) -> float:
    series = as_array(values)
    if series.size < 4:
        return 0.0
    scale = float(np.std(series)) + EPS
    return float(np.std(np.diff(series, n=2)) / scale)


def score_sample(
    context: Iterable[float],
    prediction: Iterable[float],
    high_freq_cutoff_ratio: float = 0.5,
    freq_score_mode: str = "excess_plus_spectral",
) -> Dict[str, float]:
    ctx = as_array(context)
    pred = as_array(prediction)
    context_slope = ols_slope(ctx)
    prediction_slope = ols_slope(pred)
    scale = float(np.std(ctx)) + EPS
    trend_score = abs(prediction_slope - context_slope) * max(pred.size, 1) / scale
    context_hf = high_frequency_energy_ratio(ctx, high_freq_cutoff_ratio)
    prediction_hf = high_frequency_energy_ratio(pred, high_freq_cutoff_ratio)
    spec_dist = spectral_distance(ctx, pred)
    hf_excess = max(0.0, prediction_hf - context_hf)
    context_roughness = curvature_roughness(ctx)
    prediction_roughness = curvature_roughness(pred)
    roughness_excess = max(0.0, prediction_roughness - context_roughness)
    if freq_score_mode == "relative_excess":
        freq_score = hf_excess / (context_hf + 0.02) + 0.10 * spec_dist
    elif freq_score_mode == "curvature_excess":
        freq_score = roughness_excess + 0.50 * hf_excess + 0.10 * spec_dist
    else:
        freq_score = hf_excess + 0.25 * spec_dist
    return {
        "trend_score": float(trend_score),
        "freq_score": float(freq_score),
        "context_slope": float(context_slope),
        "prediction_slope": float(prediction_slope),
        "context_hf_ratio": float(context_hf),
        "prediction_hf_ratio": float(prediction_hf),
        "context_roughness": float(context_roughness),
        "prediction_roughness": float(prediction_roughness),
        "roughness_excess": float(roughness_excess),
        "spectral_distance": float(spec_dist),
    }


def calibrate_thresholds(
    calibration_samples,
    quantile: float,
    high_freq_cutoff_ratio: float = 0.5,
    source_split: str = "val",
    freq_score_mode: str = "excess_plus_spectral",
) -> Thresholds:
    if not calibration_samples:
        raise ValueError("Cannot calibrate thresholds from an empty split.")
    scores = [
        score_sample(s["context"], s["prediction"], high_freq_cutoff_ratio, freq_score_mode=freq_score_mode)
        for s in calibration_samples
    ]
    trend_values = np.asarray([s["trend_score"] for s in scores], dtype=np.float64)
    freq_values = np.asarray([s["freq_score"] for s in scores], dtype=np.float64)
    trend_threshold = float(np.quantile(trend_values, quantile))
    freq_threshold = float(np.quantile(freq_values, quantile))
    trend_flags = trend_values > trend_threshold
    freq_flags = freq_values > freq_threshold
    return Thresholds(
        quantile=float(quantile),
        trend=trend_threshold,
        freq=freq_threshold,
        source_split=source_split,
        trend_val_rate=float(trend_flags.mean()),
        freq_val_rate=float(freq_flags.mean()),
        combined_val_rate=float(np.logical_or(trend_flags, freq_flags).mean()),
    )


def trigger_flags(score: Dict[str, float], thresholds: Thresholds) -> Tuple[bool, bool]:
    trend = bool(score["trend_score"] > thresholds.trend)
    freq = bool(score["freq_score"] > thresholds.freq)
    return trend, freq


def trend_correction(
    context: Iterable[float],
    prediction: Iterable[float],
    strength: float,
    max_adjustment_ratio: Optional[float] = 0.08,
) -> np.ndarray:
    pred = as_array(prediction).copy()
    if pred.size < 2 or strength <= 0:
        return pred
    slope_gap = ols_slope(pred) - ols_slope(context)
    centered_t = np.arange(pred.size, dtype=np.float64) - (pred.size - 1) / 2.0
    adjustment = float(strength) * slope_gap * centered_t
    if max_adjustment_ratio is not None and max_adjustment_ratio > 0:
        cap = float(max_adjustment_ratio) * (float(np.std(as_array(context))) + EPS)
        max_abs = float(np.max(np.abs(adjustment))) if adjustment.size else 0.0
        if max_abs > cap:
            adjustment = adjustment * (cap / (max_abs + EPS))
    return pred - adjustment


def frequency_correction(
    prediction: Iterable[float],
    strength: float,
    high_freq_cutoff_ratio: float = 0.5,
) -> np.ndarray:
    pred = as_array(prediction)
    if pred.size < 4 or strength <= 0:
        return pred.copy()
    coeffs = np.fft.rfft(pred)
    freqs = np.fft.rfftfreq(pred.size)
    cutoff = float(high_freq_cutoff_ratio) * float(freqs.max())
    mask = freqs >= cutoff
    if mask.size:
        mask[0] = False
    coeffs[mask] *= 1.0 - float(strength)
    return np.fft.irfft(coeffs, n=pred.size).astype(np.float64)


def naive_smoothing(prediction: Iterable[float], window: int = 5) -> np.ndarray:
    pred = as_array(prediction)
    if window <= 1 or pred.size < 3:
        return pred.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(pred, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def has_internal_turning_point(
    context: Iterable[float],
    prediction: Iterable[float],
    min_score: float = 0.25,
) -> bool:
    pred = as_array(prediction)
    if pred.size < 8:
        return False
    mid = pred.size // 2
    left_slope = ols_slope(pred[:mid])
    right_slope = ols_slope(pred[mid:])
    scale = float(np.std(as_array(context))) + EPS
    score = abs(right_slope - left_slope) * pred.size / scale
    return bool(left_slope * right_slope < 0.0 and score >= float(min_score))


def apply_correction(
    context: Iterable[float],
    prediction: Iterable[float],
    thresholds: Thresholds,
    variant: str,
    lambda_trend: float = 0.3,
    lambda_freq: float = 0.3,
    high_freq_cutoff_ratio: float = 0.5,
    smoothing_window: int = 5,
    turning_point_guard: bool = True,
    turning_point_guard_min_score: float = 0.25,
    max_trend_adjustment_ratio: Optional[float] = 0.08,
    forced_trend_trigger: Optional[bool] = None,
    forced_freq_trigger: Optional[bool] = None,
    freq_score_mode: str = "excess_plus_spectral",
) -> CorrectionResult:
    pred = as_array(prediction)
    score = score_sample(context, pred, high_freq_cutoff_ratio, freq_score_mode=freq_score_mode)
    rule_trend, rule_freq = trigger_flags(score, thresholds)
    trend_flag = rule_trend if forced_trend_trigger is None else bool(forced_trend_trigger)
    freq_flag = rule_freq if forced_freq_trigger is None else bool(forced_freq_trigger)

    corrected = pred.copy()
    applied_trend = False
    applied_freq = False
    guarded_turning_point = bool(
        turning_point_guard
        and has_internal_turning_point(context, pred, min_score=turning_point_guard_min_score)
    )

    if variant == "no_correction":
        pass
    elif variant == "naive_smoothing":
        corrected = naive_smoothing(corrected, smoothing_window)
    elif variant == "matched_smoothing_control":
        if (trend_flag or freq_flag) and not guarded_turning_point:
            corrected = naive_smoothing(corrected, smoothing_window)
            applied_freq = True
    elif variant == "trend_only":
        if trend_flag and not guarded_turning_point:
            corrected = trend_correction(context, corrected, lambda_trend, max_trend_adjustment_ratio)
            applied_trend = True
    elif variant == "frequency_only":
        if freq_flag and not guarded_turning_point:
            corrected = frequency_correction(corrected, lambda_freq, high_freq_cutoff_ratio)
            applied_freq = True
    elif variant in {"trend_frequency", "random_trigger"}:
        if trend_flag and not guarded_turning_point:
            corrected = trend_correction(context, corrected, lambda_trend, max_trend_adjustment_ratio)
            applied_trend = True
        if freq_flag and not guarded_turning_point:
            corrected = frequency_correction(corrected, lambda_freq, high_freq_cutoff_ratio)
            applied_freq = True
    else:
        raise ValueError(f"Unknown correction variant: {variant}")

    changed = bool(np.max(np.abs(corrected - pred)) > 1e-10) if corrected.size else False
    info = {
        **score,
        "rule_trend_trigger": float(rule_trend),
        "rule_freq_trigger": float(rule_freq),
        "used_trend_trigger": float(trend_flag),
        "used_freq_trigger": float(freq_flag),
        "applied_trend": float(applied_trend),
        "applied_freq": float(applied_freq),
        "turning_point_guard": float(guarded_turning_point),
        "changed": float(changed),
    }
    return CorrectionResult(prediction=corrected, info=info)
