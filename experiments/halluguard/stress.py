"""Deterministic synthetic/stress benchmark for HalluGuard MVP."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from correction import ols_slope


DATASET_NAME = "synthetic_halluguard_stress"
MODEL_NAME = "deterministic_synthetic_baseline"


def generate_synthetic_samples(config: Dict, quick: bool = False) -> List[dict]:
    data_cfg = config["data"]
    seed = int(config.get("seed", 7))
    counts = {
        "train": int(data_cfg.get("n_train", 512)),
        "val": int(data_cfg.get("n_val", 256)),
        "test": int(data_cfg.get("n_test", 512)),
    }
    if quick:
        counts = {
            "train": min(counts["train"], 80),
            "val": min(counts["val"], 80),
            "test": min(counts["test"], 120),
        }
    context_len = int(data_cfg.get("context_len", 96))
    horizon = int(data_cfg.get("horizon", 24))
    stress_types = list(data_cfg.get("stress_types", ["clean"]))
    calibration_types = list(data_cfg.get("calibration_stress_types", ["clean"]))

    samples: List[dict] = []
    for split, n_samples in counts.items():
        split_types = calibration_types if split == "val" else stress_types
        for idx in range(n_samples):
            stress_type = split_types[idx % len(split_types)]
            rng = np.random.default_rng(seed + _split_offset(split) + idx)
            sample = make_sample(rng, split, idx, context_len, horizon, stress_type)
            samples.append(sample)
    return samples


def make_sample(
    rng: np.random.Generator,
    split: str,
    idx: int,
    context_len: int,
    horizon: int,
    stress_type: str,
) -> dict:
    context, target, params = _base_context_target(rng, context_len, horizon)
    scale = float(np.std(context)) + 1e-6
    prediction = target + rng.normal(0.0, 0.055 * scale, size=horizon)
    violation_family = "none"
    is_turning_point = False

    if stress_type == "clean":
        pass
    elif stress_type == "trend_drift":
        sign = rng.choice([-1.0, 1.0])
        drift = sign * rng.uniform(1.0, 1.8) * scale * np.linspace(0.0, 1.0, horizon)
        prediction = prediction + drift
        violation_family = "trend"
    elif stress_type == "high_frequency_noise":
        t = np.arange(horizon, dtype=np.float64)
        freq = rng.uniform(0.32, 0.46)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        prediction = prediction + rng.uniform(0.45, 0.85) * scale * np.sin(2.0 * np.pi * freq * t + phase)
        violation_family = "frequency"
    elif stress_type == "local_oscillation":
        t = np.arange(horizon, dtype=np.float64)
        center = rng.uniform(0.35 * horizon, 0.70 * horizon)
        width = rng.uniform(0.10 * horizon, 0.18 * horizon)
        window = np.exp(-0.5 * ((t - center) / max(width, 1.0)) ** 2)
        freq = rng.uniform(0.26, 0.42)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        burst = window * np.sin(2.0 * np.pi * freq * t + phase)
        prediction = prediction + rng.uniform(0.75, 1.25) * scale * burst
        violation_family = "frequency"
    elif stress_type == "real_turning_point":
        target = _turning_point_target(rng, context, target, params)
        prediction = target + rng.normal(0.0, 0.055 * scale, size=horizon)
        violation_family = "true_turning_point"
        is_turning_point = True
    else:
        raise ValueError(f"Unknown synthetic stress type: {stress_type}")

    return {
        "sample_id": f"{split}_{idx:05d}",
        "dataset": DATASET_NAME,
        "model": MODEL_NAME,
        "split": split,
        "stress_type": stress_type,
        "violation_family": violation_family,
        "is_turning_point": is_turning_point,
        "context": context.astype(float).tolist(),
        "prediction": prediction.astype(float).tolist(),
        "target": target.astype(float).tolist(),
    }


def _base_context_target(
    rng: np.random.Generator,
    context_len: int,
    horizon: int,
) -> tuple:
    total = context_len + horizon
    t = np.arange(total, dtype=np.float64)
    level = rng.normal(0.0, 0.8)
    slope = rng.uniform(-0.020, 0.020)
    if abs(slope) < 0.004:
        slope += rng.choice([-1.0, 1.0]) * 0.004
    amp = rng.uniform(0.45, 1.20)
    period = float(rng.choice([24, 32, 48, 64]))
    phase = rng.uniform(0.0, 2.0 * np.pi)
    phase2 = rng.uniform(0.0, 2.0 * np.pi)
    seasonal = amp * np.sin(2.0 * np.pi * t / period + phase)
    seasonal += 0.22 * amp * np.sin(4.0 * np.pi * t / period + phase2)
    noise = rng.normal(0.0, 0.025 + 0.015 * amp, size=total)
    series = level + slope * t + seasonal + noise
    params = {
        "slope": slope,
        "amp": amp,
        "period": period,
        "phase": phase,
        "phase2": phase2,
    }
    return series[:context_len], series[context_len:], params


def _turning_point_target(
    rng: np.random.Generator,
    context: np.ndarray,
    clean_target: np.ndarray,
    params: Dict[str, float],
) -> np.ndarray:
    horizon = clean_target.size
    t = np.arange(horizon, dtype=np.float64)
    local_slope = ols_slope(context[-min(48, context.size) :])
    direction = -1.0 if local_slope >= 0.0 else 1.0
    if abs(local_slope) < 0.006:
        direction = rng.choice([-1.0, 1.0])
    scale = float(np.std(context)) + 1e-6
    turn = int(rng.integers(max(3, horizon // 4), max(4, 3 * horizon // 4)))
    bend_strength = rng.uniform(0.035, 0.065) * scale
    bend = direction * bend_strength * np.maximum(0.0, t - float(turn))
    corner = direction * 0.12 * scale * np.tanh((t - float(turn)) / max(horizon / 10.0, 1.0))
    seasonal_guard = 0.04 * params["amp"] * np.sin(2.0 * np.pi * t / max(params["period"] / 2.0, 2.0) + params["phase2"])
    return clean_target + bend + corner + seasonal_guard


def _split_offset(split: str) -> int:
    if split == "train":
        return 0
    if split == "val":
        return 100_000
    if split == "test":
        return 200_000
    return 300_000
