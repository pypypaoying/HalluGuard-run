"""Reusable HalluGuard-Dynamics API.

The API is intentionally small:

- fit_policy(validation_samples, config)
- score_sample(context, prediction, policy)
- apply_correction(context, prediction, policy)
- evaluate_table(validation_samples, evaluation_samples, config)

All policy fitting uses validation samples only. Test samples should only be
passed to apply/evaluate after the policy is frozen.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from correction import naive_smoothing
from metrics import mae, mse


EPS = 1e-12
COMPONENTS = ("boundary", "first_diff", "curvature")


@dataclass(frozen=True)
class VariantSpec:
    name: str
    score_components: Tuple[str, ...]
    repair_components: Tuple[str, ...]
    is_main: bool = False


def default_variant_specs() -> List[VariantSpec]:
    return [
        VariantSpec("dynamics_full", ("boundary", "first_diff", "curvature"), ("boundary", "first_diff"), True),
        VariantSpec("boundary_only", ("boundary",), ("boundary",)),
        VariantSpec("first_diff_only", ("first_diff",), ("first_diff",)),
        VariantSpec("curvature_only", ("curvature",), ("curvature",)),
        VariantSpec("boundary_first_diff", ("boundary", "first_diff"), ("boundary", "first_diff")),
        VariantSpec("boundary_curvature", ("boundary", "curvature"), ("boundary", "curvature")),
        VariantSpec("first_diff_curvature", ("first_diff", "curvature"), ("first_diff", "curvature")),
    ]


def arr(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def fit_policy(validation_samples: List[dict], config: Dict, variant: Optional[VariantSpec] = None) -> Dict[str, object]:
    """Fit a validation-only HalluGuard-Dynamics policy."""

    if not validation_samples:
        raise ValueError("fit_policy requires a non-empty validation split.")
    if variant is None:
        variant = default_variant_specs()[0]

    policy_cfg = config.get("policy", {}) or {}
    quantiles = [float(v) for v in policy_cfg.get("trigger_quantiles", [0.50, 0.70, 0.80, 0.90, 0.95])]
    strengths = [float(v) for v in policy_cfg.get("correction_strengths", [0.10, 0.30, 0.50, 0.80, 1.00])]
    random_seeds = [int(v) for v in policy_cfg.get("random_seeds", [1101, 2202, 3303, 4404, 5505])]
    random_weight = float(policy_cfg.get("random_separation_weight", 0.50))
    anti_smoothing_weight = float(policy_cfg.get("anti_smoothing_weight", 1.00)) if variant.is_main else 0.0
    rate_penalty = float(policy_cfg.get("correction_rate_penalty", 0.0002))

    base_predictions = np.asarray([arr(s["prediction"]) for s in validation_samples], dtype=np.float64)
    targets = np.asarray([arr(s["target"]) for s in validation_samples], dtype=np.float64)
    scores = np.asarray([score_sample(s["context"], s["prediction"], {"score_components": variant.score_components})["score"] for s in validation_samples], dtype=np.float64)
    vectors = np.asarray([correction_vector(s["context"], s["prediction"], variant.repair_components, config) for s in validation_samples], dtype=np.float64)
    baseline_mse = array_mse(base_predictions, targets)
    smoothing_window = int(policy_cfg.get("smoothing_window", 5))
    smoothed_base_predictions = np.asarray([naive_smoothing(pred, smoothing_window) for pred in base_predictions], dtype=np.float64)

    best: Optional[Dict[str, object]] = None
    for quantile in quantiles:
        threshold = float(np.quantile(scores, quantile))
        mask = scores >= threshold
        if not mask.any():
            mask[int(np.argmax(scores))] = True
        matched_predictions = base_predictions.copy()
        matched_predictions[mask] = smoothed_base_predictions[mask]
        matched_mse = array_mse(matched_predictions, targets)
        for strength in strengths:
            candidate_predictions = _apply_vectors(base_predictions, vectors, mask, strength)
            candidate_mse = array_mse(candidate_predictions, targets)

            random_mses = []
            for seed in random_seeds:
                random_mask = matched_random_mask(len(validation_samples), int(mask.sum()), seed)
                random_predictions = _apply_vectors(base_predictions, vectors, random_mask, strength)
                random_mses.append(array_mse(random_predictions, targets))
            random_mse = float(np.mean(random_mses)) if random_mses else candidate_mse

            trigger_rate = float(mask.mean())
            random_advantage = random_mse - candidate_mse
            matched_advantage = matched_mse - candidate_mse
            objective = (
                candidate_mse
                - random_weight * random_advantage
                - anti_smoothing_weight * matched_advantage
                + rate_penalty * baseline_mse * trigger_rate
            )
            record = {
                "variant": variant.name,
                "score_components": list(variant.score_components),
                "repair_components": list(variant.repair_components),
                "source_split": "val",
                "trigger_quantile": float(quantile),
                "trigger_threshold": threshold,
                "correction_strength": float(strength),
                "validation_baseline_mse": baseline_mse,
                "validation_mse": candidate_mse,
                "validation_mse_delta_pct": pct_delta(candidate_mse, baseline_mse),
                "validation_random_mse": random_mse,
                "validation_random_advantage_mse": random_advantage,
                "validation_matched_smoothing_mse": matched_mse,
                "validation_matched_advantage_mse": matched_advantage,
                "validation_trigger_rate": trigger_rate,
                "objective": objective,
                "test_threshold_leakage": False,
                "smoothing_window": smoothing_window,
                "ema_alpha": float(policy_cfg.get("ema_alpha", 0.35)),
                "median_window": int(policy_cfg.get("median_window", 5)),
                "max_adjustment_ratio": policy_cfg.get("max_adjustment_ratio", None),
            }
            if best is None or float(record["objective"]) < float(best["objective"]):
                best = record
    assert best is not None
    return best


def score_sample(context: Iterable[float], prediction: Iterable[float], policy: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    """Score local dynamics discontinuity from context to prediction."""

    components = tuple((policy or {}).get("score_components", COMPONENTS))
    signed = dynamics_signed_terms(context, prediction)
    raw = {
        "boundary": abs(signed["boundary_jump_scaled"]),
        "first_diff": abs(signed["first_diff_gap_scaled"]),
        "curvature": abs(signed["curvature_gap_scaled"]),
    }
    score = 0.0
    if "boundary" in components:
        score += raw["boundary"]
    if "first_diff" in components:
        score += 0.5 * raw["first_diff"]
    if "curvature" in components:
        score += 0.25 * raw["curvature"]
    return {
        "score": float(score),
        "boundary_score": float(raw["boundary"]),
        "first_diff_score": float(raw["first_diff"]),
        "curvature_score": float(raw["curvature"]),
        **signed,
    }


def apply_correction(
    context: Iterable[float],
    prediction: Iterable[float],
    policy: Dict[str, object],
    force_trigger: Optional[bool] = None,
    strength_scale: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply HalluGuard-Dynamics correction with a frozen policy."""

    pred = arr(prediction)
    score = score_sample(context, pred, policy)
    trigger = bool(score["score"] >= float(policy["trigger_threshold"]))
    if force_trigger is not None:
        trigger = bool(force_trigger)
    vector = correction_vector(context, pred, tuple(policy.get("repair_components", ("boundary", "first_diff"))), {"policy": policy})
    strength = float(policy.get("correction_strength", 0.0)) * float(strength_scale)
    corrected = pred + strength * vector if trigger else pred.copy()
    changed = bool(np.max(np.abs(corrected - pred)) > 1e-10) if pred.size else False
    info = {
        **score,
        "trigger": float(trigger),
        "changed": float(changed),
        "correction_strength": strength,
    }
    return corrected.astype(np.float64), info


def evaluate_table(
    validation_samples: List[dict],
    evaluation_samples: List[dict],
    config: Dict,
    variants: Optional[List[VariantSpec]] = None,
    seed: int = 23,
) -> Dict[str, object]:
    """Fit policies on validation and evaluate core dynamics variants."""

    if variants is None:
        variants = default_variant_specs()
    policies = {variant.name: fit_policy(validation_samples, config, variant) for variant in variants}
    rows = []
    for variant in variants:
        predictions, infos, latencies = apply_policy_to_samples(evaluation_samples, policies[variant.name])
        rows.append(metric_row(variant.name, evaluation_samples, predictions, infos, latencies, policies[variant.name]))
    return {"policies": policies, "rows": rows, "test_threshold_leakage": False, "seed": seed}


def apply_policy_to_samples(samples: List[dict], policy: Dict[str, object]) -> Tuple[List[np.ndarray], List[Dict[str, float]], List[float]]:
    predictions = []
    infos = []
    latencies = []
    for sample in samples:
        start = time.perf_counter()
        corrected, info = apply_correction(sample["context"], sample["prediction"], policy)
        latencies.append((time.perf_counter() - start) * 1000.0)
        predictions.append(corrected)
        infos.append(info)
    return predictions, infos, latencies


def correction_vector(context: Iterable[float], prediction: Iterable[float], repair_components: Iterable[str], config: Optional[Dict] = None) -> np.ndarray:
    ctx = arr(context)
    pred = arr(prediction)
    if pred.size == 0:
        return pred.copy()
    components = tuple(repair_components)
    signed = dynamics_signed_terms(ctx, pred)
    t = np.arange(pred.size, dtype=np.float64)
    denom = max(pred.size - 1, 1)
    u = t / float(denom)
    decay = np.exp(-t / max(4.0, pred.size / 12.0))
    vector = np.zeros_like(pred, dtype=np.float64)
    if "boundary" in components:
        vector += -signed["boundary_jump"] * decay
    if "first_diff" in components:
        vector += -signed["first_diff_gap"] * u * decay
    if "curvature" in components:
        vector += -signed["curvature_gap"] * (u**2) * decay

    max_ratio = None
    if config:
        max_ratio = (config.get("policy", {}) or {}).get("max_adjustment_ratio", None)
    if max_ratio is not None:
        cap = float(max_ratio) * (float(np.std(ctx)) + EPS)
        max_abs = float(np.max(np.abs(vector))) if vector.size else 0.0
        if max_abs > cap > 0.0:
            vector = vector * (cap / (max_abs + EPS))
    return vector.astype(np.float64)


def dynamics_signed_terms(context: Iterable[float], prediction: Iterable[float]) -> Dict[str, float]:
    ctx = arr(context)
    pred = arr(prediction)
    scale = float(np.std(ctx)) + EPS
    ctx_last = float(ctx[-1]) if ctx.size else 0.0
    pred_first = float(pred[0]) if pred.size else ctx_last
    last_diff = float(ctx[-1] - ctx[-2]) if ctx.size >= 2 else 0.0
    pred_first_diff = float(pred[1] - pred[0]) if pred.size >= 2 else 0.0
    boundary_jump = pred_first - (ctx_last + last_diff)
    first_diff_gap = pred_first_diff - last_diff
    ctx_tail = ctx[-min(ctx.size, 16) :]
    pred_head = pred[: min(pred.size, 16)]
    ctx_curvature = float(np.mean(np.diff(ctx_tail, n=2))) if ctx_tail.size >= 3 else 0.0
    pred_curvature = float(np.mean(np.diff(pred_head, n=2))) if pred_head.size >= 3 else 0.0
    curvature_gap = pred_curvature - ctx_curvature
    return {
        "boundary_jump": float(boundary_jump),
        "first_diff_gap": float(first_diff_gap),
        "curvature_gap": float(curvature_gap),
        "boundary_jump_scaled": float(boundary_jump / scale),
        "first_diff_gap_scaled": float(first_diff_gap / scale),
        "curvature_gap_scaled": float(curvature_gap / scale),
        "context_scale": float(scale),
    }


def trigger_mask_for_samples(samples: List[dict], policy: Dict[str, object]) -> np.ndarray:
    scores = np.asarray([score_sample(s["context"], s["prediction"], policy)["score"] for s in samples], dtype=np.float64)
    mask = scores >= float(policy["trigger_threshold"])
    if not mask.any() and mask.size:
        mask[int(np.argmax(scores))] = True
    return mask


def vectors_for_samples(samples: List[dict], policy: Dict[str, object], config: Optional[Dict] = None) -> np.ndarray:
    return np.asarray(
        [correction_vector(s["context"], s["prediction"], tuple(policy.get("repair_components", ("boundary", "first_diff"))), config or {"policy": policy}) for s in samples],
        dtype=np.float64,
    )


def matched_random_mask(n: int, count: int, seed: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    if n <= 0 or count <= 0:
        return mask
    rng = np.random.default_rng(seed)
    mask[rng.choice(n, size=min(count, n), replace=False)] = True
    return mask


def shuffled_score_mask(samples: List[dict], policy: Dict[str, object], seed: int) -> np.ndarray:
    scores = np.asarray([score_sample(s["context"], s["prediction"], policy)["score"] for s in samples], dtype=np.float64)
    rng = np.random.default_rng(seed)
    shuffled = scores.copy()
    rng.shuffle(shuffled)
    mask = shuffled >= float(policy["trigger_threshold"])
    if not mask.any() and mask.size:
        mask[int(np.argmax(shuffled))] = True
    return mask


def apply_vector_predictions(samples: List[dict], policy: Dict[str, object], mask: np.ndarray, strength_scale: float = 1.0, config: Optional[Dict] = None) -> Tuple[List[np.ndarray], List[Dict[str, float]], List[float]]:
    predictions = []
    infos = []
    latencies = []
    strength = float(policy.get("correction_strength", 0.0)) * float(strength_scale)
    for idx, sample in enumerate(samples):
        start = time.perf_counter()
        pred = arr(sample["prediction"])
        vector = correction_vector(sample["context"], pred, tuple(policy.get("repair_components", ("boundary", "first_diff"))), config or {"policy": policy})
        corrected = pred + strength * vector if bool(mask[idx]) else pred.copy()
        score = score_sample(sample["context"], pred, policy)
        changed = bool(np.max(np.abs(corrected - pred)) > 1e-10) if pred.size else False
        latencies.append((time.perf_counter() - start) * 1000.0)
        predictions.append(corrected)
        infos.append({**score, "trigger": float(bool(mask[idx])), "changed": float(changed), "correction_strength": strength})
    return predictions, infos, latencies


def ema_smoothing(prediction: Iterable[float], alpha: float = 0.35) -> np.ndarray:
    pred = arr(prediction)
    if pred.size <= 1:
        return pred.copy()
    out = pred.copy()
    alpha = float(alpha)
    for idx in range(1, pred.size):
        out[idx] = alpha * pred[idx] + (1.0 - alpha) * out[idx - 1]
    return out


def median_smoothing(prediction: Iterable[float], window: int = 5) -> np.ndarray:
    pred = arr(prediction)
    if pred.size < 3 or int(window) <= 1:
        return pred.copy()
    window = int(window)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(pred, (pad, pad), mode="edge")
    out = np.empty_like(pred, dtype=np.float64)
    for idx in range(pred.size):
        out[idx] = float(np.median(padded[idx : idx + window]))
    return out


def smoothing_predictions(samples: List[dict], kind: str, config: Dict) -> Tuple[List[np.ndarray], List[Dict[str, float]], List[float]]:
    policy_cfg = config.get("policy", {}) or {}
    predictions = []
    infos = []
    latencies = []
    for sample in samples:
        start = time.perf_counter()
        if kind == "naive_smoothing":
            pred = naive_smoothing(sample["prediction"], int(policy_cfg.get("smoothing_window", 5)))
        elif kind == "ema_smoothing":
            pred = ema_smoothing(sample["prediction"], float(policy_cfg.get("ema_alpha", 0.35)))
        elif kind == "median_smoothing":
            pred = median_smoothing(sample["prediction"], int(policy_cfg.get("median_window", 5)))
        else:
            raise ValueError(f"Unknown smoothing kind: {kind}")
        latencies.append((time.perf_counter() - start) * 1000.0)
        predictions.append(pred)
        infos.append({"trigger": 1.0, "changed": float(np.max(np.abs(pred - arr(sample["prediction"]))) > 1e-10)})
    return predictions, infos, latencies


def metric_row(
    variant: str,
    samples: List[dict],
    predictions: List[np.ndarray],
    infos: List[Dict[str, float]],
    latencies_ms: List[float],
    policy: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    base_predictions = np.asarray([arr(s["prediction"]) for s in samples], dtype=np.float64) if samples else np.asarray([], dtype=np.float64)
    targets = np.asarray([arr(s["target"]) for s in samples], dtype=np.float64) if samples else np.asarray([], dtype=np.float64)
    pred_array = np.asarray(predictions, dtype=np.float64) if predictions else np.asarray([], dtype=np.float64)
    base_mse = array_mse(base_predictions, targets) if samples else 0.0
    base_mae = array_mae(base_predictions, targets) if samples else 0.0
    value_mse = array_mse(pred_array, targets) if samples else 0.0
    value_mae = array_mae(pred_array, targets) if samples else 0.0
    changed = [bool(info.get("changed", 0.0)) for info in infos]
    triggered = [bool(info.get("trigger", 0.0)) for info in infos]
    return {
        "variant": variant,
        "n_samples": len(samples),
        "mse": value_mse,
        "mae": value_mae,
        "mse_delta_pct_vs_no_correction": pct_delta(value_mse, base_mse),
        "mae_delta_pct_vs_no_correction": pct_delta(value_mae, base_mae),
        "correction_rate": float(np.mean(changed)) if changed else 0.0,
        "trigger_rate": float(np.mean(triggered)) if triggered else 0.0,
        "inference_latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
        "threshold_quantile": (policy or {}).get("trigger_quantile", ""),
        "correction_strength": (policy or {}).get("correction_strength", ""),
        "validation_mse_delta_pct": (policy or {}).get("validation_mse_delta_pct", ""),
        "test_threshold_leakage": False,
    }


def pct_delta(value: float, baseline: float) -> float:
    if abs(float(baseline)) <= EPS:
        return 0.0
    return 100.0 * (float(value) - float(baseline)) / float(baseline)


def array_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    if predictions.size == 0:
        return 0.0
    return float(np.mean((np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64)) ** 2))


def array_mae(predictions: np.ndarray, targets: np.ndarray) -> float:
    if predictions.size == 0:
        return 0.0
    return float(np.mean(np.abs(np.asarray(predictions, dtype=np.float64) - np.asarray(targets, dtype=np.float64))))


def _apply_vectors(base_predictions: np.ndarray, vectors: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    corrected = np.asarray(base_predictions, dtype=np.float64).copy()
    if corrected.size == 0:
        return corrected
    corrected[mask] = corrected[mask] + float(strength) * vectors[mask]
    return corrected
