"""Stage 14 signal-preserving component router.

This module adds a small output-space router that tries to distinguish
unsupported pseudo-noise from context-supported high-frequency or turning
structure. All thresholds and action choices are fit on validation samples only.
"""

from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import halluguard_router as router
from halluguard_dynamics import arr, array_mse, array_mae, metric_row


EPS = 1e-12
DEFAULT_ACTIONS = (
    "no_correction",
    "boundary_only",
    "dynamics_full",
    "median_smoothing",
    "ema_smoothing",
    "naive_smoothing",
)
SIGNAL_VARIANTS = (
    "signal_preserve_router",
    "component_router_without_signal_support",
    "signal_support_only_ablation",
)


def extract_component_features(sample: dict, policy_context: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    """Extract target-free component and signal-support features."""

    policy_context = policy_context or {}
    base = router.extract_router_features(sample, policy_context)
    ctx = arr(sample["context"])
    pred = arr(sample["prediction"])
    tail = context_tail(ctx, pred.size)
    pred_hf = high_frequency_ratio(pred)
    ctx_hf = high_frequency_ratio(tail)
    pred_diff = np.diff(pred) if pred.size >= 2 else np.asarray([0.0])
    ctx_diff = np.diff(tail) if tail.size >= 2 else np.asarray([0.0])
    pred_diff_std = float(np.std(pred_diff)) + EPS
    ctx_diff_std = float(np.std(ctx_diff)) + EPS
    pred_turn = turning_rate(pred)
    ctx_turn = turning_rate(tail)
    pred_curv_energy = curvature_energy(pred)
    ctx_curv_energy = curvature_energy(tail)
    spectral_distance = float(base["spectral_distance"])
    spectral_support = float(math.exp(-3.0 * spectral_distance))
    hf_support = float(min(1.0, (ctx_hf + EPS) / (pred_hf + EPS)))
    volatility_support = float(min(1.0, ctx_diff_std / pred_diff_std))
    turning_support = float(min(1.0, (ctx_turn + 0.02) / (pred_turn + 0.02)))
    curvature_support = float(min(1.0, (ctx_curv_energy + EPS) / (pred_curv_energy + EPS)))
    phase_support = diff_phase_support(tail, pred)
    periodic_support = autocorr_peak(tail)
    signal_support = float(
        0.22 * hf_support
        + 0.18 * volatility_support
        + 0.18 * spectral_support
        + 0.16 * turning_support
        + 0.14 * curvature_support
        + 0.07 * phase_support
        + 0.05 * periodic_support
    )
    diff_std_excess = float(max(0.0, pred_diff_std / ctx_diff_std - 1.0))
    roughness_score = float(
        0.45 * max(0.0, pred_hf - ctx_hf)
        + 0.25 * spectral_distance
        + 0.20 * diff_std_excess
        + 0.10 * max(0.0, pred_turn - ctx_turn)
    )
    out = dict(base)
    out.update(
        {
            "context_hf_ratio": float(ctx_hf),
            "prediction_hf_ratio": float(pred_hf),
            "hf_support": hf_support,
            "volatility_support": volatility_support,
            "spectral_support": spectral_support,
            "turning_support": turning_support,
            "curvature_support": curvature_support,
            "phase_support": phase_support,
            "periodic_support": periodic_support,
            "signal_support_score": signal_support,
            "roughness_score": roughness_score,
            "diff_std_excess": diff_std_excess,
            "context_turning_rate": float(ctx_turn),
            "prediction_turning_rate": float(pred_turn),
            "context_curvature_energy": float(ctx_curv_energy),
            "prediction_curvature_energy": float(pred_curv_energy),
        }
    )
    return out


def prepare_signal_training(validation_samples: List[dict], config: Dict) -> Dict[str, object]:
    """Cache validation-only action policies, component features, and action errors."""

    if not validation_samples:
        raise ValueError("prepare_signal_training requires non-empty validation samples.")
    cfg = config.get("signal_preserve", {}) or {}
    actions = list((config.get("method", {}) or {}).get("candidate_actions", DEFAULT_ACTIONS))
    action_context = router.fit_action_context(validation_samples, config, actions)
    features = [extract_component_features(sample, action_context) for sample in validation_samples]
    action_errors, action_mae_errors = action_loss_matrices(validation_samples, action_context, actions)
    validation_action_mse = {action: float(np.mean(action_errors[:, idx])) for idx, action in enumerate(actions)}
    smoothing_candidates = [a for a in cfg.get("smoothing_actions", ["median_smoothing", "ema_smoothing", "naive_smoothing"]) if a in actions]
    smoothing_action = router.best_global_action(action_errors, actions, smoothing_candidates) if smoothing_candidates else "no_correction"
    best_single_action = min(validation_action_mse, key=validation_action_mse.get)
    base_predictions = np.asarray([sample["prediction"] for sample in validation_samples], dtype=np.float64)
    targets = np.asarray([sample["target"] for sample in validation_samples], dtype=np.float64)
    return {
        "validation_samples": validation_samples,
        "actions": actions,
        "action_context": action_context,
        "features": features,
        "action_errors": action_errors,
        "action_mae_errors": action_mae_errors,
        "validation_action_mse": validation_action_mse,
        "smoothing_action": smoothing_action,
        "best_single_action": best_single_action,
        "targets": targets,
        "baseline_mse": array_mse(base_predictions, targets),
        "baseline_mae": array_mae(base_predictions, targets),
    }



def action_loss_matrices(samples: List[dict], action_context: Dict[str, object], actions: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-sample per-action MSE/MAE once for fast validation grid search."""

    mse_errors = np.zeros((len(samples), len(actions)), dtype=np.float64)
    mae_errors = np.zeros((len(samples), len(actions)), dtype=np.float64)
    for row_idx, sample in enumerate(samples):
        target = arr(sample["target"])
        for col_idx, action in enumerate(actions):
            pred = router.apply_action(sample, action, action_context).prediction
            diff = pred - target
            mse_errors[row_idx, col_idx] = float(np.mean(diff ** 2))
            mae_errors[row_idx, col_idx] = float(np.mean(np.abs(diff)))
    return mse_errors, mae_errors


def fit_signal_policy(validation_samples: List[dict], config: Dict, variant: str = "signal_preserve_router", prepared: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Fit a validation-only signal-preserving policy."""

    if not validation_samples:
        raise ValueError("fit_signal_policy requires non-empty validation samples.")
    if variant not in SIGNAL_VARIANTS:
        raise ValueError(f"Unknown signal policy variant: {variant}")
    cfg = config.get("signal_preserve", {}) or {}
    if prepared is None:
        prepared = prepare_signal_training(validation_samples, config)
    actions = list(prepared["actions"])
    action_context = prepared["action_context"]
    features = list(prepared["features"])
    action_errors = np.asarray(prepared["action_errors"], dtype=np.float64)
    action_mae_errors = np.asarray(prepared["action_mae_errors"], dtype=np.float64)
    validation_action_mse = dict(prepared["validation_action_mse"])
    smoothing_action = str(prepared["smoothing_action"])
    best_single_action = str(prepared["best_single_action"])
    targets = np.asarray(prepared["targets"], dtype=np.float64)
    baseline_mse = float(prepared["baseline_mse"])
    baseline_mae = float(prepared["baseline_mae"])
    random_seeds = [int(v) for v in (cfg.get("random_seeds") or (config.get("router", {}) or {}).get("random_seeds", [1101, 2202, 3303, 4404, 5505]))]
    boundary_quantiles = [float(v) for v in cfg.get("boundary_quantiles", [0.70, 0.75, 0.80])]
    roughness_quantiles = [float(v) for v in cfg.get("roughness_quantiles", [0.70, 0.80, 0.85])]
    support_quantiles = [float(v) for v in cfg.get("support_quantiles", [0.35, 0.45, 0.55])]
    if variant == "signal_preserve_router":
        extreme_roughness_quantiles = [float(v) for v in cfg.get("extreme_roughness_quantiles", [0.85, 0.90])]
    else:
        extreme_roughness_quantiles = [1.01]
    random_weight = float(cfg.get("random_separation_weight", 0.35))
    matched_weight = float(cfg.get("matched_smoothing_weight", 0.60))
    rate_penalty = float(cfg.get("correction_rate_penalty", 0.0002))
    degeneracy_penalty = float(cfg.get("degeneracy_penalty", 0.40))
    max_single_action_rate = float(cfg.get("max_single_action_rate", 0.90))
    action_to_idx = {action: idx for idx, action in enumerate(actions)}
    no_idx = action_to_idx.get("no_correction", 0)
    smoothing_idx = action_to_idx.get(smoothing_action, no_idx)
    row_ids = np.arange(len(validation_samples))

    best: Optional[Dict[str, object]] = None
    for bq in boundary_quantiles:
        bthr = quantile(features, "boundary_score", bq)
        for rq in roughness_quantiles:
            rthr = quantile(features, "roughness_score", rq)
            for sq in support_quantiles:
                sthr = quantile(features, "signal_support_score", sq)
                for eq in extreme_roughness_quantiles:
                    ethr = quantile(features, "roughness_score", eq) if eq <= 1.0 else float("inf")
                    policy = {
                        "variant": variant,
                        "source_split": "val",
                        "action_context": action_context,
                        "candidate_actions": actions,
                        "smoothing_action": smoothing_action,
                        "validation_best_single_action": best_single_action,
                        "validation_action_mse": validation_action_mse,
                        "boundary_threshold": float(bthr),
                        "roughness_threshold": float(rthr),
                        "support_threshold": float(sthr),
                        "extreme_roughness_threshold": float(ethr),
                        "boundary_quantile": float(bq),
                        "roughness_quantile": float(rq),
                        "support_quantile": float(sq),
                        "extreme_roughness_quantile": float(eq),
                        "test_threshold_leakage": False,
                    }
                    actions_chosen = [route_signal_decision(row, policy)[0] for row in features]
                    chosen_idx = np.asarray([action_to_idx.get(action, no_idx) for action in actions_chosen], dtype=np.int64)
                    mse = float(np.mean(action_errors[row_ids, chosen_idx]))
                    mae = float(np.mean(action_mae_errors[row_ids, chosen_idx]))
                    trigger_rate = float(np.mean([a != "no_correction" for a in actions_chosen])) if actions_chosen else 0.0
                    dominant = dominant_action_rate(actions_chosen)
                    matched_idx = np.asarray([smoothing_idx if action != "no_correction" else no_idx for action in actions_chosen], dtype=np.int64)
                    matched_mse = float(np.mean(action_errors[row_ids, matched_idx]))
                    random_mses = []
                    for seed in random_seeds:
                        random_actions = shuffled_actions(actions_chosen, seed)
                        random_idx = np.asarray([action_to_idx.get(action, no_idx) for action in random_actions], dtype=np.int64)
                        random_mses.append(float(np.mean(action_errors[row_ids, random_idx])))
                    random_mse = float(np.mean(random_mses)) if random_mses else mse
                    random_advantage = random_mse - mse
                    matched_advantage = matched_mse - mse
                    objective = mse - random_weight * random_advantage - matched_weight * matched_advantage + rate_penalty * baseline_mse * trigger_rate
                    if dominant > max_single_action_rate:
                        objective += degeneracy_penalty * baseline_mse * (dominant - max_single_action_rate)
                    record = dict(policy)
                    record.update(
                        {
                            "validation_baseline_mse": baseline_mse,
                            "validation_baseline_mae": baseline_mae,
                            "validation_mse": mse,
                            "validation_mae": mae,
                            "validation_mse_delta_pct": pct_delta(mse, baseline_mse),
                            "validation_mae_delta_pct": pct_delta(mae, baseline_mae),
                            "validation_random_action_mse": random_mse,
                            "validation_random_advantage_mse": random_advantage,
                            "validation_matched_smoothing_mse": matched_mse,
                            "validation_matched_advantage_mse": matched_advantage,
                            "validation_trigger_rate": trigger_rate,
                            "validation_dominant_action_rate": dominant,
                            "validation_action_distribution": router.compact_distribution(actions_chosen),
                            "objective": objective,
                        }
                    )
                    if best is None or float(record["objective"]) < float(best["objective"]):
                        best = record
    assert best is not None
    return best



def route_signal_decision(features: Dict[str, float], policy: Dict[str, object]) -> Tuple[str, str, bool, bool, bool, bool]:
    variant = str(policy.get("variant", "signal_preserve_router"))
    boundary_high = features["boundary_score"] >= float(policy["boundary_threshold"])
    rough_high = features["roughness_score"] >= float(policy["roughness_threshold"])
    supported = features["signal_support_score"] >= float(policy["support_threshold"])
    extreme_roughness = features["roughness_score"] >= float(policy.get("extreme_roughness_threshold", float("inf")))
    smoothing_action = str(policy.get("smoothing_action", "median_smoothing"))
    if variant == "component_router_without_signal_support":
        if boundary_high:
            return "boundary_only", "boundary_high", boundary_high, rough_high, supported, extreme_roughness
        if rough_high:
            return smoothing_action, "roughness_high_no_support_gate", boundary_high, rough_high, supported, extreme_roughness
        return "no_correction", "abstain", boundary_high, rough_high, supported, extreme_roughness
    if variant == "signal_support_only_ablation":
        if rough_high and not supported:
            return smoothing_action, "unsupported_roughness", boundary_high, rough_high, supported, extreme_roughness
        return "no_correction", "supported_or_low_risk", boundary_high, rough_high, supported, extreme_roughness
    if boundary_high:
        return "boundary_only", "boundary_high", boundary_high, rough_high, supported, extreme_roughness
    if rough_high and (not supported or extreme_roughness):
        return smoothing_action, "unsupported_or_extreme_roughness", boundary_high, rough_high, supported, extreme_roughness
    return "no_correction", "supported_or_low_risk", boundary_high, rough_high, supported, extreme_roughness


def route_signal_action(sample: dict, policy: Dict[str, object]) -> Tuple[str, Dict[str, float]]:
    features = extract_component_features(sample, policy.get("action_context", {}))
    variant = str(policy.get("variant", "signal_preserve_router"))
    boundary_high = features["boundary_score"] >= float(policy["boundary_threshold"])
    rough_high = features["roughness_score"] >= float(policy["roughness_threshold"])
    supported = features["signal_support_score"] >= float(policy["support_threshold"])
    extreme_roughness = features["roughness_score"] >= float(policy.get("extreme_roughness_threshold", float("inf")))
    smoothing_action = str(policy.get("smoothing_action", "median_smoothing"))
    if variant == "component_router_without_signal_support":
        if boundary_high:
            action = "boundary_only"
            reason = "boundary_high"
        elif rough_high:
            action = smoothing_action
            reason = "roughness_high_no_support_gate"
        else:
            action = "no_correction"
            reason = "abstain"
    elif variant == "signal_support_only_ablation":
        if rough_high and not supported:
            action = smoothing_action
            reason = "unsupported_roughness"
        else:
            action = "no_correction"
            reason = "supported_or_low_risk"
    else:
        if boundary_high:
            action = "boundary_only"
            reason = "boundary_high"
        elif rough_high and (not supported or extreme_roughness):
            action = smoothing_action
            reason = "unsupported_or_extreme_roughness"
        else:
            action = "no_correction"
            reason = "supported_or_low_risk"
    info = dict(features)
    info.update(
        {
            "action": action,
            "route_reason": reason,
            "boundary_zone": float(boundary_high),
            "roughness_zone": float(rough_high),
            "signal_supported": float(supported),
            "extreme_roughness": float(extreme_roughness),
            "unsupported_high_frequency": float(rough_high and not supported),
        }
    )
    return action, info


def apply_signal_policy(sample: dict, policy: Dict[str, object]) -> Tuple[np.ndarray, Dict[str, object]]:
    start = time.perf_counter()
    action, route_info = route_signal_action(sample, policy)
    result = router.apply_action(sample, action, policy["action_context"])
    elapsed = (time.perf_counter() - start) * 1000.0
    base = arr(sample["prediction"])
    changed = bool(np.max(np.abs(result.prediction - base)) > 1e-10) if base.size else False
    info = dict(route_info)
    info.update(result.info)
    info.update({"action": action, "changed": float(changed), "trigger": float(action != "no_correction"), "latency_ms": elapsed})
    return result.prediction, info


def apply_signal_policy_to_samples(samples: List[dict], policy: Dict[str, object]) -> Tuple[List[np.ndarray], List[Dict[str, object]], List[float]]:
    preds: List[np.ndarray] = []
    infos: List[Dict[str, object]] = []
    latencies: List[float] = []
    for sample in samples:
        start = time.perf_counter()
        pred, info = apply_signal_policy(sample, policy)
        latencies.append((time.perf_counter() - start) * 1000.0)
        preds.append(pred)
        infos.append(info)
    return preds, infos, latencies


def signal_metric_row(variant: str, samples: List[dict], predictions: List[np.ndarray], infos: List[Dict[str, object]], latencies_ms: List[float], policy: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    row = router.router_metric_row(variant, samples, predictions, infos, latencies_ms, policy)
    row.update(
        {
            "boundary_quantile": (policy or {}).get("boundary_quantile", ""),
            "roughness_quantile": (policy or {}).get("roughness_quantile", ""),
            "support_quantile": (policy or {}).get("support_quantile", ""),
            "boundary_threshold": (policy or {}).get("boundary_threshold", ""),
            "roughness_threshold": (policy or {}).get("roughness_threshold", ""),
            "support_threshold": (policy or {}).get("support_threshold", ""),
            "extreme_roughness_threshold": (policy or {}).get("extreme_roughness_threshold", ""),
            "extreme_roughness_quantile": (policy or {}).get("extreme_roughness_quantile", ""),
            "smoothing_action": (policy or {}).get("smoothing_action", ""),
            "validation_mse_delta_pct": (policy or {}).get("validation_mse_delta_pct", ""),
            "validation_random_advantage_mse": (policy or {}).get("validation_random_advantage_mse", ""),
            "validation_matched_advantage_mse": (policy or {}).get("validation_matched_advantage_mse", ""),
            "validation_action_distribution": (policy or {}).get("validation_action_distribution", ""),
        }
    )
    return row


def matched_sparse_smoothing_outputs(samples: List[dict], action_context: Dict[str, object], actions: Sequence[str], smoothing_action: str) -> Tuple[List[np.ndarray], List[Dict[str, object]], List[float]]:
    preds: List[np.ndarray] = []
    infos: List[Dict[str, object]] = []
    latencies: List[float] = []
    for sample, action in zip(samples, actions):
        start = time.perf_counter()
        use_smoothing = str(action) != "no_correction"
        actual_action = smoothing_action if use_smoothing else "no_correction"
        result = router.apply_action(sample, actual_action, action_context)
        base = arr(sample["prediction"])
        preds.append(result.prediction)
        infos.append({"action": actual_action, "trigger": float(use_smoothing), "changed": float(np.max(np.abs(result.prediction - base)) > 1e-10)})
        latencies.append((time.perf_counter() - start) * 1000.0)
    return preds, infos, latencies


def matched_sparse_smoothing_mse(samples: List[dict], action_context: Dict[str, object], actions: Sequence[str], smoothing_action: str, targets: np.ndarray) -> float:
    preds, _, _ = matched_sparse_smoothing_outputs(samples, action_context, actions, smoothing_action)
    return array_mse(np.asarray(preds, dtype=np.float64), targets)


def random_action_outputs(samples: List[dict], action_context: Dict[str, object], matched_actions: Sequence[str], seed: int) -> Tuple[List[np.ndarray], List[Dict[str, object]], List[float]]:
    actions = shuffled_actions(matched_actions, seed)
    preds: List[np.ndarray] = []
    infos: List[Dict[str, object]] = []
    latencies: List[float] = []
    for sample, action in zip(samples, actions):
        start = time.perf_counter()
        result = router.apply_action(sample, action, action_context)
        base = arr(sample["prediction"])
        preds.append(result.prediction)
        infos.append({"action": action, "trigger": float(action != "no_correction"), "changed": float(np.max(np.abs(result.prediction - base)) > 1e-10)})
        latencies.append((time.perf_counter() - start) * 1000.0)
    return preds, infos, latencies


def shuffled_actions(actions: Sequence[str], seed: int) -> List[str]:
    out = [str(a) for a in actions]
    rng = np.random.default_rng(seed)
    if out:
        rng.shuffle(out)
    return out


def signal_alignment_rows(samples: List[dict], policy: Dict[str, object], infos: List[Dict[str, object]], meta: Dict[str, object]) -> List[Dict[str, object]]:
    features = [extract_component_features(sample, policy.get("action_context", {})) for sample in samples]
    actions = [str(info.get("action", "")) for info in infos]
    rows: List[Dict[str, object]] = []
    specs = [
        ("boundary_score", ["boundary_only", "dynamics_full"], "boundary_action_rate", False),
        ("roughness_score", [str(policy.get("smoothing_action", "median_smoothing")), "median_smoothing", "ema_smoothing", "naive_smoothing"], "smoothing_action_rate", False),
        ("signal_support_score", ["no_correction"], "preserve_action_rate", False),
        ("signal_support_score", [str(policy.get("smoothing_action", "median_smoothing")), "median_smoothing", "ema_smoothing", "naive_smoothing"], "smoothing_action_rate", True),
    ]
    for feature_name, action_set, metric_name, reverse in specs:
        values = np.asarray([float(row.get(feature_name, 0.0)) for row in features], dtype=np.float64)
        if values.size == 0:
            continue
        order_values = -values if reverse else values
        qs = np.quantile(order_values, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        for idx, label in enumerate(["low", "mid", "high"]):
            if idx == 2:
                mask = (order_values >= qs[idx]) & (order_values <= qs[idx + 1])
            else:
                mask = (order_values >= qs[idx]) & (order_values < qs[idx + 1])
            selected = [actions[i] for i, keep in enumerate(mask) if bool(keep)]
            row = dict(meta)
            row.update(
                {
                    "alignment_feature": feature_name,
                    "bin": label,
                    "n_samples": len(selected),
                    metric_name: router.action_rate(selected, action_set),
                    "action_distribution": router.compact_distribution(selected),
                }
            )
            rows.append(row)
    return rows


def validation_policy_rows(policy: Dict[str, object], meta: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for action, value in sorted((policy.get("validation_action_mse", {}) or {}).items()):
        row = dict(meta)
        row.update(
            {
                "variant": policy.get("variant", ""),
                "action": action,
                "validation_action_mse": value,
                "validation_best_single_action": policy.get("validation_best_single_action", ""),
                "validation_mse_delta_pct": policy.get("validation_mse_delta_pct", ""),
                "validation_action_distribution": policy.get("validation_action_distribution", ""),
                "boundary_quantile": policy.get("boundary_quantile", ""),
                "roughness_quantile": policy.get("roughness_quantile", ""),
                "support_quantile": policy.get("support_quantile", ""),
                "test_threshold_leakage": False,
            }
        )
        rows.append(row)
    return rows


def quantile(rows: List[Dict[str, float]], key: str, q: float) -> float:
    values = np.asarray([float(row.get(key, 0.0)) for row in rows], dtype=np.float64)
    return float(np.quantile(values, float(q))) if values.size else 0.0


def context_tail(context: np.ndarray, horizon: int) -> np.ndarray:
    if context.size == 0:
        return np.zeros(max(1, horizon), dtype=np.float64)
    n = max(1, int(horizon))
    if context.size >= n:
        return context[-n:].astype(np.float64)
    return np.pad(context.astype(np.float64), (n - context.size, 0), mode="edge")


def normalized_power(values: Iterable[float]) -> np.ndarray:
    values = arr(values)
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    centered = values - float(np.mean(values))
    power = np.abs(np.fft.rfft(centered)) ** 2
    return power / (float(np.sum(power)) + EPS)


def high_frequency_ratio(values: Iterable[float]) -> float:
    power = normalized_power(values)
    if power.size <= 3:
        return 0.0
    cutoff = int(math.ceil(0.60 * power.size))
    return float(np.sum(power[cutoff:]))


def turning_rate(values: Iterable[float]) -> float:
    values = arr(values)
    if values.size < 4:
        return 0.0
    diffs = np.diff(values)
    signs = np.sign(diffs)
    signs[np.abs(diffs) < 1e-10] = 0.0
    changes = 0
    total = 0
    prev = 0.0
    for sign in signs:
        if sign == 0.0:
            continue
        if prev != 0.0:
            total += 1
            if sign != prev:
                changes += 1
        prev = sign
    return float(changes / total) if total else 0.0


def curvature_energy(values: Iterable[float]) -> float:
    values = arr(values)
    if values.size < 3:
        return 0.0
    return float(np.mean(np.diff(values, n=2) ** 2))


def diff_phase_support(context_tail_values: Iterable[float], prediction: Iterable[float]) -> float:
    ctx = arr(context_tail_values)
    pred = arr(prediction)
    if ctx.size < 3 or pred.size < 3:
        return 0.5
    k = int(min(24, ctx.size - 1, pred.size - 1))
    if k <= 1:
        return 0.5
    ctx_d = np.diff(ctx[-(k + 1) :])
    pred_d = np.diff(pred[: k + 1])
    ctx_s = np.sign(ctx_d)
    pred_s = np.sign(pred_d)
    valid = (ctx_s != 0.0) & (pred_s != 0.0)
    if not np.any(valid):
        return 0.5
    return float(np.mean(ctx_s[valid] == pred_s[valid]))


def autocorr_peak(values: Iterable[float]) -> float:
    values = arr(values)
    if values.size < 8:
        return 0.0
    centered = values - float(np.mean(values))
    denom = float(np.dot(centered, centered)) + EPS
    max_lag = int(min(24, max(2, values.size // 3)))
    peaks = []
    for lag in range(2, max_lag + 1):
        a = centered[:-lag]
        b = centered[lag:]
        if a.size and b.size:
            peaks.append(float(np.dot(a, b) / denom))
    return float(max(0.0, max(peaks))) if peaks else 0.0


def dominant_action_rate(actions: Sequence[str]) -> float:
    if not actions:
        return 0.0
    counts: Dict[str, int] = {}
    for action in actions:
        counts[str(action)] = counts.get(str(action), 0) + 1
    return float(max(counts.values()) / len(actions))


def pct_delta(value: float, baseline: float) -> float:
    if abs(float(baseline)) <= EPS:
        return 0.0
    return 100.0 * (float(value) - float(baseline)) / float(baseline)
