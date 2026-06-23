"""Reusable Stage 13 adaptive HalluGuard router.

The router is deliberately small and external-facing. It fits all thresholds,
action policies, and learned routing parameters on validation samples only.
Test samples are passed to ``apply_router`` only after the returned policy is
frozen.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from correction import naive_smoothing
from halluguard_dynamics import (
    VariantSpec,
    apply_correction,
    arr,
    array_mae,
    array_mse,
    default_variant_specs,
    ema_smoothing,
    fit_policy as fit_dynamics_policy,
    median_smoothing,
    metric_row,
    score_sample as dynamics_score_sample,
)


EPS = 1e-12
DEFAULT_ACTIONS = (
    "no_correction",
    "boundary_only",
    "dynamics_full",
    "median_smoothing",
    "ema_smoothing",
    "naive_smoothing",
)
OPTIONAL_ACTIONS = ("boundary_then_ema", "boundary_then_median", "boundary_then_selective_median")
DEPLOYABLE_ROUTERS = ("rule_router", "shallow_tree_router", "logistic_router", "margin_abstain_router", "smoothing_risk_guard_router", "smoothing_benefit_guard_router", "stable_forecast_guard_router", "stable_selective_fallback_router", "spectral_support_guard_router", "smoothing_cap_selective_router", "stable_smoothing_cap_router", "conditional_stable_cap_router", "selective_smoothing_alias_router", "harm_aware_router", "oracle_val_policy")
FEATURE_NAMES = (
    "boundary_score",
    "first_diff_score",
    "curvature_score",
    "high_frequency_excess",
    "spectral_distance",
    "pred_context_variance_ratio",
    "pred_context_diff_std_ratio",
    "context_volatility",
    "horizon",
)


@dataclass(frozen=True)
class ActionResult:
    prediction: np.ndarray
    info: Dict[str, float]
    latency_ms: float


def extract_router_features(sample: dict, policy_context: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    """Extract target-free router features from one prediction sample."""

    policy_context = policy_context or {}
    context = arr(sample["context"])
    prediction = arr(sample["prediction"])
    score_policy = policy_context.get("score_policy", {"score_components": ("boundary", "first_diff", "curvature")})
    dyn = dynamics_score_sample(context, prediction, score_policy)
    context_tail = _context_tail(context, prediction.size)
    pred_power = _normalized_power(prediction)
    ctx_power = _normalized_power(context_tail)
    spectral_distance = float(np.mean(np.abs(pred_power - ctx_power))) if pred_power.size and ctx_power.size else 0.0
    pred_hf = _high_frequency_ratio(prediction)
    ctx_hf = _high_frequency_ratio(context_tail)
    diff_ctx = np.diff(context_tail) if context_tail.size >= 2 else np.asarray([0.0])
    diff_pred = np.diff(prediction) if prediction.size >= 2 else np.asarray([0.0])
    ctx_std = float(np.std(context_tail)) + EPS
    ctx_diff_std = float(np.std(diff_ctx)) + EPS
    return {
        "boundary_score": float(dyn["boundary_score"]),
        "first_diff_score": float(dyn["first_diff_score"]),
        "curvature_score": float(dyn["curvature_score"]),
        "high_frequency_excess": float(max(0.0, pred_hf - ctx_hf)),
        "spectral_distance": spectral_distance,
        "pred_context_variance_ratio": float((np.var(prediction) + EPS) / (np.var(context_tail) + EPS)) if prediction.size else 1.0,
        "pred_context_diff_std_ratio": float((np.std(diff_pred) + EPS) / ctx_diff_std),
        "context_volatility": float(ctx_diff_std / ctx_std),
        "horizon": float(prediction.size),
        "score": float(dyn["score"]),
    }


def fit_router(validation_samples: List[dict], config: Dict, candidate_actions: Optional[Sequence[str]] = None, router_type: Optional[str] = None) -> Dict[str, object]:
    """Fit a validation-only router policy."""

    if not validation_samples:
        raise ValueError("fit_router requires non-empty validation samples.")
    method_cfg = config.get("method", {}) or {}
    actions = list(candidate_actions or method_cfg.get("candidate_actions", DEFAULT_ACTIONS))
    if not actions:
        raise ValueError("fit_router requires at least one candidate action.")
    router_type = str(router_type or method_cfg.get("main_router", "harm_aware_router"))
    prepared = prepare_router_training(validation_samples, config, actions)
    return fit_router_from_prepared(prepared, config, router_type)


def prepare_router_training(validation_samples: List[dict], config: Dict, candidate_actions: Sequence[str]) -> Dict[str, object]:
    """Precompute validation-only action outputs and router features once."""

    if not validation_samples:
        raise ValueError("prepare_router_training requires non-empty validation samples.")
    router_cfg = config.get("router", {}) or {}
    actions = list(candidate_actions)
    action_context = fit_action_context(validation_samples, config, actions)
    feature_names = list(router_cfg.get("feature_names", FEATURE_NAMES))
    features = feature_matrix(validation_samples, action_context, feature_names)
    feature_stats = fit_feature_stats(features)
    z_features = standardize(features, feature_stats)
    action_errors = action_error_matrix(validation_samples, action_context, actions)
    labels = label_best_safe_actions(action_errors, actions, config)
    label_indices = np.asarray([actions.index(label) for label in labels], dtype=np.int64)
    validation_action_mse = {
        action: float(np.mean(action_errors[:, idx])) if action_errors.size else 0.0
        for idx, action in enumerate(actions)
    }
    best_single_action = min(validation_action_mse, key=validation_action_mse.get)
    return {
        "actions": actions,
        "action_context": action_context,
        "feature_names": feature_names,
        "features": features,
        "feature_stats": feature_stats,
        "z_features": z_features,
        "action_errors": action_errors,
        "labels": labels,
        "label_indices": label_indices,
        "validation_action_mse": validation_action_mse,
        "best_single_action": best_single_action,
    }


def fit_router_from_prepared(prepared: Dict[str, object], config: Dict, router_type: str) -> Dict[str, object]:
    """Fit one router from a shared validation preparation bundle."""

    router_cfg = config.get("router", {}) or {}
    actions = list(prepared["actions"])
    features = np.asarray(prepared["features"], dtype=np.float64)
    z_features = np.asarray(prepared["z_features"], dtype=np.float64)
    action_errors = np.asarray(prepared["action_errors"], dtype=np.float64)
    feature_names = list(prepared["feature_names"])
    label_indices = np.asarray(prepared["label_indices"], dtype=np.int64)
    if router_type == "rule_router":
        model = fit_rule_router(features, action_errors, actions, feature_names, config)
    elif router_type == "shallow_tree_router":
        model = fit_tree_router(z_features, label_indices, actions, config)
    elif router_type == "logistic_router":
        model = fit_logistic_router(z_features, label_indices, actions, config)
    elif router_type == "capped_logistic_router":
        model = fit_capped_logistic_router(z_features, label_indices, actions, config)
    elif router_type == "margin_abstain_router":
        model = fit_margin_abstain_router(z_features, label_indices, action_errors, actions, config)
    elif router_type == "smoothing_risk_guard_router":
        model = fit_smoothing_risk_guard_router(features, z_features, label_indices, action_errors, actions, feature_names, config)
    elif router_type == "smoothing_benefit_guard_router":
        model = fit_smoothing_benefit_guard_router(features, z_features, label_indices, action_errors, actions, feature_names, config)
    elif router_type == "stable_forecast_guard_router":
        model = fit_stable_forecast_guard_router(z_features, label_indices, action_errors, actions, config)
    elif router_type == "stable_selective_fallback_router":
        model = fit_stable_selective_fallback_router(z_features, label_indices, action_errors, actions, config)
    elif router_type == "spectral_support_guard_router":
        model = fit_spectral_support_guard_router(features, z_features, label_indices, action_errors, actions, feature_names, config)
    elif router_type == "smoothing_cap_selective_router":
        model = fit_smoothing_cap_selective_router(z_features, label_indices, action_errors, actions, config)
    elif router_type == "stable_smoothing_cap_router":
        model = fit_stable_smoothing_cap_router(z_features, label_indices, action_errors, actions, config)
    elif router_type == "conditional_stable_cap_router":
        model = fit_conditional_stable_cap_router(features, z_features, label_indices, action_errors, actions, feature_names, config)
    elif router_type == "selective_smoothing_alias_router":
        model = fit_selective_smoothing_alias_router(z_features, label_indices, action_errors, actions, config)
    elif router_type == "harm_aware_router":
        base = fit_tree_router(z_features, label_indices, actions, config)
        model = {
            "kind": "harm_aware_router",
            "base_kind": "shallow_tree_router",
            "base_model": base,
            "margin": float(router_cfg.get("harm_abstain_margin_pct", 0.03)) / 100.0,
        }
    elif router_type in {"oracle_val_policy", "validation_best_single_action"}:
        model = {"kind": "oracle_val_policy", "action": str(prepared["best_single_action"])}
    else:
        raise ValueError(f"Unknown router_type={router_type}")

    policy = {
        "router_type": router_type,
        "candidate_actions": actions,
        "feature_names": feature_names,
        "feature_stats": prepared["feature_stats"],
        "model": model,
        "action_context": prepared["action_context"],
        "validation_action_mse": prepared["validation_action_mse"],
        "validation_best_single_action": prepared["best_single_action"],
        "validation_label_distribution": histogram(prepared["labels"]),
        "source_split": "val",
        "test_threshold_leakage": False,
    }
    return policy


def apply_router(sample: dict, router_policy: Dict[str, object], override_features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[str, object]]:
    """Apply a frozen router policy to one sample."""

    start = time.perf_counter()
    action, route_info = route_action(sample, router_policy, override_features=override_features)
    result = apply_action(sample, action, router_policy["action_context"])
    latency_ms = (time.perf_counter() - start) * 1000.0
    changed = bool(np.max(np.abs(result.prediction - arr(sample["prediction"]))) > 1e-10) if result.prediction.size else False
    info = {
        **result.info,
        **route_info,
        "action": action,
        "changed": float(changed),
        "trigger": float(action != "no_correction"),
        "latency_ms": latency_ms,
    }
    return result.prediction, info


def score_sample(context: Iterable[float], prediction: Iterable[float], policy: Optional[Dict[str, object]] = None) -> Dict[str, float]:
    sample = {"context": list(context), "prediction": list(prediction)}
    return extract_router_features(sample, policy or {})


def evaluate_router(validation_samples: List[dict], test_samples: List[dict], config: Dict, router_type: Optional[str] = None) -> Dict[str, object]:
    """Fit one router on validation samples and evaluate it on test samples."""

    policy = fit_router(validation_samples, config, router_type=router_type)
    predictions, infos, latencies = apply_router_to_samples(test_samples, policy)
    row = metric_row(str(policy["router_type"]), test_samples, predictions, infos, latencies, policy)
    row["action_entropy"] = action_entropy([str(info.get("action", "")) for info in infos])
    row["action_distribution"] = compact_distribution([str(info.get("action", "")) for info in infos])
    return {"policy": policy, "row": row, "infos": infos, "predictions": predictions}


def fit_action_context(validation_samples: List[dict], config: Dict, actions: Sequence[str]) -> Dict[str, object]:
    variants = {spec.name: spec for spec in default_variant_specs()}
    policies: Dict[str, Dict[str, object]] = {}
    for action in actions:
        if action in variants and action not in policies:
            policies[action] = fit_dynamics_policy(validation_samples, config, variants[action])
        elif action in OPTIONAL_ACTIONS:
            base_action = "boundary_only"
            if base_action not in policies:
                policies[base_action] = fit_dynamics_policy(validation_samples, config, variants[base_action])
    if "boundary_only" not in policies:
        policies["boundary_only"] = fit_dynamics_policy(validation_samples, config, variants["boundary_only"])
    if "dynamics_full" not in policies:
        policies["dynamics_full"] = fit_dynamics_policy(validation_samples, config, variants["dynamics_full"])
    return {
        "config": config,
        "dynamics_policies": policies,
        "score_policy": policies.get("dynamics_full"),
        "smoothing_window": int((config.get("policy", {}) or {}).get("smoothing_window", 5)),
        "ema_alpha": float((config.get("policy", {}) or {}).get("ema_alpha", 0.35)),
        "median_window": int((config.get("policy", {}) or {}).get("median_window", 5)),
    }


def apply_action(sample: dict, action: str, action_context: Dict[str, object]) -> ActionResult:
    start = time.perf_counter()
    pred = arr(sample["prediction"])
    policies = action_context.get("dynamics_policies", {})
    if action == "no_correction":
        out = pred.copy()
        info = {"action_triggered": 0.0}
    elif action in policies:
        out, dyn_info = apply_correction(sample["context"], pred, policies[action])
        info = {f"dynamics_{k}": v for k, v in dyn_info.items() if isinstance(v, (int, float))}
        info["action_triggered"] = float(dyn_info.get("trigger", 0.0))
    elif action == "naive_smoothing":
        out = naive_smoothing(pred, int(action_context.get("smoothing_window", 5)))
        info = {"action_triggered": 1.0}
    elif action == "ema_smoothing":
        out = ema_smoothing(pred, float(action_context.get("ema_alpha", 0.35)))
        info = {"action_triggered": 1.0}
    elif action == "median_smoothing":
        out = median_smoothing(pred, int(action_context.get("median_window", 5)))
        info = {"action_triggered": 1.0}
    elif action == "boundary_then_ema":
        boundary, _ = apply_correction(sample["context"], pred, policies["boundary_only"])
        out = ema_smoothing(boundary, float(action_context.get("ema_alpha", 0.35)))
        info = {"action_triggered": 1.0}
    elif action == "boundary_then_median":
        boundary, _ = apply_correction(sample["context"], pred, policies["boundary_only"])
        out = median_smoothing(boundary, int(action_context.get("median_window", 5)))
        info = {"action_triggered": 1.0}
    elif action == "boundary_then_selective_median":
        boundary, dyn_info = apply_correction(sample["context"], pred, policies["boundary_only"])
        out, selective_info = selective_residual_median(sample["context"], boundary, action_context)
        info = {f"dynamics_{k}": v for k, v in dyn_info.items() if isinstance(v, (int, float))}
        info.update(selective_info)
        info["action_triggered"] = 1.0
    else:
        raise ValueError(f"Unknown router action: {action}")
    return ActionResult(out.astype(np.float64), info, (time.perf_counter() - start) * 1000.0)


def selective_residual_median(context_values: Iterable[float], prediction_values: Iterable[float], action_context: Dict[str, object]) -> Tuple[np.ndarray, Dict[str, float]]:
    """Damp only prediction residual spikes that exceed context-supported roughness."""

    pred = arr(prediction_values)
    if pred.size == 0:
        return pred.copy(), {"selective_smoothing_rate": 0.0}
    cfg = (action_context.get("config", {}) or {}).get("policy", {}) or {}
    window = int(action_context.get("median_window", 5))
    strength = float(cfg.get("selective_smoothing_strength", 0.65))
    residual_quantile = float(cfg.get("selective_residual_quantile", 0.75))
    residual_quantile = min(max(residual_quantile, 0.0), 1.0)

    smoothed = median_smoothing(pred, window)
    residual = pred - smoothed
    context = arr(context_values)
    tail = _context_tail(context, pred.size)
    context_smooth = median_smoothing(tail, window)
    context_residual = np.abs(tail - context_smooth)
    threshold = float(np.quantile(context_residual, residual_quantile)) + EPS
    mask = np.abs(residual) > threshold
    if mask.size >= 3:
        mask = mask | np.r_[False, mask[:-1]] | np.r_[mask[1:], False]
    out = pred.copy()
    if np.any(mask):
        out[mask] = pred[mask] + strength * (smoothed[mask] - pred[mask])
    return out.astype(np.float64), {
        "selective_smoothing_rate": float(np.mean(mask)) if mask.size else 0.0,
        "selective_residual_threshold": threshold,
        "selective_smoothing_strength": strength,
    }


def route_action(sample: dict, router_policy: Dict[str, object], override_features: Optional[np.ndarray] = None) -> Tuple[str, Dict[str, object]]:
    feature_names = list(router_policy["feature_names"])
    if override_features is None:
        raw_features = extract_router_features(sample, router_policy["action_context"])
        raw = np.asarray([raw_features[name] for name in feature_names], dtype=np.float64)
    else:
        raw = np.asarray(override_features, dtype=np.float64)
        raw_features = {name: float(raw[idx]) for idx, name in enumerate(feature_names)}
    z = standardize(raw.reshape(1, -1), router_policy["feature_stats"])[0]
    model = router_policy["model"]
    action, probs = predict_action_from_model(raw, z, model, list(router_policy["candidate_actions"]), feature_names)
    info: Dict[str, object] = {
        "router_type": router_policy["router_type"],
        "route_confidence": float(np.max(probs)) if probs.size else 1.0,
        "route_margin": probability_margin(probs),
    }
    info.update(raw_features)
    return action, info


def apply_router_to_samples(samples: List[dict], router_policy: Dict[str, object], shuffled_features: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[Dict[str, object]], List[float]]:
    predictions = []
    infos = []
    latencies = []
    for idx, sample in enumerate(samples):
        start = time.perf_counter()
        override = shuffled_features[idx] if shuffled_features is not None else None
        pred, info = apply_router(sample, router_policy, override_features=override)
        elapsed = (time.perf_counter() - start) * 1000.0
        predictions.append(pred)
        infos.append(info)
        latencies.append(elapsed)
    enforce_deploy_action_cap(samples, predictions, infos, router_policy)
    return predictions, infos, latencies


def enforce_deploy_action_cap(samples: List[dict], predictions: List[np.ndarray], infos: List[Dict[str, object]], router_policy: Dict[str, object]) -> None:
    model = router_policy.get("model", {})
    cap_value = model.get("deploy_action_cap", None) if isinstance(model, dict) else None
    if cap_value is None or not infos:
        return
    cap = float(cap_value)
    if cap <= 0.0 or cap >= 1.0:
        return
    no_action = "no_correction"
    total = len(infos)
    max_count = max(0, int(math.floor(cap * total)))
    actions = [str(info.get("action", no_action)) for info in infos]
    for action in sorted(set(actions)):
        if action == no_action:
            continue
        indices = [idx for idx, value in enumerate(actions) if value == action]
        if len(indices) <= max_count:
            continue
        indices.sort(key=lambda idx: float(infos[idx].get("route_margin", 0.0)))
        for idx in indices[: len(indices) - max_count]:
            predictions[idx] = arr(samples[idx]["prediction"]).copy()
            infos[idx]["action"] = no_action
            infos[idx]["changed"] = 0.0
            infos[idx]["trigger"] = 0.0
            infos[idx]["deploy_cap_abstained"] = 1.0


def action_error_matrix(samples: List[dict], action_context: Dict[str, object], actions: Sequence[str]) -> np.ndarray:
    errors = np.zeros((len(samples), len(actions)), dtype=np.float64)
    for row_idx, sample in enumerate(samples):
        target = arr(sample["target"])
        for col_idx, action in enumerate(actions):
            pred = apply_action(sample, action, action_context).prediction
            errors[row_idx, col_idx] = float(np.mean((pred - target) ** 2))
    return errors


def label_best_safe_actions(errors: np.ndarray, actions: Sequence[str], config: Dict) -> List[str]:
    if errors.size == 0:
        return []
    router_cfg = config.get("router", {}) or {}
    no_idx = actions.index("no_correction") if "no_correction" in actions else 0
    margin = float(router_cfg.get("benefit_margin_pct", 0.02)) / 100.0
    prefer_no = float(router_cfg.get("prefer_no_action_margin_pct", 0.005)) / 100.0
    labels = []
    for row in errors:
        base = float(row[no_idx])
        best_idx = int(np.argmin(row))
        best_value = float(row[best_idx])
        if best_idx != no_idx and base - best_value <= margin * max(base, EPS):
            labels.append("no_correction")
        elif best_idx != no_idx and abs(best_value - base) <= prefer_no * max(base, EPS):
            labels.append("no_correction")
        else:
            labels.append(str(actions[best_idx]))
    return labels


def fit_rule_router(features: np.ndarray, errors: np.ndarray, actions: Sequence[str], feature_names: Sequence[str], config: Dict) -> Dict[str, object]:
    router_cfg = config.get("router", {}) or {}
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    boundary = features[:, name_to_idx["boundary_score"]]
    noise = 0.5 * features[:, name_to_idx["high_frequency_excess"]] + 0.5 * features[:, name_to_idx["pred_context_diff_std_ratio"]]
    risk = boundary + features[:, name_to_idx["first_diff_score"]] + features[:, name_to_idx["curvature_score"]] + noise
    smoothing_candidates = [a for a in ("median_smoothing", "ema_smoothing", "naive_smoothing") if a in actions]
    smoothing_action = best_global_action(errors, actions, smoothing_candidates) if smoothing_candidates else "no_correction"
    configured_fallback = str(router_cfg.get("rule_fallback_action", "") or "")
    fallback_action = configured_fallback if configured_fallback in actions else best_global_action(errors, actions, actions)
    return {
        "kind": "rule_router",
        "boundary_threshold": float(np.quantile(boundary, float(router_cfg.get("rule_boundary_quantile", 0.75)))),
        "noise_threshold": float(np.quantile(noise, float(router_cfg.get("rule_noise_quantile", 0.75)))),
        "low_risk_threshold": float(np.quantile(risk, float(router_cfg.get("rule_low_risk_quantile", 0.35)))),
        "smoothing_action": smoothing_action,
        "fallback_action": fallback_action,
    }


def fit_tree_router(features: np.ndarray, label_indices: np.ndarray, actions: Sequence[str], config: Dict) -> Dict[str, object]:
    router_cfg = config.get("router", {}) or {}
    max_depth = int(router_cfg.get("tree_max_depth", 3))
    min_leaf = int(router_cfg.get("min_samples_leaf", 24))
    tree = build_tree(features, label_indices, len(actions), max_depth, min_leaf)
    return {"kind": "shallow_tree_router", "tree": tree}


def fit_logistic_router(features: np.ndarray, label_indices: np.ndarray, actions: Sequence[str], config: Dict) -> Dict[str, object]:
    router_cfg = config.get("router", {}) or {}
    n_samples, n_features = features.shape
    n_classes = len(actions)
    x = np.c_[np.ones(n_samples), features]
    y = np.zeros((n_samples, n_classes), dtype=np.float64)
    y[np.arange(n_samples), label_indices] = 1.0
    weights = np.zeros((n_features + 1, n_classes), dtype=np.float64)
    lr = float(router_cfg.get("logistic_learning_rate", 0.08))
    l2 = float(router_cfg.get("logistic_l2", 0.05))
    steps = int(router_cfg.get("logistic_steps", 500))
    for _ in range(max(1, steps)):
        probs = softmax(x @ weights)
        grad = (x.T @ (probs - y)) / max(n_samples, 1)
        grad[1:] += l2 * weights[1:]
        weights -= lr * grad
    return {"kind": "logistic_router", "weights": weights}


def fit_capped_logistic_router(features: np.ndarray, label_indices: np.ndarray, actions: Sequence[str], config: Dict) -> Dict[str, object]:
    router_cfg = config.get("router", {}) or {}
    base = fit_logistic_router(features, label_indices, actions, config)
    x = np.c_[np.ones(features.shape[0]), features]
    probs = softmax(x @ np.asarray(base["weights"], dtype=np.float64))
    pred_idx = np.argmax(probs, axis=1) if probs.size else np.asarray([], dtype=np.int64)
    cap = float(router_cfg.get("max_single_action_rate", 0.86))
    dominant_idx = int(np.bincount(pred_idx, minlength=len(actions)).argmax()) if pred_idx.size else 0
    dominant_rate = float(np.mean(pred_idx == dominant_idx)) if pred_idx.size else 0.0
    threshold = 0.0
    if dominant_rate > cap:
        dominant_probs = probs[pred_idx == dominant_idx, dominant_idx]
        keep_fraction = min(1.0, max(0.0, cap / max(dominant_rate, EPS)))
        threshold = float(np.quantile(dominant_probs, 1.0 - keep_fraction)) if dominant_probs.size else 1.0
    return {
        "kind": "capped_logistic_router",
        "base_model": base,
        "dominant_action": str(actions[dominant_idx]) if actions else "",
        "dominant_action_index": dominant_idx,
        "dominant_action_rate": dominant_rate,
        "cap": cap,
        "dominant_probability_threshold": threshold,
    }


def fit_margin_abstain_router(features: np.ndarray, label_indices: np.ndarray, action_errors: np.ndarray, actions: Sequence[str], config: Dict) -> Dict[str, object]:
    """Fit a logistic router with a validation-selected confidence abstention."""

    router_cfg = config.get("router", {}) or {}
    base = fit_logistic_router(features, label_indices, actions, config)
    x = np.c_[np.ones(features.shape[0]), features]
    probs = softmax(x @ np.asarray(base["weights"], dtype=np.float64))
    pred_idx = np.argmax(probs, axis=1) if probs.size else np.asarray([], dtype=np.int64)
    margins = np.asarray([probability_margin(row) for row in probs], dtype=np.float64)
    no_idx = actions.index("no_correction") if "no_correction" in actions else 0
    active = pred_idx != no_idx
    candidates = [0.0]
    if np.any(active):
        quantiles = router_cfg.get("margin_abstain_quantiles", [0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
        candidates.extend(float(np.quantile(margins[active], float(q))) for q in quantiles)
    rate_penalty = float(router_cfg.get("margin_abstain_rate_penalty", 0.0))
    degeneracy_penalty = float(router_cfg.get("margin_abstain_degeneracy_penalty", 0.0))
    best_threshold = 0.0
    best_score = float("inf")
    best_rate = 0.0
    best_dominant = 0.0
    for threshold in sorted(set(candidates)):
        selected = pred_idx.copy()
        selected[(pred_idx != no_idx) & (margins < threshold)] = no_idx
        errors = action_errors[np.arange(action_errors.shape[0]), selected] if action_errors.size else np.asarray([0.0])
        correction_rate = float(np.mean(selected != no_idx)) if selected.size else 0.0
        dominant_rate = float(np.max(np.bincount(selected, minlength=len(actions))) / max(selected.size, 1)) if selected.size else 0.0
        score = float(np.mean(errors)) + rate_penalty * correction_rate + degeneracy_penalty * max(0.0, dominant_rate - 0.90)
        if score < best_score - 1e-12:
            best_score = score
            best_threshold = float(threshold)
            best_rate = correction_rate
            best_dominant = dominant_rate
    return {
        "kind": "margin_abstain_router",
        "base_model": base,
        "margin_threshold": best_threshold,
        "deploy_action_cap": router_cfg.get("margin_abstain_deploy_action_cap", None),
        "validation_score": best_score,
        "validation_correction_rate": best_rate,
        "validation_dominant_action_rate": best_dominant,
    }


def fit_smoothing_risk_guard_router(raw_features: np.ndarray, features: np.ndarray, label_indices: np.ndarray, action_errors: np.ndarray, actions: Sequence[str], feature_names: Sequence[str], config: Dict) -> Dict[str, object]:
    """Fit a margin router that abstains low-support smoothing actions.

    The guard score is target-free at test time. Validation targets are used
    only to choose the smoothing-support threshold.
    """

    router_cfg = config.get("router", {}) or {}
    base = fit_margin_abstain_router(features, label_indices, action_errors, actions, config)
    base = dict(base)
    base["deploy_action_cap"] = None
    probs = []
    selected = []
    for row, z_row in zip(raw_features, features):
        action, prob = predict_action_from_model(row, z_row, base, list(actions), feature_names)
        selected.append(action)
        probs.append(prob)
    selected_idx = np.asarray([actions.index(action) if action in actions else 0 for action in selected], dtype=np.int64)
    no_idx = actions.index("no_correction") if "no_correction" in actions else 0
    smoothing_idx = {actions.index(action) for action in ("median_smoothing", "ema_smoothing", "naive_smoothing") if action in actions}
    smoothing_mask = np.asarray([idx in smoothing_idx for idx in selected_idx], dtype=bool)
    support = smoothing_support_score(features, feature_names)
    candidates = [float("-inf")]
    if np.any(smoothing_mask):
        quantiles = router_cfg.get("smoothing_guard_quantiles", [0.10, 0.25, 0.40, 0.55, 0.70])
        candidates.extend(float(np.quantile(support[smoothing_mask], float(q))) for q in quantiles)
        min_quantile = router_cfg.get("smoothing_guard_min_quantile", None)
        if min_quantile is not None:
            min_threshold = float(np.quantile(support[smoothing_mask], float(min_quantile)))
            candidates = [max(float(value), min_threshold) for value in candidates]
            candidates.append(min_threshold)
    rate_penalty = float(router_cfg.get("smoothing_guard_rate_penalty", 0.0))
    degeneracy_penalty = float(router_cfg.get("smoothing_guard_degeneracy_penalty", 0.0))
    best_threshold = float("-inf")
    best_score = float("inf")
    best_rate = 0.0
    best_dominant = 0.0
    for threshold in sorted(set(candidates)):
        guarded = selected_idx.copy()
        guarded[smoothing_mask & (support < threshold)] = no_idx
        errors = action_errors[np.arange(action_errors.shape[0]), guarded] if action_errors.size else np.asarray([0.0])
        correction_rate = float(np.mean(guarded != no_idx)) if guarded.size else 0.0
        dominant_rate = float(np.max(np.bincount(guarded, minlength=len(actions))) / max(guarded.size, 1)) if guarded.size else 0.0
        score = float(np.mean(errors)) + rate_penalty * correction_rate + degeneracy_penalty * max(0.0, dominant_rate - 0.90)
        if score < best_score - 1e-12:
            best_score = score
            best_threshold = float(threshold)
            best_rate = correction_rate
            best_dominant = dominant_rate
    return {
        "kind": "smoothing_risk_guard_router",
        "base_model": base,
        "smoothing_support_threshold": best_threshold,
        "deploy_action_cap": router_cfg.get("margin_abstain_deploy_action_cap", None),
        "validation_score": best_score,
        "validation_correction_rate": best_rate,
        "validation_dominant_action_rate": best_dominant,
    }


def fit_smoothing_benefit_guard_router(raw_features: np.ndarray, features: np.ndarray, label_indices: np.ndarray, action_errors: np.ndarray, actions: Sequence[str], feature_names: Sequence[str], config: Dict) -> Dict[str, object]:
    """Fit a validation-only model of smoothing benefit versus no correction."""

    router_cfg = config.get("router", {}) or {}
    base = fit_margin_abstain_router(features, label_indices, action_errors, actions, config)
    base = dict(base)
    base["deploy_action_cap"] = None
    selected = []
    for row, z_row in zip(raw_features, features):
        action, _ = predict_action_from_model(row, z_row, base, list(actions), feature_names)
        selected.append(action)
    selected_idx = np.asarray([actions.index(action) if action in actions else 0 for action in selected], dtype=np.int64)
    no_idx = actions.index("no_correction") if "no_correction" in actions else 0
    smoothing_actions = [action for action in ("median_smoothing", "ema_smoothing", "naive_smoothing") if action in actions]
    smoothing_idx = {actions.index(action) for action in smoothing_actions}
    smoothing_mask = np.asarray([idx in smoothing_idx for idx in selected_idx], dtype=bool)
    x_all = benefit_guard_design(features, selected_idx, actions, smoothing_actions)
    weights = np.zeros(x_all.shape[1], dtype=np.float64)
    if np.any(smoothing_mask):
        y = action_errors[smoothing_mask, no_idx] - action_errors[np.where(smoothing_mask)[0], selected_idx[smoothing_mask]]
        x = x_all[smoothing_mask]
        l2 = float(router_cfg.get("smoothing_benefit_l2", 0.01))
        gram = x.T @ x
        ridge = l2 * np.eye(gram.shape[0], dtype=np.float64)
        ridge[0, 0] = 0.0
        weights = np.linalg.pinv(gram + ridge) @ x.T @ y
    predicted_benefit = x_all @ weights
    candidates = [float("-inf"), 0.0]
    if np.any(smoothing_mask):
        quantiles = router_cfg.get("smoothing_benefit_quantiles", [0.10, 0.25, 0.40, 0.55, 0.70])
        candidates.extend(float(np.quantile(predicted_benefit[smoothing_mask], float(q))) for q in quantiles)
    min_threshold = router_cfg.get("smoothing_benefit_min_threshold", None)
    if min_threshold is not None:
        min_value = float(min_threshold)
        candidates = [max(float(value), min_value) for value in candidates]
        candidates.append(min_value)
    rate_penalty = float(router_cfg.get("smoothing_benefit_rate_penalty", 0.0))
    degeneracy_penalty = float(router_cfg.get("smoothing_benefit_degeneracy_penalty", 0.0))
    best_threshold = float("-inf")
    best_score = float("inf")
    best_rate = 0.0
    best_dominant = 0.0
    for threshold in sorted(set(candidates)):
        guarded = selected_idx.copy()
        guarded[smoothing_mask & (predicted_benefit <= threshold)] = no_idx
        errors = action_errors[np.arange(action_errors.shape[0]), guarded] if action_errors.size else np.asarray([0.0])
        correction_rate = float(np.mean(guarded != no_idx)) if guarded.size else 0.0
        dominant_rate = float(np.max(np.bincount(guarded, minlength=len(actions))) / max(guarded.size, 1)) if guarded.size else 0.0
        score = float(np.mean(errors)) + rate_penalty * correction_rate + degeneracy_penalty * max(0.0, dominant_rate - 0.90)
        if score < best_score - 1e-12:
            best_score = score
            best_threshold = float(threshold)
            best_rate = correction_rate
            best_dominant = dominant_rate
    return {
        "kind": "smoothing_benefit_guard_router",
        "base_model": base,
        "smoothing_actions": smoothing_actions,
        "benefit_weights": weights,
        "benefit_threshold": best_threshold,
        "deploy_action_cap": router_cfg.get("margin_abstain_deploy_action_cap", None),
        "validation_score": best_score,
        "validation_correction_rate": best_rate,
        "validation_dominant_action_rate": best_dominant,
    }


def fit_stable_forecast_guard_router(features: np.ndarray, label_indices: np.ndarray, action_errors: np.ndarray, actions: Sequence[str], config: Dict) -> Dict[str, object]:
    """Fit a base router with a target-free stable-forecast smoothing guard."""

    router_cfg = config.get("router", {}) or {}
    base = fit_margin_abstain_router(features, label_indices, action_errors, actions, config)
    base = dict(base)
    base["deploy_action_cap"] = None
    return {
        "kind": "stable_forecast_guard_router",
        "base_model": base,
        "diff_std_ratio_threshold": float(router_cfg.get("stable_guard_min_diff_std_ratio", 1.0)),
        "deploy_action_cap": router_cfg.get("margin_abstain_deploy_action_cap", None),
    }


def fit_stable_selective_fallback_router(features: np.ndarray, label_indices: np.ndarray, action_errors: np.ndarray, actions: Sequence[str], config: Dict) -> Dict[str, object]:
    """Fit a stable-forecast guard that falls back to selective residual repair."""

    model = fit_stable_forecast_guard_router(features, label_indices, action_errors, actions, config)
    model = dict(model)
    model["kind"] = "stable_selective_fallback_router"
    model["fallback_action"] = "boundary_then_selective_median" if "boundary_then_selective_median" in actions else "no_correction"
    return model


def fit_spectral_support_guard_router(raw_features: np.ndarray, features: np.ndarray, label_indices: np.ndarray, action_errors: np.ndarray, actions: Sequence[str], feature_names: Sequence[str], config: Dict) -> Dict[str, object]:
    """Allow full smoothing only when prediction roughness is unsupported by context."""

    router_cfg = config.get("router", {}) or {}
    base = fit_margin_abstain_router(features, label_indices, action_errors, actions, config)
    base = dict(base)
    base["deploy_action_cap"] = None
    selected = []
    for row, z_row in zip(raw_features, features):
        action, _ = predict_action_from_model(row, z_row, base, list(actions), feature_names)
        selected.append(action)
    selected_idx = np.asarray([actions.index(action) if action in actions else 0 for action in selected], dtype=np.int64)
    no_idx = actions.index("no_correction") if "no_correction" in actions else 0
    fallback = "boundary_then_selective_median" if "boundary_then_selective_median" in actions else "no_correction"
    fallback_idx = actions.index(fallback) if fallback in actions else no_idx
    smoothing_idx = {actions.index(action) for action in ("median_smoothing", "ema_smoothing", "naive_smoothing") if action in actions}
    smoothing_mask = np.asarray([idx in smoothing_idx for idx in selected_idx], dtype=bool)
    unsupported = unsupported_noise_score(raw_features, feature_names)
    candidates = [float("-inf")]
    if np.any(smoothing_mask):
        quantiles = router_cfg.get("spectral_support_guard_quantiles", router_cfg.get("smoothing_guard_quantiles", [0.10, 0.25, 0.40, 0.55, 0.70]))
        candidates.extend(float(np.quantile(unsupported[smoothing_mask], float(q))) for q in quantiles)
        min_quantile = router_cfg.get("spectral_support_min_quantile", None)
        if min_quantile is not None:
            min_threshold = float(np.quantile(unsupported[smoothing_mask], float(min_quantile)))
            candidates = [max(float(value), min_threshold) for value in candidates]
            candidates.append(min_threshold)
    rate_penalty = float(router_cfg.get("spectral_support_rate_penalty", router_cfg.get("smoothing_guard_rate_penalty", 0.0)))
    degeneracy_penalty = float(router_cfg.get("spectral_support_degeneracy_penalty", router_cfg.get("smoothing_guard_degeneracy_penalty", 0.0)))
    best_threshold = float("-inf")
    best_score = float("inf")
    best_rate = 0.0
    best_dominant = 0.0
    for threshold in sorted(set(candidates)):
        guarded = selected_idx.copy()
        guarded[smoothing_mask & (unsupported < threshold)] = fallback_idx
        errors = action_errors[np.arange(action_errors.shape[0]), guarded] if action_errors.size else np.asarray([0.0])
        correction_rate = float(np.mean(guarded != no_idx)) if guarded.size else 0.0
        dominant_rate = float(np.max(np.bincount(guarded, minlength=len(actions))) / max(guarded.size, 1)) if guarded.size else 0.0
        score = float(np.mean(errors)) + rate_penalty * correction_rate + degeneracy_penalty * max(0.0, dominant_rate - 0.90)
        if score < best_score - 1e-12:
            best_score = score
            best_threshold = float(threshold)
            best_rate = correction_rate
            best_dominant = dominant_rate
    return {
        "kind": "spectral_support_guard_router",
        "base_model": base,
        "fallback_action": fallback,
        "unsupported_noise_threshold": best_threshold,
        "deploy_action_cap": router_cfg.get("margin_abstain_deploy_action_cap", None),
        "validation_score": best_score,
        "validation_correction_rate": best_rate,
        "validation_dominant_action_rate": best_dominant,
    }


def fit_smoothing_cap_selective_router(features: np.ndarray, label_indices: np.ndarray, action_errors: np.ndarray, actions: Sequence[str], config: Dict) -> Dict[str, object]:
    """Cap low-confidence full smoothing by falling back to selective repair."""

    router_cfg = config.get("router", {}) or {}
    base = fit_margin_abstain_router(features, label_indices, action_errors, actions, config)
    base = dict(base)
    base["deploy_action_cap"] = None
    x = np.c_[np.ones(features.shape[0]), features]
    probs = softmax(x @ np.asarray(base["base_model"]["weights"], dtype=np.float64))
    pred_idx = np.argmax(probs, axis=1) if probs.size else np.asarray([], dtype=np.int64)
    margins = np.asarray([probability_margin(row) for row in probs], dtype=np.float64)
    no_idx = actions.index("no_correction") if "no_correction" in actions else 0
    if float(base.get("margin_threshold", 0.0)) > 0.0:
        pred_idx[(pred_idx != no_idx) & (margins < float(base.get("margin_threshold", 0.0)))] = no_idx
    smoothing_idx = {actions.index(action) for action in ("median_smoothing", "ema_smoothing", "naive_smoothing") if action in actions}
    smoothing_mask = np.asarray([idx in smoothing_idx for idx in pred_idx], dtype=bool)
    fallback = "boundary_then_selective_median" if "boundary_then_selective_median" in actions else "no_correction"
    fallback_idx = actions.index(fallback) if fallback in actions else no_idx
    candidates = [float("-inf")]
    if np.any(smoothing_mask):
        quantiles = router_cfg.get("smoothing_cap_margin_quantiles", [0.10, 0.25, 0.40, 0.55, 0.70])
        candidates.extend(float(np.quantile(margins[smoothing_mask], float(q))) for q in quantiles)
    rate_penalty = float(router_cfg.get("smoothing_cap_rate_penalty", 0.0))
    degeneracy_penalty = float(router_cfg.get("smoothing_cap_degeneracy_penalty", 0.0))
    best_threshold = float("-inf")
    best_score = float("inf")
    best_rate = 0.0
    best_dominant = 0.0
    for threshold in sorted(set(candidates)):
        selected = pred_idx.copy()
        selected[smoothing_mask & (margins < threshold)] = fallback_idx
        errors = action_errors[np.arange(action_errors.shape[0]), selected] if action_errors.size else np.asarray([0.0])
        correction_rate = float(np.mean(selected != no_idx)) if selected.size else 0.0
        dominant_rate = float(np.max(np.bincount(selected, minlength=len(actions))) / max(selected.size, 1)) if selected.size else 0.0
        score = float(np.mean(errors)) + rate_penalty * correction_rate + degeneracy_penalty * max(0.0, dominant_rate - 0.90)
        if score < best_score - 1e-12:
            best_score = score
            best_threshold = float(threshold)
            best_rate = correction_rate
            best_dominant = dominant_rate
    return {
        "kind": "smoothing_cap_selective_router",
        "base_model": base,
        "fallback_action": fallback,
        "smoothing_margin_threshold": best_threshold,
        "deploy_action_cap": router_cfg.get("margin_abstain_deploy_action_cap", None),
        "validation_score": best_score,
        "validation_correction_rate": best_rate,
        "validation_dominant_action_rate": best_dominant,
    }


def fit_stable_smoothing_cap_router(features: np.ndarray, label_indices: np.ndarray, action_errors: np.ndarray, actions: Sequence[str], config: Dict) -> Dict[str, object]:
    """Add a stable-forecast veto after the smoothing confidence cap."""

    router_cfg = config.get("router", {}) or {}
    base = fit_smoothing_cap_selective_router(features, label_indices, action_errors, actions, config)
    base = dict(base)
    base["deploy_action_cap"] = None
    return {
        "kind": "stable_smoothing_cap_router",
        "base_model": base,
        "fallback_action": "boundary_then_selective_median" if "boundary_then_selective_median" in actions else "no_correction",
        "diff_std_ratio_threshold": float(router_cfg.get("stable_guard_min_diff_std_ratio", 1.0)),
        "deploy_action_cap": router_cfg.get("margin_abstain_deploy_action_cap", None),
        "validation_score": base.get("validation_score", 0.0),
        "validation_correction_rate": base.get("validation_correction_rate", 0.0),
        "validation_dominant_action_rate": base.get("validation_dominant_action_rate", 0.0),
    }


def fit_conditional_stable_cap_router(raw_features: np.ndarray, features: np.ndarray, label_indices: np.ndarray, action_errors: np.ndarray, actions: Sequence[str], feature_names: Sequence[str], config: Dict) -> Dict[str, object]:
    """Veto stable raw-smoothing only when low unsupported-noise supports it."""

    router_cfg = config.get("router", {}) or {}
    base = fit_smoothing_cap_selective_router(features, label_indices, action_errors, actions, config)
    base = dict(base)
    base["deploy_action_cap"] = None
    selected = []
    for row, z_row in zip(raw_features, features):
        action, _ = predict_action_from_model(row, z_row, base, list(actions), feature_names)
        selected.append(action)
    selected_idx = np.asarray([actions.index(action) if action in actions else 0 for action in selected], dtype=np.int64)
    no_idx = actions.index("no_correction") if "no_correction" in actions else 0
    fallback = "boundary_then_selective_median" if "boundary_then_selective_median" in actions else "no_correction"
    fallback_idx = actions.index(fallback) if fallback in actions else no_idx
    smoothing_idx = {actions.index(action) for action in ("median_smoothing", "ema_smoothing", "naive_smoothing") if action in actions}
    smoothing_mask = np.asarray([idx in smoothing_idx for idx in selected_idx], dtype=bool)
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    diff_idx = name_to_idx.get("pred_context_diff_std_ratio", 0)
    diff_ratio = np.asarray(raw_features[:, diff_idx], dtype=np.float64) if raw_features.size else np.asarray([], dtype=np.float64)
    stable_mask = diff_ratio < float(router_cfg.get("stable_guard_min_diff_std_ratio", 1.0))
    unsupported = unsupported_noise_score(raw_features, feature_names)
    eligible = smoothing_mask & stable_mask
    candidates = [float("-inf")]
    if np.any(eligible):
        quantiles = router_cfg.get("conditional_stable_veto_quantiles", [0.10, 0.25, 0.40, 0.55, 0.70])
        candidates.extend(float(np.quantile(unsupported[eligible], float(q))) for q in quantiles)
        min_quantile = router_cfg.get("conditional_stable_veto_min_quantile", None)
        if min_quantile is not None:
            min_threshold = float(np.quantile(unsupported[eligible], float(min_quantile)))
            candidates = [max(float(value), min_threshold) for value in candidates]
            candidates.append(min_threshold)
    rate_penalty = float(router_cfg.get("conditional_stable_veto_rate_penalty", 0.0))
    degeneracy_penalty = float(router_cfg.get("conditional_stable_veto_degeneracy_penalty", 0.0))
    best_threshold = float("-inf")
    best_score = float("inf")
    best_rate = 0.0
    best_dominant = 0.0
    for threshold in sorted(set(candidates)):
        guarded = selected_idx.copy()
        guarded[eligible & (unsupported <= threshold)] = fallback_idx
        errors = action_errors[np.arange(action_errors.shape[0]), guarded] if action_errors.size else np.asarray([0.0])
        correction_rate = float(np.mean(guarded != no_idx)) if guarded.size else 0.0
        dominant_rate = float(np.max(np.bincount(guarded, minlength=len(actions))) / max(guarded.size, 1)) if guarded.size else 0.0
        score = float(np.mean(errors)) + rate_penalty * correction_rate + degeneracy_penalty * max(0.0, dominant_rate - 0.90)
        if score < best_score - 1e-12:
            best_score = score
            best_threshold = float(threshold)
            best_rate = correction_rate
            best_dominant = dominant_rate
    return {
        "kind": "conditional_stable_cap_router",
        "base_model": base,
        "fallback_action": fallback,
        "diff_std_ratio_threshold": float(router_cfg.get("stable_guard_min_diff_std_ratio", 1.0)),
        "unsupported_noise_threshold": best_threshold,
        "deploy_action_cap": router_cfg.get("margin_abstain_deploy_action_cap", None),
        "validation_score": best_score,
        "validation_correction_rate": best_rate,
        "validation_dominant_action_rate": best_dominant,
    }


def fit_selective_smoothing_alias_router(features: np.ndarray, label_indices: np.ndarray, action_errors: np.ndarray, actions: Sequence[str], config: Dict) -> Dict[str, object]:
    """Fit a margin router but deploy raw smoothing labels as selective repair."""

    base = fit_margin_abstain_router(features, label_indices, action_errors, actions, config)
    base = dict(base)
    base["deploy_action_cap"] = None
    router_cfg = config.get("router", {}) or {}
    return {
        "kind": "selective_smoothing_alias_router",
        "base_model": base,
        "smoothing_actions": [action for action in ("median_smoothing", "ema_smoothing", "naive_smoothing") if action in actions],
        "alias_action": "boundary_then_selective_median" if "boundary_then_selective_median" in actions else "no_correction",
        "deploy_action_cap": router_cfg.get("margin_abstain_deploy_action_cap", None),
    }


def predict_action_from_model(raw_features: np.ndarray, features: np.ndarray, model: Dict[str, object], actions: List[str], feature_names: Sequence[str]) -> Tuple[str, np.ndarray]:
    kind = str(model["kind"])
    if kind == "rule_router":
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        boundary = float(raw_features[name_to_idx["boundary_score"]])
        noise = 0.5 * float(raw_features[name_to_idx["high_frequency_excess"]]) + 0.5 * float(raw_features[name_to_idx["pred_context_diff_std_ratio"]])
        risk = boundary + float(raw_features[name_to_idx["first_diff_score"]]) + float(raw_features[name_to_idx["curvature_score"]]) + noise
        if boundary >= float(model["boundary_threshold"]) and boundary >= noise:
            action = "boundary_only" if "boundary_only" in actions else actions[0]
        elif noise >= float(model["noise_threshold"]):
            action = str(model["smoothing_action"])
        elif risk <= float(model["low_risk_threshold"]):
            action = "no_correction" if "no_correction" in actions else actions[0]
        else:
            action = str(model["fallback_action"])
        return action, one_hot_probs(actions, action)
    if kind == "shallow_tree_router":
        probs = predict_tree(np.asarray(features, dtype=np.float64), model["tree"])
        return actions[int(np.argmax(probs))], probs
    if kind == "logistic_router":
        x = np.r_[1.0, np.asarray(features, dtype=np.float64)]
        probs = softmax(x @ np.asarray(model["weights"], dtype=np.float64))
        return actions[int(np.argmax(probs))], probs
    if kind == "capped_logistic_router":
        action, probs = predict_action_from_model(raw_features, features, model["base_model"], actions, feature_names)
        dominant = str(model.get("dominant_action", ""))
        if action == dominant:
            idx = int(model.get("dominant_action_index", 0))
            if idx < probs.size and float(probs[idx]) < float(model.get("dominant_probability_threshold", 0.0)):
                no_action = "no_correction" if "no_correction" in actions else actions[0]
                return no_action, probs
        return action, probs
    if kind == "margin_abstain_router":
        action, probs = predict_action_from_model(raw_features, features, model["base_model"], actions, feature_names)
        no_action = "no_correction" if "no_correction" in actions else actions[0]
        if action != no_action and probability_margin(probs) < float(model.get("margin_threshold", 0.0)):
            return no_action, probs
        return action, probs
    if kind == "smoothing_risk_guard_router":
        action, probs = predict_action_from_model(raw_features, features, model["base_model"], actions, feature_names)
        if action in {"median_smoothing", "ema_smoothing", "naive_smoothing"}:
            support = float(smoothing_support_score(np.asarray(features, dtype=np.float64).reshape(1, -1), feature_names)[0])
            if support < float(model.get("smoothing_support_threshold", float("-inf"))):
                no_action = "no_correction" if "no_correction" in actions else actions[0]
                return no_action, probs
        return action, probs
    if kind == "smoothing_benefit_guard_router":
        action, probs = predict_action_from_model(raw_features, features, model["base_model"], actions, feature_names)
        if action in set(model.get("smoothing_actions", [])):
            idx = actions.index(action) if action in actions else 0
            x = benefit_guard_design(np.asarray(features, dtype=np.float64).reshape(1, -1), np.asarray([idx], dtype=np.int64), actions, model.get("smoothing_actions", []))[0]
            benefit = float(x @ np.asarray(model.get("benefit_weights", []), dtype=np.float64))
            if benefit <= float(model.get("benefit_threshold", float("-inf"))):
                no_action = "no_correction" if "no_correction" in actions else actions[0]
                return no_action, probs
        return action, probs
    if kind == "stable_forecast_guard_router":
        action, probs = predict_action_from_model(raw_features, features, model["base_model"], actions, feature_names)
        if action in {"median_smoothing", "ema_smoothing", "naive_smoothing"}:
            name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
            diff_ratio = float(raw_features[name_to_idx.get("pred_context_diff_std_ratio", 0)])
            if diff_ratio < float(model.get("diff_std_ratio_threshold", 1.0)):
                no_action = "no_correction" if "no_correction" in actions else actions[0]
                return no_action, probs
        return action, probs
    if kind == "stable_selective_fallback_router":
        action, probs = predict_action_from_model(raw_features, features, model["base_model"], actions, feature_names)
        if action in {"median_smoothing", "ema_smoothing", "naive_smoothing"}:
            name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
            diff_ratio = float(raw_features[name_to_idx.get("pred_context_diff_std_ratio", 0)])
            if diff_ratio < float(model.get("diff_std_ratio_threshold", 1.0)):
                fallback = str(model.get("fallback_action", "no_correction"))
                if fallback in actions:
                    return fallback, probs
                no_action = "no_correction" if "no_correction" in actions else actions[0]
                return no_action, probs
        return action, probs
    if kind == "spectral_support_guard_router":
        action, probs = predict_action_from_model(raw_features, features, model["base_model"], actions, feature_names)
        if action in {"median_smoothing", "ema_smoothing", "naive_smoothing"}:
            unsupported = float(unsupported_noise_score(np.asarray(raw_features, dtype=np.float64).reshape(1, -1), feature_names)[0])
            if unsupported < float(model.get("unsupported_noise_threshold", float("-inf"))):
                fallback = str(model.get("fallback_action", "no_correction"))
                if fallback in actions:
                    return fallback, probs
                no_action = "no_correction" if "no_correction" in actions else actions[0]
                return no_action, probs
        return action, probs
    if kind == "smoothing_cap_selective_router":
        action, probs = predict_action_from_model(raw_features, features, model["base_model"], actions, feature_names)
        if action in {"median_smoothing", "ema_smoothing", "naive_smoothing"} and probability_margin(probs) < float(model.get("smoothing_margin_threshold", float("-inf"))):
            fallback = str(model.get("fallback_action", "no_correction"))
            if fallback in actions:
                return fallback, probs
            no_action = "no_correction" if "no_correction" in actions else actions[0]
            return no_action, probs
        return action, probs
    if kind == "stable_smoothing_cap_router":
        action, probs = predict_action_from_model(raw_features, features, model["base_model"], actions, feature_names)
        if action in {"median_smoothing", "ema_smoothing", "naive_smoothing"}:
            name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
            diff_ratio = float(raw_features[name_to_idx.get("pred_context_diff_std_ratio", 0)])
            if diff_ratio < float(model.get("diff_std_ratio_threshold", 1.0)):
                fallback = str(model.get("fallback_action", "no_correction"))
                if fallback in actions:
                    return fallback, probs
                no_action = "no_correction" if "no_correction" in actions else actions[0]
                return no_action, probs
        return action, probs
    if kind == "conditional_stable_cap_router":
        action, probs = predict_action_from_model(raw_features, features, model["base_model"], actions, feature_names)
        if action in {"median_smoothing", "ema_smoothing", "naive_smoothing"}:
            name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
            diff_ratio = float(raw_features[name_to_idx.get("pred_context_diff_std_ratio", 0)])
            unsupported = float(unsupported_noise_score(np.asarray(raw_features, dtype=np.float64).reshape(1, -1), feature_names)[0])
            if diff_ratio < float(model.get("diff_std_ratio_threshold", 1.0)) and unsupported <= float(model.get("unsupported_noise_threshold", float("-inf"))):
                fallback = str(model.get("fallback_action", "no_correction"))
                if fallback in actions:
                    return fallback, probs
                no_action = "no_correction" if "no_correction" in actions else actions[0]
                return no_action, probs
        return action, probs
    if kind == "selective_smoothing_alias_router":
        action, probs = predict_action_from_model(raw_features, features, model["base_model"], actions, feature_names)
        if action in set(model.get("smoothing_actions", [])):
            alias = str(model.get("alias_action", "no_correction"))
            if alias in actions:
                return alias, probs
        return action, probs
    if kind == "harm_aware_router":
        action, probs = predict_action_from_model(raw_features, features, model["base_model"], actions, feature_names)
        no_action = "no_correction" if "no_correction" in actions else actions[0]
        if probability_margin(probs) < float(model.get("margin", 0.0003)):
            return no_action, probs
        return action, probs
    if kind == "oracle_val_policy":
        action = str(model["action"])
        return action, one_hot_probs(actions, action)
    raise ValueError(f"Unknown router model kind: {kind}")


def build_tree(features: np.ndarray, labels: np.ndarray, n_classes: int, max_depth: int, min_leaf: int) -> Dict[str, object]:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    probs = counts / max(float(counts.sum()), 1.0)
    node = {"probs": probs, "class": int(np.argmax(probs)), "n": int(labels.size)}
    if max_depth <= 0 or labels.size < 2 * min_leaf or np.max(counts) == labels.size:
        node["leaf"] = True
        return node
    split = best_tree_split(features, labels, n_classes, min_leaf)
    if split is None:
        node["leaf"] = True
        return node
    feature_idx, threshold, left_mask = split
    node.update(
        {
            "leaf": False,
            "feature_idx": int(feature_idx),
            "threshold": float(threshold),
            "left": build_tree(features[left_mask], labels[left_mask], n_classes, max_depth - 1, min_leaf),
            "right": build_tree(features[~left_mask], labels[~left_mask], n_classes, max_depth - 1, min_leaf),
        }
    )
    return node


def best_tree_split(features: np.ndarray, labels: np.ndarray, n_classes: int, min_leaf: int) -> Optional[Tuple[int, float, np.ndarray]]:
    best = None
    best_impurity = gini(labels, n_classes)
    for feature_idx in range(features.shape[1]):
        values = features[:, feature_idx]
        thresholds = np.unique(np.quantile(values, np.linspace(0.10, 0.90, 9)))
        for threshold in thresholds:
            left = values <= threshold
            left_n = int(left.sum())
            right_n = int((~left).sum())
            if left_n < min_leaf or right_n < min_leaf:
                continue
            impurity = (left_n * gini(labels[left], n_classes) + right_n * gini(labels[~left], n_classes)) / labels.size
            if impurity + 1e-12 < best_impurity:
                best_impurity = impurity
                best = (feature_idx, float(threshold), left.copy())
    return best


def predict_tree(features: np.ndarray, tree: Dict[str, object]) -> np.ndarray:
    node = tree
    while not bool(node.get("leaf", False)):
        if float(features[int(node["feature_idx"])]) <= float(node["threshold"]):
            node = node["left"]
        else:
            node = node["right"]
    return np.asarray(node["probs"], dtype=np.float64)


def gini(labels: np.ndarray, n_classes: int) -> float:
    if labels.size == 0:
        return 0.0
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    probs = counts / float(labels.size)
    return float(1.0 - np.sum(probs**2))


def feature_matrix(samples: List[dict], action_context: Dict[str, object], feature_names: Sequence[str]) -> np.ndarray:
    rows = []
    for sample in samples:
        feats = extract_router_features(sample, action_context)
        rows.append([feats[name] for name in feature_names])
    return np.asarray(rows, dtype=np.float64)


def fit_feature_stats(features: np.ndarray) -> Dict[str, np.ndarray]:
    mean = np.mean(features, axis=0) if features.size else np.asarray([])
    std = np.std(features, axis=0) if features.size else np.asarray([])
    std = np.where(std < EPS, 1.0, std)
    return {"mean": mean, "std": std}


def standardize(features: np.ndarray, stats: Dict[str, np.ndarray]) -> np.ndarray:
    return (np.asarray(features, dtype=np.float64) - np.asarray(stats["mean"], dtype=np.float64)) / np.asarray(stats["std"], dtype=np.float64)


def softmax(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        shifted = values - np.max(values)
        exp = np.exp(shifted)
        return exp / max(float(np.sum(exp)), EPS)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(np.sum(exp, axis=1, keepdims=True), EPS)


def one_hot_probs(actions: Sequence[str], action: str) -> np.ndarray:
    probs = np.zeros(len(actions), dtype=np.float64)
    probs[actions.index(action) if action in actions else 0] = 1.0
    return probs


def probability_margin(probs: np.ndarray) -> float:
    probs = np.sort(np.asarray(probs, dtype=np.float64))[::-1]
    if probs.size <= 1:
        return 1.0
    return float(probs[0] - probs[1])


def smoothing_support_score(features: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    def col(name: str) -> np.ndarray:
        if name not in name_to_idx:
            return np.zeros(features.shape[0], dtype=np.float64)
        return features[:, name_to_idx[name]]

    hf = col("high_frequency_excess")
    spectral = col("spectral_distance")
    diff_std = col("pred_context_diff_std_ratio")
    boundary = col("boundary_score")
    return np.maximum(0.0, hf) + np.maximum(0.0, spectral) + np.maximum(0.0, diff_std) + 0.25 * np.maximum(0.0, boundary)


def unsupported_noise_score(features: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Score target-free evidence that prediction roughness exceeds context support."""

    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}

    def col(name: str, default: float = 0.0) -> np.ndarray:
        if name not in name_to_idx:
            return np.full(features.shape[0], default, dtype=np.float64)
        return features[:, name_to_idx[name]]

    hf_excess = np.maximum(0.0, col("high_frequency_excess"))
    spectral = np.maximum(0.0, col("spectral_distance"))
    diff_excess = np.maximum(0.0, col("pred_context_diff_std_ratio", 1.0) - 1.0)
    variance_excess = np.maximum(0.0, col("pred_context_variance_ratio", 1.0) - 1.0)
    return hf_excess + 0.50 * spectral + 0.35 * diff_excess + 0.15 * variance_excess


def benefit_guard_design(features: np.ndarray, selected_idx: np.ndarray, actions: Sequence[str], smoothing_actions: Sequence[str]) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    selected_idx = np.asarray(selected_idx, dtype=np.int64).reshape(-1)
    action_to_idx = {action: idx for idx, action in enumerate(actions)}
    one_hot = np.zeros((features.shape[0], len(smoothing_actions)), dtype=np.float64)
    for col_idx, action in enumerate(smoothing_actions):
        action_idx = action_to_idx.get(str(action), -1)
        one_hot[:, col_idx] = (selected_idx == action_idx).astype(np.float64)
    return np.c_[np.ones(features.shape[0], dtype=np.float64), features, one_hot]


def best_global_action(errors: np.ndarray, actions: Sequence[str], candidates: Sequence[str]) -> str:
    action_to_idx = {action: idx for idx, action in enumerate(actions)}
    valid = [a for a in candidates if a in action_to_idx]
    if not valid:
        return str(actions[0])
    return min(valid, key=lambda action: float(np.mean(errors[:, action_to_idx[action]])))


def _context_tail(context: np.ndarray, horizon: int) -> np.ndarray:
    if context.size == 0:
        return np.zeros(max(1, horizon), dtype=np.float64)
    n = max(1, int(horizon))
    if context.size >= n:
        return context[-n:].astype(np.float64)
    return np.pad(context.astype(np.float64), (n - context.size, 0), mode="edge")


def _normalized_power(values: np.ndarray) -> np.ndarray:
    values = arr(values)
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    centered = values - float(np.mean(values))
    power = np.abs(np.fft.rfft(centered)) ** 2
    total = float(np.sum(power)) + EPS
    return power / total


def _high_frequency_ratio(values: np.ndarray) -> float:
    power = _normalized_power(values)
    if power.size <= 3:
        return 0.0
    cutoff = int(math.ceil(0.60 * power.size))
    return float(np.sum(power[cutoff:]))


def histogram(values: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def compact_distribution(values: Sequence[str]) -> str:
    counts = histogram(values)
    total = max(sum(counts.values()), 1)
    return ";".join(f"{key}:{counts[key] / total:.4f}" for key in sorted(counts))


def action_entropy(values: Sequence[str]) -> float:
    counts = histogram(values)
    total = float(max(sum(counts.values()), 1))
    entropy = 0.0
    for count in counts.values():
        p = float(count) / total
        entropy -= p * math.log(p + EPS)
    return float(entropy)


def action_rate(values: Sequence[str], action_set: Sequence[str]) -> float:
    if not values:
        return 0.0
    allowed = set(action_set)
    return float(np.mean([str(v) in allowed for v in values]))


def router_metric_row(variant: str, samples: List[dict], predictions: List[np.ndarray], infos: List[Dict[str, object]], latencies_ms: List[float], policy: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    row = metric_row(variant, samples, predictions, infos, latencies_ms, policy)
    actions = [str(info.get("action", "")) for info in infos]
    row.update(
        {
            "action_distribution": compact_distribution(actions),
            "action_entropy": action_entropy(actions),
            "no_correction_action_rate": action_rate(actions, ["no_correction"]),
            "boundary_action_rate": action_rate(actions, ["boundary_only", "dynamics_full"]),
            "smoothing_action_rate": action_rate(actions, ["median_smoothing", "ema_smoothing", "naive_smoothing", "boundary_then_ema", "boundary_then_median"]),
        }
    )
    return row


def oracle_test_ceiling(samples: List[dict], action_context: Dict[str, object], actions: Sequence[str]) -> Tuple[List[np.ndarray], List[Dict[str, object]], List[float]]:
    predictions = []
    infos = []
    latencies = []
    for sample in samples:
        start = time.perf_counter()
        target = arr(sample["target"])
        best_action = "no_correction"
        best_error = float("inf")
        best_prediction = arr(sample["prediction"])
        for action in actions:
            pred = apply_action(sample, action, action_context).prediction
            value = float(np.mean((pred - target) ** 2))
            if value < best_error:
                best_error = value
                best_action = action
                best_prediction = pred
        latencies.append((time.perf_counter() - start) * 1000.0)
        predictions.append(best_prediction)
        infos.append({"action": best_action, "changed": float(best_action != "no_correction"), "trigger": float(best_action != "no_correction"), "oracle_test_target_used": 1.0})
    return predictions, infos, latencies
