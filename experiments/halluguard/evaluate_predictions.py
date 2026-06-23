"""Evaluate HalluGuard corrections on prediction samples."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

from correction import Thresholds, apply_correction, calibrate_thresholds, score_sample, trigger_flags
from metrics import aggregate_metrics, stress_slice_metrics


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return validate_prediction_samples(samples, source=str(path))


def load_csv(path: Path) -> List[dict]:
    samples = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample = dict(row)
            for key in ["context", "prediction", "target"]:
                sample[key] = json.loads(sample[key])
            samples.append(sample)
    return validate_prediction_samples(samples, source=str(path))


def load_prediction_file(path: Path) -> List[dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".csv":
        return load_csv(path)
    raise ValueError(f"Unsupported prediction file extension: {path.suffix}. Use .jsonl or .csv.")


def validate_prediction_samples(samples: List[dict], source: str = "<memory>") -> List[dict]:
    required = ["sample_id", "dataset", "model", "split", "context", "prediction", "target"]
    for idx, sample in enumerate(samples):
        missing = [key for key in required if key not in sample]
        if missing:
            raise ValueError(f"{source} sample {idx} is missing required fields: {missing}")
        for key in ["context", "prediction", "target"]:
            if not isinstance(sample[key], list) or not sample[key]:
                raise ValueError(f"{source} sample {sample['sample_id']} field {key} must be a non-empty list.")
            sample[key] = [float(v) for v in sample[key]]
        if len(sample["prediction"]) != len(sample["target"]):
            raise ValueError(
                f"{source} sample {sample['sample_id']} prediction and target lengths differ: "
                f"{len(sample['prediction'])} vs {len(sample['target'])}"
            )
    return samples


def write_jsonl(samples: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def split_samples(samples: Iterable[dict], split: str) -> List[dict]:
    return [s for s in samples if s.get("split") == split]


def sample_error(sample: dict, prediction: Iterable[float], metric: str = "mse") -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(sample["target"], dtype=np.float64)
    if metric == "mae":
        return float(np.mean(np.abs(pred - target)))
    return float(np.mean((pred - target) ** 2))


def calibrate_evaluation_policy(
    calibration_samples: List[dict],
    config: Dict,
    source_split: str,
) -> Tuple[Thresholds, Dict[str, object]]:
    """Select a trigger/strength policy from validation only.

    The default path returns the original Stage 7 behavior. When
    `validation_calibrated_policy.enabled` is true, the search uses validation
    targets to choose a common threshold quantile plus component-specific
    margins and lambdas. Test samples are never read here.
    """

    corr_cfg = config["correction"]
    policy_cfg = corr_cfg.get("validation_calibrated_policy", {}) or {}
    high_freq_cutoff_ratio = float(corr_cfg.get("high_freq_cutoff_ratio", 0.5))
    freq_score_mode = str(corr_cfg.get("freq_score_mode", "excess_plus_spectral"))
    default_quantile = float(config["thresholds"]["default_quantile"])

    if not bool(policy_cfg.get("enabled", False)):
        thresholds = calibrate_thresholds(
            calibration_samples,
            quantile=default_quantile,
            high_freq_cutoff_ratio=high_freq_cutoff_ratio,
            source_split=source_split,
            freq_score_mode=freq_score_mode,
        )
        return thresholds, {"enabled": False}

    quantiles = _float_list(policy_cfg.get("candidate_quantiles", config["thresholds"].get("quantiles", [default_quantile])))
    if default_quantile not in quantiles:
        quantiles.append(default_quantile)
    quantiles = sorted(set(float(q) for q in quantiles))
    margins = _float_list(policy_cfg.get("margin_factors", [0.0, 0.25, 0.5]))
    trend_lambdas = _float_list(policy_cfg.get("lambda_trend_values", corr_cfg.get("lambda_trend_values", [corr_cfg["default_lambda_trend"]])))
    freq_lambdas = _float_list(policy_cfg.get("lambda_freq_values", corr_cfg.get("lambda_freq_values", [corr_cfg["default_lambda_freq"]])))
    if 0.0 not in trend_lambdas:
        trend_lambdas.insert(0, 0.0)
    if 0.0 not in freq_lambdas:
        freq_lambdas.insert(0, 0.0)

    objective_metric = str(policy_cfg.get("objective", "mse"))
    correction_rate_penalty = float(policy_cfg.get("correction_rate_penalty", 0.0))
    require_positive_gain = bool(policy_cfg.get("require_positive_gain", True))
    min_relative_gain_pct = float(policy_cfg.get("min_relative_gain_pct", 0.0))

    baseline = float(np.mean([sample_error(s, s["prediction"], objective_metric) for s in calibration_samples]))
    best = None
    for quantile in quantiles:
        thresholds = calibrate_thresholds(
            calibration_samples,
            quantile=quantile,
            high_freq_cutoff_ratio=high_freq_cutoff_ratio,
            source_split=source_split,
            freq_score_mode=freq_score_mode,
        )
        trend_choice = _best_component_policy(
            calibration_samples,
            thresholds,
            component="trend",
            margins=margins,
            lambdas=trend_lambdas,
            config=config,
            objective_metric=objective_metric,
            baseline=baseline,
            require_positive_gain=require_positive_gain,
            min_relative_gain_pct=min_relative_gain_pct,
        )
        freq_choice = _best_component_policy(
            calibration_samples,
            thresholds,
            component="freq",
            margins=margins,
            lambdas=freq_lambdas,
            config=config,
            objective_metric=objective_metric,
            baseline=baseline,
            require_positive_gain=require_positive_gain,
            min_relative_gain_pct=min_relative_gain_pct,
        )
        policy = {
            "enabled": True,
            "source_split": source_split,
            "objective": objective_metric,
            "selected_quantile": float(quantile),
            "trend_margin_factor": trend_choice["margin_factor"],
            "freq_margin_factor": freq_choice["margin_factor"],
            "lambda_trend": trend_choice["lambda"],
            "lambda_freq": freq_choice["lambda"],
            "trend_threshold": thresholds.trend * (1.0 + trend_choice["margin_factor"]),
            "freq_threshold": thresholds.freq * (1.0 + freq_choice["margin_factor"]),
            "trend_val_delta_pct": trend_choice["delta_pct"],
            "freq_val_delta_pct": freq_choice["delta_pct"],
            "baseline_val_objective": baseline,
        }
        combined_value, combined_rate = _policy_objective(
            calibration_samples,
            thresholds,
            policy,
            config,
            objective_metric=objective_metric,
            variant="trend_frequency",
        )
        random_weight = float(policy_cfg.get("random_separation_weight", 0.0))
        random_advantage = 0.0
        random_value = ""
        if random_weight > 0:
            random_value = _random_policy_objective(
                calibration_samples,
                thresholds,
                policy,
                config,
                objective_metric=objective_metric,
                seeds=[int(s) for s in policy_cfg.get("random_separation_seeds", policy_cfg.get("random_seeds", [101, 202, 303]))],
            )
            random_advantage = float(random_value) - float(combined_value)
        objective = combined_value + correction_rate_penalty * baseline * combined_rate - random_weight * random_advantage
        record = {
            "objective": objective,
            "combined_value": combined_value,
            "combined_rate": combined_rate,
            "random_value": random_value,
            "random_advantage": random_advantage,
            "thresholds": thresholds,
            "policy": policy,
        }
        if best is None or record["objective"] < best["objective"]:
            best = record

    assert best is not None
    policy = dict(best["policy"])
    policy["combined_val_objective"] = float(best["combined_value"])
    policy["combined_val_delta_pct"] = _pct_delta(float(best["combined_value"]), baseline)
    policy["combined_val_correction_rate"] = float(best["combined_rate"])
    policy["freq_score_mode"] = freq_score_mode
    policy["random_separation_val_objective"] = best.get("random_value", "")
    policy["random_separation_val_advantage"] = best.get("random_advantage", 0.0)
    policy["search_space"] = {
        "quantiles": quantiles,
        "margin_factors": margins,
        "lambda_trend_values": trend_lambdas,
        "lambda_freq_values": freq_lambdas,
    }
    return best["thresholds"], policy


def _float_list(values: Iterable[float]) -> List[float]:
    return [float(v) for v in values]


def _best_component_policy(
    samples: List[dict],
    thresholds: Thresholds,
    component: str,
    margins: List[float],
    lambdas: List[float],
    config: Dict,
    objective_metric: str,
    baseline: float,
    require_positive_gain: bool,
    min_relative_gain_pct: float,
) -> Dict[str, float]:
    best = None
    for margin_factor in margins:
        for strength in lambdas:
            policy = {
                "enabled": True,
                "trend_margin_factor": margin_factor if component == "trend" else 10.0,
                "freq_margin_factor": margin_factor if component == "freq" else 10.0,
                "lambda_trend": strength if component == "trend" else 0.0,
                "lambda_freq": strength if component == "freq" else 0.0,
                "trend_threshold": thresholds.trend * (1.0 + (margin_factor if component == "trend" else 10.0)),
                "freq_threshold": thresholds.freq * (1.0 + (margin_factor if component == "freq" else 10.0)),
            }
            value, rate = _policy_objective(
                samples,
                thresholds,
                policy,
                config,
                objective_metric=objective_metric,
                variant="trend_only" if component == "trend" else "frequency_only",
            )
            objective = value + 1e-7 * rate
            record = {
                "objective": objective,
                "value": value,
                "rate": rate,
                "margin_factor": float(margin_factor),
                "lambda": float(strength),
                "delta_pct": _pct_delta(float(value), baseline),
            }
            if best is None or record["objective"] < best["objective"]:
                best = record
    assert best is not None
    if require_positive_gain and best["delta_pct"] >= -abs(min_relative_gain_pct):
        return {"margin_factor": 10.0, "lambda": 0.0, "delta_pct": 0.0, "value": baseline, "rate": 0.0}
    return best


def _policy_objective(
    samples: List[dict],
    thresholds: Thresholds,
    policy: Dict[str, object],
    config: Dict,
    objective_metric: str,
    variant: str,
) -> Tuple[float, float]:
    predictions, _, infos = correct_samples(
        samples=samples,
        thresholds=thresholds,
        variant=variant,
        config=config,
        seed=int(config.get("seed", 7)) + 37,
        lambda_trend=float(policy.get("lambda_trend", config["correction"]["default_lambda_trend"])),
        lambda_freq=float(policy.get("lambda_freq", config["correction"]["default_lambda_freq"])),
        policy=policy,
    )
    values = [sample_error(sample, pred, objective_metric) for sample, pred in zip(samples, predictions)]
    changed = [info["changed"] for info in infos]
    return float(np.mean(values)), float(np.mean(changed)) if changed else 0.0


def _random_policy_objective(
    samples: List[dict],
    thresholds: Thresholds,
    policy: Dict[str, object],
    config: Dict,
    objective_metric: str,
    seeds: List[int],
) -> float:
    values = []
    for seed in seeds:
        predictions, _, _ = correct_samples(
            samples=samples,
            thresholds=thresholds,
            variant="random_trigger",
            config=config,
            seed=seed,
            lambda_trend=float(policy.get("lambda_trend", config["correction"]["default_lambda_trend"])),
            lambda_freq=float(policy.get("lambda_freq", config["correction"]["default_lambda_freq"])),
            policy=policy,
        )
        values.append(float(np.mean([sample_error(sample, pred, objective_metric) for sample, pred in zip(samples, predictions)])))
    return float(np.mean(values)) if values else 0.0


def policy_trigger_flags(score: Dict[str, float], thresholds: Thresholds, policy: Optional[Dict[str, object]]) -> Tuple[bool, bool]:
    if policy and bool(policy.get("enabled", False)):
        trend_threshold = float(policy.get("trend_threshold", thresholds.trend))
        freq_threshold = float(policy.get("freq_threshold", thresholds.freq))
        return bool(score["trend_score"] > trend_threshold), bool(score["freq_score"] > freq_threshold)
    return trigger_flags(score, thresholds)


def correct_samples(
    samples: List[dict],
    thresholds: Thresholds,
    variant: str,
    config: Dict,
    seed: int,
    lambda_trend: float,
    lambda_freq: float,
    policy: Optional[Dict[str, object]] = None,
) -> Tuple[List[np.ndarray], List[float], List[Dict[str, float]]]:
    corr_cfg = config["correction"]
    high_freq_cutoff_ratio = float(corr_cfg.get("high_freq_cutoff_ratio", 0.5))
    freq_score_mode = str(corr_cfg.get("freq_score_mode", "excess_plus_spectral"))
    smoothing_window = int(corr_cfg.get("smoothing_window", 5))
    turning_point_guard = bool(corr_cfg.get("turning_point_guard", True))
    turning_point_guard_min_score = float(corr_cfg.get("turning_point_guard_min_score", 0.25))
    max_trend_adjustment_ratio = corr_cfg.get("max_trend_adjustment_ratio", 0.08)
    if max_trend_adjustment_ratio is not None:
        max_trend_adjustment_ratio = float(max_trend_adjustment_ratio)

    effective_lambda_trend = float(policy.get("lambda_trend", lambda_trend)) if policy else float(lambda_trend)
    effective_lambda_freq = float(policy.get("lambda_freq", lambda_freq)) if policy else float(lambda_freq)
    random_trend, random_freq = _random_masks_like_rule(
        samples,
        thresholds,
        high_freq_cutoff_ratio,
        seed,
        policy=policy,
        freq_score_mode=freq_score_mode,
    )
    corrected_predictions = []
    latencies_ms = []
    infos = []

    for idx, sample in enumerate(samples):
        score = score_sample(sample["context"], sample["prediction"], high_freq_cutoff_ratio, freq_score_mode=freq_score_mode)
        policy_trend, policy_freq = policy_trigger_flags(score, thresholds, policy)
        if variant == "random_trigger":
            forced_trend = random_trend[idx]
            forced_freq = random_freq[idx]
        else:
            forced_trend = policy_trend
            forced_freq = policy_freq
        start = time.perf_counter()
        result = apply_correction(
            sample["context"],
            sample["prediction"],
            thresholds,
            variant=variant,
            lambda_trend=effective_lambda_trend,
            lambda_freq=effective_lambda_freq,
            high_freq_cutoff_ratio=high_freq_cutoff_ratio,
            smoothing_window=smoothing_window,
            turning_point_guard=turning_point_guard,
            turning_point_guard_min_score=turning_point_guard_min_score,
            max_trend_adjustment_ratio=max_trend_adjustment_ratio,
            forced_trend_trigger=forced_trend,
            forced_freq_trigger=forced_freq,
            freq_score_mode=freq_score_mode,
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        result.info["policy_rule_trend_trigger"] = float(policy_trend)
        result.info["policy_rule_freq_trigger"] = float(policy_freq)
        result.info["policy_rule_hallucination"] = float(policy_trend or policy_freq)
        corrected_predictions.append(result.prediction)
        infos.append(result.info)
    return corrected_predictions, latencies_ms, infos


def evaluate_variant(
    samples: List[dict],
    thresholds: Thresholds,
    variant: str,
    config: Dict,
    seed: int,
    lambda_trend: float,
    lambda_freq: float,
    policy: Optional[Dict[str, object]] = None,
) -> Dict:
    high_freq_cutoff_ratio = float(config["correction"].get("high_freq_cutoff_ratio", 0.5))
    freq_score_mode = str(config["correction"].get("freq_score_mode", "excess_plus_spectral"))
    corrected_predictions, latencies_ms, infos = correct_samples(
        samples=samples,
        thresholds=thresholds,
        variant=variant,
        config=config,
        seed=seed,
        lambda_trend=lambda_trend,
        lambda_freq=lambda_freq,
        policy=policy,
    )

    metrics = aggregate_metrics(
        samples=samples,
        corrected_predictions=corrected_predictions,
        thresholds=thresholds,
        high_freq_cutoff_ratio=high_freq_cutoff_ratio,
        latencies_ms=latencies_ms,
        variant=variant,
        lambda_trend=lambda_trend,
        lambda_freq=lambda_freq,
        freq_score_mode=freq_score_mode,
    )
    metrics["pre_correction_rule_trend_rate"] = float(np.mean([i["policy_rule_trend_trigger"] for i in infos])) if infos else 0.0
    metrics["pre_correction_rule_freq_rate"] = float(np.mean([i["policy_rule_freq_trigger"] for i in infos])) if infos else 0.0
    metrics["pre_correction_rule_hallucination_rate"] = (
        float(np.mean([bool(i["policy_rule_hallucination"]) for i in infos])) if infos else 0.0
    )
    metrics["slices"] = stress_slice_metrics(samples, corrected_predictions)
    if policy and bool(policy.get("enabled", False)):
        metrics["policy"] = {
            "enabled": True,
            "selected_quantile": policy.get("selected_quantile"),
            "trend_margin_factor": policy.get("trend_margin_factor"),
            "freq_margin_factor": policy.get("freq_margin_factor"),
            "lambda_trend": policy.get("lambda_trend"),
            "lambda_freq": policy.get("lambda_freq"),
            "combined_val_delta_pct": policy.get("combined_val_delta_pct"),
            "freq_score_mode": policy.get("freq_score_mode"),
            "random_separation_val_advantage": policy.get("random_separation_val_advantage"),
        }
    return metrics


def flatten_metric_row(table: str, metric: Dict) -> Dict[str, object]:
    keys = [
        "variant",
        "n_samples",
        "threshold_quantile",
        "lambda_trend",
        "lambda_freq",
        "mse",
        "mae",
        "mse_delta_pct_vs_original",
        "mae_delta_pct_vs_original",
        "hallucination_rate",
        "trend_violation_rate",
        "freq_violation_rate",
        "spectral_consistency",
        "turning_point_false_correction_rate",
        "turning_point_correction_rate",
        "correction_rate",
        "inference_latency_ms",
        "pre_correction_rule_hallucination_rate",
    ]
    row = {"table": table}
    for key in keys:
        row[key] = metric.get(key, "")
    return row


def write_metrics_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pct_delta(value: float, baseline: float) -> float:
    if abs(float(baseline)) <= 1e-12:
        return 0.0
    return 100.0 * (float(value) - float(baseline)) / float(baseline)


def build_diagnostics(
    calibration_samples: List[dict],
    evaluation_samples: List[dict],
    thresholds: Thresholds,
    policy: Dict[str, object],
    config: Dict,
    metrics: List[Dict],
    seed: int,
) -> Dict[str, List[Dict[str, object]]]:
    lambda_trend = float(policy.get("lambda_trend", config["correction"]["default_lambda_trend"])) if policy else float(config["correction"]["default_lambda_trend"])
    lambda_freq = float(policy.get("lambda_freq", config["correction"]["default_lambda_freq"])) if policy else float(config["correction"]["default_lambda_freq"])
    trend_val, _, trend_infos = correct_samples(
        calibration_samples, thresholds, "trend_only", config, seed, lambda_trend, lambda_freq, policy
    )
    freq_val, _, freq_infos = correct_samples(
        calibration_samples, thresholds, "frequency_only", config, seed, lambda_trend, lambda_freq, policy
    )
    full_val, _, full_val_infos = correct_samples(
        calibration_samples, thresholds, "trend_frequency", config, seed, lambda_trend, lambda_freq, policy
    )
    full_test, _, full_infos = correct_samples(
        evaluation_samples, thresholds, "trend_frequency", config, seed, lambda_trend, lambda_freq, policy
    )
    naive_test, _, naive_infos = correct_samples(
        evaluation_samples, thresholds, "naive_smoothing", config, seed, lambda_trend, lambda_freq, policy
    )
    random_rows = rule_vs_random_paired_rows(evaluation_samples, thresholds, policy, config, seed, full_test)
    return {
        "trigger_precision_proxy": trigger_precision_proxy_rows(
            calibration_samples,
            [
                ("trend", trend_val, trend_infos),
                ("freq", freq_val, freq_infos),
                ("combined", full_val, full_val_infos),
            ],
        ),
        "error_conditioned_analysis": error_conditioned_rows(
            evaluation_samples,
            full_test,
            naive_test,
            full_infos,
            naive_infos,
        ),
        "score_bin_table": score_bin_rows(evaluation_samples, full_test, full_infos, config),
        "rule_vs_random_paired": random_rows,
        "naive_smoothing_comparison": naive_smoothing_comparison_rows(metrics),
    }


def trigger_precision_proxy_rows(
    samples: List[dict],
    component_predictions: List[Tuple[str, List[np.ndarray], List[Dict[str, float]]]],
) -> List[Dict[str, object]]:
    rows = []
    base_errors = np.asarray([sample_error(s, s["prediction"], "mse") for s in samples], dtype=np.float64)
    for component, predictions, infos in component_predictions:
        if component == "trend":
            trigger = np.asarray([bool(info["policy_rule_trend_trigger"]) for info in infos], dtype=bool)
            scores = np.asarray([info["trend_score"] for info in infos], dtype=np.float64)
        elif component == "freq":
            trigger = np.asarray([bool(info["policy_rule_freq_trigger"]) for info in infos], dtype=bool)
            scores = np.asarray([info["freq_score"] for info in infos], dtype=np.float64)
        else:
            trigger = np.asarray([bool(info["policy_rule_hallucination"]) for info in infos], dtype=bool)
            scores = np.asarray([max(info["trend_score"], info["freq_score"]) for info in infos], dtype=np.float64)
        corrected_errors = np.asarray([sample_error(s, p, "mse") for s, p in zip(samples, predictions)], dtype=np.float64)
        idx = trigger if trigger.any() else np.zeros_like(trigger, dtype=bool)
        rows.append(
            {
                "component": component,
                "n_samples": len(samples),
                "trigger_rate": float(trigger.mean()) if trigger.size else 0.0,
                "score_p50": float(np.quantile(scores, 0.50)) if scores.size else 0.0,
                "score_p90": float(np.quantile(scores, 0.90)) if scores.size else 0.0,
                "score_p95": float(np.quantile(scores, 0.95)) if scores.size else 0.0,
                "score_p99": float(np.quantile(scores, 0.99)) if scores.size else 0.0,
                "triggered_baseline_mse": float(base_errors[idx].mean()) if idx.any() else "",
                "triggered_corrected_mse": float(corrected_errors[idx].mean()) if idx.any() else "",
                "triggered_mse_delta_pct": _pct_delta(float(corrected_errors[idx].mean()), float(base_errors[idx].mean())) if idx.any() else "",
                "untriggered_baseline_mse": float(base_errors[~idx].mean()) if (~idx).any() else "",
            }
        )
    return rows


def error_conditioned_rows(
    samples: List[dict],
    full_predictions: List[np.ndarray],
    naive_predictions: List[np.ndarray],
    full_infos: List[Dict[str, float]],
    naive_infos: List[Dict[str, float]],
) -> List[Dict[str, object]]:
    base_errors = np.asarray([sample_error(s, s["prediction"], "mse") for s in samples], dtype=np.float64)
    full_errors = np.asarray([sample_error(s, p, "mse") for s, p in zip(samples, full_predictions)], dtype=np.float64)
    naive_errors = np.asarray([sample_error(s, p, "mse") for s, p in zip(samples, naive_predictions)], dtype=np.float64)
    full_changed = np.asarray([bool(info["changed"]) for info in full_infos], dtype=bool)
    naive_changed = np.asarray([bool(info["changed"]) for info in naive_infos], dtype=bool)
    return _quantile_bin_rows(
        base_errors,
        [
            ("trend_frequency", full_errors, full_changed),
            ("naive_smoothing", naive_errors, naive_changed),
        ],
        score_name="no_correction_error",
        bin_edges=[0.0, 0.25, 0.50, 0.75, 0.90, 1.0],
    )


def score_bin_rows(
    samples: List[dict],
    full_predictions: List[np.ndarray],
    full_infos: List[Dict[str, float]],
    config: Dict,
) -> List[Dict[str, object]]:
    high_freq_cutoff_ratio = float(config["correction"].get("high_freq_cutoff_ratio", 0.5))
    freq_score_mode = str(config["correction"].get("freq_score_mode", "excess_plus_spectral"))
    base_errors = np.asarray([sample_error(s, s["prediction"], "mse") for s in samples], dtype=np.float64)
    full_errors = np.asarray([sample_error(s, p, "mse") for s, p in zip(samples, full_predictions)], dtype=np.float64)
    changed = np.asarray([bool(info["changed"]) for info in full_infos], dtype=bool)
    scores = [score_sample(s["context"], s["prediction"], high_freq_cutoff_ratio, freq_score_mode=freq_score_mode) for s in samples]
    rows = []
    for score_name in ["trend_score", "freq_score"]:
        score_values = np.asarray([s[score_name] for s in scores], dtype=np.float64)
        rows.extend(
            _quantile_bin_rows(
                score_values,
                [("trend_frequency", full_errors, changed)],
                score_name=score_name,
                bin_edges=[0.0, 0.50, 0.80, 0.90, 0.95, 1.0],
                baseline_errors=base_errors,
            )
        )
    return rows


def _quantile_bin_rows(
    driver: np.ndarray,
    variant_errors: List[Tuple[str, np.ndarray, np.ndarray]],
    score_name: str,
    bin_edges: List[float],
    baseline_errors: Optional[np.ndarray] = None,
) -> List[Dict[str, object]]:
    if baseline_errors is None:
        baseline_errors = driver
    quantiles = np.quantile(driver, bin_edges)
    rows = []
    n = driver.size
    for idx in range(len(bin_edges) - 1):
        lo_q = bin_edges[idx]
        hi_q = bin_edges[idx + 1]
        lo = quantiles[idx]
        hi = quantiles[idx + 1]
        if idx == len(bin_edges) - 2:
            mask = (driver >= lo) & (driver <= hi)
        else:
            mask = (driver >= lo) & (driver < hi)
        for variant, errors, changed in variant_errors:
            rows.append(
                {
                    "score_name": score_name,
                    "bin": f"{lo_q:.2f}-{hi_q:.2f}",
                    "variant": variant,
                    "n_samples": int(mask.sum()),
                    "baseline_mse": float(baseline_errors[mask].mean()) if mask.any() else "",
                    "corrected_mse": float(errors[mask].mean()) if mask.any() else "",
                    "mse_delta_pct": _pct_delta(float(errors[mask].mean()), float(baseline_errors[mask].mean())) if mask.any() else "",
                    "correction_rate": float(changed[mask].mean()) if mask.any() else "",
                    "score_min": float(driver[mask].min()) if mask.any() else "",
                    "score_max": float(driver[mask].max()) if mask.any() else "",
                    "sample_fraction": float(mask.sum() / max(n, 1)),
                }
            )
    return rows


def rule_vs_random_paired_rows(
    samples: List[dict],
    thresholds: Thresholds,
    policy: Dict[str, object],
    config: Dict,
    seed: int,
    rule_predictions: List[np.ndarray],
) -> List[Dict[str, object]]:
    policy_cfg = (config["correction"].get("validation_calibrated_policy", {}) or {})
    random_seeds = [int(s) for s in policy_cfg.get("random_seeds", [seed + 101, seed + 202, seed + 303, seed + 404, seed + 505])]
    lambda_trend = float(policy.get("lambda_trend", config["correction"]["default_lambda_trend"])) if policy else float(config["correction"]["default_lambda_trend"])
    lambda_freq = float(policy.get("lambda_freq", config["correction"]["default_lambda_freq"])) if policy else float(config["correction"]["default_lambda_freq"])
    rule_mse = float(np.mean([sample_error(s, p, "mse") for s, p in zip(samples, rule_predictions)]))
    rule_mae = float(np.mean([sample_error(s, p, "mae") for s, p in zip(samples, rule_predictions)]))
    rows = []
    for random_seed in random_seeds:
        random_predictions, _, random_infos = correct_samples(
            samples=samples,
            thresholds=thresholds,
            variant="random_trigger",
            config=config,
            seed=random_seed,
            lambda_trend=lambda_trend,
            lambda_freq=lambda_freq,
            policy=policy,
        )
        random_mse = float(np.mean([sample_error(s, p, "mse") for s, p in zip(samples, random_predictions)]))
        random_mae = float(np.mean([sample_error(s, p, "mae") for s, p in zip(samples, random_predictions)]))
        rows.append(
            {
                "random_seed": random_seed,
                "rule_mse": rule_mse,
                "random_mse": random_mse,
                "rule_minus_random_mse": rule_mse - random_mse,
                "rule_beats_random_mse": bool(rule_mse < random_mse),
                "rule_mae": rule_mae,
                "random_mae": random_mae,
                "rule_minus_random_mae": rule_mae - random_mae,
                "rule_beats_random_mae": bool(rule_mae < random_mae),
                "random_correction_rate": float(np.mean([info["changed"] for info in random_infos])) if random_infos else 0.0,
            }
        )
    return rows


def naive_smoothing_comparison_rows(metrics: List[Dict]) -> List[Dict[str, object]]:
    by_variant = {m["variant"]: m for m in metrics}
    no = by_variant.get("no_correction", {})
    full = by_variant.get("trend_frequency", {})
    naive = by_variant.get("naive_smoothing", {})
    return [
        {
            "no_correction_mse": no.get("mse", ""),
            "trend_frequency_mse": full.get("mse", ""),
            "naive_smoothing_mse": naive.get("mse", ""),
            "trend_frequency_delta_pct": full.get("mse_delta_pct_vs_original", ""),
            "naive_smoothing_delta_pct": naive.get("mse_delta_pct_vs_original", ""),
            "mse_gap_vs_naive_pct_points": (
                float(full.get("mse_delta_pct_vs_original", 0.0)) - float(naive.get("mse_delta_pct_vs_original", 0.0))
                if full and naive
                else ""
            ),
            "trend_frequency_correction_rate": full.get("correction_rate", ""),
            "naive_smoothing_correction_rate": naive.get("correction_rate", ""),
        }
    ]


def write_diagnostics_csv(diagnostics: Dict[str, List[Dict[str, object]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in diagnostics.items():
        write_metrics_csv(rows, output_dir / f"{name}.csv")


def write_external_outputs(payload: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = payload["main_ablation"]
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_metrics_csv([flatten_metric_row("main_ablation", m) for m in metrics], output_dir / "metrics.csv")
    (output_dir / "ablation_table.md").write_text(_markdown_table(metrics), encoding="utf-8")
    (output_dir / "summary.md").write_text(_summary_markdown(payload), encoding="utf-8")
    if "diagnostics" in payload:
        write_diagnostics_csv(payload["diagnostics"], output_dir / "diagnostics")


def _random_masks_like_rule(
    samples: List[dict],
    thresholds: Thresholds,
    high_freq_cutoff_ratio: float,
    seed: int,
    policy: Optional[Dict[str, object]] = None,
    freq_score_mode: str = "excess_plus_spectral",
) -> Tuple[List[bool], List[bool]]:
    n = len(samples)
    rng = np.random.default_rng(seed)
    rule_trend = []
    rule_freq = []
    for sample in samples:
        score = score_sample(sample["context"], sample["prediction"], high_freq_cutoff_ratio, freq_score_mode=freq_score_mode)
        trend, freq = policy_trigger_flags(score, thresholds, policy)
        rule_trend.append(trend)
        rule_freq.append(freq)
    trend_count = int(np.sum(rule_trend))
    freq_count = int(np.sum(rule_freq))
    trend_mask = np.zeros(n, dtype=bool)
    freq_mask = np.zeros(n, dtype=bool)
    if trend_count > 0:
        trend_mask[rng.choice(n, size=trend_count, replace=False)] = True
    if freq_count > 0:
        freq_mask[rng.choice(n, size=freq_count, replace=False)] = True
    return trend_mask.tolist(), freq_mask.tolist()


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HalluGuard corrections on JSONL prediction samples.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    samples = load_prediction_file(args.input)
    calibration = split_samples(samples, args.calibration_split)
    evaluation = split_samples(samples, args.split)
    if not calibration:
        raise ValueError(f"No samples found for calibration split {args.calibration_split!r}.")
    if not evaluation:
        raise ValueError(f"No samples found for evaluation split {args.split!r}.")
    thresholds, policy = calibrate_evaluation_policy(
        calibration_samples=calibration,
        config=config,
        source_split=args.calibration_split,
    )
    lambda_trend = float(policy.get("lambda_trend", config["correction"]["default_lambda_trend"])) if policy else float(config["correction"]["default_lambda_trend"])
    lambda_freq = float(policy.get("lambda_freq", config["correction"]["default_lambda_freq"])) if policy else float(config["correction"]["default_lambda_freq"])
    metrics = []
    for variant in config["variants"]:
        metrics.append(
            evaluate_variant(
                evaluation,
                thresholds,
                variant=variant,
                config=config,
                seed=int(config.get("seed", 7)) + 991,
                lambda_trend=lambda_trend,
                lambda_freq=lambda_freq,
                policy=policy,
            )
        )

    payload = {
        "run_id": "external_prediction_evaluation",
        "input_path": str(args.input),
        "threshold_source_split": args.calibration_split,
        "evaluation_split": args.split,
        "test_threshold_leakage": False,
        "thresholds": thresholds.to_dict(),
        "validation_policy": policy,
        "main_ablation": metrics,
    }
    if bool((config["correction"].get("validation_calibrated_policy", {}) or {}).get("write_diagnostics", False)):
        payload["diagnostics"] = build_diagnostics(
            calibration_samples=calibration,
            evaluation_samples=evaluation,
            thresholds=thresholds,
            policy=policy,
            config=config,
            metrics=metrics,
            seed=int(config.get("seed", 7)) + 991,
        )
    if args.output_dir:
        write_external_outputs(payload, args.output_dir)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.output_csv:
        write_metrics_csv([flatten_metric_row("main_ablation", m) for m in metrics], args.output_csv)
    if not args.output_dir and not args.output_json and not args.output_csv:
        print(json.dumps(payload, indent=2))


def _markdown_table(metrics: List[Dict]) -> str:
    headers = [
        "variant",
        "MSE",
        "MAE",
        "HallucinationRate",
        "TrendViolationRate",
        "FreqViolationRate",
        "SpectralConsistency",
        "TPFalseCorrection",
        "LatencyMs",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for metric in metrics:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(metric["variant"]),
                    f"{metric['mse']:.6f}",
                    f"{metric['mae']:.6f}",
                    f"{metric['hallucination_rate']:.3f}",
                    f"{metric['trend_violation_rate']:.3f}",
                    f"{metric['freq_violation_rate']:.3f}",
                    f"{metric['spectral_consistency']:.3f}",
                    f"{metric['turning_point_false_correction_rate']:.3f}",
                    f"{metric['inference_latency_ms']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _summary_markdown(payload: Dict) -> str:
    by_variant = {m["variant"]: m for m in payload["main_ablation"]}
    no = by_variant.get("no_correction", {})
    full = by_variant.get("trend_frequency", {})
    random = by_variant.get("random_trigger", {})
    return f"""# External Prediction Evaluation Summary

## Run

- Input: `{payload['input_path']}`
- Threshold source split: `{payload['threshold_source_split']}`
- Evaluation split: `{payload['evaluation_split']}`
- Test threshold leakage: `{payload['test_threshold_leakage']}`

## Headline

- no_correction MSE: {no.get('mse', 0.0):.6f}
- trend_frequency MSE: {full.get('mse', 0.0):.6f}
- random_trigger MSE: {random.get('mse', 0.0):.6f}
- trend_frequency HallucinationRate: {full.get('hallucination_rate', 0.0):.3f}

## Ablation

{_markdown_table(payload['main_ablation'])}
"""


if __name__ == "__main__":
    _cli()
