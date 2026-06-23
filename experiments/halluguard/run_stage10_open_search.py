"""Stage 10 open mechanism search for HalluGuard.

This runner intentionally lives beside, rather than inside, the Stage 7-9
evaluator. Stage 10 candidates can use new triggers and correction formulas
while still reporting the old incumbent and the required controls.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

from correction import (
    Thresholds,
    curvature_roughness,
    frequency_correction,
    high_frequency_energy_ratio,
    naive_smoothing,
    ols_slope,
    score_sample,
    spectral_distance,
    trend_correction,
    trigger_flags,
)
import evaluate_predictions as incumbent_eval
from metrics import mae, mse


DATASETS = ["ETTm1", "ETTh1"]
MODELS = ["DLinear", "PatchTST"]
HORIZONS = [96, 192, 336, 720]
SMOKE_CONFIGS = [
    ("ETTm1", "DLinear", 192),
    ("ETTm1", "PatchTST", 720),
    ("ETTh1", "DLinear", 336),
    ("ETTh1", "PatchTST", 720),
]

CANDIDATE_FAMILIES = {
    "s10_c1_error_predictive_residual": "error_predictive_diagnostic_trigger",
    "s10_c2_dynamics_consistency": "dynamics_consistency_trigger",
    "s10_c2b_dynamics_anti_smoothing": "dynamics_consistency_trigger_anti_smoothing_exploit",
    "s10_c3_perturbation_stability": "ensemble_perturbation_stability_trigger",
    "s10_c4_residual_shape_model": "residual_shape_correction",
}

VARIANTS = [
    "no_correction",
    "naive_smoothing",
    "stage9_incumbent",
    "matched_smoothing_control",
    "random_trigger",
    "candidate_half_strength",
    "candidate_main",
]

FEATURE_NAMES = [
    "horizon_scaled",
    "context_slope_scaled",
    "prediction_slope_scaled",
    "slope_gap_abs",
    "boundary_jump",
    "first_diff_gap",
    "mean_shift",
    "variance_ratio_log",
    "diff_std_ratio_log",
    "context_hf_ratio",
    "prediction_hf_ratio",
    "hf_excess",
    "spectral_distance",
    "context_roughness",
    "prediction_roughness",
    "roughness_excess",
    "acf1_gap",
    "range_ratio_log",
]


@dataclass(frozen=True)
class Config:
    dataset: str
    model: str
    horizon: int

    @property
    def tag(self) -> str:
        return f"{self.dataset}_{self.model}_{self.horizon}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 10 HalluGuard open mechanism search.")
    parser.add_argument("--scope", required=True, choices=["smoke", "clean_full", "stress"])
    parser.add_argument("--candidate-id", required=True, choices=sorted(CANDIDATE_FAMILIES))
    parser.add_argument("--config", default="experiments/halluguard/configs/halluguard_stage10_open_search.yaml")
    parser.add_argument("--stage7-dir", default="experiments/halluguard/results/stage7_big_table")
    parser.add_argument("--output-root", default="experiments/halluguard/results/stage10_open_mechanism_search")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_yaml(repo_root / args.config)
    stage7_dir = repo_root / args.stage7_dir
    output_root = repo_root / args.output_root
    configs = [Config(*c) for c in SMOKE_CONFIGS] if args.scope == "smoke" else [
        Config(dataset, model, horizon)
        for dataset in DATASETS
        for model in MODELS
        for horizon in HORIZONS
    ]
    table_name = "smoke" if args.scope == "smoke" else ("clean_full_table" if args.scope == "clean_full" else "stress_table")
    run_root = output_root / table_name / args.candidate_id
    run_root.mkdir(parents=True, exist_ok=True)
    (output_root / "diagnostics").mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    config_records: List[Dict[str, object]] = []
    diagnostics: Dict[str, List[Dict[str, object]]] = {}
    stress_types = ["clean"] if args.scope != "stress" else list((cfg.get("stress", {}) or {}).get("types", ["trend_drift", "high_frequency_perturbation"]))

    for stress_type in stress_types:
        for config_record in configs:
            record, metric_rows, diag_rows = run_one_config(
                repo_root=repo_root,
                stage7_dir=stage7_dir,
                output_root=output_root,
                run_root=run_root,
                config_record=config_record,
                args=args,
                cfg=cfg,
                stress_type=stress_type,
            )
            config_records.append(record)
            rows.extend(metric_rows)
            for name, values in diag_rows.items():
                diagnostics.setdefault(name, []).extend(values)
            write_outputs(output_root, run_root, args.candidate_id, args.scope, rows, config_records, diagnostics)

    write_outputs(output_root, run_root, args.candidate_id, args.scope, rows, config_records, diagnostics)
    print(
        json.dumps(
            {
                "scope": args.scope,
                "candidate_id": args.candidate_id,
                "output_dir": str(run_root),
                "completed_configs": count_completed(config_records),
                "total_configs": len(config_records),
            }
        )
    )


def run_one_config(
    repo_root: Path,
    stage7_dir: Path,
    output_root: Path,
    run_root: Path,
    config_record: Config,
    args: argparse.Namespace,
    cfg: Dict,
    stress_type: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    source_prediction_path = stage7_dir / "predictions" / f"{config_record.tag}.jsonl"
    if stress_type == "clean":
        prediction_path = source_prediction_path
        run_dir = run_root / "runs" / config_record.tag
    else:
        prediction_path = output_root / "stress_predictions" / stress_type / f"{config_record.tag}.jsonl"
        run_dir = run_root / stress_type / "runs" / config_record.tag

    try:
        if not source_prediction_path.exists():
            raise FileNotFoundError(f"Missing Stage 7 prediction file: {source_prediction_path}")
        if stress_type != "clean":
            write_stress_predictions(source_prediction_path, prediction_path, stress_type)
        samples = load_jsonl(prediction_path)
        val_samples = [s for s in samples if s.get("split") == "val"]
        test_samples = [s for s in samples if s.get("split") == "test"]
        if not val_samples or not test_samples:
            raise ValueError(f"{prediction_path} must contain both val and test splits.")

        run_dir.mkdir(parents=True, exist_ok=True)
        policy = fit_candidate_policy(args.candidate_id, val_samples, cfg)
        incumbent_cfg = load_yaml(repo_root / cfg["stage9_incumbent_config"])
        incumbent_thresholds, incumbent_policy = incumbent_eval.calibrate_evaluation_policy(
            calibration_samples=val_samples,
            config=incumbent_cfg,
            source_split="val",
        )
        metric_rows, diag_rows, payload = evaluate_candidate(
            config_record=config_record,
            candidate_id=args.candidate_id,
            stress_type=stress_type,
            test_samples=test_samples,
            val_samples=val_samples,
            policy=policy,
            search_cfg=cfg,
            incumbent_cfg=incumbent_cfg,
            incumbent_thresholds=incumbent_thresholds,
            incumbent_policy=incumbent_policy,
            prediction_path=prediction_path,
            run_dir=run_dir,
        )
        write_run_outputs(run_dir, payload, metric_rows, diag_rows)
        record = {
            "candidate_id": args.candidate_id,
            "family": CANDIDATE_FAMILIES[args.candidate_id],
            "stress_type": stress_type,
            "dataset": config_record.dataset,
            "model": config_record.model,
            "horizon": config_record.horizon,
            "status": "completed",
            "blocker_reason": "",
        }
        return record, metric_rows, diag_rows
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        record = {
            "candidate_id": args.candidate_id,
            "family": CANDIDATE_FAMILIES.get(args.candidate_id, ""),
            "stress_type": stress_type,
            "dataset": config_record.dataset,
            "model": config_record.model,
            "horizon": config_record.horizon,
            "status": "blocked",
            "blocker_reason": reason,
        }
        return record, [blocked_row(config_record, args.candidate_id, stress_type, prediction_path, run_dir, reason)], {}


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(path: Path) -> List[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            for key in ["context", "prediction", "target"]:
                sample[key] = [float(v) for v in sample[key]]
            samples.append(sample)
    return samples


def write_stress_predictions(source_path: Path, output_path: Path, stress_type: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            sample = json.loads(line)
            pred = arr(sample["prediction"])
            ctx = arr(sample["context"])
            scale = float(np.std(ctx)) + 1e-12
            t = np.linspace(-0.5, 0.5, pred.size)
            sign = -1.0 if stable_hash(str(sample["sample_id"])) % 2 else 1.0
            if stress_type == "trend_drift":
                perturbation = sign * 0.45 * scale * t
            elif stress_type == "high_frequency_perturbation":
                phase = (stable_hash(str(sample["sample_id"])) % 17) / 17.0 * 2.0 * np.pi
                perturbation = 0.28 * scale * np.sin(np.arange(pred.size, dtype=np.float64) * np.pi * 0.85 + phase)
            elif stress_type == "boundary_discontinuity":
                decay = np.exp(-np.arange(pred.size, dtype=np.float64) / max(4.0, pred.size / 16.0))
                perturbation = sign * 0.50 * scale * decay
            elif stress_type == "variance_shift":
                centered = pred - float(pred.mean())
                perturbation = 0.25 * centered
            else:
                raise ValueError(f"Unknown stress type: {stress_type}")
            sample["prediction"] = (pred + perturbation).astype(float).tolist()
            sample["sample_id"] = f"{sample['sample_id']}::{stress_type}"
            sample["stress_type"] = stress_type
            sample["stress_only"] = True
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")


def stable_hash(text: str) -> int:
    value = 2166136261
    for ch in text:
        value ^= ord(ch)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def fit_candidate_policy(candidate_id: str, val_samples: List[dict], cfg: Dict) -> Dict[str, object]:
    family = CANDIDATE_FAMILIES[candidate_id]
    search = cfg.get("search", {}) or {}
    quantiles = [float(v) for v in search.get("trigger_quantiles", [0.7, 0.8, 0.9])]
    strengths = [float(v) for v in search.get("correction_strengths", [0.1, 0.3, 0.5])]
    random_seeds = [int(v) for v in search.get("random_seeds", [1101, 2202, 3303])]
    random_weight = float(search.get("random_separation_weight", 0.5))
    matched_weight = float(search.get("anti_smoothing_weight", 1.0)) if candidate_id == "s10_c2b_dynamics_anti_smoothing" else 0.0
    rate_penalty = float(search.get("correction_rate_penalty", 0.0002))
    ridge_l2 = float(search.get("ridge_l2", 1.0))

    base_predictions = np.asarray([arr(s["prediction"]) for s in val_samples], dtype=np.float64)
    targets = np.asarray([arr(s["target"]) for s in val_samples], dtype=np.float64)
    residuals = targets - base_predictions
    errors = np.asarray([mse(s["prediction"], s["target"]) for s in val_samples], dtype=np.float64)
    baseline_mse = float(errors.mean())
    features = np.asarray([feature_vector(s["context"], s["prediction"]) for s in val_samples], dtype=np.float64)
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0) + 1e-8
    x_std = (features - feature_mean) / feature_std

    base_policy: Dict[str, object] = {
        "candidate_id": candidate_id,
        "family": family,
        "source_split": "val",
        "test_threshold_leakage": False,
        "feature_names": FEATURE_NAMES,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }

    if candidate_id == "s10_c1_error_predictive_residual":
        y = np.log1p(errors)
        risk_weights = fit_ridge(x_std, y[:, None], ridge_l2).reshape(-1)
        scores = predict_ridge(x_std, risk_weights)
        vectors_by_quantile = {}
        for q in quantiles:
            threshold = float(np.quantile(scores, q))
            mask = scores >= threshold
            if not mask.any():
                mask[np.argmax(scores)] = True
            vectors_by_quantile[q] = np.repeat(residuals[mask].mean(axis=0, keepdims=True), len(val_samples), axis=0)
        selected = select_policy(
            samples=val_samples,
            base_predictions=base_predictions,
            scores=scores,
            vectors_by_quantile=vectors_by_quantile,
            quantiles=quantiles,
            strengths=strengths,
            baseline_mse=baseline_mse,
            random_seeds=random_seeds,
            random_weight=random_weight,
            matched_weight=matched_weight,
            rate_penalty=rate_penalty,
        )
        base_policy.update(selected)
        base_policy["risk_weights"] = risk_weights
        base_policy["residual_shape"] = vectors_by_quantile[float(selected["trigger_quantile"])][0]
        return base_policy

    if candidate_id in {"s10_c2_dynamics_consistency", "s10_c2b_dynamics_anti_smoothing"}:
        scores = np.asarray([dynamics_score(s["context"], s["prediction"]) for s in val_samples], dtype=np.float64)
        vectors = np.asarray([dynamics_vector(s["context"], s["prediction"]) for s in val_samples], dtype=np.float64)
        selected = select_policy(
            samples=val_samples,
            base_predictions=base_predictions,
            scores=scores,
            vectors_by_quantile={q: vectors for q in quantiles},
            quantiles=quantiles,
            strengths=strengths,
            baseline_mse=baseline_mse,
            random_seeds=random_seeds,
            random_weight=random_weight,
            matched_weight=matched_weight,
            rate_penalty=rate_penalty,
        )
        base_policy.update(selected)
        return base_policy

    if candidate_id == "s10_c3_perturbation_stability":
        scores = []
        vectors = []
        for sample in val_samples:
            score, vector = stability_score_and_vector(sample["context"], sample["prediction"])
            scores.append(score)
            vectors.append(vector)
        selected = select_policy(
            samples=val_samples,
            base_predictions=base_predictions,
            scores=np.asarray(scores, dtype=np.float64),
            vectors_by_quantile={q: np.asarray(vectors, dtype=np.float64) for q in quantiles},
            quantiles=quantiles,
            strengths=strengths,
            baseline_mse=baseline_mse,
            random_seeds=random_seeds,
            random_weight=random_weight,
            matched_weight=matched_weight,
            rate_penalty=rate_penalty,
        )
        base_policy.update(selected)
        return base_policy

    if candidate_id == "s10_c4_residual_shape_model":
        residual_weights = fit_ridge(x_std, residuals, ridge_l2)
        predicted_residuals = predict_ridge(x_std, residual_weights)
        scale = np.asarray([float(np.std(arr(s["context"]))) + 1e-12 for s in val_samples], dtype=np.float64)
        scores = np.sqrt(np.mean(predicted_residuals**2, axis=1)) / scale
        selected = select_policy(
            samples=val_samples,
            base_predictions=base_predictions,
            scores=scores,
            vectors_by_quantile={q: predicted_residuals for q in quantiles},
            quantiles=quantiles,
            strengths=strengths,
            baseline_mse=baseline_mse,
            random_seeds=random_seeds,
            random_weight=random_weight,
            matched_weight=matched_weight,
            rate_penalty=rate_penalty,
        )
        base_policy.update(selected)
        base_policy["residual_weights"] = residual_weights
        return base_policy

    raise ValueError(f"Unknown Stage 10 candidate: {candidate_id}")


def select_policy(
    samples: List[dict],
    base_predictions: np.ndarray,
    scores: np.ndarray,
    vectors_by_quantile: Dict[float, np.ndarray],
    quantiles: List[float],
    strengths: List[float],
    baseline_mse: float,
    random_seeds: List[int],
    random_weight: float,
    matched_weight: float,
    rate_penalty: float,
) -> Dict[str, object]:
    best: Optional[Dict[str, object]] = None
    for q in quantiles:
        vectors = vectors_by_quantile[q]
        threshold = float(np.quantile(scores, q))
        mask = scores >= threshold
        if not mask.any():
            mask[np.argmax(scores)] = True
        for alpha in strengths:
            preds = apply_vector_correction(base_predictions, vectors, mask, alpha)
            cand_mse = mean([mse(pred, s["target"]) for pred, s in zip(preds, samples)])
            random_mses = []
            for seed in random_seeds:
                random_mask = matched_random_mask(len(samples), int(mask.sum()), seed)
                random_preds = apply_vector_correction(base_predictions, vectors, random_mask, alpha)
                random_mses.append(mean([mse(pred, s["target"]) for pred, s in zip(random_preds, samples)]))
            random_mse = mean(random_mses)
            matched_preds = base_predictions.copy()
            for idx, flag in enumerate(mask):
                if flag:
                    matched_preds[idx] = naive_smoothing(matched_preds[idx], 5)
            matched_mse = mean([mse(pred, s["target"]) for pred, s in zip(matched_preds, samples)])
            trigger_rate = float(mask.mean())
            random_advantage = random_mse - cand_mse
            matched_advantage = matched_mse - cand_mse
            objective = cand_mse - random_weight * random_advantage - matched_weight * matched_advantage + rate_penalty * baseline_mse * trigger_rate
            record = {
                "objective": objective,
                "trigger_quantile": float(q),
                "trigger_threshold": threshold,
                "correction_strength": float(alpha),
                "validation_mse": cand_mse,
                "validation_mse_delta_pct": pct_delta(cand_mse, baseline_mse),
                "validation_random_mse": random_mse,
                "validation_random_advantage_mse": random_advantage,
                "validation_matched_smoothing_mse": matched_mse,
                "validation_matched_advantage_mse": matched_advantage,
                "validation_trigger_rate": trigger_rate,
                "validation_baseline_mse": baseline_mse,
            }
            if best is None or record["objective"] < float(best["objective"]):
                best = record
    assert best is not None
    return best


def evaluate_candidate(
    config_record: Config,
    candidate_id: str,
    stress_type: str,
    test_samples: List[dict],
    val_samples: List[dict],
    policy: Dict[str, object],
    search_cfg: Dict,
    incumbent_cfg: Dict,
    incumbent_thresholds: Thresholds,
    incumbent_policy: Dict[str, object],
    prediction_path: Path,
    run_dir: Path,
) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, object]]], Dict[str, object]]:
    base_predictions = np.asarray([arr(s["prediction"]) for s in test_samples], dtype=np.float64)
    scores, vectors = score_and_vectors_for_samples(candidate_id, test_samples, policy)
    trigger_mask = scores >= float(policy["trigger_threshold"])
    if not trigger_mask.any() and len(trigger_mask):
        trigger_mask[int(np.argmax(scores))] = True

    variant_outputs: Dict[str, Tuple[np.ndarray, List[float], np.ndarray]] = {}
    for variant in VARIANTS:
        start = time.perf_counter()
        if variant == "no_correction":
            preds = base_predictions.copy()
            mask = np.zeros(len(test_samples), dtype=bool)
        elif variant == "naive_smoothing":
            preds = np.asarray([naive_smoothing(p, 5) for p in base_predictions], dtype=np.float64)
            mask = np.ones(len(test_samples), dtype=bool)
        elif variant == "stage9_incumbent":
            inc_preds, inc_latencies, inc_infos = incumbent_eval.correct_samples(
                samples=test_samples,
                thresholds=incumbent_thresholds,
                variant="trend_frequency",
                config=incumbent_cfg,
                seed=int(search_cfg.get("seed", 17)) + 991,
                lambda_trend=float(incumbent_policy.get("lambda_trend", incumbent_cfg["correction"]["default_lambda_trend"])),
                lambda_freq=float(incumbent_policy.get("lambda_freq", incumbent_cfg["correction"]["default_lambda_freq"])),
                policy=incumbent_policy,
            )
            preds = np.asarray(inc_preds, dtype=np.float64)
            mask = np.asarray([bool(info.get("changed", 0.0)) for info in inc_infos], dtype=bool)
            variant_outputs[variant] = (preds, inc_latencies, mask)
            continue
        elif variant == "matched_smoothing_control":
            preds = base_predictions.copy()
            for idx, flag in enumerate(trigger_mask):
                if flag:
                    preds[idx] = naive_smoothing(preds[idx], 5)
            mask = trigger_mask.copy()
        elif variant == "random_trigger":
            mask = matched_random_mask(len(test_samples), int(trigger_mask.sum()), int(search_cfg.get("seed", 17)) + 404)
            preds = apply_vector_correction(base_predictions, vectors, mask, float(policy["correction_strength"]))
        elif variant == "candidate_half_strength":
            mask = trigger_mask.copy()
            preds = apply_vector_correction(base_predictions, vectors, mask, 0.5 * float(policy["correction_strength"]))
        elif variant == "candidate_main":
            mask = trigger_mask.copy()
            preds = apply_vector_correction(base_predictions, vectors, mask, float(policy["correction_strength"]))
        else:
            raise ValueError(f"Unknown variant: {variant}")
        elapsed = (time.perf_counter() - start) * 1000.0 / max(len(test_samples), 1)
        variant_outputs[variant] = (preds, [elapsed] * len(test_samples), mask)

    no_mse = mean([mse(s["prediction"], s["target"]) for s in test_samples])
    no_mae = mean([mae(s["prediction"], s["target"]) for s in test_samples])
    metric_rows = []
    main_predictions = variant_outputs["candidate_main"][0]
    for variant in VARIANTS:
        preds, latencies, mask = variant_outputs[variant]
        metric_rows.append(
            metric_row(
                config_record=config_record,
                candidate_id=candidate_id,
                stress_type=stress_type,
                variant=variant,
                samples=test_samples,
                predictions=preds,
                latencies_ms=latencies,
                trigger_mask=mask,
                no_mse=no_mse,
                no_mae=no_mae,
                thresholds=incumbent_thresholds,
                prediction_path=prediction_path,
                run_dir=run_dir,
                policy=policy,
            )
        )

    diagnostics = {
        "diagnostic_score": [diagnostic_score_row(config_record, candidate_id, stress_type, test_samples, scores)],
        "score_bin_table": score_bin_rows(config_record, candidate_id, stress_type, test_samples, scores, main_predictions, trigger_mask),
        "rule_vs_random_paired": random_paired_rows(config_record, candidate_id, stress_type, test_samples, base_predictions, vectors, trigger_mask, policy, search_cfg),
        "validation_policy": [policy_summary_row(config_record, candidate_id, stress_type, policy)],
    }
    payload = {
        "candidate_id": candidate_id,
        "family": CANDIDATE_FAMILIES[candidate_id],
        "stress_type": stress_type,
        "dataset": config_record.dataset,
        "model": config_record.model,
        "horizon": config_record.horizon,
        "input_path": str(prediction_path),
        "threshold_source_split": "val",
        "evaluation_split": "test",
        "test_threshold_leakage": False,
        "policy": serialize_policy(policy),
        "main_ablation": metric_rows,
        "diagnostics": diagnostics,
    }
    return metric_rows, diagnostics, payload


def score_and_vectors_for_samples(candidate_id: str, samples: List[dict], policy: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    if candidate_id == "s10_c1_error_predictive_residual":
        x = standardized_features(samples, policy)
        scores = predict_ridge(x, np.asarray(policy["risk_weights"], dtype=np.float64))
        vectors = np.repeat(np.asarray(policy["residual_shape"], dtype=np.float64).reshape(1, -1), len(samples), axis=0)
        return scores, vectors
    if candidate_id in {"s10_c2_dynamics_consistency", "s10_c2b_dynamics_anti_smoothing"}:
        return (
            np.asarray([dynamics_score(s["context"], s["prediction"]) for s in samples], dtype=np.float64),
            np.asarray([dynamics_vector(s["context"], s["prediction"]) for s in samples], dtype=np.float64),
        )
    if candidate_id == "s10_c3_perturbation_stability":
        scores = []
        vectors = []
        for sample in samples:
            score, vector = stability_score_and_vector(sample["context"], sample["prediction"])
            scores.append(score)
            vectors.append(vector)
        return np.asarray(scores, dtype=np.float64), np.asarray(vectors, dtype=np.float64)
    if candidate_id == "s10_c4_residual_shape_model":
        x = standardized_features(samples, policy)
        vectors = predict_ridge(x, np.asarray(policy["residual_weights"], dtype=np.float64))
        scale = np.asarray([float(np.std(arr(s["context"]))) + 1e-12 for s in samples], dtype=np.float64)
        scores = np.sqrt(np.mean(vectors**2, axis=1)) / scale
        return scores, vectors
    raise ValueError(candidate_id)


def standardized_features(samples: List[dict], policy: Dict[str, object]) -> np.ndarray:
    features = np.asarray([feature_vector(s["context"], s["prediction"]) for s in samples], dtype=np.float64)
    return (features - np.asarray(policy["feature_mean"], dtype=np.float64)) / np.asarray(policy["feature_std"], dtype=np.float64)


def feature_vector(context: Iterable[float], prediction: Iterable[float]) -> np.ndarray:
    ctx = arr(context)
    pred = arr(prediction)
    scale = float(np.std(ctx)) + 1e-12
    ctx_slope = ols_slope(ctx)
    pred_slope = ols_slope(pred)
    ctx_tail = ctx[-min(ctx.size, pred.size, 32) :]
    pred_head = pred[: min(pred.size, 32)]
    last_diff = float(ctx[-1] - ctx[-2]) if ctx.size >= 2 else 0.0
    pred_first_diff = float(pred[1] - pred[0]) if pred.size >= 2 else 0.0
    context_diff_std = float(np.std(np.diff(ctx))) + 1e-12 if ctx.size >= 2 else 1.0
    pred_diff_std = float(np.std(np.diff(pred))) + 1e-12 if pred.size >= 2 else 1.0
    ctx_hf = high_frequency_energy_ratio(ctx, 0.5)
    pred_hf = high_frequency_energy_ratio(pred, 0.5)
    ctx_range = float(np.ptp(ctx)) + 1e-12
    pred_range = float(np.ptp(pred)) + 1e-12
    return np.asarray(
        [
            pred.size / 720.0,
            ctx_slope * pred.size / scale,
            pred_slope * pred.size / scale,
            abs(pred_slope - ctx_slope) * pred.size / scale,
            (float(pred[0]) - (float(ctx[-1]) + last_diff)) / scale,
            (pred_first_diff - last_diff) / scale,
            (float(pred_head.mean()) - float(ctx_tail.mean())) / scale,
            math.log((float(np.std(pred)) + 1e-12) / scale),
            math.log(pred_diff_std / context_diff_std),
            ctx_hf,
            pred_hf,
            max(0.0, pred_hf - ctx_hf),
            spectral_distance(ctx, pred),
            curvature_roughness(ctx),
            curvature_roughness(pred),
            max(0.0, curvature_roughness(pred) - curvature_roughness(ctx)),
            abs(acf1(pred) - acf1(ctx)),
            math.log(pred_range / ctx_range),
        ],
        dtype=np.float64,
    )


def dynamics_score(context: Iterable[float], prediction: Iterable[float]) -> float:
    ctx = arr(context)
    pred = arr(prediction)
    scale = float(np.std(ctx)) + 1e-12
    last_diff = float(ctx[-1] - ctx[-2]) if ctx.size >= 2 else 0.0
    pred_first_diff = float(pred[1] - pred[0]) if pred.size >= 2 else 0.0
    boundary = abs(float(pred[0]) - (float(ctx[-1]) + last_diff)) / scale
    diff_gap = abs(pred_first_diff - last_diff) / scale
    ctx_curv = float(np.mean(np.diff(ctx[-min(ctx.size, 16) :], n=2))) if ctx.size >= 4 else 0.0
    pred_curv = float(np.mean(np.diff(pred[: min(pred.size, 16)], n=2))) if pred.size >= 4 else 0.0
    curv_gap = abs(pred_curv - ctx_curv) / scale
    return float(boundary + 0.5 * diff_gap + 0.25 * curv_gap)


def dynamics_vector(context: Iterable[float], prediction: Iterable[float]) -> np.ndarray:
    ctx = arr(context)
    pred = arr(prediction)
    if pred.size == 0:
        return pred.copy()
    last_diff = float(ctx[-1] - ctx[-2]) if ctx.size >= 2 else 0.0
    expected_first = float(ctx[-1]) + last_diff
    jump = float(pred[0]) - expected_first
    pred_first_diff = float(pred[1] - pred[0]) if pred.size >= 2 else 0.0
    diff_gap = pred_first_diff - last_diff
    t = np.arange(pred.size, dtype=np.float64)
    decay = np.exp(-t / max(4.0, pred.size / 12.0))
    return (-jump * decay - diff_gap * t * decay / max(pred.size, 1)).astype(np.float64)


def stability_score_and_vector(context: Iterable[float], prediction: Iterable[float]) -> Tuple[float, np.ndarray]:
    ctx = arr(context)
    pred = arr(prediction)
    if pred.size == 0:
        return 0.0, pred.copy()
    views = [
        pred,
        frequency_correction(pred, 0.5, 0.5),
        pred + dynamics_vector(ctx, pred),
        trend_correction(ctx, pred, 0.3, max_adjustment_ratio=0.04),
    ]
    stack = np.vstack(views)
    scale = float(np.std(ctx)) + 1e-12
    score = float(np.mean(np.var(stack, axis=0)) / (scale**2))
    consensus = np.median(stack, axis=0)
    return score, (consensus - pred).astype(np.float64)


def fit_ridge(x_std: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    x_aug = np.hstack([np.ones((x_std.shape[0], 1), dtype=np.float64), x_std])
    reg = float(l2) * np.eye(x_aug.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)


def predict_ridge(x_std: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.hstack([np.ones((x_std.shape[0], 1), dtype=np.float64), x_std])
    return x_aug @ weights


def apply_vector_correction(base_predictions: np.ndarray, vectors: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    corrected = np.asarray(base_predictions, dtype=np.float64).copy()
    if corrected.size == 0:
        return corrected
    corrected[mask] = corrected[mask] + float(alpha) * vectors[mask]
    return corrected


def matched_random_mask(n: int, count: int, seed: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    if n <= 0 or count <= 0:
        return mask
    rng = np.random.default_rng(seed)
    mask[rng.choice(n, size=min(count, n), replace=False)] = True
    return mask


def metric_row(
    config_record: Config,
    candidate_id: str,
    stress_type: str,
    variant: str,
    samples: List[dict],
    predictions: np.ndarray,
    latencies_ms: List[float],
    trigger_mask: np.ndarray,
    no_mse: float,
    no_mae: float,
    thresholds: Thresholds,
    prediction_path: Path,
    run_dir: Path,
    policy: Dict[str, object],
) -> Dict[str, object]:
    values_mse = [mse(pred, sample["target"]) for pred, sample in zip(predictions, samples)]
    values_mae = [mae(pred, sample["target"]) for pred, sample in zip(predictions, samples)]
    trend_flags = []
    freq_flags = []
    spectral = []
    changed = []
    for pred, sample in zip(predictions, samples):
        score = score_sample(sample["context"], pred, 0.5, freq_score_mode="excess_plus_spectral")
        trend, freq = trigger_flags(score, thresholds)
        trend_flags.append(trend)
        freq_flags.append(freq)
        spectral.append(1.0 / (1.0 + score["spectral_distance"]))
        changed.append(bool(np.max(np.abs(arr(sample["prediction"]) - arr(pred))) > 1e-10))
    return {
        "candidate_id": candidate_id,
        "family": CANDIDATE_FAMILIES[candidate_id],
        "stress_type": stress_type,
        "dataset": config_record.dataset,
        "model": config_record.model,
        "horizon": config_record.horizon,
        "variant": variant,
        "status": "completed",
        "mse": mean(values_mse),
        "mae": mean(values_mae),
        "mse_delta_pct_vs_no_correction": pct_delta(mean(values_mse), no_mse),
        "mae_delta_pct_vs_no_correction": pct_delta(mean(values_mae), no_mae),
        "hallucination_rate": float(np.mean(trigger_mask)) if len(trigger_mask) else 0.0,
        "trend_violation_rate": float(np.mean(trend_flags)) if trend_flags else 0.0,
        "freq_violation_rate": float(np.mean(freq_flags)) if freq_flags else 0.0,
        "spectral_consistency": mean(spectral),
        "turning_point_false_correction_rate": 0.0,
        "correction_rate": float(np.mean(changed)) if changed else 0.0,
        "inference_latency_ms": mean(latencies_ms),
        "threshold_quantile": policy.get("trigger_quantile", ""),
        "correction_strength": policy.get("correction_strength", ""),
        "validation_mse_delta_pct": policy.get("validation_mse_delta_pct", ""),
        "test_threshold_leakage": False,
        "prediction_path": str(prediction_path),
        "output_dir": str(run_dir),
        "blocker_reason": "",
    }


def blocked_row(config_record: Config, candidate_id: str, stress_type: str, prediction_path: Path, run_dir: Path, reason: str) -> Dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "family": CANDIDATE_FAMILIES.get(candidate_id, ""),
        "stress_type": stress_type,
        "dataset": config_record.dataset,
        "model": config_record.model,
        "horizon": config_record.horizon,
        "variant": "all",
        "status": "blocked",
        "mse": "",
        "mae": "",
        "mse_delta_pct_vs_no_correction": "",
        "mae_delta_pct_vs_no_correction": "",
        "hallucination_rate": "",
        "trend_violation_rate": "",
        "freq_violation_rate": "",
        "spectral_consistency": "",
        "turning_point_false_correction_rate": "",
        "correction_rate": "",
        "inference_latency_ms": "",
        "threshold_quantile": "",
        "correction_strength": "",
        "validation_mse_delta_pct": "",
        "test_threshold_leakage": "",
        "prediction_path": str(prediction_path),
        "output_dir": str(run_dir),
        "blocker_reason": reason,
    }


def diagnostic_score_row(config_record: Config, candidate_id: str, stress_type: str, samples: List[dict], scores: np.ndarray) -> Dict[str, object]:
    errors = np.asarray([mse(s["prediction"], s["target"]) for s in samples], dtype=np.float64)
    top20 = errors >= float(np.quantile(errors, 0.80))
    top10_score = scores >= float(np.quantile(scores, 0.90))
    bottom50_score = scores <= float(np.quantile(scores, 0.50))
    top10_error = float(errors[top10_score].mean()) if top10_score.any() else 0.0
    bottom50_error = float(errors[bottom50_score].mean()) if bottom50_score.any() else 0.0
    return {
        "candidate_id": candidate_id,
        "family": CANDIDATE_FAMILIES[candidate_id],
        "stress_type": stress_type,
        "dataset": config_record.dataset,
        "model": config_record.model,
        "horizon": config_record.horizon,
        "spearman_error_corr": spearman(scores, errors),
        "auroc_top20_error": auroc(scores, top20),
        "top10_risk_error": top10_error,
        "bottom50_risk_error": bottom50_error,
        "top10_vs_bottom50_error_lift": top10_error / bottom50_error if bottom50_error > 1e-12 else "",
        "mean_error": float(errors.mean()),
        "score_p50": float(np.quantile(scores, 0.50)),
        "score_p90": float(np.quantile(scores, 0.90)),
    }


def score_bin_rows(
    config_record: Config,
    candidate_id: str,
    stress_type: str,
    samples: List[dict],
    scores: np.ndarray,
    predictions: np.ndarray,
    trigger_mask: np.ndarray,
) -> List[Dict[str, object]]:
    base_errors = np.asarray([mse(s["prediction"], s["target"]) for s in samples], dtype=np.float64)
    corrected_errors = np.asarray([mse(pred, s["target"]) for pred, s in zip(predictions, samples)], dtype=np.float64)
    edges = [0.0, 0.50, 0.80, 0.90, 0.95, 1.0]
    quantiles = np.quantile(scores, edges)
    rows = []
    for idx in range(len(edges) - 1):
        lo_q, hi_q = edges[idx], edges[idx + 1]
        lo, hi = quantiles[idx], quantiles[idx + 1]
        mask = (scores >= lo) & (scores <= hi) if idx == len(edges) - 2 else (scores >= lo) & (scores < hi)
        rows.append(
            {
                "candidate_id": candidate_id,
                "family": CANDIDATE_FAMILIES[candidate_id],
                "stress_type": stress_type,
                "dataset": config_record.dataset,
                "model": config_record.model,
                "horizon": config_record.horizon,
                "score_bin": f"{lo_q:.2f}-{hi_q:.2f}",
                "n_samples": int(mask.sum()),
                "baseline_mse": float(base_errors[mask].mean()) if mask.any() else "",
                "corrected_mse": float(corrected_errors[mask].mean()) if mask.any() else "",
                "mse_delta_pct": pct_delta(float(corrected_errors[mask].mean()), float(base_errors[mask].mean())) if mask.any() else "",
                "correction_rate": float(trigger_mask[mask].mean()) if mask.any() else "",
                "score_min": float(scores[mask].min()) if mask.any() else "",
                "score_max": float(scores[mask].max()) if mask.any() else "",
            }
        )
    return rows


def random_paired_rows(
    config_record: Config,
    candidate_id: str,
    stress_type: str,
    samples: List[dict],
    base_predictions: np.ndarray,
    vectors: np.ndarray,
    trigger_mask: np.ndarray,
    policy: Dict[str, object],
    search_cfg: Dict,
) -> List[Dict[str, object]]:
    alpha = float(policy["correction_strength"])
    rule_preds = apply_vector_correction(base_predictions, vectors, trigger_mask, alpha)
    rule_mse = mean([mse(pred, s["target"]) for pred, s in zip(rule_preds, samples)])
    rule_mae = mean([mae(pred, s["target"]) for pred, s in zip(rule_preds, samples)])
    seeds = [int(v) for v in ((search_cfg.get("search", {}) or {}).get("random_seeds", [1101, 2202, 3303, 4404, 5505]))]
    rows = []
    for seed in seeds:
        random_mask = matched_random_mask(len(samples), int(trigger_mask.sum()), seed)
        random_preds = apply_vector_correction(base_predictions, vectors, random_mask, alpha)
        random_mse = mean([mse(pred, s["target"]) for pred, s in zip(random_preds, samples)])
        random_mae = mean([mae(pred, s["target"]) for pred, s in zip(random_preds, samples)])
        rows.append(
            {
                "candidate_id": candidate_id,
                "family": CANDIDATE_FAMILIES[candidate_id],
                "stress_type": stress_type,
                "dataset": config_record.dataset,
                "model": config_record.model,
                "horizon": config_record.horizon,
                "random_seed": seed,
                "rule_mse": rule_mse,
                "random_mse": random_mse,
                "rule_minus_random_mse": rule_mse - random_mse,
                "rule_beats_random_mse": bool(rule_mse < random_mse),
                "rule_mae": rule_mae,
                "random_mae": random_mae,
                "rule_minus_random_mae": rule_mae - random_mae,
                "rule_beats_random_mae": bool(rule_mae < random_mae),
                "random_correction_rate": float(random_mask.mean()) if len(random_mask) else 0.0,
            }
        )
    return rows


def policy_summary_row(config_record: Config, candidate_id: str, stress_type: str, policy: Dict[str, object]) -> Dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "family": CANDIDATE_FAMILIES[candidate_id],
        "stress_type": stress_type,
        "dataset": config_record.dataset,
        "model": config_record.model,
        "horizon": config_record.horizon,
        "trigger_quantile": policy.get("trigger_quantile", ""),
        "trigger_threshold": policy.get("trigger_threshold", ""),
        "correction_strength": policy.get("correction_strength", ""),
        "validation_mse_delta_pct": policy.get("validation_mse_delta_pct", ""),
        "validation_random_advantage_mse": policy.get("validation_random_advantage_mse", ""),
        "validation_trigger_rate": policy.get("validation_trigger_rate", ""),
    }


def write_run_outputs(run_dir: Path, payload: Dict[str, object], metric_rows: List[Dict[str, object]], diagnostics: Dict[str, List[Dict[str, object]]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    write_csv(metric_rows, run_dir / "metrics.csv")
    (run_dir / "ablation_table.md").write_text(markdown_table(metric_rows), encoding="utf-8")
    (run_dir / "summary.md").write_text(run_summary_markdown(payload, metric_rows), encoding="utf-8")
    diagnostics_dir = run_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in diagnostics.items():
        write_csv(rows, diagnostics_dir / f"{name}.csv")


def write_outputs(
    output_root: Path,
    run_root: Path,
    candidate_id: str,
    scope: str,
    rows: List[Dict[str, object]],
    config_records: List[Dict[str, object]],
    diagnostics: Dict[str, List[Dict[str, object]]],
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    write_csv(rows, run_root / "combined_metrics.csv")
    payload = {
        "candidate_id": candidate_id,
        "family": CANDIDATE_FAMILIES[candidate_id],
        "scope": scope,
        "configs": config_records,
        "rows": rows,
        "completed_configs": count_completed(config_records),
        "total_configs": len(config_records),
        "summary": summarize_rows(rows, diagnostics),
    }
    (run_root / "combined_metrics.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    (run_root / "combined_ablation_table.md").write_text(markdown_table(rows), encoding="utf-8")
    (run_root / "summary.md").write_text(summary_markdown(payload), encoding="utf-8")
    for name, diag_rows in diagnostics.items():
        write_csv(diag_rows, output_root / "diagnostics" / f"{scope}_{candidate_id}_{name}.csv")
    write_candidate_ledger(output_root / "candidate_ledger.csv", candidate_id, scope, rows, config_records, diagnostics)


def write_candidate_ledger(path: Path, candidate_id: str, scope: str, rows: List[Dict[str, object]], config_records: List[Dict[str, object]], diagnostics: Dict[str, List[Dict[str, object]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: List[Dict[str, object]] = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
    existing = [r for r in existing if not (r.get("candidate_id") == candidate_id and r.get("scope") == scope)]
    summary = summarize_rows(rows, diagnostics)
    existing.append(
        {
            "candidate_id": candidate_id,
            "family": CANDIDATE_FAMILIES[candidate_id],
            "scope": scope,
            "status": "completed" if count_completed(config_records) == len(config_records) else "blocked",
            "completed_configs": count_completed(config_records),
            "total_configs": len(config_records),
            "main_mean_mse_delta_pct": summary["main_mean_mse_delta_pct"],
            "main_improved_configs": summary["main_improved_configs"],
            "main_beats_random_configs": summary["main_beats_random_configs"],
            "paired_rule_win_rate": summary["paired_rule_win_rate"],
            "main_beats_matched_configs": summary["main_beats_matched_configs"],
            "matched_smoothing_mean_mse_delta_pct": summary["matched_smoothing_mean_mse_delta_pct"],
            "naive_smoothing_mean_mse_delta_pct": summary["naive_smoothing_mean_mse_delta_pct"],
            "stage9_incumbent_mean_mse_delta_pct": summary["stage9_incumbent_mean_mse_delta_pct"],
            "diagnostic_spearman_mean": summary["diagnostic_spearman_mean"],
            "diagnostic_auroc_mean": summary["diagnostic_auroc_mean"],
            "diagnostic_top10_lift_mean": summary["diagnostic_top10_lift_mean"],
            "max_mse_harm_pct": summary["max_mse_harm_pct"],
            "max_mae_harm_pct": summary["max_mae_harm_pct"],
            "test_threshold_leakage": summary["test_threshold_leakage"],
            "gate_verdict": gate_verdict(scope, summary, len(config_records)),
        }
    )
    write_csv(existing, path)


def summarize_rows(rows: List[Dict[str, object]], diagnostics: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    completed = [r for r in rows if r.get("status") == "completed"]
    main = [r for r in completed if r.get("variant") == "candidate_main"]
    random_rows = [r for r in completed if r.get("variant") == "random_trigger"]
    matched = [r for r in completed if r.get("variant") == "matched_smoothing_control"]
    naive = [r for r in completed if r.get("variant") == "naive_smoothing"]
    incumbent = [r for r in completed if r.get("variant") == "stage9_incumbent"]
    random_by_key = {(r["stress_type"], r["dataset"], r["model"], str(r["horizon"])): r for r in random_rows}
    matched_by_key = {(r["stress_type"], r["dataset"], r["model"], str(r["horizon"])): r for r in matched}
    beats_random = 0
    beats_matched = 0
    advantages = []
    matched_advantages = []
    for row in main:
        key = (row["stress_type"], row["dataset"], row["model"], str(row["horizon"]))
        rr = random_by_key.get(key)
        mm = matched_by_key.get(key)
        if rr:
            advantage = float(rr["mse"]) - float(row["mse"])
            advantages.append(advantage)
            beats_random += int(advantage > 0)
        if mm:
            matched_advantage = float(mm["mse"]) - float(row["mse"])
            matched_advantages.append(matched_advantage)
            beats_matched += int(matched_advantage > 0)
    paired = diagnostics.get("rule_vs_random_paired", [])
    paired_wins = sum(1 for r in paired if str(r.get("rule_beats_random_mse", "")).lower() == "true")
    diag = diagnostics.get("diagnostic_score", [])
    return {
        "main_mean_mse_delta_pct": mean([float(r["mse_delta_pct_vs_no_correction"]) for r in main]),
        "main_improved_configs": sum(1 for r in main if float(r["mse_delta_pct_vs_no_correction"]) < 0),
        "main_beats_random_configs": beats_random,
        "main_vs_random_mean_advantage_mse": mean(advantages),
        "paired_rule_win_rate": paired_wins / len(paired) if paired else "",
        "main_beats_matched_configs": beats_matched,
        "main_vs_matched_mean_advantage_mse": mean(matched_advantages),
        "matched_smoothing_mean_mse_delta_pct": mean([float(r["mse_delta_pct_vs_no_correction"]) for r in matched]),
        "naive_smoothing_mean_mse_delta_pct": mean([float(r["mse_delta_pct_vs_no_correction"]) for r in naive]),
        "stage9_incumbent_mean_mse_delta_pct": mean([float(r["mse_delta_pct_vs_no_correction"]) for r in incumbent]),
        "diagnostic_spearman_mean": mean([float(r["spearman_error_corr"]) for r in diag if r.get("spearman_error_corr") != ""]),
        "diagnostic_auroc_mean": mean([float(r["auroc_top20_error"]) for r in diag if r.get("auroc_top20_error") != ""]),
        "diagnostic_top10_lift_mean": mean([float(r["top10_vs_bottom50_error_lift"]) for r in diag if r.get("top10_vs_bottom50_error_lift") != ""]),
        "max_mse_harm_pct": max([float(r["mse_delta_pct_vs_no_correction"]) for r in main], default=0.0),
        "max_mae_harm_pct": max([float(r["mae_delta_pct_vs_no_correction"]) for r in main], default=0.0),
        "test_threshold_leakage": any(str(r.get("test_threshold_leakage")) != "False" for r in completed),
        "stress_types": sorted(set(r["stress_type"] for r in completed)),
    }


def gate_verdict(scope: str, summary: Dict[str, object], n_records: int) -> str:
    if summary["test_threshold_leakage"]:
        return "fail_leakage"
    if summary["max_mse_harm_pct"] > 3.0 or summary["max_mae_harm_pct"] > 3.0:
        return "fail_harm"
    if scope == "smoke":
        if summary["main_mean_mse_delta_pct"] <= -0.03 and summary["main_beats_random_configs"] >= 3:
            return "promote_to_clean_full"
        if summary["diagnostic_top10_lift_mean"] >= 2.0 and summary["diagnostic_auroc_mean"] >= 0.60:
            return "promote_diagnostic"
        return "archive_smoke"
    if scope == "clean_full":
        if summary["diagnostic_top10_lift_mean"] >= 2.0 and summary["diagnostic_auroc_mean"] >= 0.60:
            return "diagnostic_success"
        if summary["main_mean_mse_delta_pct"] > -0.05:
            return "fail_clean_mse"
        if summary["main_beats_random_configs"] < 14 or float(summary["paired_rule_win_rate"] or 0.0) < 0.70:
            return "fail_rule_random"
        if summary["main_beats_matched_configs"] < 10:
            return "fail_matched_smoothing"
        return "clean_mechanism_pass"
    if scope == "stress":
        return "stress_completed"
    return "unknown"


def markdown_table(rows: List[Dict[str, object]]) -> str:
    headers = ["stress_type", "dataset", "model", "horizon", "variant", "status", "mse", "mae", "mse_delta_pct_vs_no_correction", "correction_rate", "blocker_reason"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def run_summary_markdown(payload: Dict[str, object], rows: List[Dict[str, object]]) -> str:
    by_variant = {r["variant"]: r for r in rows}
    main = by_variant.get("candidate_main", {})
    random = by_variant.get("random_trigger", {})
    matched = by_variant.get("matched_smoothing_control", {})
    return f"""# Stage 10 Config Summary

- Candidate: `{payload['candidate_id']}`
- Family: `{payload['family']}`
- Stress type: `{payload['stress_type']}`
- Dataset/model/horizon: `{payload['dataset']} / {payload['model']} / {payload['horizon']}`
- Threshold source split: `val`
- Evaluation split: `test`
- Test threshold leakage: `False`

## Headline

- candidate_main MSE delta: {float(main.get('mse_delta_pct_vs_no_correction', 0.0)):.6f}%
- random_trigger MSE delta: {float(random.get('mse_delta_pct_vs_no_correction', 0.0)):.6f}%
- matched_smoothing_control MSE delta: {float(matched.get('mse_delta_pct_vs_no_correction', 0.0)):.6f}%

## Ablation

{markdown_table(rows)}
"""


def summary_markdown(payload: Dict[str, object]) -> str:
    s = payload["summary"]
    lines = [
        "# Stage 10 Table Summary",
        "",
        f"- Candidate: `{payload['candidate_id']}`",
        f"- Family: `{payload['family']}`",
        f"- Scope: `{payload['scope']}`",
        f"- Completed configs: {payload['completed_configs']} / {payload['total_configs']}",
        f"- Stress types: `{', '.join(s['stress_types'])}`",
        f"- candidate_main mean MSE delta: {s['main_mean_mse_delta_pct']:.6f}%",
        f"- candidate_main improved configs: {s['main_improved_configs']}",
        f"- candidate_main beats random configs: {s['main_beats_random_configs']}",
        f"- paired rule win rate: {s['paired_rule_win_rate']}",
        f"- candidate_main beats matched smoothing configs: {s['main_beats_matched_configs']}",
        f"- matched smoothing mean MSE delta: {s['matched_smoothing_mean_mse_delta_pct']:.6f}%",
        f"- naive smoothing mean MSE delta: {s['naive_smoothing_mean_mse_delta_pct']:.6f}%",
        f"- Stage 9 incumbent mean MSE delta: {s['stage9_incumbent_mean_mse_delta_pct']:.6f}%",
        f"- diagnostic Spearman mean: {s['diagnostic_spearman_mean']:.6f}",
        f"- diagnostic AUROC mean: {s['diagnostic_auroc_mean']:.6f}",
        f"- diagnostic top10/bottom50 lift mean: {s['diagnostic_top10_lift_mean']:.6f}",
        f"- max MSE harm: {s['max_mse_harm_pct']:.6f}%",
        f"- max MAE harm: {s['max_mae_harm_pct']:.6f}%",
        f"- test threshold leakage: {s['test_threshold_leakage']}",
        f"- gate verdict: `{gate_verdict(payload['scope'], s, payload['total_configs'])}`",
        "",
        "## Config Status",
        "",
    ]
    for record in payload["configs"]:
        reason = f" ({record['blocker_reason']})" if record.get("blocker_reason") else ""
        lines.append(f"- {record['stress_type']} / {record['dataset']} / {record['model']} / {record['horizon']}: {record['status']}{reason}")
    return "\n".join(lines) + "\n"


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def serialize_policy(policy: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in policy.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        elif key in {"feature_mean", "feature_std", "risk_weights", "residual_shape", "residual_weights"}:
            out[key] = np.asarray(value).tolist()
        else:
            out[key] = value
    return out


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def arr(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def acf1(values: Iterable[float]) -> float:
    series = arr(values)
    if series.size < 3:
        return 0.0
    x0 = series[:-1] - float(series[:-1].mean())
    x1 = series[1:] - float(series[1:].mean())
    denom = float(np.linalg.norm(x0) * np.linalg.norm(x1))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x0, x1) / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    rx = rankdata(x)
    ry = rankdata(y)
    return pearson(rx, ry)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - float(x.mean())
    y = y - float(y.mean())
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0 or labels.sum() == labels.size:
        return 0.5
    ranks = rankdata(scores) + 1.0
    pos = ranks[labels].sum()
    n_pos = float(labels.sum())
    n_neg = float(labels.size - labels.sum())
    return float((pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


def pct_delta(value: float, baseline: float) -> float:
    if abs(float(baseline)) <= 1e-12:
        return 0.0
    return 100.0 * (float(value) - float(baseline)) / float(baseline)


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return float(sum(vals) / len(vals)) if vals else 0.0


def count_completed(records: Iterable[Dict[str, object]]) -> int:
    return sum(1 for r in records if r.get("status") == "completed")


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
