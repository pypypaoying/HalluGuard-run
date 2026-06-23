"""Stage 13 adaptive HalluGuard router runner."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

import evaluate_predictions as incumbent_eval
import halluguard_router as router
import run_stage11_dynamics as stage11
from correction import Thresholds, naive_smoothing
from halluguard_dynamics import (
    apply_vector_predictions,
    array_mae,
    array_mse,
    matched_random_mask,
    metric_row,
    trigger_mask_for_samples,
)


DATASETS = ["ETTm1", "ETTh1"]
MODELS = ["DLinear", "PatchTST"]
HORIZONS = [96, 192, 336, 720]
SMOKE_CONFIGS = [
    ("ETTm1", "DLinear", 192),
    ("ETTm1", "PatchTST", 720),
    ("ETTh1", "DLinear", 336),
    ("ETTh1", "PatchTST", 720),
]
DEFAULT_CONFIG = "experiments/halluguard/configs/halluguard_stage13_adaptive_router.yaml"


@dataclass(frozen=True)
class ConfigRecord:
    dataset: str
    model: str
    horizon: int

    @property
    def tag(self) -> str:
        return f"{self.dataset}_{self.model}_{self.horizon}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 13 adaptive HalluGuard router.")
    parser.add_argument("--scope", required=True, choices=["smoke", "clean_full", "stress", "external_batch"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--stage7-dir", default=None)
    parser.add_argument("--external-input-dir", default=None)
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_yaml(resolve_path(repo_root, Path(args.config)))
    candidate_id = str(cfg.get("candidate_id", "s13_adaptive_halluguard_router"))
    output_root = resolve_path(repo_root, Path(args.output_root)) if args.output_root else resolve_path(repo_root, Path((cfg.get("outputs", {}) or {}).get("results_dir", "experiments/halluguard/results/stage13_adaptive_router")))
    output_root.mkdir(parents=True, exist_ok=True)
    stage7_dir = resolve_path(repo_root, Path(args.stage7_dir)) if args.stage7_dir else resolve_path(repo_root, Path((cfg.get("data", {}) or {}).get("stage7_prediction_dir", "experiments/halluguard/results/stage7_big_table/predictions"))).parent

    if args.scope == "external_batch":
        input_dir = resolve_path(repo_root, Path(args.external_input_dir)) if args.external_input_dir else resolve_path(repo_root, Path((cfg.get("data", {}) or {}).get("external_fixture_dir", "experiments/halluguard/results/stage12_external_batch/fixtures/external_batch_clean")))
        files = sorted([p for p in input_dir.glob("*.jsonl") if p.name.lower() != "manifest.jsonl"] + [p for p in input_dir.glob("*.csv") if p.name.lower() != "manifest.csv"])
        if args.limit_files is not None:
            files = files[: int(args.limit_files)]
        scopes = [("external_batch", record_from_prediction_file(path), path) for path in files]
        run_root = output_root / "external_batch" / candidate_id
    else:
        records = [ConfigRecord(*c) for c in SMOKE_CONFIGS] if args.scope == "smoke" else [
            ConfigRecord(dataset, model, horizon)
            for dataset in DATASETS
            for model in MODELS
            for horizon in HORIZONS
        ]
        if args.scope == "stress":
            stress_types = list((cfg.get("stress", {}) or {}).get("types", []))
            run_root = output_root / "stress_table" / candidate_id
        else:
            stress_types = ["clean"]
            table_name = "smoke" if args.scope == "smoke" else "clean_full_table"
            run_root = output_root / table_name / candidate_id
        scopes = []
        for stress_type in stress_types:
            for record in records:
                source = stage7_dir / "predictions" / f"{record.tag}.jsonl" if stage7_dir.name != "predictions" else stage7_dir / f"{record.tag}.jsonl"
                if stress_type == "clean":
                    prediction_path = source
                else:
                    prediction_path = output_root / "stress_predictions" / stress_type / f"{record.tag}.jsonl"
                    stage11.write_stress_predictions(source, prediction_path, stress_type)
                scopes.append((stress_type, record, prediction_path))

    rows: List[Dict[str, object]] = []
    config_records: List[Dict[str, object]] = []
    diagnostics: Dict[str, List[Dict[str, object]]] = {}
    for stress_type, record, prediction_path in scopes:
        run_group = run_root / "runs" if stress_type in {"clean", "external_batch"} else run_root / stress_type / "runs"
        run_dir = run_group / record.tag
        config_record, metric_rows, diag_rows = run_one_config(repo_root, cfg, candidate_id, record, prediction_path, run_dir, stress_type)
        config_records.append(config_record)
        rows.extend(metric_rows)
        for name, values in diag_rows.items():
            diagnostics.setdefault(name, []).extend(values)
        write_outputs(output_root, run_root, args.scope, candidate_id, cfg, rows, config_records, diagnostics)
        print(
            json.dumps(
                {
                    "progress": "stage13_config_done",
                    "scope": args.scope,
                    "stress_type": stress_type,
                    "config": record.tag,
                    "status": config_record["status"],
                    "completed_configs": count_completed(config_records),
                    "total_seen": len(config_records),
                }
            ),
            flush=True,
        )

    write_outputs(output_root, run_root, args.scope, candidate_id, cfg, rows, config_records, diagnostics)
    print(json.dumps({"scope": args.scope, "candidate_id": candidate_id, "output_dir": str(run_root), "completed_configs": count_completed(config_records), "total_configs": len(config_records)}))


def run_one_config(repo_root: Path, cfg: Dict, candidate_id: str, record: ConfigRecord, prediction_path: Path, run_dir: Path, stress_type: str):
    try:
        samples = stage11.load_prediction_file(prediction_path)
        val_samples = [s for s in samples if s.get("split") == "val"]
        test_samples = [s for s in samples if s.get("split") == "test"]
        if not val_samples or not test_samples:
            raise ValueError(f"{prediction_path} must contain both val and test samples.")
        run_dir.mkdir(parents=True, exist_ok=True)
        metric_rows, diag_rows, payload = evaluate_router_ablation_set(repo_root, cfg, candidate_id, record, stress_type, prediction_path, run_dir, val_samples, test_samples)
        stage11.write_run_outputs(run_dir, payload, metric_rows, diag_rows)
        return {
            "candidate_id": candidate_id,
            "stress_type": stress_type,
            "dataset": record.dataset,
            "model": record.model,
            "horizon": record.horizon,
            "status": "completed",
            "blocker_reason": "",
        }, metric_rows, diag_rows
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return {
            "candidate_id": candidate_id,
            "stress_type": stress_type,
            "dataset": record.dataset,
            "model": record.model,
            "horizon": record.horizon,
            "status": "blocked",
            "blocker_reason": reason,
        }, [blocked_row(candidate_id, record, stress_type, prediction_path, run_dir, reason)], {}


def evaluate_router_ablation_set(
    repo_root: Path,
    cfg: Dict,
    candidate_id: str,
    record: ConfigRecord,
    stress_type: str,
    prediction_path: Path,
    run_dir: Path,
    val_samples: List[dict],
    test_samples: List[dict],
) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, object]]], Dict[str, object]]:
    method_cfg = cfg.get("method", {}) or {}
    router_cfg = cfg.get("router", {}) or {}
    candidate_actions = list(method_cfg.get("candidate_actions", router.DEFAULT_ACTIONS))
    optional_actions = list(method_cfg.get("optional_actions", []))
    eval_actions = list(dict.fromkeys(candidate_actions + optional_actions))
    router_variants = list(method_cfg.get("router_variants", router.DEPLOYABLE_ROUTERS))
    main_router = str(method_cfg.get("main_router", "harm_aware_router"))
    seed = int(cfg.get("seed", 23))

    prepared = router.prepare_router_training(val_samples, cfg, candidate_actions)
    router_policies = {
        variant: router.fit_router_from_prepared(prepared, cfg, router_type=variant)
        for variant in router_variants
        if variant != "oracle_test_ceiling"
    }
    if main_router not in router_policies:
        raise ValueError(f"main_router={main_router} was not fit.")
    main_policy = router_policies[main_router]
    action_context = main_policy["action_context"]

    variant_outputs: Dict[str, Tuple[List[np.ndarray], List[Dict[str, object]], List[float], Optional[Dict[str, object]]]] = {}
    for action in eval_actions:
        preds, infos, latencies = apply_action_to_samples(test_samples, action, action_context)
        variant_outputs[action] = (preds, infos, latencies, None)

    boundary_policy = action_context["dynamics_policies"]["boundary_only"]
    boundary_mask = trigger_mask_for_samples(test_samples, boundary_policy)
    matched_preds, matched_infos = sparse_smoothing_outputs(test_samples, boundary_mask, cfg)
    variant_outputs["matched_smoothing_control"] = (matched_preds, matched_infos, [0.0 for _ in test_samples], boundary_policy)

    random_mask = matched_random_mask(len(test_samples), int(boundary_mask.sum()), seed + 404)
    random_preds, random_infos, random_latencies = apply_vector_predictions(test_samples, boundary_policy, random_mask, config=cfg)
    for info in random_infos:
        info["action"] = "boundary_only"
    variant_outputs["random_trigger"] = (random_preds, random_infos, random_latencies, boundary_policy)

    incumbent_cfg = load_yaml(repo_root / cfg["stage9_incumbent_config"])
    incumbent_thresholds, incumbent_policy = incumbent_eval.calibrate_evaluation_policy(val_samples, incumbent_cfg, "val")
    inc_preds, inc_latencies, inc_infos = incumbent_eval.correct_samples(
        samples=test_samples,
        thresholds=incumbent_thresholds,
        variant="trend_frequency",
        config=incumbent_cfg,
        seed=seed + 991,
        lambda_trend=float(incumbent_policy.get("lambda_trend", incumbent_cfg["correction"]["default_lambda_trend"])),
        lambda_freq=float(incumbent_policy.get("lambda_freq", incumbent_cfg["correction"]["default_lambda_freq"])),
        policy=incumbent_policy,
    )
    variant_outputs["stage9_incumbent"] = (
        [np.asarray(p, dtype=np.float64) for p in inc_preds],
        [{"trigger": float(bool(i.get("policy_rule_hallucination", 0.0))), "changed": float(bool(i.get("changed", 0.0))), "action": "stage9_incumbent"} for i in inc_infos],
        inc_latencies,
        None,
    )

    for router_name, policy in router_policies.items():
        preds, infos, latencies = router.apply_router_to_samples(test_samples, policy)
        variant_outputs[router_name] = (preds, infos, latencies, policy)
        if router_name == "oracle_val_policy":
            variant_outputs["validation_best_single_action"] = (preds, infos, latencies, policy)

    main_preds, main_infos, _, _ = variant_outputs[main_router]
    random_action_preds, random_action_infos, random_action_latencies = random_action_outputs(test_samples, action_context, [str(info.get("action", "no_correction")) for info in main_infos], seed + 707)
    variant_outputs["random_action_router"] = (random_action_preds, random_action_infos, random_action_latencies, main_policy)

    feature_names = list(main_policy["feature_names"])
    features = router.feature_matrix(test_samples, action_context, feature_names)
    rng = np.random.default_rng(seed + 808)
    shuffled = features.copy()
    if shuffled.shape[0] > 1:
        shuffled = shuffled[rng.permutation(shuffled.shape[0])]
    shuffled_preds, shuffled_infos, shuffled_latencies = router.apply_router_to_samples(test_samples, main_policy, shuffled_features=shuffled)
    variant_outputs["shuffled_feature_router"] = (shuffled_preds, shuffled_infos, shuffled_latencies, main_policy)

    if "oracle_test_ceiling" in router_variants:
        oracle_preds, oracle_infos, oracle_latencies = router.oracle_test_ceiling(test_samples, action_context, candidate_actions)
        variant_outputs["oracle_test_ceiling"] = (oracle_preds, oracle_infos, oracle_latencies, main_policy)

    metric_rows = []
    for variant_name, (preds, infos, latencies, policy) in variant_outputs.items():
        if variant_name in router_variants or variant_name in {"random_action_router", "shuffled_feature_router", "validation_best_single_action", "oracle_test_ceiling"}:
            row = router.router_metric_row(variant_name, test_samples, preds, infos, latencies, policy)
        else:
            row = metric_row(variant_name, test_samples, preds, infos, latencies, policy)
            actions = [str(info.get("action", variant_name)) for info in infos]
            row.update({"action_distribution": router.compact_distribution(actions), "action_entropy": router.action_entropy(actions)})
        row.update(
            {
                "candidate_id": candidate_id,
                "stress_type": stress_type,
                "dataset": record.dataset,
                "model": record.model,
                "horizon": record.horizon,
                "status": "diagnostic_only" if variant_name == "oracle_test_ceiling" else "completed",
                "prediction_path": str(prediction_path),
                "output_dir": str(run_dir),
                "blocker_reason": "",
                "main_router": main_router,
                "parent_method": str(method_cfg.get("parent_method", "boundary_only")),
                "test_threshold_leakage": "diagnostic_only" if variant_name == "oracle_test_ceiling" else False,
            }
        )
        metric_rows.append(row)

    diagnostics = {
        "router_paired_random_action": paired_random_action_rows(record, stress_type, test_samples, variant_outputs, action_context, cfg),
        "action_alignment": action_alignment_rows(record, stress_type, test_samples, main_policy, main_infos),
        "router_validation": router_validation_rows(record, stress_type, router_policies),
    }
    payload = {
        "candidate_id": candidate_id,
        "stress_type": stress_type,
        "dataset": record.dataset,
        "model": record.model,
        "horizon": record.horizon,
        "input_path": str(prediction_path),
        "threshold_source_split": "val",
        "evaluation_split": "test",
        "test_threshold_leakage": False,
        "main_router": main_router,
        "candidate_actions": candidate_actions,
        "main_ablation": metric_rows,
        "diagnostics": diagnostics,
    }
    return metric_rows, diagnostics, payload


def apply_action_to_samples(samples: List[dict], action: str, action_context: Dict[str, object]) -> Tuple[List[np.ndarray], List[Dict[str, object]], List[float]]:
    preds = []
    infos = []
    latencies = []
    for sample in samples:
        result = router.apply_action(sample, action, action_context)
        base = np.asarray(sample["prediction"], dtype=np.float64)
        changed = bool(np.max(np.abs(result.prediction - base)) > 1e-10) if base.size else False
        preds.append(result.prediction)
        infos.append({**result.info, "action": action, "changed": float(changed), "trigger": float(action != "no_correction")})
        latencies.append(result.latency_ms)
    return preds, infos, latencies


def sparse_smoothing_outputs(samples: List[dict], mask: np.ndarray, cfg: Dict) -> Tuple[List[np.ndarray], List[Dict[str, object]]]:
    preds = []
    infos = []
    window = int((cfg.get("policy", {}) or {}).get("smoothing_window", 5))
    for idx, sample in enumerate(samples):
        base = np.asarray(sample["prediction"], dtype=np.float64)
        if bool(mask[idx]):
            pred = naive_smoothing(base, window)
            action = "naive_smoothing"
        else:
            pred = base.copy()
            action = "no_correction"
        preds.append(pred)
        infos.append({"trigger": float(bool(mask[idx])), "changed": float(np.max(np.abs(pred - base)) > 1e-10), "action": action})
    return preds, infos


def random_action_outputs(samples: List[dict], action_context: Dict[str, object], matched_actions: Sequence[str], seed: int) -> Tuple[List[np.ndarray], List[Dict[str, object]], List[float]]:
    rng = np.random.default_rng(seed)
    actions = list(matched_actions)
    if actions:
        rng.shuffle(actions)
    preds = []
    infos = []
    latencies = []
    for idx, sample in enumerate(samples):
        action = actions[idx] if idx < len(actions) else "no_correction"
        result = router.apply_action(sample, action, action_context)
        base = np.asarray(sample["prediction"], dtype=np.float64)
        preds.append(result.prediction)
        infos.append({**result.info, "action": action, "changed": float(np.max(np.abs(result.prediction - base)) > 1e-10), "trigger": float(action != "no_correction")})
        latencies.append(result.latency_ms)
    return preds, infos, latencies


def paired_random_action_rows(
    record: ConfigRecord,
    stress_type: str,
    samples: List[dict],
    variant_outputs: Dict[str, Tuple[List[np.ndarray], List[Dict[str, object]], List[float], Optional[Dict[str, object]]]],
    action_context: Dict[str, object],
    cfg: Dict,
) -> List[Dict[str, object]]:
    rows = []
    seeds = [int(v) for v in ((cfg.get("router", {}) or {}).get("random_seeds", [1101, 2202, 3303, 4404, 5505]))]
    targets = np.asarray([s["target"] for s in samples], dtype=np.float64)
    router_variants = set((cfg.get("method", {}) or {}).get("router_variants", router.DEPLOYABLE_ROUTERS))
    for variant, (preds, infos, _, _) in variant_outputs.items():
        if variant not in router_variants:
            continue
        action_list = [str(info.get("action", "no_correction")) for info in infos]
        rule_preds = np.asarray(preds, dtype=np.float64)
        rule_mse = array_mse(rule_preds, targets)
        rule_mae = array_mae(rule_preds, targets)
        for seed in seeds:
            random_preds, random_infos, _ = random_action_outputs(samples, action_context, action_list, seed)
            random_array = np.asarray(random_preds, dtype=np.float64)
            random_mse = array_mse(random_array, targets)
            random_mae = array_mae(random_array, targets)
            rows.append(
                {
                    "variant": variant,
                    "stress_type": stress_type,
                    "dataset": record.dataset,
                    "model": record.model,
                    "horizon": record.horizon,
                    "random_seed": seed,
                    "rule_mse": rule_mse,
                    "random_action_mse": random_mse,
                    "rule_minus_random_mse": rule_mse - random_mse,
                    "rule_beats_random_action_mse": bool(rule_mse < random_mse),
                    "rule_mae": rule_mae,
                    "random_action_mae": random_mae,
                    "rule_minus_random_mae": rule_mae - random_mae,
                    "rule_beats_random_action_mae": bool(rule_mae < random_mae),
                    "random_action_distribution": router.compact_distribution([str(info.get("action", "")) for info in random_infos]),
                }
            )
    return rows


def action_alignment_rows(record: ConfigRecord, stress_type: str, samples: List[dict], policy: Dict[str, object], infos: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    feature_names = list(policy["feature_names"])
    action_context = policy["action_context"]
    feature_rows = [router.extract_router_features(sample, action_context) for sample in samples]
    specs = [
        ("boundary_score", ["boundary_only", "dynamics_full"], "boundary_action_rate"),
        ("high_frequency_excess", ["median_smoothing", "ema_smoothing", "naive_smoothing", "boundary_then_ema", "boundary_then_median"], "smoothing_action_rate"),
        ("score", ["no_correction"], "no_correction_rate"),
    ]
    actions = [str(info.get("action", "")) for info in infos]
    for feature_name, action_set, metric_name in specs:
        values = np.asarray([float(row.get(feature_name, 0.0)) for row in feature_rows], dtype=np.float64)
        if values.size == 0:
            continue
        if feature_name == "score":
            order_values = -values
            bin_name = "low_risk_bin"
        else:
            order_values = values
            bin_name = f"{feature_name}_bin"
        quantiles = np.quantile(order_values, [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
        for idx, label in enumerate(["low", "mid", "high"]):
            if idx == 2:
                mask = (order_values >= quantiles[idx]) & (order_values <= quantiles[idx + 1])
            else:
                mask = (order_values >= quantiles[idx]) & (order_values < quantiles[idx + 1])
            selected = [actions[i] for i, keep in enumerate(mask) if bool(keep)]
            rows.append(
                {
                    "stress_type": stress_type,
                    "dataset": record.dataset,
                    "model": record.model,
                    "horizon": record.horizon,
                    "alignment_feature": feature_name,
                    "bin": label,
                    "bin_type": bin_name,
                    "n_samples": len(selected),
                    metric_name: router.action_rate(selected, action_set),
                    "action_distribution": router.compact_distribution(selected),
                }
            )
    return rows


def router_validation_rows(record: ConfigRecord, stress_type: str, policies: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for name, policy in policies.items():
        for action, value in sorted(policy.get("validation_action_mse", {}).items()):
            rows.append(
                {
                    "variant": name,
                    "stress_type": stress_type,
                    "dataset": record.dataset,
                    "model": record.model,
                    "horizon": record.horizon,
                    "action": action,
                    "validation_action_mse": value,
                    "validation_best_single_action": policy.get("validation_best_single_action", ""),
                    "validation_label_distribution": json.dumps(policy.get("validation_label_distribution", {}), sort_keys=True),
                    "test_threshold_leakage": False,
                }
            )
    return rows


def write_outputs(output_root: Path, run_root: Path, scope: str, candidate_id: str, cfg: Dict, rows: List[Dict[str, object]], config_records: List[Dict[str, object]], diagnostics: Dict[str, List[Dict[str, object]]]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows, diagnostics, str((cfg.get("method", {}) or {}).get("main_router", "harm_aware_router")))
    payload = {
        "candidate_id": candidate_id,
        "scope": scope,
        "configs": config_records,
        "rows": rows,
        "completed_configs": count_completed(config_records),
        "total_configs": len(config_records),
        "summary": summary,
    }
    write_csv(rows, run_root / "combined_metrics.csv")
    write_csv(summary["variant_summary"], run_root / "variant_summary.csv")
    (run_root / "combined_metrics.json").write_text(json.dumps(payload, indent=2, default=stage11.json_default), encoding="utf-8")
    (run_root / "combined_ablation_table.md").write_text(markdown_table(rows), encoding="utf-8")
    (run_root / "summary.md").write_text(table_summary(payload), encoding="utf-8")
    for name, diag_rows in diagnostics.items():
        write_csv(diag_rows, output_root / "diagnostics" / f"{scope}_{candidate_id}_{name}.csv")
    write_candidate_ledger(output_root / "candidate_ledger.csv", scope, payload)


def summarize_rows(rows: List[Dict[str, object]], diagnostics: Dict[str, List[Dict[str, object]]], main_router: str) -> Dict[str, object]:
    completed = [r for r in rows if r.get("status") == "completed"]
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in completed:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    by_key_variant = {(key_for(row), str(row["variant"])): row for row in completed}
    paired = diagnostics.get("router_paired_random_action", [])
    paired_by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in paired:
        paired_by_variant.setdefault(str(row["variant"]), []).append(row)
    variant_summary = []
    for variant, variant_rows in sorted(by_variant.items()):
        paired_rows = paired_by_variant.get(variant, [])
        wins = sum(1 for r in paired_rows if str(r.get("rule_beats_random_action_mse", "")).lower() == "true")
        beats_matched = compare_variant_count(variant_rows, by_key_variant, "matched_smoothing_control")
        beats_boundary = compare_variant_count(variant_rows, by_key_variant, "boundary_only")
        beats_val_best = compare_variant_count(variant_rows, by_key_variant, "validation_best_single_action")
        beats_random_metric = compare_variant_count(variant_rows, by_key_variant, "random_action_router")
        variant_summary.append(
            {
                "variant": variant,
                "mean_mse_delta_pct": mean([float(r["mse_delta_pct_vs_no_correction"]) for r in variant_rows]),
                "mean_mae_delta_pct": mean([float(r["mae_delta_pct_vs_no_correction"]) for r in variant_rows]),
                "improved_configs": sum(1 for r in variant_rows if float(r["mse_delta_pct_vs_no_correction"]) < 0.0),
                "beats_random_action_configs": config_level_random_action_wins(paired_rows) if paired_rows else beats_random_metric,
                "paired_rule_vs_random_win_rate": wins / len(paired_rows) if paired_rows else "",
                "beats_matched_smoothing_configs": beats_matched,
                "beats_boundary_only_configs": beats_boundary,
                "beats_validation_best_single_action_configs": beats_val_best,
                "max_mse_harm_pct": max([float(r["mse_delta_pct_vs_no_correction"]) for r in variant_rows], default=0.0),
                "max_mae_harm_pct": max([float(r["mae_delta_pct_vs_no_correction"]) for r in variant_rows], default=0.0),
                "mean_correction_rate": mean([float(r["correction_rate"]) for r in variant_rows]),
                "mean_latency_ms": mean([float(r["inference_latency_ms"]) for r in variant_rows]),
                "mean_action_entropy": mean([float(r.get("action_entropy", 0.0)) for r in variant_rows]),
                "dominant_action_rate_max": max_action_rate(variant_rows),
                "test_threshold_leakage": any(str(r.get("test_threshold_leakage")) != "False" for r in variant_rows),
            }
        )
    main = next((r for r in variant_summary if r["variant"] == main_router), {})
    return {
        "variant_summary": variant_summary,
        "main": main,
        "main_router": main_router,
        "stress_types": sorted(set(str(r.get("stress_type", "")) for r in completed)),
        "test_threshold_leakage": any(str(r.get("test_threshold_leakage")) != "False" for r in completed),
    }


def compare_variant_count(rows: List[Dict[str, object]], by_key_variant: Dict[Tuple[Tuple[str, str, str, str], str], Dict[str, object]], baseline_variant: str) -> int:
    count = 0
    for row in rows:
        base = by_key_variant.get((key_for(row), baseline_variant))
        if base and float(row["mse"]) < float(base["mse"]):
            count += 1
    return count


def config_level_random_action_wins(paired_rows: List[Dict[str, object]]) -> int:
    grouped: Dict[Tuple[str, str, str, str], List[bool]] = {}
    for row in paired_rows:
        key = (str(row["stress_type"]), str(row["dataset"]), str(row["model"]), str(row["horizon"]))
        grouped.setdefault(key, []).append(str(row.get("rule_beats_random_action_mse", "")).lower() == "true")
    return sum(1 for values in grouped.values() if sum(values) > 0.5 * len(values))


def max_action_rate(rows: List[Dict[str, object]]) -> float:
    max_rate = 0.0
    for row in rows:
        dist = parse_distribution(str(row.get("action_distribution", "")))
        if dist:
            max_rate = max(max_rate, max(dist.values()))
    return max_rate


def parse_distribution(text: str) -> Dict[str, float]:
    out = {}
    for part in text.split(";"):
        if not part or ":" not in part:
            continue
        key, value = part.rsplit(":", 1)
        try:
            out[key] = float(value)
        except ValueError:
            pass
    return out


def table_summary(payload: Dict[str, object]) -> str:
    s = payload["summary"]
    main = s.get("main", {})
    lines = [
        "# Stage 13 Adaptive Router Summary",
        "",
        f"- Candidate: `{payload['candidate_id']}`",
        f"- Scope: `{payload['scope']}`",
        f"- Completed configs: {payload['completed_configs']} / {payload['total_configs']}",
        f"- Stress types: `{', '.join(s.get('stress_types', []))}`",
        f"- Main router: `{s.get('main_router', '')}`",
        f"- Main mean MSE delta: {float(main.get('mean_mse_delta_pct', 0.0)):.6f}%",
        f"- Main improved configs: {main.get('improved_configs', '')}",
        f"- Main beats random action configs: {main.get('beats_random_action_configs', '')}",
        f"- Main paired rule-vs-random win rate: {main.get('paired_rule_vs_random_win_rate', '')}",
        f"- Main beats matched smoothing configs: {main.get('beats_matched_smoothing_configs', '')}",
        f"- Main beats boundary_only configs: {main.get('beats_boundary_only_configs', '')}",
        f"- Main dominant action max rate: {main.get('dominant_action_rate_max', '')}",
        f"- Test threshold leakage: {s.get('test_threshold_leakage', '')}",
        f"- Gate verdict: `{gate_verdict(payload['scope'], s)}`",
        "",
        "## Variant Summary",
        "",
    ]
    for row in s.get("variant_summary", []):
        lines.append(
            f"- `{row['variant']}`: mean MSE delta {float(row['mean_mse_delta_pct']):.6f}%, improved {row['improved_configs']}, "
            f"beats random action {row['beats_random_action_configs']}, paired win {row['paired_rule_vs_random_win_rate']}, "
            f"beats matched {row['beats_matched_smoothing_configs']}, beats boundary {row['beats_boundary_only_configs']}"
        )
    return "\n".join(lines) + "\n"


def gate_verdict(scope: str, summary: Dict[str, object]) -> str:
    main = summary.get("main", {})
    if summary.get("test_threshold_leakage"):
        return "fail_leakage"
    if not main:
        return "no_main"
    if safe_float(main.get("max_mse_harm_pct", 999.0), 999.0) > 3.0 or safe_float(main.get("max_mae_harm_pct", 999.0), 999.0) > 3.0:
        return "fail_harm"
    if scope == "clean_full":
        if safe_float(main.get("mean_mse_delta_pct", 999.0), 999.0) >= 0.0:
            return "fail_clean_mse"
        if safe_int(main.get("beats_random_action_configs", 0), 0) < 14 or safe_float(main.get("paired_rule_vs_random_win_rate", 0.0), 0.0) < 0.80:
            return "fail_random_action"
        if safe_int(main.get("beats_matched_smoothing_configs", 0), 0) < 10:
            return "fail_matched_smoothing"
        return "clean_router_gate_pass"
    if scope == "stress":
        return "stress_completed"
    return "completed"


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def markdown_table(rows: List[Dict[str, object]]) -> str:
    headers = ["stress_type", "dataset", "model", "horizon", "variant", "status", "mse", "mae", "mse_delta_pct_vs_no_correction", "correction_rate", "action_distribution", "inference_latency_ms", "blocker_reason"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def write_candidate_ledger(path: Path, scope: str, payload: Dict[str, object]) -> None:
    rows = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [r for r in rows if not (r.get("candidate_id") == payload["candidate_id"] and r.get("scope") == scope)]
    main = payload["summary"].get("main", {})
    rows.append(
        {
            "candidate_id": payload["candidate_id"],
            "scope": scope,
            "status": "completed" if payload["completed_configs"] == payload["total_configs"] else "blocked",
            "completed_configs": payload["completed_configs"],
            "total_configs": payload["total_configs"],
            "main_router": payload["summary"].get("main_router", ""),
            "main_mean_mse_delta_pct": main.get("mean_mse_delta_pct", ""),
            "main_improved_configs": main.get("improved_configs", ""),
            "main_beats_random_action_configs": main.get("beats_random_action_configs", ""),
            "paired_rule_win_rate": main.get("paired_rule_vs_random_win_rate", ""),
            "main_beats_matched_configs": main.get("beats_matched_smoothing_configs", ""),
            "main_beats_boundary_configs": main.get("beats_boundary_only_configs", ""),
            "dominant_action_rate_max": main.get("dominant_action_rate_max", ""),
            "max_mse_harm_pct": main.get("max_mse_harm_pct", ""),
            "max_mae_harm_pct": main.get("max_mae_harm_pct", ""),
            "test_threshold_leakage": payload["summary"].get("test_threshold_leakage", ""),
            "gate_verdict": gate_verdict(scope, payload["summary"]),
        }
    )
    write_csv(rows, path)


def blocked_row(candidate_id: str, record: ConfigRecord, stress_type: str, prediction_path: Path, run_dir: Path, reason: str) -> Dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "stress_type": stress_type,
        "dataset": record.dataset,
        "model": record.model,
        "horizon": record.horizon,
        "variant": "all",
        "status": "blocked",
        "mse": "",
        "mae": "",
        "mse_delta_pct_vs_no_correction": "",
        "mae_delta_pct_vs_no_correction": "",
        "correction_rate": "",
        "inference_latency_ms": "",
        "test_threshold_leakage": "",
        "prediction_path": str(prediction_path),
        "output_dir": str(run_dir),
        "blocker_reason": reason,
    }


def record_from_prediction_file(path: Path) -> ConfigRecord:
    samples = stage11.load_prediction_file(path)
    if not samples:
        return ConfigRecord("external", "external", 0)
    first = samples[0]
    return ConfigRecord(str(first.get("dataset", "external")), str(first.get("model", "external")), int(len(first.get("prediction", []))))


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


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


def count_completed(records: Iterable[Dict[str, object]]) -> int:
    return sum(1 for r in records if r.get("status") == "completed")


def mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return float(sum(values) / len(values)) if values else 0.0


def key_for(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    return (str(row.get("stress_type", "")), str(row.get("dataset", "")), str(row.get("model", "")), str(row.get("horizon", "")))


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
