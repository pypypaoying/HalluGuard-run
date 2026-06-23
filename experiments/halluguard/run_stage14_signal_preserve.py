"""Stage 14 signal-preserving HalluGuard router runner.

This runner keeps the Stage 13 evaluation contract while adding a component-wise
signal-support router. Validation rows fit every threshold/action policy; test
rows are only used after policies are frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

import evaluate_predictions as incumbent_eval
import halluguard_router as router
import halluguard_signal_preserve as signal
import run_stage11_dynamics as stage11
import run_stage13_adaptive_router as stage13
from halluguard_dynamics import array_mae, array_mse, metric_row


DATASETS = ["ETTm1", "ETTh1"]
MODELS = ["DLinear", "PatchTST"]
HORIZONS = [96, 192, 336, 720]
SMOKE_CONFIGS = [
    ("ETTm1", "DLinear", 96),
    ("ETTm1", "PatchTST", 720),
    ("ETTh1", "DLinear", 720),
    ("ETTh1", "PatchTST", 336),
]
DEFAULT_CONFIG = "experiments/halluguard/configs/halluguard_stage14_signal_preserve.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 14 signal-preserving HalluGuard router.")
    parser.add_argument("--scope", required=True, choices=["smoke", "clean_full", "stress", "external_batch"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--stage7-prediction-dir", default=None)
    parser.add_argument("--external-input-dir", default=None)
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_yaml(resolve_path(repo_root, Path(args.config)))
    candidate_id = str(cfg.get("candidate_id", "s14_signal_preserve_router"))
    output_root = resolve_path(repo_root, Path(args.output_root)) if args.output_root else resolve_path(
        repo_root, Path((cfg.get("outputs", {}) or {}).get("results_dir", "experiments/halluguard/results/stage14_signal_preserve"))
    )
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_dir = resolve_path(repo_root, Path(args.stage7_prediction_dir)) if args.stage7_prediction_dir else resolve_path(
        repo_root, Path((cfg.get("data", {}) or {}).get("stage7_prediction_dir", "experiments/halluguard/results/stage7_big_table/predictions"))
    )

    if args.scope == "external_batch":
        input_dir = resolve_path(repo_root, Path(args.external_input_dir)) if args.external_input_dir else resolve_path(
            repo_root, Path((cfg.get("data", {}) or {}).get("external_fixture_dir", "experiments/halluguard/results/stage12_external_batch/fixtures/external_batch_clean"))
        )
        files = sorted([p for p in input_dir.glob("*.jsonl") if p.name.lower() != "manifest.jsonl"] + [p for p in input_dir.glob("*.csv") if p.name.lower() != "manifest.csv"])
        if args.limit_files is not None:
            files = files[: int(args.limit_files)]
        scopes = [("external_batch", record_from_prediction_file(path), path) for path in files]
        run_root = output_root / "external_batch" / candidate_id
    else:
        records = [stage13.ConfigRecord(*c) for c in SMOKE_CONFIGS] if args.scope == "smoke" else [
            stage13.ConfigRecord(dataset, model, horizon)
            for dataset in DATASETS
            for model in MODELS
            for horizon in HORIZONS
        ]
        stress_types = ["clean"] if args.scope != "stress" else list((cfg.get("stress", {}) or {}).get("types", []))
        table_name = "smoke" if args.scope == "smoke" else ("clean_full_table" if args.scope == "clean_full" else "stress_table")
        run_root = output_root / table_name / candidate_id
        scopes = []
        for stress_type in stress_types:
            for record in records:
                source = prediction_dir / f"{record.tag}.jsonl"
                prediction_path = source if stress_type == "clean" else output_root / "stress_predictions" / stress_type / f"{record.tag}.jsonl"
                if stress_type != "clean":
                    stage11.write_stress_predictions(source, prediction_path, stress_type)
                scopes.append((stress_type, record, prediction_path))

    rows = []
    config_records = []
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
        print(json.dumps({"progress": "stage14_config_done", "scope": args.scope, "stress_type": stress_type, "config": record.tag, "status": config_record["status"], "completed_configs": stage13.count_completed(config_records), "total_seen": len(config_records)}), flush=True)

    write_outputs(output_root, run_root, args.scope, candidate_id, cfg, rows, config_records, diagnostics)
    print(json.dumps({"scope": args.scope, "candidate_id": candidate_id, "output_dir": str(run_root), "completed_configs": stage13.count_completed(config_records), "total_configs": len(config_records)}))


def run_one_config(repo_root: Path, cfg: Dict, candidate_id: str, record: stage13.ConfigRecord, prediction_path: Path, run_dir: Path, stress_type: str):
    try:
        samples = stage11.load_prediction_file(prediction_path)
        val_samples = [s for s in samples if s.get("split") == "val"]
        test_samples = [s for s in samples if s.get("split") == "test"]
        if not val_samples or not test_samples:
            raise ValueError(f"{prediction_path} must contain both val and test samples.")
        run_dir.mkdir(parents=True, exist_ok=True)
        metric_rows, diag_rows, payload = evaluate_signal_ablation_set(repo_root, cfg, candidate_id, record, stress_type, prediction_path, run_dir, val_samples, test_samples)
        stage11.write_run_outputs(run_dir, payload, metric_rows, diag_rows)
        return {"candidate_id": candidate_id, "stress_type": stress_type, "dataset": record.dataset, "model": record.model, "horizon": record.horizon, "status": "completed", "blocker_reason": ""}, metric_rows, diag_rows
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        return {"candidate_id": candidate_id, "stress_type": stress_type, "dataset": record.dataset, "model": record.model, "horizon": record.horizon, "status": "blocked", "blocker_reason": reason}, [stage13.blocked_row(candidate_id, record, stress_type, prediction_path, run_dir, reason)], {}


def evaluate_signal_ablation_set(
    repo_root: Path,
    cfg: Dict,
    candidate_id: str,
    record: stage13.ConfigRecord,
    stress_type: str,
    prediction_path: Path,
    run_dir: Path,
    val_samples: List[dict],
    test_samples: List[dict],
):
    method_cfg = cfg.get("method", {}) or {}
    signal_variants = list(method_cfg.get("signal_variants", signal.SIGNAL_VARIANTS))
    candidate_actions = list(method_cfg.get("candidate_actions", signal.DEFAULT_ACTIONS))
    main_router = str(method_cfg.get("main_router", "signal_preserve_router"))
    seed = int(cfg.get("seed", 23))

    prepared_signal = signal.prepare_signal_training(val_samples, cfg)
    signal_policies = {
        variant: signal.fit_signal_policy(val_samples, cfg, variant=variant, prepared=prepared_signal)
        for variant in signal_variants
    }
    if main_router not in signal_policies:
        raise ValueError(f"main_router={main_router} was not fit as a signal policy.")
    main_policy = signal_policies[main_router]
    action_context = main_policy["action_context"]
    stage13_policy = router.fit_router(val_samples, cfg, candidate_actions, router_type="rule_router")

    variant_outputs: Dict[str, Tuple[List[np.ndarray], List[Dict[str, object]], List[float], Optional[Dict[str, object]]]] = {}
    for action in candidate_actions:
        preds, infos, latencies = stage13.apply_action_to_samples(test_samples, action, action_context)
        variant_outputs[action] = (preds, infos, latencies, None)

    for variant, policy in signal_policies.items():
        preds, infos, latencies = signal.apply_signal_policy_to_samples(test_samples, policy)
        variant_outputs[variant] = (preds, infos, latencies, policy)

    s13_preds, s13_infos, s13_latencies = router.apply_router_to_samples(test_samples, stage13_policy)
    variant_outputs["stage13_rule_router"] = (s13_preds, s13_infos, s13_latencies, stage13_policy)

    main_preds, main_infos, _, _ = variant_outputs[main_router]
    main_actions = [str(info.get("action", "no_correction")) for info in main_infos]
    matched_preds, matched_infos, matched_latencies = signal.matched_sparse_smoothing_outputs(test_samples, action_context, main_actions, str(main_policy.get("smoothing_action", "median_smoothing")))
    variant_outputs["matched_smoothing_control"] = (matched_preds, matched_infos, matched_latencies, main_policy)

    random_preds, random_infos, random_latencies = signal.random_action_outputs(test_samples, action_context, main_actions, seed + 707)
    variant_outputs["random_action_router"] = (random_preds, random_infos, random_latencies, main_policy)

    val_best = str(main_policy.get("validation_best_single_action", "no_correction"))
    vb_preds, vb_infos, vb_latencies = stage13.apply_action_to_samples(test_samples, val_best, action_context)
    for info in vb_infos:
        info["action"] = val_best
    variant_outputs["validation_best_single_action"] = (vb_preds, vb_infos, vb_latencies, main_policy)

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

    oracle_preds, oracle_infos, oracle_latencies = router.oracle_test_ceiling(test_samples, action_context, candidate_actions)
    variant_outputs["oracle_test_ceiling"] = (oracle_preds, oracle_infos, oracle_latencies, main_policy)

    metric_rows = []
    for variant_name, (preds, infos, latencies, policy) in variant_outputs.items():
        if variant_name in signal_variants:
            row = signal.signal_metric_row(variant_name, test_samples, preds, infos, latencies, policy)
        elif variant_name in {"stage13_rule_router", "random_action_router", "validation_best_single_action", "oracle_test_ceiling"}:
            row = router.router_metric_row(variant_name, test_samples, preds, infos, latencies, policy)
        else:
            row = metric_row(variant_name, test_samples, preds, infos, latencies, policy)
            actions = [str(info.get("action", variant_name)) for info in infos]
            row.update({"action_distribution": router.compact_distribution(actions), "action_entropy": router.action_entropy(actions)})
        row.update({
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
            "parent_method": str(method_cfg.get("parent_method", "stage13_rule_router")),
            "test_threshold_leakage": "diagnostic_only" if variant_name == "oracle_test_ceiling" else False,
        })
        metric_rows.append(row)

    meta = {"stress_type": stress_type, "dataset": record.dataset, "model": record.model, "horizon": record.horizon}
    diagnostics = {
        "router_paired_random_action": paired_random_action_rows(record, stress_type, test_samples, variant_outputs, cfg),
        "signal_alignment": signal.signal_alignment_rows(test_samples, main_policy, main_infos, meta),
        "signal_validation": [row for policy in signal_policies.values() for row in signal.validation_policy_rows(policy, meta)],
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


def paired_random_action_rows(
    record: stage13.ConfigRecord,
    stress_type: str,
    samples: List[dict],
    variant_outputs: Dict[str, Tuple[List[np.ndarray], List[Dict[str, object]], List[float], Optional[Dict[str, object]]]],
    cfg: Dict,
) -> List[Dict[str, object]]:
    rows = []
    seeds = [int(v) for v in ((cfg.get("signal_preserve", {}) or {}).get("random_seeds", [1101, 2202, 3303, 4404, 5505]))]
    targets = np.asarray([s["target"] for s in samples], dtype=np.float64)
    variants = set((cfg.get("method", {}) or {}).get("signal_variants", signal.SIGNAL_VARIANTS)) | {"stage13_rule_router"}
    for variant, (preds, infos, _, policy) in variant_outputs.items():
        if variant not in variants or policy is None:
            continue
        action_context = policy.get("action_context", {})
        action_list = [str(info.get("action", "no_correction")) for info in infos]
        rule_preds = np.asarray(preds, dtype=np.float64)
        rule_mse = array_mse(rule_preds, targets)
        rule_mae = array_mae(rule_preds, targets)
        for seed in seeds:
            random_preds, random_infos, _ = signal.random_action_outputs(samples, action_context, action_list, seed)
            random_array = np.asarray(random_preds, dtype=np.float64)
            random_mse = array_mse(random_array, targets)
            random_mae = array_mae(random_array, targets)
            rows.append({
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
            })
    return rows


def write_outputs(output_root: Path, run_root: Path, scope: str, candidate_id: str, cfg: Dict, rows: List[Dict[str, object]], config_records: List[Dict[str, object]], diagnostics: Dict[str, List[Dict[str, object]]]) -> None:
    stage13.write_outputs(output_root, run_root, scope, candidate_id, cfg, rows, config_records, diagnostics)
    summary = summarize_model_groups(rows)
    stage13.write_csv(summary, run_root / "model_group_summary.csv")
    if rows:
        enrich_summary_markdown(run_root / "summary.md", scope, rows, summary, str((cfg.get("method", {}) or {}).get("main_router", "signal_preserve_router")))


def summarize_model_groups(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    completed = [r for r in rows if r.get("status") == "completed"]
    grouped: Dict[Tuple[str, str, str], List[Dict[str, object]]] = {}
    for row in completed:
        grouped.setdefault((str(row.get("stress_type", "")), str(row.get("variant", "")), str(row.get("model", ""))), []).append(row)
    out = []
    for (stress_type, variant, model), values in sorted(grouped.items()):
        out.append({
            "stress_type": stress_type,
            "variant": variant,
            "model": model,
            "configs": len(values),
            "mean_mse_delta_pct": mean([float(r["mse_delta_pct_vs_no_correction"]) for r in values]),
            "mean_mae_delta_pct": mean([float(r["mae_delta_pct_vs_no_correction"]) for r in values]),
            "harmed_configs": sum(1 for r in values if float(r["mse_delta_pct_vs_no_correction"]) > 0.0),
            "max_mse_harm_pct": max([float(r["mse_delta_pct_vs_no_correction"]) for r in values], default=0.0),
            "mean_correction_rate": mean([float(r["correction_rate"]) for r in values]),
            "dominant_action_rate_max": stage13.max_action_rate(values),
        })
    return out


def enrich_summary_markdown(path: Path, scope: str, rows: List[Dict[str, object]], model_summary: List[Dict[str, object]], main_variant: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else "# Stage 14 Signal-Preserving Router Summary\n"
    lines = [text.rstrip(), "", "## Stage 14 Model Groups", ""]
    for row in model_summary:
        if row["variant"] == main_variant:
            lines.append(
                f"- `{row['stress_type']}` / `{row['model']}`: mean MSE delta {float(row['mean_mse_delta_pct']):.6f}%, "
                f"harmed {row['harmed_configs']} / {row['configs']}, max harm {float(row['max_mse_harm_pct']):.6f}%, "
                f"dominant action max {float(row['dominant_action_rate_max']):.4f}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def record_from_prediction_file(path: Path) -> stage13.ConfigRecord:
    samples = stage11.load_prediction_file(path)
    if not samples:
        return stage13.ConfigRecord("external", "external", 0)
    first = samples[0]
    return stage13.ConfigRecord(str(first.get("dataset", "external")), str(first.get("model", "external")), int(len(first.get("prediction", []))))


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def mean(values: Sequence[float]) -> float:
    return float(sum(float(v) for v in values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
