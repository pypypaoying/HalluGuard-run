"""Run Stage 11 HalluGuard-Dynamics tables and external smoke tests."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import yaml

import evaluate_predictions as incumbent_eval
from correction import Thresholds, naive_smoothing
from halluguard_dynamics import (
    VariantSpec,
    array_mae,
    array_mse,
    apply_vector_predictions,
    correction_vector,
    default_variant_specs,
    fit_policy,
    matched_random_mask,
    median_smoothing,
    metric_row,
    score_sample as dynamics_score_sample,
    shuffled_score_mask,
    smoothing_predictions,
    trigger_mask_for_samples,
)
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
CANDIDATE_ID = "s11_halluguard_dynamics"
MAIN_VARIANT = "dynamics_full"


@dataclass(frozen=True)
class ConfigRecord:
    dataset: str
    model: str
    horizon: int

    @property
    def tag(self) -> str:
        return f"{self.dataset}_{self.model}_{self.horizon}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 11 HalluGuard-Dynamics runner.")
    parser.add_argument("--scope", required=True, choices=["smoke", "clean_full", "stress", "external_smoke", "external_eval"])
    parser.add_argument("--config", default="experiments/halluguard/configs/halluguard_stage11_dynamics.yaml")
    parser.add_argument("--stage7-dir", default="experiments/halluguard/results/stage7_big_table")
    parser.add_argument("--output-root", default="experiments/halluguard/results/stage11_dynamics")
    parser.add_argument("--input", type=Path, default=None, help="External JSONL/CSV prediction file for external_smoke.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_yaml(repo_root / args.config)
    stage7_dir = repo_root / args.stage7_dir
    output_root = repo_root / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if args.scope in {"external_smoke", "external_eval"}:
        if args.scope == "external_eval" and args.input is None:
            raise ValueError("--input is required when --scope external_eval.")
        prediction_path = args.input if args.input else write_external_fixture(stage7_dir, output_root)
        run_root = output_root / "external_smoke" / CANDIDATE_ID
        if args.scope == "external_eval":
            run_root = output_root / "external_eval" / CANDIDATE_ID
        records = [record_from_prediction_file(prediction_path)]
        scopes = [("external", records[0], prediction_path)]
    else:
        table_name = "smoke" if args.scope == "smoke" else ("clean_full_table" if args.scope == "clean_full" else "stress_table")
        run_root = output_root / table_name / CANDIDATE_ID
        base_records = [ConfigRecord(*c) for c in SMOKE_CONFIGS] if args.scope == "smoke" else [
            ConfigRecord(dataset, model, horizon)
            for dataset in DATASETS
            for model in MODELS
            for horizon in HORIZONS
        ]
        stress_types = ["clean"] if args.scope != "stress" else list((cfg.get("stress", {}) or {}).get("types", []))
        scopes = []
        for stress_type in stress_types:
            for record in base_records:
                source = stage7_dir / "predictions" / f"{record.tag}.jsonl"
                prediction_path = source if stress_type == "clean" else output_root / "stress_predictions" / stress_type / f"{record.tag}.jsonl"
                if stress_type != "clean":
                    write_stress_predictions(source, prediction_path, stress_type)
                scopes.append((stress_type, record, prediction_path))

    rows: List[Dict[str, object]] = []
    config_records: List[Dict[str, object]] = []
    diagnostics: Dict[str, List[Dict[str, object]]] = {}
    for stress_type, record, prediction_path in scopes:
        run_group = run_root / "runs" if stress_type in {"clean", "external"} else run_root / stress_type / "runs"
        run_dir = run_group / record.tag
        config_record, metric_rows, diag_rows = run_one_config(repo_root, cfg, record, prediction_path, run_dir, stress_type)
        config_records.append(config_record)
        rows.extend(metric_rows)
        for name, values in diag_rows.items():
            diagnostics.setdefault(name, []).extend(values)
        write_outputs(output_root, run_root, args.scope, rows, config_records, diagnostics)
        print(
            json.dumps(
                {
                    "progress": "stage11_config_done",
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

    write_outputs(output_root, run_root, args.scope, rows, config_records, diagnostics)
    print(json.dumps({"scope": args.scope, "candidate_id": CANDIDATE_ID, "output_dir": str(run_root), "completed_configs": count_completed(config_records), "total_configs": len(config_records)}))


def run_one_config(repo_root: Path, cfg: Dict, record: ConfigRecord, prediction_path: Path, run_dir: Path, stress_type: str):
    try:
        samples = load_prediction_file(prediction_path)
        val_samples = [s for s in samples if s.get("split") == "val"]
        test_samples = [s for s in samples if s.get("split") == "test"]
        if not val_samples or not test_samples:
            raise ValueError(f"{prediction_path} must contain both val and test samples.")
        run_dir.mkdir(parents=True, exist_ok=True)
        variants = default_variant_specs()
        policies = {spec.name: fit_policy(val_samples, cfg, spec) for spec in variants}
        full_policy = policies[MAIN_VARIANT]

        incumbent_cfg = load_yaml(repo_root / cfg["stage9_incumbent_config"])
        incumbent_thresholds, incumbent_policy = incumbent_eval.calibrate_evaluation_policy(val_samples, incumbent_cfg, "val")

        metric_rows, diag_rows, payload = evaluate_ablation_set(
            record=record,
            stress_type=stress_type,
            prediction_path=prediction_path,
            run_dir=run_dir,
            cfg=cfg,
            test_samples=test_samples,
            policies=policies,
            variants=variants,
            full_policy=full_policy,
            incumbent_cfg=incumbent_cfg,
            incumbent_thresholds=incumbent_thresholds,
            incumbent_policy=incumbent_policy,
        )
        write_run_outputs(run_dir, payload, metric_rows, diag_rows)
        config_record = {
            "candidate_id": CANDIDATE_ID,
            "stress_type": stress_type,
            "dataset": record.dataset,
            "model": record.model,
            "horizon": record.horizon,
            "status": "completed",
            "blocker_reason": "",
        }
        return config_record, metric_rows, diag_rows
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        config_record = {
            "candidate_id": CANDIDATE_ID,
            "stress_type": stress_type,
            "dataset": record.dataset,
            "model": record.model,
            "horizon": record.horizon,
            "status": "blocked",
            "blocker_reason": reason,
        }
        return config_record, [blocked_row(record, stress_type, prediction_path, run_dir, reason)], {}


def evaluate_ablation_set(
    record: ConfigRecord,
    stress_type: str,
    prediction_path: Path,
    run_dir: Path,
    cfg: Dict,
    test_samples: List[dict],
    policies: Dict[str, Dict[str, object]],
    variants: List[VariantSpec],
    full_policy: Dict[str, object],
    incumbent_cfg: Dict,
    incumbent_thresholds: Thresholds,
    incumbent_policy: Dict[str, object],
) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, object]]], Dict[str, object]]:
    variant_outputs = {}
    full_mask = trigger_mask_for_samples(test_samples, full_policy)
    seed = int(cfg.get("seed", 23))

    variant_outputs["no_correction"] = (
        [np.asarray(s["prediction"], dtype=np.float64) for s in test_samples],
        [{"trigger": 0.0, "changed": 0.0} for _ in test_samples],
        [0.0 for _ in test_samples],
        None,
    )
    for smoothing_name in ["naive_smoothing", "ema_smoothing", "median_smoothing"]:
        preds, infos, latencies = smoothing_predictions(test_samples, smoothing_name, cfg)
        variant_outputs[smoothing_name] = (preds, infos, latencies, None)

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
        [{"trigger": float(bool(i.get("policy_rule_hallucination", 0.0))), "changed": float(bool(i.get("changed", 0.0)))} for i in inc_infos],
        inc_latencies,
        None,
    )

    # Controls tied to the full HalluGuard-Dynamics trigger.
    matched_predictions = []
    matched_infos = []
    for idx, sample in enumerate(test_samples):
        if bool(full_mask[idx]):
            pred = naive_smoothing(sample["prediction"], int((cfg.get("policy", {}) or {}).get("smoothing_window", 5)))
        else:
            pred = np.asarray(sample["prediction"], dtype=np.float64)
        matched_predictions.append(pred)
        matched_infos.append({"trigger": float(bool(full_mask[idx])), "changed": float(np.max(np.abs(pred - np.asarray(sample["prediction"], dtype=np.float64))) > 1e-10)})
    variant_outputs["matched_smoothing_control"] = (matched_predictions, matched_infos, [0.0 for _ in test_samples], full_policy)

    random_mask = matched_random_mask(len(test_samples), int(full_mask.sum()), seed + 404)
    random_preds, random_infos, random_latencies = apply_vector_predictions(test_samples, full_policy, random_mask, config=cfg)
    variant_outputs["random_trigger"] = (random_preds, random_infos, random_latencies, full_policy)

    shuffled_mask = shuffled_score_mask(test_samples, full_policy, seed + 808)
    shuffled_preds, shuffled_infos, shuffled_latencies = apply_vector_predictions(test_samples, full_policy, shuffled_mask, config=cfg)
    variant_outputs["shuffled_score_correction"] = (shuffled_preds, shuffled_infos, shuffled_latencies, full_policy)

    trigger_only_infos = []
    for idx, sample in enumerate(test_samples):
        score = dynamics_score_sample(sample["context"], sample["prediction"], full_policy)
        trigger_only_infos.append({**score, "trigger": float(bool(full_mask[idx])), "changed": 0.0})
    variant_outputs["trigger_only_no_correction"] = (
        [np.asarray(s["prediction"], dtype=np.float64) for s in test_samples],
        trigger_only_infos,
        [0.0 for _ in test_samples],
        full_policy,
    )

    for spec in variants:
        policy = policies[spec.name]
        mask = trigger_mask_for_samples(test_samples, policy)
        preds, infos, latencies = apply_vector_predictions(test_samples, policy, mask, config=cfg)
        variant_outputs[spec.name] = (preds, infos, latencies, policy)

    metric_rows = []
    for variant_name, (preds, infos, latencies, policy) in variant_outputs.items():
        row = metric_row(variant_name, test_samples, preds, infos, latencies, policy)
        row.update(
            {
                "candidate_id": CANDIDATE_ID,
                "stress_type": stress_type,
                "dataset": record.dataset,
                "model": record.model,
                "horizon": record.horizon,
                "status": "completed",
                "prediction_path": str(prediction_path),
                "output_dir": str(run_dir),
                "blocker_reason": "",
            }
        )
        metric_rows.append(row)

    diagnostics = {
        "variant_paired_random": paired_random_rows(record, stress_type, test_samples, policies, variants, cfg),
        "score_components": score_component_rows(record, stress_type, test_samples, full_policy),
        "validation_policy": validation_policy_rows(record, stress_type, policies),
    }
    payload = {
        "candidate_id": CANDIDATE_ID,
        "stress_type": stress_type,
        "dataset": record.dataset,
        "model": record.model,
        "horizon": record.horizon,
        "input_path": str(prediction_path),
        "threshold_source_split": "val",
        "evaluation_split": "test",
        "test_threshold_leakage": False,
        "main_variant": MAIN_VARIANT,
        "main_ablation": metric_rows,
        "diagnostics": diagnostics,
    }
    return metric_rows, diagnostics, payload


def paired_random_rows(record: ConfigRecord, stress_type: str, samples: List[dict], policies: Dict[str, Dict[str, object]], variants: List[VariantSpec], cfg: Dict) -> List[Dict[str, object]]:
    rows = []
    seeds = [int(v) for v in ((cfg.get("policy", {}) or {}).get("random_seeds", [1101, 2202, 3303, 4404, 5505]))]
    base_predictions = np.asarray([s["prediction"] for s in samples], dtype=np.float64)
    targets = np.asarray([s["target"] for s in samples], dtype=np.float64)
    for spec in variants:
        policy = policies[spec.name]
        mask = trigger_mask_for_samples(samples, policy)
        vectors = np.asarray(
            [correction_vector(s["context"], s["prediction"], tuple(policy.get("repair_components", ("boundary", "first_diff"))), cfg) for s in samples],
            dtype=np.float64,
        )
        strength = float(policy.get("correction_strength", 0.0))
        rule_preds = base_predictions.copy()
        rule_preds[mask] = rule_preds[mask] + strength * vectors[mask]
        rule_mse = array_mse(rule_preds, targets)
        rule_mae = array_mae(rule_preds, targets)
        for seed in seeds:
            random_mask = matched_random_mask(len(samples), int(mask.sum()), seed)
            random_preds = base_predictions.copy()
            random_preds[random_mask] = random_preds[random_mask] + strength * vectors[random_mask]
            random_mse = array_mse(random_preds, targets)
            random_mae = array_mae(random_preds, targets)
            rows.append(
                {
                    "candidate_id": CANDIDATE_ID,
                    "variant": spec.name,
                    "stress_type": stress_type,
                    "dataset": record.dataset,
                    "model": record.model,
                    "horizon": record.horizon,
                    "random_seed": seed,
                    "rule_mse": rule_mse,
                    "random_mse": random_mse,
                    "rule_minus_random_mse": rule_mse - random_mse,
                    "rule_beats_random_mse": bool(rule_mse < random_mse),
                    "rule_mae": rule_mae,
                    "random_mae": random_mae,
                    "rule_minus_random_mae": rule_mae - random_mae,
                    "rule_beats_random_mae": bool(rule_mae < random_mae),
                    "random_correction_rate": float(random_mask.mean()) if random_mask.size else 0.0,
                }
            )
    return rows


def score_component_rows(record: ConfigRecord, stress_type: str, samples: List[dict], policy: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for sample in samples:
        score = dynamics_score_sample(sample["context"], sample["prediction"], policy)
        rows.append(
            {
                "candidate_id": CANDIDATE_ID,
                "stress_type": stress_type,
                "dataset": record.dataset,
                "model": record.model,
                "horizon": record.horizon,
                "sample_id": sample.get("sample_id", ""),
                "score": score["score"],
                "boundary_score": score["boundary_score"],
                "first_diff_score": score["first_diff_score"],
                "curvature_score": score["curvature_score"],
                "no_correction_mse": mse(sample["prediction"], sample["target"]),
            }
        )
    return rows


def validation_policy_rows(record: ConfigRecord, stress_type: str, policies: Dict[str, Dict[str, object]]) -> List[Dict[str, object]]:
    rows = []
    for variant, policy in policies.items():
        rows.append(
            {
                "candidate_id": CANDIDATE_ID,
                "variant": variant,
                "stress_type": stress_type,
                "dataset": record.dataset,
                "model": record.model,
                "horizon": record.horizon,
                "score_components": ",".join(policy.get("score_components", [])),
                "repair_components": ",".join(policy.get("repair_components", [])),
                "trigger_quantile": policy.get("trigger_quantile", ""),
                "correction_strength": policy.get("correction_strength", ""),
                "validation_mse_delta_pct": policy.get("validation_mse_delta_pct", ""),
                "validation_random_advantage_mse": policy.get("validation_random_advantage_mse", ""),
                "validation_matched_advantage_mse": policy.get("validation_matched_advantage_mse", ""),
                "validation_trigger_rate": policy.get("validation_trigger_rate", ""),
            }
        )
    return rows


def write_stress_predictions(source_path: Path, output_path: Path, stress_type: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            sample = json.loads(line)
            pred = np.asarray(sample["prediction"], dtype=np.float64)
            ctx = np.asarray(sample["context"], dtype=np.float64)
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
                perturbation = sign * 0.65 * scale * decay
            elif stress_type == "variance_shift":
                perturbation = 0.30 * (pred - float(pred.mean()))
            elif stress_type == "slope_break":
                ramp = np.maximum(0.0, np.linspace(-0.5, 0.5, pred.size))
                perturbation = sign * 0.70 * scale * ramp
            elif stress_type == "delayed_level_shift":
                step = np.zeros(pred.size, dtype=np.float64)
                step[int(0.35 * pred.size) :] = 1.0
                perturbation = sign * 0.45 * scale * step
            else:
                raise ValueError(f"Unknown stress type: {stress_type}")
            sample["prediction"] = (pred + perturbation).astype(float).tolist()
            sample["sample_id"] = f"{sample['sample_id']}::{stress_type}"
            sample["stress_type"] = stress_type
            sample["stress_only"] = True
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")


def write_external_fixture(stage7_dir: Path, output_root: Path) -> Path:
    source = stage7_dir / "predictions" / "ETTm1_DLinear_192.jsonl"
    out = output_root / "external_smoke" / "fixtures" / "external_predictions_smoke.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    val_count = 0
    test_count = 0
    with source.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            sample = json.loads(line)
            if sample.get("split") == "val" and val_count < 16:
                val_count += 1
            elif sample.get("split") == "test" and test_count < 16:
                test_count += 1
            else:
                continue
            sample["sample_id"] = f"external_smoke_{sample['split']}_{val_count if sample['split']=='val' else test_count:03d}"
            f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
            if val_count >= 16 and test_count >= 16:
                break
    return out


def record_from_prediction_file(path: Path) -> ConfigRecord:
    samples = load_prediction_file(path)
    if not samples:
        return ConfigRecord("external", "external", 0)
    first = samples[0]
    horizon = len(first.get("prediction", []))
    return ConfigRecord(str(first.get("dataset", "external")), str(first.get("model", "external")), int(horizon))


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prediction_file(path: Path) -> List[dict]:
    if path.suffix.lower() == ".jsonl":
        samples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                normalize_sample(sample)
                samples.append(sample)
        return samples
    if path.suffix.lower() == ".csv":
        samples = []
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                sample = dict(row)
                for key in ["context", "prediction", "target"]:
                    sample[key] = json.loads(sample[key])
                normalize_sample(sample)
                samples.append(sample)
        return samples
    raise ValueError(f"Unsupported prediction file extension: {path.suffix}")


def normalize_sample(sample: dict) -> None:
    for key in ["context", "prediction", "target"]:
        sample[key] = [float(v) for v in sample[key]]


def write_run_outputs(run_dir: Path, payload: Dict[str, object], rows: List[Dict[str, object]], diagnostics: Dict[str, List[Dict[str, object]]]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    write_csv(rows, run_dir / "metrics.csv")
    (run_dir / "ablation_table.md").write_text(markdown_table(rows), encoding="utf-8")
    (run_dir / "summary.md").write_text(run_summary(payload, rows), encoding="utf-8")
    diag_dir = run_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    for name, diag_rows in diagnostics.items():
        write_csv(diag_rows, diag_dir / f"{name}.csv")


def write_outputs(output_root: Path, run_root: Path, scope: str, rows: List[Dict[str, object]], config_records: List[Dict[str, object]], diagnostics: Dict[str, List[Dict[str, object]]]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    summary = summarize_rows(rows, diagnostics)
    payload = {
        "candidate_id": CANDIDATE_ID,
        "scope": scope,
        "configs": config_records,
        "rows": rows,
        "completed_configs": count_completed(config_records),
        "total_configs": len(config_records),
        "summary": summary,
    }
    write_csv(rows, run_root / "combined_metrics.csv")
    write_csv(summary["variant_summary"], run_root / "variant_summary.csv")
    (run_root / "combined_metrics.json").write_text(json.dumps(payload, indent=2, default=json_default), encoding="utf-8")
    (run_root / "combined_ablation_table.md").write_text(markdown_table(rows), encoding="utf-8")
    (run_root / "summary.md").write_text(table_summary(payload), encoding="utf-8")
    for name, diag_rows in diagnostics.items():
        write_csv(diag_rows, output_root / "diagnostics" / f"{scope}_{CANDIDATE_ID}_{name}.csv")
    write_candidate_ledger(output_root / "candidate_ledger.csv", scope, payload)


def summarize_rows(rows: List[Dict[str, object]], diagnostics: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    completed = [r for r in rows if r.get("status") == "completed"]
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in completed:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    matched_by_key = {
        key_for(row): row for row in completed if row.get("variant") == "matched_smoothing_control"
    }
    paired = diagnostics.get("variant_paired_random", [])
    paired_by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in paired:
        paired_by_variant.setdefault(str(row["variant"]), []).append(row)
    variant_summary = []
    for variant, variant_rows in sorted(by_variant.items()):
        if not variant_rows:
            continue
        beats_matched = 0
        for row in variant_rows:
            matched = matched_by_key.get(key_for(row))
            if matched and float(row["mse"]) < float(matched["mse"]):
                beats_matched += 1
        paired_rows = paired_by_variant.get(variant, [])
        wins = sum(1 for r in paired_rows if str(r.get("rule_beats_random_mse", "")).lower() == "true")
        variant_summary.append(
            {
                "variant": variant,
                "mean_mse_delta_pct": mean([float(r["mse_delta_pct_vs_no_correction"]) for r in variant_rows]),
                "mean_mae_delta_pct": mean([float(r["mae_delta_pct_vs_no_correction"]) for r in variant_rows]),
                "improved_configs": sum(1 for r in variant_rows if float(r["mse_delta_pct_vs_no_correction"]) < 0.0),
                "beats_random_configs": config_level_rule_wins(paired_rows),
                "paired_rule_vs_random_win_rate": wins / len(paired_rows) if paired_rows else "",
                "beats_matched_smoothing_configs": beats_matched,
                "max_mse_harm_pct": max([float(r["mse_delta_pct_vs_no_correction"]) for r in variant_rows], default=0.0),
                "max_mae_harm_pct": max([float(r["mae_delta_pct_vs_no_correction"]) for r in variant_rows], default=0.0),
                "mean_correction_rate": mean([float(r["correction_rate"]) for r in variant_rows]),
                "mean_latency_ms": mean([float(r["inference_latency_ms"]) for r in variant_rows]),
                "test_threshold_leakage": any(str(r.get("test_threshold_leakage")) != "False" for r in variant_rows),
            }
        )
    main_summary = next((r for r in variant_summary if r["variant"] == MAIN_VARIANT), {})
    return {
        "variant_summary": variant_summary,
        "main": main_summary,
        "stress_types": sorted(set(str(r.get("stress_type", "")) for r in completed)),
        "test_threshold_leakage": any(str(r.get("test_threshold_leakage")) != "False" for r in completed),
    }


def config_level_rule_wins(paired_rows: List[Dict[str, object]]) -> int:
    by_key: Dict[Tuple[str, str, str, str], List[bool]] = {}
    for row in paired_rows:
        key = (str(row["stress_type"]), str(row["dataset"]), str(row["model"]), str(row["horizon"]))
        by_key.setdefault(key, []).append(str(row.get("rule_beats_random_mse", "")).lower() == "true")
    return sum(1 for wins in by_key.values() if sum(wins) > 0.5 * len(wins))


def key_for(row: Dict[str, object]) -> Tuple[str, str, str, str]:
    return (str(row.get("stress_type", "")), str(row.get("dataset", "")), str(row.get("model", "")), str(row.get("horizon", "")))


def write_candidate_ledger(path: Path, scope: str, payload: Dict[str, object]) -> None:
    existing = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
    existing = [r for r in existing if not (r.get("candidate_id") == CANDIDATE_ID and r.get("scope") == scope)]
    main = payload["summary"].get("main", {})
    existing.append(
        {
            "candidate_id": CANDIDATE_ID,
            "scope": scope,
            "status": "completed" if payload["completed_configs"] == payload["total_configs"] else "blocked",
            "completed_configs": payload["completed_configs"],
            "total_configs": payload["total_configs"],
            "main_mean_mse_delta_pct": main.get("mean_mse_delta_pct", ""),
            "main_improved_configs": main.get("improved_configs", ""),
            "main_beats_random_configs": main.get("beats_random_configs", ""),
            "paired_rule_win_rate": main.get("paired_rule_vs_random_win_rate", ""),
            "main_beats_matched_configs": main.get("beats_matched_smoothing_configs", ""),
            "matched_smoothing_mean_mse_delta_pct": variant_metric(payload, "matched_smoothing_control", "mean_mse_delta_pct"),
            "naive_smoothing_mean_mse_delta_pct": variant_metric(payload, "naive_smoothing", "mean_mse_delta_pct"),
            "stage9_incumbent_mean_mse_delta_pct": variant_metric(payload, "stage9_incumbent", "mean_mse_delta_pct"),
            "max_mse_harm_pct": main.get("max_mse_harm_pct", ""),
            "max_mae_harm_pct": main.get("max_mae_harm_pct", ""),
            "test_threshold_leakage": payload["summary"].get("test_threshold_leakage", ""),
            "gate_verdict": gate_verdict(scope, payload["summary"]),
        }
    )
    write_csv(existing, path)


def variant_metric(payload: Dict[str, object], variant: str, metric: str) -> object:
    for row in payload["summary"].get("variant_summary", []):
        if row.get("variant") == variant:
            return row.get(metric, "")
    return ""


def gate_verdict(scope: str, summary: Dict[str, object]) -> str:
    main = summary.get("main", {})
    if summary.get("test_threshold_leakage"):
        return "fail_leakage"
    if not main:
        return "no_main"
    if float(main.get("max_mse_harm_pct", 999.0)) > 3.0 or float(main.get("max_mae_harm_pct", 999.0)) > 3.0:
        return "fail_harm"
    if scope == "clean_full":
        if float(main.get("mean_mse_delta_pct", 999.0)) >= 0.0:
            return "fail_clean_mse"
        if int(main.get("improved_configs", 0)) < 14:
            return "fail_improved_configs"
        if int(main.get("beats_random_configs", 0)) < 14 or float(main.get("paired_rule_vs_random_win_rate", 0.0)) < 0.80:
            return "fail_rule_random"
        if int(main.get("beats_matched_smoothing_configs", 0)) < 10:
            return "fail_matched_smoothing"
        return "clean_gate_pass"
    if scope == "stress":
        return "stress_completed"
    return "completed"


def markdown_table(rows: List[Dict[str, object]]) -> str:
    headers = ["stress_type", "dataset", "model", "horizon", "variant", "status", "mse", "mae", "mse_delta_pct_vs_no_correction", "correction_rate", "inference_latency_ms", "blocker_reason"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def run_summary(payload: Dict[str, object], rows: List[Dict[str, object]]) -> str:
    by_variant = {r["variant"]: r for r in rows}
    main = by_variant.get(MAIN_VARIANT, {})
    matched = by_variant.get("matched_smoothing_control", {})
    naive = by_variant.get("naive_smoothing", {})
    return f"""# Stage 11 Config Summary

- Candidate: `{payload['candidate_id']}`
- Stress type: `{payload['stress_type']}`
- Dataset/model/horizon: `{payload['dataset']} / {payload['model']} / {payload['horizon']}`
- Threshold source split: `val`
- Evaluation split: `test`
- Test threshold leakage: `False`

## Headline

- HalluGuard-Dynamics MSE delta: {float(main.get('mse_delta_pct_vs_no_correction', 0.0)):.6f}%
- matched_smoothing_control MSE delta: {float(matched.get('mse_delta_pct_vs_no_correction', 0.0)):.6f}%
- naive_smoothing MSE delta: {float(naive.get('mse_delta_pct_vs_no_correction', 0.0)):.6f}%

## Ablation

{markdown_table(rows)}
"""


def table_summary(payload: Dict[str, object]) -> str:
    s = payload["summary"]
    main = s.get("main", {})
    lines = [
        "# Stage 11 Table Summary",
        "",
        f"- Candidate: `{payload['candidate_id']}`",
        f"- Scope: `{payload['scope']}`",
        f"- Completed configs: {payload['completed_configs']} / {payload['total_configs']}",
        f"- Stress types: `{', '.join(s.get('stress_types', []))}`",
        f"- Main variant: `{MAIN_VARIANT}`",
        f"- Main mean MSE delta: {float(main.get('mean_mse_delta_pct', 0.0)):.6f}%",
        f"- Main improved configs: {main.get('improved_configs', '')}",
        f"- Main beats random configs: {main.get('beats_random_configs', '')}",
        f"- Main paired rule-vs-random win rate: {main.get('paired_rule_vs_random_win_rate', '')}",
        f"- Main beats matched smoothing configs: {main.get('beats_matched_smoothing_configs', '')}",
        f"- Main max MSE harm: {float(main.get('max_mse_harm_pct', 0.0)):.6f}%",
        f"- Main max MAE harm: {float(main.get('max_mae_harm_pct', 0.0)):.6f}%",
        f"- Test threshold leakage: {s.get('test_threshold_leakage', '')}",
        f"- Gate verdict: `{gate_verdict(payload['scope'], s)}`",
        "",
        "## Variant Summary",
        "",
    ]
    for row in s.get("variant_summary", []):
        lines.append(
            f"- `{row['variant']}`: mean MSE delta {float(row['mean_mse_delta_pct']):.6f}%, improved {row['improved_configs']}, "
            f"beats random {row['beats_random_configs']}, paired win {row['paired_rule_vs_random_win_rate']}, "
            f"beats matched {row['beats_matched_smoothing_configs']}"
        )
    return "\n".join(lines) + "\n"


def blocked_row(record: ConfigRecord, stress_type: str, prediction_path: Path, run_dir: Path, reason: str) -> Dict[str, object]:
    return {
        "candidate_id": CANDIDATE_ID,
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


def stable_hash(text: str) -> int:
    value = 2166136261
    for ch in text:
        value ^= ord(ch)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def count_completed(records: Iterable[Dict[str, object]]) -> int:
    return sum(1 for r in records if r.get("status") == "completed")


def mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return float(sum(values) / len(values)) if values else 0.0


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()
