"""Run Stage 9 mechanism-separation clean and stress tables."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


DATASETS = ["ETTm1", "ETTh1"]
MODELS = ["DLinear", "PatchTST"]
HORIZONS = [96, 192, 336, 720]
SMOKE_CONFIGS = [
    ("ETTm1", "DLinear", 192),
    ("ETTm1", "PatchTST", 720),
    ("ETTh1", "DLinear", 336),
    ("ETTh1", "PatchTST", 720),
]
STRESS_TYPES = ["trend_drift", "high_frequency_perturbation"]


@dataclass(frozen=True)
class Config:
    dataset: str
    model: str
    horizon: int

    @property
    def tag(self) -> str:
        return f"{self.dataset}_{self.model}_{self.horizon}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 9 HalluGuard mechanism-separation tables.")
    parser.add_argument("--scope", required=True, choices=["smoke", "clean_full", "stress"])
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage7-dir", default="experiments/halluguard/results/stage7_big_table")
    parser.add_argument("--output-root", default="experiments/halluguard/results/stage9_mechanism_separation")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
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
    stress_values = ["clean"] if args.scope != "stress" else STRESS_TYPES
    for stress_type in stress_values:
        for cfg in configs:
            record, metric_rows, diag_rows = run_config(repo_root, stage7_dir, output_root, run_root, cfg, args, stress_type)
            config_records.append(record)
            rows.extend(metric_rows)
            for name, values in diag_rows.items():
                diagnostics.setdefault(name, []).extend(values)
            write_outputs(output_root, run_root, args.candidate_id, args.scope, rows, config_records, diagnostics)

    write_outputs(output_root, run_root, args.candidate_id, args.scope, rows, config_records, diagnostics)
    print(json.dumps({"scope": args.scope, "candidate_id": args.candidate_id, "output_dir": str(run_root), "completed_configs": count_completed(config_records), "total_configs": len(config_records)}))


def run_config(
    repo_root: Path,
    stage7_dir: Path,
    output_root: Path,
    run_root: Path,
    cfg: Config,
    args: argparse.Namespace,
    stress_type: str,
):
    source_prediction_path = stage7_dir / "predictions" / f"{cfg.tag}.jsonl"
    if stress_type == "clean":
        prediction_path = source_prediction_path
        run_dir = run_root / "runs" / cfg.tag
    else:
        prediction_path = output_root / "stress_predictions" / stress_type / f"{cfg.tag}.jsonl"
        run_dir = run_root / stress_type / "runs" / cfg.tag
    evaluator = repo_root / "experiments" / "halluguard" / "evaluate_predictions.py"
    try:
        if not source_prediction_path.exists():
            raise FileNotFoundError(f"Missing Stage 7 prediction file: {source_prediction_path}")
        if stress_type != "clean":
            write_stress_predictions(source_prediction_path, prediction_path, stress_type)
        run_dir.mkdir(parents=True, exist_ok=True)
        eval_cmd = [
            sys.executable,
            str(evaluator),
            "--config",
            str(repo_root / args.config),
            "--input",
            str(prediction_path),
            "--calibration-split",
            "val",
            "--split",
            "test",
            "--output-dir",
            str(run_dir),
        ]
        subprocess.run(eval_cmd, cwd=str(repo_root), check=True)
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        metric_rows = completed_rows(cfg, args.candidate_id, stress_type, prediction_path, run_dir, metrics)
        diag_rows = read_diagnostics(cfg, args.candidate_id, stress_type, run_dir / "diagnostics")
        record = {
            "candidate_id": args.candidate_id,
            "dataset": cfg.dataset,
            "model": cfg.model,
            "horizon": cfg.horizon,
            "stress_type": stress_type,
            "status": "completed",
            "blocker_reason": "",
        }
        return record, metric_rows, diag_rows
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        record = {
            "candidate_id": args.candidate_id,
            "dataset": cfg.dataset,
            "model": cfg.model,
            "horizon": cfg.horizon,
            "stress_type": stress_type,
            "status": "blocked",
            "blocker_reason": reason,
        }
        return record, [blocked_row(cfg, args.candidate_id, stress_type, prediction_path, run_dir, reason)], {}


def write_stress_predictions(source_path: Path, output_path: Path, stress_type: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8") as f_in, output_path.open("w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
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
                alternating = np.sin(np.arange(pred.size, dtype=np.float64) * np.pi * 0.85 + phase)
                perturbation = 0.28 * scale * alternating
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


def completed_rows(cfg: Config, candidate_id: str, stress_type: str, prediction_path: Path, run_dir: Path, metrics: Dict) -> List[Dict[str, object]]:
    rows = []
    no = next(m for m in metrics["main_ablation"] if m["variant"] == "no_correction")
    policy = metrics.get("validation_policy", {}) or {}
    for metric in metrics["main_ablation"]:
        rows.append(
            {
                "candidate_id": candidate_id,
                "stress_type": stress_type,
                "dataset": cfg.dataset,
                "model": cfg.model,
                "horizon": cfg.horizon,
                "variant": metric["variant"],
                "status": "completed",
                "mse": metric["mse"],
                "mae": metric["mae"],
                "mse_delta_pct_vs_no_correction": pct_delta(metric["mse"], no["mse"]),
                "mae_delta_pct_vs_no_correction": pct_delta(metric["mae"], no["mae"]),
                "hallucination_rate": metric["hallucination_rate"],
                "trend_violation_rate": metric["trend_violation_rate"],
                "freq_violation_rate": metric["freq_violation_rate"],
                "spectral_consistency": metric["spectral_consistency"],
                "turning_point_false_correction_rate": metric["turning_point_false_correction_rate"],
                "correction_rate": metric["correction_rate"],
                "inference_latency_ms": metric["inference_latency_ms"],
                "threshold_quantile": metric["threshold_quantile"],
                "lambda_trend": metric["lambda_trend"],
                "lambda_freq": metric["lambda_freq"],
                "policy_trend_margin_factor": policy.get("trend_margin_factor", ""),
                "policy_freq_margin_factor": policy.get("freq_margin_factor", ""),
                "policy_combined_val_delta_pct": policy.get("combined_val_delta_pct", ""),
                "policy_random_val_advantage": policy.get("random_separation_val_advantage", ""),
                "freq_score_mode": policy.get("freq_score_mode", ""),
                "test_threshold_leakage": metrics["test_threshold_leakage"],
                "prediction_path": str(prediction_path),
                "output_dir": str(run_dir),
                "blocker_reason": "",
            }
        )
    return rows


def blocked_row(cfg: Config, candidate_id: str, stress_type: str, prediction_path: Path, run_dir: Path, reason: str) -> Dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "stress_type": stress_type,
        "dataset": cfg.dataset,
        "model": cfg.model,
        "horizon": cfg.horizon,
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
        "lambda_trend": "",
        "lambda_freq": "",
        "policy_trend_margin_factor": "",
        "policy_freq_margin_factor": "",
        "policy_combined_val_delta_pct": "",
        "policy_random_val_advantage": "",
        "freq_score_mode": "",
        "test_threshold_leakage": "",
        "prediction_path": str(prediction_path),
        "output_dir": str(run_dir),
        "blocker_reason": reason,
    }


def read_diagnostics(cfg: Config, candidate_id: str, stress_type: str, diagnostics_dir: Path) -> Dict[str, List[Dict[str, object]]]:
    out: Dict[str, List[Dict[str, object]]] = {}
    if not diagnostics_dir.exists():
        return out
    for path in diagnostics_dir.glob("*.csv"):
        rows = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append({"candidate_id": candidate_id, "stress_type": stress_type, "dataset": cfg.dataset, "model": cfg.model, "horizon": cfg.horizon, **row})
        out[path.stem] = rows
    return out


def write_outputs(output_root: Path, run_root: Path, candidate_id: str, scope: str, rows: List[Dict[str, object]], config_records: List[Dict[str, object]], diagnostics: Dict[str, List[Dict[str, object]]]) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    write_csv(rows, run_root / "combined_metrics.csv")
    payload = {
        "candidate_id": candidate_id,
        "scope": scope,
        "configs": config_records,
        "rows": rows,
        "completed_configs": count_completed(config_records),
        "total_configs": len(config_records),
        "summary": summarize_rows(rows, diagnostics),
    }
    (run_root / "combined_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
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
            "scope": scope,
            "status": "completed" if count_completed(config_records) == len(config_records) else "blocked",
            "completed_configs": count_completed(config_records),
            "total_configs": len(config_records),
            "trend_frequency_mean_mse_delta_pct": summary["trend_frequency_mean_mse_delta_pct"],
            "trend_frequency_improved_configs": summary["trend_frequency_improved_configs"],
            "rule_beats_random_configs": summary["rule_beats_random_configs"],
            "paired_rule_win_rate": summary["paired_rule_win_rate"],
            "rule_vs_random_mean_advantage_mse": summary["rule_vs_random_mean_advantage_mse"],
            "matched_smoothing_mean_mse_delta_pct": summary["matched_smoothing_mean_mse_delta_pct"],
            "naive_smoothing_mean_mse_delta_pct": summary["naive_smoothing_mean_mse_delta_pct"],
            "max_mse_harm_pct": summary["max_mse_harm_pct"],
            "max_mae_harm_pct": summary["max_mae_harm_pct"],
            "test_threshold_leakage": summary["test_threshold_leakage"],
            "gate_verdict": gate_verdict(scope, summary, len(config_records)),
        }
    )
    write_csv(existing, path)


def summarize_rows(rows: List[Dict[str, object]], diagnostics: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    completed = [r for r in rows if r.get("status") == "completed"]
    tf = [r for r in completed if r["variant"] == "trend_frequency"]
    random = [r for r in completed if r["variant"] == "random_trigger"]
    naive = [r for r in completed if r["variant"] == "naive_smoothing"]
    matched = [r for r in completed if r["variant"] == "matched_smoothing_control"]
    random_by_key = {(r["stress_type"], r["dataset"], r["model"], str(r["horizon"])): r for r in random}
    rule_beats = 0
    advantages = []
    for row in tf:
        rr = random_by_key.get((row["stress_type"], row["dataset"], row["model"], str(row["horizon"])))
        if rr:
            advantage = float(rr["mse"]) - float(row["mse"])
            advantages.append(advantage)
            if advantage > 0:
                rule_beats += 1
    paired = diagnostics.get("rule_vs_random_paired", [])
    paired_wins = sum(1 for r in paired if str(r.get("rule_beats_random_mse", "")).lower() == "true")
    paired_total = len(paired)
    return {
        "trend_frequency_mean_mse_delta_pct": mean([float(r["mse_delta_pct_vs_no_correction"]) for r in tf]),
        "trend_frequency_improved_configs": sum(1 for r in tf if float(r["mse_delta_pct_vs_no_correction"]) < 0),
        "rule_beats_random_configs": rule_beats,
        "rule_vs_random_mean_advantage_mse": mean(advantages),
        "paired_rule_win_rate": paired_wins / paired_total if paired_total else "",
        "matched_smoothing_mean_mse_delta_pct": mean([float(r["mse_delta_pct_vs_no_correction"]) for r in matched]),
        "naive_smoothing_mean_mse_delta_pct": mean([float(r["mse_delta_pct_vs_no_correction"]) for r in naive]),
        "max_mse_harm_pct": max([float(r["mse_delta_pct_vs_no_correction"]) for r in tf], default=0.0),
        "max_mae_harm_pct": max([float(r["mae_delta_pct_vs_no_correction"]) for r in tf], default=0.0),
        "test_threshold_leakage": any(str(r.get("test_threshold_leakage")) != "False" for r in completed),
        "stress_types": sorted(set(r["stress_type"] for r in completed)),
    }


def gate_verdict(scope: str, summary: Dict[str, object], n_records: int) -> str:
    if summary["test_threshold_leakage"]:
        return "fail_leakage"
    if summary["max_mse_harm_pct"] > 3.0 or summary["max_mae_harm_pct"] > 3.0:
        return "fail_harm"
    if scope == "clean_full":
        if summary["trend_frequency_mean_mse_delta_pct"] > -0.05:
            return "fail_clean_mse"
        if summary["rule_beats_random_configs"] < 14 or float(summary["paired_rule_win_rate"] or 0.0) < 0.65:
            return "fail_rule_random"
    if scope == "stress":
        if summary["rule_beats_random_configs"] < max(1, int(0.75 * n_records / 7)):
            return "weak_stress_rule_random"
    return "pass_or_smoke_ok"


def markdown_table(rows: List[Dict[str, object]]) -> str:
    headers = ["stress_type", "dataset", "model", "horizon", "variant", "status", "mse", "mae", "mse_delta_pct_vs_no_correction", "correction_rate", "blocker_reason"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def summary_markdown(payload: Dict[str, object]) -> str:
    s = payload["summary"]
    lines = [
        "# Stage 9 Table Summary",
        "",
        f"- Candidate: `{payload['candidate_id']}`",
        f"- Scope: `{payload['scope']}`",
        f"- Completed configs: {payload['completed_configs']} / {payload['total_configs']}",
        f"- stress types: `{', '.join(s['stress_types'])}`",
        f"- trend_frequency mean MSE delta: {s['trend_frequency_mean_mse_delta_pct']:.6f}%",
        f"- trend_frequency improved configs: {s['trend_frequency_improved_configs']}",
        f"- rule beats random configs: {s['rule_beats_random_configs']}",
        f"- paired rule win rate: {s['paired_rule_win_rate']}",
        f"- rule-vs-random mean MSE advantage: {s['rule_vs_random_mean_advantage_mse']:.9f}",
        f"- matched smoothing mean MSE delta: {s['matched_smoothing_mean_mse_delta_pct']:.6f}%",
        f"- naive smoothing mean MSE delta: {s['naive_smoothing_mean_mse_delta_pct']:.6f}%",
        f"- max MSE harm: {s['max_mse_harm_pct']:.6f}%",
        f"- max MAE harm: {s['max_mae_harm_pct']:.6f}%",
        f"- test threshold leakage: {s['test_threshold_leakage']}",
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


def count_completed(records: Iterable[Dict[str, object]]) -> int:
    return sum(1 for r in records if r["status"] == "completed")


def pct_delta(value: float, baseline: float) -> float:
    if abs(float(baseline)) <= 1e-12:
        return 0.0
    return 100.0 * (float(value) - float(baseline)) / float(baseline)


def mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
