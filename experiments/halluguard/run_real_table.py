"""Run real ETT prediction export + HalluGuard evaluation tables."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


DATASETS = ["ETTm1", "ETTh1"]
MODELS = ["DLinear", "PatchTST"]
HORIZONS = [96, 192, 336, 720]
VARIANTS = ["no_correction", "naive_smoothing", "trend_only", "frequency_only", "trend_frequency", "random_trigger"]


@dataclass(frozen=True)
class Config:
    dataset: str
    model: str
    horizon: int

    @property
    def tag(self) -> str:
        return f"{self.dataset}_{self.model}_{self.horizon}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 6 or Stage 7 real HalluGuard tables.")
    parser.add_argument("--scope", required=True, choices=["stage6", "stage7"])
    parser.add_argument("--config", default="experiments/halluguard/configs/halluguard_mvp.yaml")
    parser.add_argument("--data-root", default="external/ETDataset")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-windows", type=int, default=4096)
    parser.add_argument("--max-eval-windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    if args.scope == "stage6":
        out_dir = repo_root / "experiments" / "halluguard" / "results" / "stage6_small_table"
        configs = [Config("ETTm1", "DLinear", h) for h in HORIZONS]
    else:
        out_dir = repo_root / "experiments" / "halluguard" / "results" / "stage7_big_table"
        configs = [Config(dataset, model, horizon) for dataset in DATASETS for model in MODELS for horizon in HORIZONS]

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    config_records = []
    for cfg in configs:
        record, metric_rows = run_config(repo_root, out_dir, cfg, args)
        config_records.append(record)
        rows.extend(metric_rows)
        write_combined(out_dir, rows, config_records)

    write_combined(out_dir, rows, config_records)
    print(json.dumps({"scope": args.scope, "output_dir": str(out_dir), "completed_configs": count_completed(config_records), "total_configs": len(config_records)}))


def run_config(repo_root: Path, out_dir: Path, cfg: Config, args: argparse.Namespace):
    prediction_dir = out_dir / "predictions"
    run_dir = out_dir / "runs" / cfg.tag
    prediction_path = prediction_dir / f"{cfg.tag}.jsonl"
    export_script = repo_root / "external" / "halluguard_real_pipeline" / "export_predictions.py"
    evaluator = repo_root / "experiments" / "halluguard" / "evaluate_predictions.py"

    try:
        prediction_dir.mkdir(parents=True, exist_ok=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        if not prediction_path.exists():
            batch_size = "128" if cfg.model == "PatchTST" else "256"
            max_train = str(min(args.max_train_windows, 3072 if cfg.model == "PatchTST" else args.max_train_windows))
            export_cmd = [
                sys.executable,
                str(export_script),
                "--dataset",
                cfg.dataset,
                "--model",
                cfg.model,
                "--horizon",
                str(cfg.horizon),
                "--data-root",
                str(repo_root / args.data_root),
                "--output",
                str(prediction_path),
                "--epochs",
                str(args.epochs),
                "--max-train-windows",
                max_train,
                "--max-eval-windows",
                str(args.max_eval_windows),
                "--batch-size",
                batch_size,
                "--seed",
                str(args.seed + cfg.horizon + (100 if cfg.model == "PatchTST" else 0) + (1000 if cfg.dataset == "ETTh1" else 0)),
                "--device",
                args.device,
            ]
            subprocess.run(export_cmd, cwd=str(repo_root), check=True)

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
        metric_rows = completed_rows(cfg, prediction_path, run_dir, metrics)
        record = {"dataset": cfg.dataset, "model": cfg.model, "horizon": cfg.horizon, "status": "completed", "blocker_reason": ""}
        return record, metric_rows
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        record = {"dataset": cfg.dataset, "model": cfg.model, "horizon": cfg.horizon, "status": "blocked", "blocker_reason": reason}
        return record, [blocked_row(cfg, prediction_path, run_dir, reason)]


def completed_rows(cfg: Config, prediction_path: Path, run_dir: Path, metrics: Dict) -> List[Dict[str, object]]:
    rows = []
    no = next(m for m in metrics["main_ablation"] if m["variant"] == "no_correction")
    for metric in metrics["main_ablation"]:
        rows.append(
            {
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
                "test_threshold_leakage": metrics["test_threshold_leakage"],
                "prediction_path": str(prediction_path),
                "output_dir": str(run_dir),
                "blocker_reason": "",
            }
        )
    return rows


def blocked_row(cfg: Config, prediction_path: Path, run_dir: Path, reason: str) -> Dict[str, object]:
    return {
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
        "test_threshold_leakage": "",
        "prediction_path": str(prediction_path),
        "output_dir": str(run_dir),
        "blocker_reason": reason,
    }


def write_combined(out_dir: Path, rows: List[Dict[str, object]], config_records: List[Dict[str, object]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "combined_metrics.csv"
    fieldnames = list(blocked_row(Config("dataset", "model", 0), Path("prediction"), Path("output"), "").keys())
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {"configs": config_records, "rows": rows, "completed_configs": count_completed(config_records), "total_configs": len(config_records)}
    (out_dir / "combined_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "combined_ablation_table.md").write_text(markdown_table(rows), encoding="utf-8")
    (out_dir / "summary.md").write_text(summary(payload), encoding="utf-8")


def markdown_table(rows: List[Dict[str, object]]) -> str:
    headers = ["dataset", "model", "horizon", "variant", "status", "mse", "mae", "mse_delta_pct_vs_no_correction", "hallucination_rate", "blocker_reason"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(h, "")) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def summary(payload: Dict) -> str:
    rows = payload["rows"]
    completed_configs = payload["completed_configs"]
    total_configs = payload["total_configs"]
    tf_rows = [r for r in rows if r["status"] == "completed" and r["variant"] == "trend_frequency"]
    random_rows = [r for r in rows if r["status"] == "completed" and r["variant"] == "random_trigger"]
    improved = sum(1 for r in tf_rows if float(r["mse_delta_pct_vs_no_correction"]) < 0)
    worsened_gt3 = sum(1 for r in tf_rows if float(r["mse_delta_pct_vs_no_correction"]) > 3)
    random_close = 0
    random_by_key = {(r["dataset"], r["model"], r["horizon"]): r for r in random_rows}
    for r in tf_rows:
        rr = random_by_key.get((r["dataset"], r["model"], r["horizon"]))
        if rr and abs(float(r["mse"]) - float(rr["mse"])) <= 0.001 * max(abs(float(r["mse"])), 1e-12):
            random_close += 1
    lines = [
        "# Real HalluGuard Table Summary",
        "",
        f"- Completed configs: {completed_configs} / {total_configs}",
        f"- trend_frequency improved MSE vs no_correction in {improved} / {len(tf_rows)} completed configs",
        f"- trend_frequency worsened MSE by >3% in {worsened_gt3} / {len(tf_rows)} completed configs",
        f"- random trigger near rule trigger by tight MSE tolerance in {random_close} / {len(tf_rows)} completed configs",
        "",
        "## Config Status",
        "",
    ]
    for record in payload["configs"]:
        status = record["status"]
        reason = f" ({record['blocker_reason']})" if record.get("blocker_reason") else ""
        lines.append(f"- {record['dataset']} / {record['model']} / {record['horizon']}: {status}{reason}")
    return "\n".join(lines) + "\n"


def count_completed(records: Iterable[Dict[str, object]]) -> int:
    return sum(1 for r in records if r["status"] == "completed")


def pct_delta(value: float, baseline: float) -> float:
    if abs(float(baseline)) <= 1e-12:
        return 0.0
    return 100.0 * (float(value) - float(baseline)) / float(baseline)


def format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    main()
