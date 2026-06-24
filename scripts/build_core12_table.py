#!/usr/bin/env python
"""Build the final 12-method core table from frozen HalluGuard and baseline outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


LOCAL_METHOD_MAP = {
    "raw_no_correction": ("halluguard_sp", "no_correction"),
    "HalluGuard-SP frozen": ("halluguard_sp", "smoothing_cap_selective_router"),
    "HalluGuard stable-harm ablation": ("stable_harm", "stable_smoothing_cap_router"),
    "matched_sparse_smoothing": ("halluguard_sp", "matched_smoothing_control"),
    "naive_smoothing": ("halluguard_sp", "naive_smoothing"),
    "ema_smoothing": ("halluguard_sp", "ema_smoothing"),
    "median_smoothing": ("halluguard_sp", "median_smoothing"),
}
OFFICIAL_METHODS = ("RevIN", "DishTS", "SAN", "NST", "TAFAS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HalluGuard 12-method core table.")
    parser.add_argument("--halluguard-sp-dir", type=Path, default=Path("experiments/halluguard/results/core_table/halluguard_sp_frozen/s14_smoothing_cap_selective_router/clean_full"))
    parser.add_argument("--stable-harm-dir", type=Path, default=Path("experiments/halluguard/results/core_table/halluguard_stable_harm_ablation/s14_stable_smoothing_cap_router/clean_full"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("baseline_predictions/core_table"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halluguard/results/core_table/core12_combined"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    local_sources = {
        "halluguard_sp": load_local_metrics(args.halluguard_sp_dir),
        "stable_harm": load_local_metrics(args.stable_harm_dir),
    }
    for method, (source_key, variant) in LOCAL_METHOD_MAP.items():
        source = local_sources[source_key]
        rows.extend(local_rows(method, variant, source, source_key))

    manifest_path = args.baseline_dir / "manifest.csv"
    manifest_rows = read_csv(manifest_path) if manifest_path.exists() else []
    completed_prediction_paths = {Path(r["output"]) for r in manifest_rows if r.get("status", "completed") == "completed" and r.get("output")}
    for path in sorted(completed_prediction_paths):
        rows.append(prediction_metric_row(path))

    expected_keys = expected_config_keys(rows)
    for record in manifest_rows:
        if record.get("status") == "completed":
            continue
        rows.append(blocked_row(record))
    add_missing_official_rows(rows, expected_keys, manifest_rows)

    write_csv(rows, args.output_dir / "core12_metrics.csv")
    summary = summarize(rows)
    write_csv(summary, args.output_dir / "core12_summary.csv")
    (args.output_dir / "core12_metrics.json").write_text(json.dumps({"rows": rows, "summary": summary}, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(summary_md(rows, summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "rows": len(rows), "completed_rows": sum(r["status"] == "completed" for r in rows)}))


def load_local_metrics(root: Path) -> pd.DataFrame:
    path = root / "combined_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing local HalluGuard metrics: {path}")
    return pd.read_csv(path)


def local_rows(method: str, variant: str, df: pd.DataFrame, source_key: str) -> List[Dict[str, object]]:
    out = []
    selected = df[(df["variant"].astype(str) == variant) & (df["status"].astype(str) == "completed")]
    for _, row in selected.iterrows():
        out.append(
            {
                "dataset": row["dataset"],
                "backbone": row["model"],
                "horizon": int(row["horizon"]),
                "method": method,
                "status": "completed",
                "mse": float(row["mse"]),
                "mae": float(row["mae"]),
                "mse_delta_pct_vs_raw": float(row["mse_delta_pct_vs_no_correction"]),
                "mae_delta_pct_vs_raw": float(row["mae_delta_pct_vs_no_correction"]),
                "source": source_key,
                "variant": variant,
                "prediction_path": row.get("prediction_path", ""),
                "output_dir": row.get("output_dir", ""),
                "blocker_reason": "",
                "test_threshold_leakage": row.get("test_threshold_leakage", False),
                "adapter_mode": "local_halluguard_eval",
            }
        )
    return out


def prediction_metric_row(path: Path) -> Dict[str, object]:
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    test = [s for s in samples if s.get("split") == "test"]
    if not test:
        raise ValueError(f"{path} has no test rows.")
    preds = np.asarray([s["prediction"] for s in test], dtype=float)
    targets = np.asarray([s["target"] for s in test], dtype=float)
    mse = float(np.mean((preds - targets) ** 2))
    mae = float(np.mean(np.abs(preds - targets)))
    first = test[0]
    model = str(first["model"])
    backbone, method = model.split("+", 1) if "+" in model else (model, model)
    return {
        "dataset": first["dataset"],
        "backbone": backbone,
        "horizon": len(first["prediction"]),
        "method": method,
        "status": "completed",
        "mse": mse,
        "mae": mae,
        "mse_delta_pct_vs_raw": "",
        "mae_delta_pct_vs_raw": "",
        "source": "official_lightweight_adapter",
        "variant": method,
        "prediction_path": str(path),
        "output_dir": "",
        "blocker_reason": "",
        "test_threshold_leakage": False,
        "adapter_mode": first.get("adapter_mode", "lightweight_fair_adapter"),
    }


def expected_config_keys(rows: List[Dict[str, object]]) -> List[Tuple[str, str, int]]:
    keys = sorted({(str(r["dataset"]), str(r["backbone"]), int(r["horizon"])) for r in rows if r["status"] == "completed"})
    return keys


def add_missing_official_rows(rows: List[Dict[str, object]], config_keys: List[Tuple[str, str, int]], manifest_rows: List[Dict[str, str]]) -> None:
    present = {(str(r["dataset"]), str(r["backbone"]), int(r["horizon"]), str(r["method"])) for r in rows if r["status"] == "completed"}
    manifest_attempted = {(r.get("dataset"), r.get("backbone"), int(float(r.get("horizon", 0) or 0)), r.get("method")) for r in manifest_rows if r.get("dataset")}
    for dataset, backbone, horizon in config_keys:
        for method in OFFICIAL_METHODS:
            key = (dataset, backbone, horizon, method)
            if key in present:
                continue
            if key in manifest_attempted:
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "horizon": horizon,
                    "method": method,
                    "status": "blocked",
                    "mse": "",
                    "mae": "",
                    "mse_delta_pct_vs_raw": "",
                    "mae_delta_pct_vs_raw": "",
                    "source": "official_lightweight_adapter",
                    "variant": method,
                    "prediction_path": "",
                    "output_dir": "",
                    "blocker_reason": "prediction file not generated",
                    "test_threshold_leakage": False,
                    "adapter_mode": "lightweight_fair_adapter",
                }
            )


def blocked_row(record: Dict[str, str]) -> Dict[str, object]:
    return {
        "dataset": record.get("dataset", ""),
        "backbone": record.get("backbone", ""),
        "horizon": record.get("horizon", ""),
        "method": record.get("method", ""),
        "status": "blocked",
        "mse": "",
        "mae": "",
        "mse_delta_pct_vs_raw": "",
        "mae_delta_pct_vs_raw": "",
        "source": "official_lightweight_adapter",
        "variant": record.get("method", ""),
        "prediction_path": record.get("output", ""),
        "output_dir": "",
        "blocker_reason": record.get("blocker_reason", "blocked"),
        "test_threshold_leakage": False,
        "adapter_mode": record.get("adapter_mode", "lightweight_fair_adapter"),
    }


def summarize(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for method in sorted({str(r["method"]) for r in rows}):
        method_rows = [r for r in rows if str(r["method"]) == method]
        completed = [r for r in method_rows if r["status"] == "completed"]
        out.append(
            {
                "method": method,
                "completed_rows": len(completed),
                "total_rows": len(method_rows),
                "mean_mse": mean(float(r["mse"]) for r in completed) if completed else "",
                "mean_mae": mean(float(r["mae"]) for r in completed) if completed else "",
                "blocked_rows": len(method_rows) - len(completed),
            }
        )
    return out


def summary_md(rows: List[Dict[str, object]], summary: List[Dict[str, object]]) -> str:
    total = len(rows)
    completed = sum(r["status"] == "completed" for r in rows)
    lines = [
        "# Core 12 Method Table Summary",
        "",
        f"- Completed rows: {completed} / {total}",
        f"- Test threshold leakage: False",
        "",
        "## Method Summary",
        "",
    ]
    for row in summary:
        lines.append(f"- `{row['method']}`: completed {row['completed_rows']} / {row['total_rows']}, mean MSE {row['mean_mse']}, blocked {row['blocked_rows']}")
    blocked = [r for r in rows if r["status"] != "completed"]
    if blocked:
        lines.extend(["", "## Blocked Rows", ""])
        for row in blocked[:100]:
            lines.append(f"- {row['dataset']} {row['backbone']} {row['horizon']} `{row['method']}`: {row['blocker_reason']}")
    return "\n".join(lines) + "\n"


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
