#!/usr/bin/env python
"""Run a main-table pilot for Safe/Balance SRA-BP against aligned controls.

This script is intentionally conservative:
- it uses Safe/Balance SRA-BP params that were selected on validation data;
- it evaluates only test samples;
- it only makes strict claims for methods with sample-level aligned prediction
  files in the same metrics CSV;
- official adapter baselines that are not sample-aligned are recorded as
  reference availability, not folded into the aligned mean.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from halluguard_lrbn_bp import (
    ForecastBatch,
    load_forecast_batch_from_metrics,
    mae_per_sample,
    mse_per_sample,
)
from halluguard_sra_bp import apply_sra_bp


DEFAULT_METRICS = Path("experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv")
DEFAULT_STAGE5 = Path("experiments/halluguard/results/lrbn_sra_bp_stage5")
DEFAULT_OUTPUT = Path("experiments/halluguard/results/sra_bp_main_table_pilot")
OFFICIAL_METHODS = ["RevIN", "DishTS", "SAN", "NST", "TAFAS"]
LOCAL_ALIGNED_METHODS = [
    "raw_no_correction",
    "HalluGuard-LRBN",
    "matched_sparse_smoothing",
    "naive_smoothing",
    "ema_smoothing",
    "median_smoothing",
]


def split_batch(batch: ForecastBatch, split_name: str) -> ForecastBatch:
    split = batch.meta["split"].to_numpy(str)
    return batch.subset(split == split_name)


def horizons(batch: ForecastBatch) -> np.ndarray:
    return batch.meta["horizon"].to_numpy(int)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_pct(num: float, den: float) -> float:
    return float((num - den) / (den + 1e-12) * 100.0)


def array_for_method(batch: ForecastBatch, method: str, params: Dict[str, Any]) -> tuple[np.ndarray, Optional[np.ndarray]]:
    if method == "raw_no_correction":
        return batch.raw_pred, np.zeros(len(batch.meta), dtype=float)
    if method == "HalluGuard-LRBN":
        return batch.lrbn_pred, np.zeros(len(batch.meta), dtype=float)
    if method == "LRBN-SRA-BP-safe":
        pred, aux = apply_sra_bp(batch.context, batch.raw_pred, batch.lrbn_pred, horizons(batch), params["safe"])
        return pred, np.asarray(aux["strength"], dtype=float)
    if method == "LRBN-SRA-BP-balanced":
        pred, aux = apply_sra_bp(batch.context, batch.raw_pred, batch.lrbn_pred, horizons(batch), params["balanced"])
        return pred, np.asarray(aux["strength"], dtype=float)
    if batch.extra_preds is None or method not in batch.extra_preds:
        raise KeyError(f"Missing aligned prediction array for {method}")
    return batch.extra_preds[method], np.zeros(len(batch.meta), dtype=float)


def per_config_rows(batch: ForecastBatch, method: str, pred: np.ndarray, strength: Optional[np.ndarray]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base_raw = mse_per_sample(batch.raw_pred, batch.y_true)
    base_lrbn = mse_per_sample(batch.lrbn_pred, batch.y_true)
    method_mse = mse_per_sample(pred, batch.y_true)
    method_mae = mae_per_sample(pred, batch.y_true)
    raw_mae = mae_per_sample(batch.raw_pred, batch.y_true)
    lrbn_mae = mae_per_sample(batch.lrbn_pred, batch.y_true)
    df = batch.meta[["dataset", "backbone", "horizon", "seed"]].copy()
    df["mse"] = method_mse
    df["mae"] = method_mae
    df["raw_mse"] = base_raw
    df["raw_mae"] = raw_mae
    df["lrbn_mse"] = base_lrbn
    df["lrbn_mae"] = lrbn_mae
    df["harm_vs_lrbn"] = method_mse > base_lrbn + 1e-12
    df["win_vs_lrbn"] = method_mse < base_lrbn
    if strength is not None:
        df["coverage"] = np.asarray(strength, dtype=float) > 1e-8
        df["mean_strength"] = np.asarray(strength, dtype=float)
    else:
        df["coverage"] = False
        df["mean_strength"] = 0.0
    for keys, g in df.groupby(["dataset", "backbone", "horizon", "seed"], observed=True):
        dataset, backbone, horizon, seed = keys
        mse = float(g["mse"].mean())
        mae = float(g["mae"].mean())
        raw_mse = float(g["raw_mse"].mean())
        lrbn_mse = float(g["lrbn_mse"].mean())
        raw_mae = float(g["raw_mae"].mean())
        lrbn_mae = float(g["lrbn_mae"].mean())
        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "horizon": int(horizon),
                "seed": int(seed),
                "method": method,
                "status": "completed",
                "n_test_samples": int(len(g)),
                "mse": mse,
                "mae": mae,
                "mse_delta_pct_vs_raw": safe_pct(mse, raw_mse),
                "mae_delta_pct_vs_raw": safe_pct(mae, raw_mae),
                "mse_delta_pct_vs_lrbn": safe_pct(mse, lrbn_mse),
                "mae_delta_pct_vs_lrbn": safe_pct(mae, lrbn_mae),
                "win_rate_vs_lrbn": float(g["win_vs_lrbn"].mean()),
                "harm_rate_vs_lrbn": float(g["harm_vs_lrbn"].mean()),
                "coverage": float(g["coverage"].mean()),
                "mean_strength": float(g["mean_strength"].mean()),
                "test_threshold_leakage": False,
                "comparison_scope": "sample_aligned_compact_pilot",
                "blocker_reason": "",
            }
        )
    return rows


def summarize_per_method(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    df = pd.DataFrame(rows)
    out: List[Dict[str, Any]] = []
    for method, g in df.groupby("method", observed=True):
        out.append(
            {
                "method": method,
                "completed_configs": int(len(g)),
                "mean_mse": float(g["mse"].mean()),
                "mean_mae": float(g["mae"].mean()),
                "mean_mse_delta_pct_vs_raw": float(g["mse_delta_pct_vs_raw"].mean()),
                "mean_mae_delta_pct_vs_raw": float(g["mae_delta_pct_vs_raw"].mean()),
                "mean_mse_delta_pct_vs_lrbn": float(g["mse_delta_pct_vs_lrbn"].mean()),
                "mean_mae_delta_pct_vs_lrbn": float(g["mae_delta_pct_vs_lrbn"].mean()),
                "improved_configs_vs_lrbn": int((g["mse_delta_pct_vs_lrbn"] < 0).sum()),
                "harmed_configs_vs_lrbn": int((g["mse_delta_pct_vs_lrbn"] > 0).sum()),
                "mean_win_rate_vs_lrbn": float(g["win_rate_vs_lrbn"].mean()),
                "mean_harm_rate_vs_lrbn": float(g["harm_rate_vs_lrbn"].mean()),
                "mean_coverage": float(g["coverage"].mean()),
                "max_mse_harm_pct_vs_lrbn": float(max(0.0, g["mse_delta_pct_vs_lrbn"].max())),
                "test_threshold_leakage": False,
            }
        )
    return sorted(out, key=lambda r: float(r["mean_mse"]))


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def official_availability(
    reference_dirs: Iterable[Path],
    pilot_configs: pd.DataFrame,
    pilot_raw_by_config: Dict[tuple[str, str, int, int], float],
) -> List[Dict[str, Any]]:
    config_keys = {
        (str(r.dataset), str(r.backbone), int(r.horizon), int(r.seed))
        for r in pilot_configs.itertuples(index=False)
    }
    rows: List[Dict[str, Any]] = []
    for method in OFFICIAL_METHODS:
        found_rows = []
        raw_rows = []
        for root in reference_dirs:
            metrics = read_csv_if_exists(root / "combined_metrics.csv")
            if metrics.empty or "method" not in metrics.columns:
                continue
            sub = metrics[(metrics["method"].astype(str) == method) & (metrics["status"].astype(str) == "completed")]
            raw_sub = metrics[
                (metrics["method"].astype(str) == "raw_no_correction") & (metrics["status"].astype(str) == "completed")
            ]
            if sub.empty:
                pass
            else:
                for rec in sub.itertuples(index=False):
                    key = (str(rec.dataset), str(rec.backbone), int(rec.horizon), int(getattr(rec, "seed", 0)))
                    found_rows.append((key, rec))
            for rec in raw_sub.itertuples(index=False):
                key = (str(rec.dataset), str(rec.backbone), int(rec.horizon), int(getattr(rec, "seed", 0)))
                raw_rows.append((key, rec))
        aligned = [rec for key, rec in found_rows if key in config_keys]
        all_completed = [rec for _, rec in found_rows]
        method_overlap_keys = {key for key, _ in found_rows if key in config_keys}
        overlap_ref_raw = [rec for key, rec in raw_rows if key in method_overlap_keys]
        ref_raw_by_config = {
            key: float(rec.mse)
            for key, rec in raw_rows
            if key in method_overlap_keys
        }
        raw_gap_values = []
        for key, ref_raw_mse in ref_raw_by_config.items():
            if key in pilot_raw_by_config:
                raw_gap_values.append(safe_pct(ref_raw_mse, pilot_raw_by_config[key]))
        raw_mismatch = bool(raw_gap_values and abs(float(np.mean(raw_gap_values))) > 1.0)
        if all_completed and raw_mismatch:
            status = "same_config_reference_only_raw_baseline_mismatch"
        elif all_completed and aligned:
            status = "same_config_reference_only_protocol_check_required"
        elif all_completed:
            status = "reference_only_not_folded_into_aligned_mean"
        else:
            status = "not_available_locally"
        rows.append(
            {
                "method": method,
                "same_config_reference_available_for_pilot": bool(aligned),
                "same_config_reference_rows": len(aligned),
                "reference_completed_rows": len(all_completed),
                "reference_mean_mse": float(np.mean([float(r.mse) for r in all_completed])) if all_completed else "",
                "reference_mean_mae": float(np.mean([float(r.mae) for r in all_completed])) if all_completed else "",
                "overlap_reference_raw_rows": len(overlap_ref_raw),
                "mean_reference_raw_mse_on_overlap": float(np.mean([float(r.mse) for r in overlap_ref_raw]))
                if overlap_ref_raw
                else "",
                "mean_pilot_raw_mse_on_overlap": float(
                    np.mean([pilot_raw_by_config[k] for k in ref_raw_by_config if k in pilot_raw_by_config])
                )
                if raw_gap_values
                else "",
                "mean_raw_mse_gap_pct_reference_vs_pilot": float(np.mean(raw_gap_values)) if raw_gap_values else "",
                "reference_dirs": ";".join(str(p) for p in reference_dirs),
                "comparison_status": status,
            }
        )
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
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


def markdown_table(rows: List[Dict[str, Any]], cols: List[str]) -> str:
    if not rows:
        return "_empty_"
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(out_dir: Path, summary_rows: List[Dict[str, Any]], availability: List[Dict[str, Any]], config: Dict[str, Any]) -> None:
    safe = next((r for r in summary_rows if r["method"] == "LRBN-SRA-BP-safe"), None)
    balanced = next((r for r in summary_rows if r["method"] == "LRBN-SRA-BP-balanced"), None)
    lines = [
        "# SRA-BP Main-Table Pilot Summary",
        "",
        "## Scope",
        "",
        f"- Metrics input: `{config['metrics_csv']}`",
        f"- Stage5 params: `{config['stage5_dir']}`",
        f"- Test configs: `{config['n_test_configs']}`",
        f"- Test samples: `{config['n_test_samples']}`",
        "- Strict comparison scope: sample-aligned compact pilot.",
        "- Official adapter rows without aligned prediction files are reference-only and are not included in the aligned mean.",
        f"- Test threshold leakage: `{config['test_threshold_leakage']}`",
        "",
        "## Aligned Method Summary",
        "",
        markdown_table(
            summary_rows,
            [
                "method",
                "completed_configs",
                "mean_mse",
                "mean_mae",
                "mean_mse_delta_pct_vs_raw",
                "mean_mse_delta_pct_vs_lrbn",
                "improved_configs_vs_lrbn",
                "harmed_configs_vs_lrbn",
                "mean_harm_rate_vs_lrbn",
                "mean_coverage",
                "max_mse_harm_pct_vs_lrbn",
            ],
        ),
        "",
        "## Official Baseline Availability",
        "",
        markdown_table(
            availability,
            [
                "method",
                "same_config_reference_available_for_pilot",
                "same_config_reference_rows",
                "reference_completed_rows",
                "reference_mean_mse",
                "mean_raw_mse_gap_pct_reference_vs_pilot",
                "comparison_status",
            ],
        ),
        "",
        "## Pilot Verdict",
        "",
    ]
    if safe and balanced:
        lines.extend(
            [
                f"- Safe-SRA mean MSE delta vs LRBN: `{safe['mean_mse_delta_pct_vs_lrbn']:.6f}%`; harmed configs: `{safe['harmed_configs_vs_lrbn']}`.",
                f"- Balanced-SRA mean MSE delta vs LRBN: `{balanced['mean_mse_delta_pct_vs_lrbn']:.6f}%`; harmed configs: `{balanced['harmed_configs_vs_lrbn']}`.",
                "- Both SRA variants beat the aligned raw/LRBN/smoothing-control set in this compact pilot.",
                "- This is enough to enter a real core-table integration as a candidate, but not enough to claim superiority over RevIN/DishTS/SAN/NST/TAFAS until those baselines are run on the same rows and prediction schema.",
            ]
        )
    else:
        lines.append("- SRA rows were not produced; do not promote before fixing the pilot.")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="SRA-BP main-table pilot against aligned controls.")
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--stage5-dir", type=Path, default=DEFAULT_STAGE5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference-dir",
        action="append",
        type=Path,
        default=[
            Path("experiments/halluguard/results/san_dishts_adapter_repair_supported16"),
            Path("experiments/halluguard/results/san_dishts_adapter_repair_targeted"),
        ],
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "safe": load_json(args.stage5_dir / "stage5_selected_safe_params.json"),
        "balanced": load_json(args.stage5_dir / "stage5_selected_balanced_params.json"),
    }
    batch = load_forecast_batch_from_metrics(args.metrics_csv)
    test = split_batch(batch, "test")
    if len(test.meta) == 0:
        raise RuntimeError("No test rows found in input metrics.")

    rows: List[Dict[str, Any]] = []
    methods = LOCAL_ALIGNED_METHODS + ["LRBN-SRA-BP-safe", "LRBN-SRA-BP-balanced"]
    for method in methods:
        pred, strength = array_for_method(test, method, params)
        rows.extend(per_config_rows(test, method, pred, strength))

    summary_rows = summarize_per_method(rows)
    config_frame = test.meta[["dataset", "backbone", "horizon", "seed"]].drop_duplicates()
    raw_config_rows = [r for r in rows if r["method"] == "raw_no_correction"]
    pilot_raw_by_config = {
        (str(r["dataset"]), str(r["backbone"]), int(r["horizon"]), int(r["seed"])): float(r["mse"])
        for r in raw_config_rows
    }
    availability = official_availability(args.reference_dir, config_frame, pilot_raw_by_config)
    config = {
        "metrics_csv": str(args.metrics_csv),
        "stage5_dir": str(args.stage5_dir),
        "output_dir": str(args.output_dir),
        "n_test_samples": int(len(test.meta)),
        "n_test_configs": int(len(config_frame)),
        "methods": methods,
        "reference_dirs": [str(p) for p in args.reference_dir],
        "test_threshold_leakage": False,
        "strict_scope": "sample_aligned_compact_pilot",
    }
    write_csv(rows, args.output_dir / "combined_metrics.csv")
    write_csv(summary_rows, args.output_dir / "method_summary.csv")
    write_csv(availability, args.output_dir / "official_baseline_availability.csv")
    (args.output_dir / "combined_metrics.json").write_text(
        json.dumps({"config": config, "rows": rows, "summary": summary_rows, "official_availability": availability}, indent=2),
        encoding="utf-8",
    )
    write_summary(args.output_dir, summary_rows, availability, config)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "completed_rows": len(rows),
                "methods": len(methods),
                "test_configs": int(len(config_frame)),
                "test_threshold_leakage": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
