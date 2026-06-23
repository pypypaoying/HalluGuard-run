"""Resume and merge Stage 14 clean-full evaluations.

The adaptive-router clean table can exceed desktop command timeouts. This
helper runs only missing clean/config pairs, then rebuilds the combined
clean_full outputs from per-run metrics and diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import run_stage13_adaptive_router as stage13
import run_stage14_signal_preserve as stage14_signal


DEFAULT_CONFIG = "experiments/halluguard/configs/halluguard_stage14_autosearch.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume missing Stage 14 clean_full runs and rebuild combined outputs.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true", help="Rerun configs even when metrics.csv exists.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg = stage14_signal.load_yaml(stage14_signal.resolve_path(repo_root, Path(args.config)))
    candidate_id = str(cfg.get("candidate_id", "s14_autosearch_candidate"))
    output_root = stage14_signal.resolve_path(
        repo_root,
        Path((cfg.get("outputs", {}) or {}).get("results_dir", "experiments/halluguard/results/stage14_autosearch")),
    )
    prediction_dir = stage14_signal.resolve_path(
        repo_root,
        Path((cfg.get("data", {}) or {}).get("stage7_prediction_dir", "experiments/halluguard/results/stage7_big_table/predictions")),
    )
    run_root = output_root / candidate_id / "clean_full"
    records = [
        stage13.ConfigRecord(dataset, model, horizon)
        for dataset in stage14_signal.DATASETS
        for model in stage14_signal.MODELS
        for horizon in stage14_signal.HORIZONS
    ]

    completed_now = 0
    skipped = 0
    for record in records:
        run_dir = run_root / "runs" / record.tag
        if not args.force and (run_dir / "metrics.csv").exists():
            skipped += 1
            continue
        prediction_path = prediction_dir / f"{record.tag}.jsonl"
        config_record, _, _ = run_missing_one(repo_root, cfg, candidate_id, record, prediction_path, run_dir)
        completed_now += int(config_record.get("status") == "completed")
        print(
            json.dumps(
                {
                    "progress": "stage14_clean_resume_config_done",
                    "candidate_id": candidate_id,
                    "config": record.tag,
                    "status": config_record.get("status"),
                    "completed_now": completed_now,
                    "skipped_existing": skipped,
                }
            ),
            flush=True,
        )

    rows, config_records, diagnostics = collect_run_outputs(run_root, candidate_id)
    if str((cfg.get("method", {}) or {}).get("evaluator", "signal_preserve")) == "adaptive_router":
        stage13.write_outputs(output_root, run_root, "clean_full", candidate_id, cfg, rows, config_records, diagnostics)
    else:
        stage14_signal.write_outputs(output_root, run_root, "clean_full", candidate_id, cfg, rows, config_records, diagnostics)
    print(
        json.dumps(
            {
                "scope": "clean_full",
                "candidate_id": candidate_id,
                "output_dir": str(run_root),
                "completed_configs": stage13.count_completed(config_records),
                "total_configs": len(config_records),
                "completed_now": completed_now,
                "skipped_existing": skipped,
            }
        )
    )


def run_missing_one(
    repo_root: Path,
    cfg: Dict,
    candidate_id: str,
    record: stage13.ConfigRecord,
    prediction_path: Path,
    run_dir: Path,
):
    evaluator = str((cfg.get("method", {}) or {}).get("evaluator", "signal_preserve"))
    if evaluator == "adaptive_router":
        return stage13.run_one_config(repo_root, cfg, candidate_id, record, prediction_path, run_dir, "clean")
    return stage14_signal.run_one_config(repo_root, cfg, candidate_id, record, prediction_path, run_dir, "clean")


def collect_run_outputs(run_root: Path, candidate_id: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    rows: List[Dict[str, object]] = []
    config_records: Dict[Tuple[str, str, str], Dict[str, object]] = {}
    diagnostics: Dict[str, List[Dict[str, object]]] = {}
    for metrics_path in sorted((run_root / "runs").glob("*/metrics.csv")):
        run_rows = read_csv(metrics_path)
        rows.extend(run_rows)
        if run_rows:
            first = run_rows[0]
            key = (str(first.get("dataset", "")), str(first.get("model", "")), str(first.get("horizon", "")))
            config_records[key] = {
                "candidate_id": candidate_id,
                "stress_type": "clean",
                "dataset": key[0],
                "model": key[1],
                "horizon": int(float(key[2])) if key[2] else "",
                "status": "completed",
                "blocker_reason": "",
            }
        diag_dir = metrics_path.parent / "diagnostics"
        if diag_dir.exists():
            for diag_path in sorted(diag_dir.glob("*.csv")):
                diagnostics.setdefault(diag_path.stem, []).extend(read_csv(diag_path))
    return rows, list(config_records.values()), diagnostics


def read_csv(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    main()
