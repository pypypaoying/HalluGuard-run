"""Resume and merge Stage 14 stress evaluations.

The normal Stage 14 stress table is long enough that desktop pipes can time
out. This helper runs only missing stress/config pairs, then rebuilds the
combined stress outputs from per-run metrics and diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import run_stage11_dynamics as stage11
import run_stage13_adaptive_router as stage13
import run_stage14_autosearch as stage14_auto
import run_stage14_signal_preserve as stage14_signal


DEFAULT_CONFIG = "experiments/halluguard/configs/halluguard_stage14_autosearch.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume missing Stage 14 stress runs and rebuild combined outputs.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stress-types", default=None, help="Comma-separated subset. Defaults to config stress.types.")
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
    candidate_root = output_root / candidate_id
    run_root = candidate_root / "stress"
    stress_types = parse_stress_types(args.stress_types, cfg)
    records = [
        stage13.ConfigRecord(dataset, model, horizon)
        for dataset in stage14_signal.DATASETS
        for model in stage14_signal.MODELS
        for horizon in stage14_signal.HORIZONS
    ]

    completed_now = 0
    skipped = 0
    for stress_type in stress_types:
        for record in records:
            run_dir = run_root / stress_type / "runs" / record.tag
            if not args.force and (run_dir / "metrics.csv").exists():
                skipped += 1
                continue
            source = prediction_dir / f"{record.tag}.jsonl"
            prediction_path = candidate_root / "stress_predictions" / stress_type / f"{record.tag}.jsonl"
            stage11.write_stress_predictions(source, prediction_path, stress_type)
            config_record, _, _ = run_missing_one(repo_root, cfg, candidate_id, record, prediction_path, run_dir, stress_type)
            completed_now += int(config_record.get("status") == "completed")
            print(
                json.dumps(
                    {
                        "progress": "stage14_stress_resume_config_done",
                        "candidate_id": candidate_id,
                        "stress_type": stress_type,
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
        stage13.write_outputs(output_root, run_root, "stress", candidate_id, cfg, rows, config_records, diagnostics)
    else:
        stage14_signal.write_outputs(output_root, run_root, "stress", candidate_id, cfg, rows, config_records, diagnostics)
    print(
        json.dumps(
            {
                "scope": "stress",
                "candidate_id": candidate_id,
                "output_dir": str(run_root),
                "completed_configs": stage13.count_completed(config_records),
                "total_configs": len(config_records),
                "completed_now": completed_now,
                "skipped_existing": skipped,
            }
        )
    )


def parse_stress_types(value: str | None, cfg: Dict) -> List[str]:
    if value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return list((cfg.get("stress", {}) or {}).get("types", []))


def run_missing_one(
    repo_root: Path,
    cfg: Dict,
    candidate_id: str,
    record: stage13.ConfigRecord,
    prediction_path: Path,
    run_dir: Path,
    stress_type: str,
):
    evaluator = str((cfg.get("method", {}) or {}).get("evaluator", "signal_preserve"))
    if evaluator == "adaptive_router":
        return stage13.run_one_config(repo_root, cfg, candidate_id, record, prediction_path, run_dir, stress_type)
    return stage14_signal.run_one_config(repo_root, cfg, candidate_id, record, prediction_path, run_dir, stress_type)


def collect_run_outputs(run_root: Path, candidate_id: str) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    rows: List[Dict[str, object]] = []
    config_records: Dict[Tuple[str, str, str, str], Dict[str, object]] = {}
    diagnostics: Dict[str, List[Dict[str, object]]] = {}
    for metrics_path in sorted(run_root.glob("*/runs/*/metrics.csv")):
        run_rows = read_csv(metrics_path)
        rows.extend(run_rows)
        if run_rows:
            first = run_rows[0]
            key = (
                str(first.get("stress_type", metrics_path.parents[2].name)),
                str(first.get("dataset", "")),
                str(first.get("model", "")),
                str(first.get("horizon", "")),
            )
            config_records[key] = {
                "candidate_id": candidate_id,
                "stress_type": key[0],
                "dataset": key[1],
                "model": key[2],
                "horizon": int(float(key[3])) if key[3] else "",
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
