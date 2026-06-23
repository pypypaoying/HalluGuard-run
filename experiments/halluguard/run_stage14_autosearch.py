"""Stage 14 autoresearch runner for signal-preserving HalluGuard variants.

This is a comparable runner around the Stage 14 signal-preserve evaluator and
the Stage 13 adaptive-router evaluator. It keeps the fixed evaluation contract,
but writes each candidate
under:

    experiments/halluguard/results/stage14_autosearch/<candidate_id>/<scope>/

Validation rows fit thresholds/policies. Test rows are used only for frozen
policy evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import run_stage11_dynamics as stage11
import run_stage13_adaptive_router as stage13
import run_stage14_signal_preserve as stage14


DEFAULT_CONFIG = "experiments/halluguard/configs/halluguard_stage14_autosearch.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 14 HalluGuard autosearch candidate.")
    parser.add_argument("--scope", required=True, choices=["smoke", "clean_full", "stress", "external_batch"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--stage7-prediction-dir", default=None)
    parser.add_argument("--external-input-dir", default=None)
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg = stage14.load_yaml(stage14.resolve_path(repo_root, Path(args.config)))
    candidate_id = str(cfg.get("candidate_id", "s14_autosearch_candidate"))
    evaluator = str((cfg.get("method", {}) or {}).get("evaluator", "signal_preserve"))
    output_root = stage14.resolve_path(repo_root, Path(args.output_root)) if args.output_root else stage14.resolve_path(
        repo_root, Path((cfg.get("outputs", {}) or {}).get("results_dir", "experiments/halluguard/results/stage14_autosearch"))
    )
    output_root.mkdir(parents=True, exist_ok=True)

    prediction_dir = stage14.resolve_path(repo_root, Path(args.stage7_prediction_dir)) if args.stage7_prediction_dir else stage14.resolve_path(
        repo_root, Path((cfg.get("data", {}) or {}).get("stage7_prediction_dir", "experiments/halluguard/results/stage7_big_table/predictions"))
    )
    candidate_root = output_root / candidate_id

    if args.scope == "external_batch":
        input_dir = stage14.resolve_path(repo_root, Path(args.external_input_dir)) if args.external_input_dir else stage14.resolve_path(
            repo_root, Path((cfg.get("data", {}) or {}).get("external_fixture_dir", "experiments/halluguard/results/stage12_external_batch/fixtures/external_batch_clean"))
        )
        files = sorted(
            [p for p in input_dir.glob("*.jsonl") if p.name.lower() != "manifest.jsonl"]
            + [p for p in input_dir.glob("*.csv") if p.name.lower() != "manifest.csv"]
        )
        if args.limit_files is not None:
            files = files[: int(args.limit_files)]
        scopes = [("external_batch", stage14.record_from_prediction_file(path), path) for path in files]
        run_root = candidate_root / "external_batch"
    else:
        records = [stage13.ConfigRecord(*c) for c in stage14.SMOKE_CONFIGS] if args.scope == "smoke" else [
            stage13.ConfigRecord(dataset, model, horizon)
            for dataset in stage14.DATASETS
            for model in stage14.MODELS
            for horizon in stage14.HORIZONS
        ]
        stress_types = ["clean"] if args.scope != "stress" else list((cfg.get("stress", {}) or {}).get("types", []))
        run_root = candidate_root / args.scope
        scopes = []
        for stress_type in stress_types:
            for record in records:
                source = prediction_dir / f"{record.tag}.jsonl"
                prediction_path = source if stress_type == "clean" else candidate_root / "stress_predictions" / stress_type / f"{record.tag}.jsonl"
                if stress_type != "clean":
                    stage11.write_stress_predictions(source, prediction_path, stress_type)
                scopes.append((stress_type, record, prediction_path))

    rows: List[Dict[str, object]] = []
    config_records: List[Dict[str, object]] = []
    diagnostics: Dict[str, List[Dict[str, object]]] = {}
    for stress_type, record, prediction_path in scopes:
        run_group = run_root / "runs" if stress_type in {"clean", "external_batch"} else run_root / stress_type / "runs"
        run_dir = run_group / record.tag
        if evaluator == "adaptive_router":
            config_record, metric_rows, diag_rows = stage13.run_one_config(
                repo_root, cfg, candidate_id, record, prediction_path, run_dir, stress_type
            )
        else:
            config_record, metric_rows, diag_rows = stage14.run_one_config(
                repo_root, cfg, candidate_id, record, prediction_path, run_dir, stress_type
            )
        config_records.append(config_record)
        rows.extend(metric_rows)
        for name, values in diag_rows.items():
            diagnostics.setdefault(name, []).extend(values)
        if evaluator == "adaptive_router":
            stage13.write_outputs(output_root, run_root, args.scope, candidate_id, cfg, rows, config_records, diagnostics)
        else:
            stage14.write_outputs(output_root, run_root, args.scope, candidate_id, cfg, rows, config_records, diagnostics)
        print(
            json.dumps(
                {
                    "progress": "stage14_autosearch_config_done",
                    "scope": args.scope,
                    "candidate_id": candidate_id,
                    "stress_type": stress_type,
                    "config": record.tag,
                    "status": config_record["status"],
                    "completed_configs": stage13.count_completed(config_records),
                    "total_seen": len(config_records),
                }
            ),
            flush=True,
        )

    if evaluator == "adaptive_router":
        stage13.write_outputs(output_root, run_root, args.scope, candidate_id, cfg, rows, config_records, diagnostics)
    else:
        stage14.write_outputs(output_root, run_root, args.scope, candidate_id, cfg, rows, config_records, diagnostics)
    print(
        json.dumps(
            {
                "scope": args.scope,
                "candidate_id": candidate_id,
                "output_dir": str(run_root),
                "completed_configs": stage13.count_completed(config_records),
                "total_configs": len(config_records),
            }
        )
    )


if __name__ == "__main__":
    main()
