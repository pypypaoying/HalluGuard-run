"""Stage 12 external batch evaluation for HalluGuard-Dynamics.

This runner evaluates external forecast prediction files without importing or
training the source forecasting framework. It accepts one JSONL/CSV file, a
directory of files, or a manifest of files. Policies are fit only on
split="val" rows and evaluated only on split="test" rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

import run_stage11_dynamics as stage11


DEFAULT_CONFIG = "experiments/halluguard/configs/halluguard_stage12_external_batch.yaml"
DEFAULT_OUTPUT_ROOT = "experiments/halluguard/results/stage12_external_batch"
REQUIRED_FIELDS = ("sample_id", "dataset", "model", "split", "context", "prediction", "target")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HalluGuard-Dynamics on external prediction batches.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=None, help="One JSONL/CSV prediction file.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Directory containing JSONL/CSV prediction files.")
    parser.add_argument("--manifest", type=Path, default=None, help="TXT/CSV/JSONL manifest listing prediction files.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --input-dir.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record failed files and continue instead of raising.")
    parser.add_argument("--make-fixtures", action="store_true", help="Create the Stage 12 external-style fixture set before evaluation.")
    parser.add_argument("--fixture-output-dir", type=Path, default=None)
    parser.add_argument("--stage7-dir", type=Path, default=None, help="Stage 7 prediction directory used to build fixtures.")
    parser.add_argument("--samples-per-split", type=int, default=None, help="Rows per split when building fixtures.")
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_yaml(resolve_path(repo_root, Path(args.config)))
    output_root = resolve_path(repo_root, args.output_dir) if args.output_dir else resolve_path(repo_root, Path((cfg.get("outputs", {}) or {}).get("results_dir", DEFAULT_OUTPUT_ROOT)))
    output_root.mkdir(parents=True, exist_ok=True)

    candidate_id = str(cfg.get("candidate_id", "s12_halluguard_dynamics_batch"))
    main_variant = str((cfg.get("method", {}) or {}).get("main_variant", "boundary_only"))
    stage11.CANDIDATE_ID = candidate_id
    stage11.MAIN_VARIANT = main_variant

    fixture_dir = None
    if args.make_fixtures:
        fixture_dir = make_fixtures(repo_root, cfg, args)
        if args.input is None and args.input_dir is None and args.manifest is None:
            args.input_dir = fixture_dir

    input_files = collect_input_files(repo_root, args)
    if args.limit_files is not None:
        input_files = input_files[: int(args.limit_files)]
    if not input_files:
        raise ValueError("No input files found. Provide --input, --input-dir, --manifest, or --make-fixtures.")

    run_root = output_root / "batch_eval" / candidate_id
    run_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    file_records: List[Dict[str, object]] = []
    config_records: List[Dict[str, object]] = []
    diagnostics: Dict[str, List[Dict[str, object]]] = {}

    for index, prediction_path in enumerate(input_files, start=1):
        try:
            file_record, group_records, metric_rows, diag_rows = evaluate_file(
                repo_root=repo_root,
                cfg=cfg,
                candidate_id=candidate_id,
                main_variant=main_variant,
                prediction_path=prediction_path,
                run_root=run_root,
            )
            file_records.append(file_record)
            config_records.extend(group_records)
            all_rows.extend(metric_rows)
            for name, values in diag_rows.items():
                diagnostics.setdefault(name, []).extend(values)
            write_batch_outputs(output_root, run_root, candidate_id, main_variant, all_rows, file_records, config_records, diagnostics)
            print(
                json.dumps(
                    {
                        "progress": "stage12_file_done",
                        "file": str(prediction_path),
                        "status": file_record["status"],
                        "groups": file_record.get("groups", 0),
                        "completed_files": sum(1 for r in file_records if r["status"] == "completed"),
                        "total_seen": index,
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            failed_record = {
                "candidate_id": candidate_id,
                "file_id": safe_stem(prediction_path),
                "prediction_path": str(prediction_path),
                "status": "failed",
                "groups": 0,
                "blocker_reason": reason,
            }
            file_records.append(failed_record)
            all_rows.append(blocked_row(candidate_id, prediction_path, reason))
            write_batch_outputs(output_root, run_root, candidate_id, main_variant, all_rows, file_records, config_records, diagnostics)
            if not args.continue_on_error:
                raise

    write_batch_outputs(output_root, run_root, candidate_id, main_variant, all_rows, file_records, config_records, diagnostics)
    failed = [r for r in file_records if r["status"] != "completed"]
    print(
        json.dumps(
            {
                "candidate_id": candidate_id,
                "main_variant": main_variant,
                "output_dir": str(run_root),
                "completed_files": sum(1 for r in file_records if r["status"] == "completed"),
                "total_files": len(file_records),
                "completed_groups": sum(1 for r in config_records if r["status"] == "completed"),
                "failed_files": len(failed),
                "test_threshold_leakage": False,
            }
        )
    )


def evaluate_file(
    repo_root: Path,
    cfg: Dict,
    candidate_id: str,
    main_variant: str,
    prediction_path: Path,
    run_root: Path,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]], Dict[str, List[Dict[str, object]]]]:
    prediction_path = prediction_path.resolve()
    samples = stage11.load_prediction_file(prediction_path)
    validate_samples(samples, prediction_path)
    groups = group_samples(samples)
    file_id = safe_stem(prediction_path)
    file_run_root = run_root / "files" / file_id

    rows: List[Dict[str, object]] = []
    group_records: List[Dict[str, object]] = []
    diagnostics: Dict[str, List[Dict[str, object]]] = {}

    for group_key, group_samples_list in sorted(groups.items()):
        dataset, model, horizon = group_key
        val_samples = [s for s in group_samples_list if s.get("split") == "val"]
        test_samples = [s for s in group_samples_list if s.get("split") == "test"]
        if not val_samples or not test_samples:
            raise ValueError(f"{prediction_path} group {group_key} must contain both val and test rows.")

        record = stage11.ConfigRecord(dataset, model, int(horizon))
        run_dir = file_run_root / "runs" / record.tag
        run_dir.mkdir(parents=True, exist_ok=True)

        variants = stage11.default_variant_specs()
        policies = {spec.name: stage11.fit_policy(val_samples, cfg, spec) for spec in variants}
        if main_variant not in policies:
            raise ValueError(f"main_variant={main_variant} is not a HalluGuard-Dynamics policy variant.")
        main_policy = policies[main_variant]

        incumbent_cfg = stage11.load_yaml(repo_root / cfg["stage9_incumbent_config"])
        incumbent_thresholds, incumbent_policy = stage11.incumbent_eval.calibrate_evaluation_policy(val_samples, incumbent_cfg, "val")

        metric_rows, diag_rows, payload = stage11.evaluate_ablation_set(
            record=record,
            stress_type="external_batch",
            prediction_path=prediction_path,
            run_dir=run_dir,
            cfg=cfg,
            test_samples=test_samples,
            policies=policies,
            variants=variants,
            full_policy=main_policy,
            incumbent_cfg=incumbent_cfg,
            incumbent_thresholds=incumbent_thresholds,
            incumbent_policy=incumbent_policy,
        )
        payload["stage12_file_id"] = file_id
        payload["main_variant"] = main_variant
        stage11.write_run_outputs(run_dir, payload, metric_rows, diag_rows)

        for row in metric_rows:
            row.update(
                {
                    "file_id": file_id,
                    "source_file": str(prediction_path),
                    "group_id": record.tag,
                    "main_variant": main_variant,
                }
            )
        rows.extend(metric_rows)
        for name, values in diag_rows.items():
            for diag_row in values:
                diag_row.update({"file_id": file_id, "source_file": str(prediction_path), "group_id": record.tag})
            diagnostics.setdefault(name, []).extend(values)

        group_records.append(
            {
                "candidate_id": candidate_id,
                "file_id": file_id,
                "prediction_path": str(prediction_path),
                "dataset": dataset,
                "model": model,
                "horizon": int(horizon),
                "status": "completed",
                "val_rows": len(val_samples),
                "test_rows": len(test_samples),
                "output_dir": str(run_dir),
                "blocker_reason": "",
                "test_threshold_leakage": False,
            }
        )

    file_record = {
        "candidate_id": candidate_id,
        "file_id": file_id,
        "prediction_path": str(prediction_path),
        "status": "completed",
        "groups": len(group_records),
        "blocker_reason": "",
        "test_threshold_leakage": False,
    }
    return file_record, group_records, rows, diagnostics


def validate_samples(samples: List[dict], prediction_path: Path) -> None:
    if not samples:
        raise ValueError(f"{prediction_path} contains no samples.")
    for index, sample in enumerate(samples):
        missing = [field for field in REQUIRED_FIELDS if field not in sample]
        if missing:
            raise ValueError(f"{prediction_path} sample #{index} is missing fields: {missing}")
        split = str(sample["split"])
        if split not in {"val", "test"}:
            raise ValueError(f"{prediction_path} sample #{index} has unsupported split={split!r}; expected 'val' or 'test'.")
        if len(sample["prediction"]) != len(sample["target"]):
            raise ValueError(f"{prediction_path} sample #{index} prediction and target lengths differ.")
        if len(sample["prediction"]) == 0:
            raise ValueError(f"{prediction_path} sample #{index} has empty prediction horizon.")
    splits = {str(sample["split"]) for sample in samples}
    if "val" not in splits or "test" not in splits:
        raise ValueError(f"{prediction_path} must contain both split='val' and split='test'.")


def group_samples(samples: List[dict]) -> Dict[Tuple[str, str, int], List[dict]]:
    groups: Dict[Tuple[str, str, int], List[dict]] = {}
    for sample in samples:
        key = (str(sample["dataset"]), str(sample["model"]), int(len(sample["prediction"])))
        groups.setdefault(key, []).append(sample)
    return groups


def collect_input_files(repo_root: Path, args: argparse.Namespace) -> List[Path]:
    files: List[Path] = []
    if args.input is not None:
        files.append(resolve_path(repo_root, args.input))
    if args.input_dir is not None:
        input_dir = resolve_path(repo_root, args.input_dir)
        pattern = "**/*" if args.recursive else "*"
        files.extend(p for p in input_dir.glob(pattern) if is_prediction_file_candidate(p))
    if args.manifest is not None:
        files.extend(read_manifest(repo_root, resolve_path(repo_root, args.manifest)))
    deduped: List[Path] = []
    seen = set()
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return sorted(deduped)


def is_prediction_file_candidate(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() not in {".jsonl", ".csv"}:
        return False
    return path.name.lower() not in {"manifest.csv", "manifest.jsonl", "manifest.txt"}


def read_manifest(repo_root: Path, manifest_path: Path) -> List[Path]:
    suffix = manifest_path.suffix.lower()
    files: List[Path] = []
    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = row.get("path") or row.get("prediction_path") or row.get("input_path") or row.get("file")
                if value:
                    files.append(resolve_manifest_path(repo_root, manifest_path, value))
    elif suffix == ".jsonl":
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                value = row.get("path") or row.get("prediction_path") or row.get("input_path") or row.get("file")
                if value:
                    files.append(resolve_manifest_path(repo_root, manifest_path, value))
    else:
        with manifest_path.open("r", encoding="utf-8") as f:
            for line in f:
                value = line.strip()
                if value and not value.startswith("#"):
                    files.append(resolve_manifest_path(repo_root, manifest_path, value))
    return files


def make_fixtures(repo_root: Path, cfg: Dict, args: argparse.Namespace) -> Path:
    fixture_cfg = cfg.get("fixtures", {}) or {}
    source_dir = resolve_path(repo_root, args.stage7_dir) if args.stage7_dir else resolve_path(repo_root, Path(fixture_cfg.get("source_stage7_prediction_dir", "experiments/halluguard/results/stage7_big_table/predictions")))
    output_dir = resolve_path(repo_root, args.fixture_output_dir) if args.fixture_output_dir else resolve_path(repo_root, Path(fixture_cfg.get("output_dir", "experiments/halluguard/results/stage12_external_batch/fixtures/external_batch_clean")))
    samples_per_split = int(args.samples_per_split if args.samples_per_split is not None else fixture_cfg.get("samples_per_split", 32))
    output_dir.mkdir(parents=True, exist_ok=True)

    expected = [
        f"{dataset}_{model}_{horizon}.jsonl"
        for dataset in ["ETTm1", "ETTh1"]
        for model in ["DLinear", "PatchTST"]
        for horizon in [96, 192, 336, 720]
    ]
    manifest_rows = []
    for name in expected:
        source = source_dir / name
        if not source.exists():
            raise FileNotFoundError(f"Missing Stage 7 prediction file for fixture: {source}")
        out = output_dir / name
        counts = {"val": 0, "test": 0}
        with source.open("r", encoding="utf-8") as f_in, out.open("w", encoding="utf-8") as f_out:
            for line in f_in:
                if not line.strip():
                    continue
                sample = json.loads(line)
                split = sample.get("split")
                if split not in counts or counts[split] >= samples_per_split:
                    continue
                counts[split] += 1
                sample["sample_id"] = f"stage12_external_{Path(name).stem}_{sample['split']}_{counts[split]:04d}"
                sample["external_source"] = "stage7_big_table_prediction_fixture"
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                if counts["val"] >= samples_per_split and counts["test"] >= samples_per_split:
                    break
        if counts["val"] < samples_per_split or counts["test"] < samples_per_split:
            raise ValueError(f"Fixture source {source} did not provide enough val/test rows: {counts}")
        dataset, model, horizon = Path(name).stem.split("_")
        manifest_rows.append(
            {
                "path": str(out),
                "dataset": dataset,
                "model": model,
                "horizon": horizon,
                "val_rows": counts["val"],
                "test_rows": counts["test"],
            }
        )
    write_csv(manifest_rows, output_dir / "manifest.csv")
    return output_dir


def write_batch_outputs(
    output_root: Path,
    run_root: Path,
    candidate_id: str,
    main_variant: str,
    rows: List[Dict[str, object]],
    file_records: List[Dict[str, object]],
    config_records: List[Dict[str, object]],
    diagnostics: Dict[str, List[Dict[str, object]]],
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    summary = summarize_batch_rows(rows, diagnostics, main_variant)
    payload = {
        "candidate_id": candidate_id,
        "main_variant": main_variant,
        "files": file_records,
        "configs": config_records,
        "rows": rows,
        "completed_files": sum(1 for r in file_records if r.get("status") == "completed"),
        "total_files": len(file_records),
        "completed_configs": sum(1 for r in config_records if r.get("status") == "completed"),
        "total_configs": len(config_records),
        "summary": summary,
        "test_threshold_leakage": False,
    }
    write_csv(rows, run_root / "batch_metrics.csv")
    write_csv(file_records, run_root / "batch_files.csv")
    write_csv(config_records, run_root / "batch_configs.csv")
    write_csv(summary.get("variant_summary", []), run_root / "batch_variant_summary.csv")
    (run_root / "batch_metrics.json").write_text(json.dumps(payload, indent=2, default=stage11.json_default), encoding="utf-8")
    (run_root / "batch_report.md").write_text(batch_report(payload), encoding="utf-8")
    for name, diag_rows in diagnostics.items():
        write_csv(diag_rows, output_root / "diagnostics" / f"batch_{candidate_id}_{name}.csv")
    write_batch_ledger(output_root / "batch_ledger.csv", payload)


def summarize_batch_rows(rows: List[Dict[str, object]], diagnostics: Dict[str, List[Dict[str, object]]], main_variant: str) -> Dict[str, object]:
    completed = [r for r in rows if r.get("status") == "completed"]
    by_variant: Dict[str, List[Dict[str, object]]] = {}
    for row in completed:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    matched_by_key = {
        batch_key_for(row): row for row in completed if row.get("variant") == "matched_smoothing_control"
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
            matched = matched_by_key.get(batch_key_for(row))
            if matched and float(row["mse"]) < float(matched["mse"]):
                beats_matched += 1
        paired_rows = paired_by_variant.get(variant, [])
        wins = sum(1 for row in paired_rows if str(row.get("rule_beats_random_mse", "")).lower() == "true")
        variant_summary.append(
            {
                "variant": variant,
                "mean_mse_delta_pct": mean(float(r["mse_delta_pct_vs_no_correction"]) for r in variant_rows),
                "mean_mae_delta_pct": mean(float(r["mae_delta_pct_vs_no_correction"]) for r in variant_rows),
                "improved_configs": sum(1 for r in variant_rows if float(r["mse_delta_pct_vs_no_correction"]) < 0.0),
                "beats_random_configs": batch_config_level_rule_wins(paired_rows),
                "paired_rule_vs_random_win_rate": wins / len(paired_rows) if paired_rows else "",
                "beats_matched_smoothing_configs": beats_matched,
                "max_mse_harm_pct": max([float(r["mse_delta_pct_vs_no_correction"]) for r in variant_rows], default=0.0),
                "max_mae_harm_pct": max([float(r["mae_delta_pct_vs_no_correction"]) for r in variant_rows], default=0.0),
                "mean_correction_rate": mean(float(r["correction_rate"]) for r in variant_rows),
                "mean_latency_ms": mean(float(r["inference_latency_ms"]) for r in variant_rows),
                "test_threshold_leakage": any(str(r.get("test_threshold_leakage")) != "False" for r in variant_rows),
            }
        )
    main_summary = next((r for r in variant_summary if r["variant"] == main_variant), {})
    return {
        "variant_summary": variant_summary,
        "main": main_summary,
        "stress_types": sorted(set(str(r.get("stress_type", "")) for r in completed)),
        "test_threshold_leakage": any(str(r.get("test_threshold_leakage")) != "False" for r in completed),
    }


def batch_config_level_rule_wins(paired_rows: List[Dict[str, object]]) -> int:
    by_key: Dict[Tuple[str, str, str, str, str], List[bool]] = {}
    for row in paired_rows:
        key = (
            str(row.get("file_id", "")),
            str(row["stress_type"]),
            str(row["dataset"]),
            str(row["model"]),
            str(row["horizon"]),
        )
        by_key.setdefault(key, []).append(str(row.get("rule_beats_random_mse", "")).lower() == "true")
    return sum(1 for wins in by_key.values() if sum(wins) > 0.5 * len(wins))


def batch_key_for(row: Dict[str, object]) -> Tuple[str, str, str, str, str]:
    return (
        str(row.get("file_id", "")),
        str(row.get("stress_type", "")),
        str(row.get("dataset", "")),
        str(row.get("model", "")),
        str(row.get("horizon", "")),
    )


def batch_report(payload: Dict[str, object]) -> str:
    summary = payload["summary"]
    main = summary.get("main", {})
    lines = [
        "# Stage 12 External Batch Report",
        "",
        f"- Candidate: `{payload['candidate_id']}`",
        f"- Main variant: `{payload['main_variant']}`",
        f"- Completed files: {payload['completed_files']} / {payload['total_files']}",
        f"- Completed groups: {payload['completed_configs']} / {payload['total_configs']}",
        f"- Test threshold leakage: {payload['test_threshold_leakage']}",
        "",
        "## Main Variant",
        "",
        f"- Mean MSE delta: {float(main.get('mean_mse_delta_pct', 0.0)):.6f}%",
        f"- Mean MAE delta: {float(main.get('mean_mae_delta_pct', 0.0)):.6f}%",
        f"- Improved configs: {main.get('improved_configs', '')}",
        f"- Beats random configs: {main.get('beats_random_configs', '')}",
        f"- Paired rule-vs-random win rate: {main.get('paired_rule_vs_random_win_rate', '')}",
        f"- Beats matched smoothing configs: {main.get('beats_matched_smoothing_configs', '')}",
        f"- Max MSE harm: {float(main.get('max_mse_harm_pct', 0.0)):.6f}%",
        f"- Max MAE harm: {float(main.get('max_mae_harm_pct', 0.0)):.6f}%",
        "",
        "## Variant Summary",
        "",
    ]
    for row in summary.get("variant_summary", []):
        lines.append(
            f"- `{row['variant']}`: mean MSE delta {float(row['mean_mse_delta_pct']):.6f}%, "
            f"improved {row['improved_configs']}, beats random {row['beats_random_configs']}, "
            f"paired win {row['paired_rule_vs_random_win_rate']}, beats matched {row['beats_matched_smoothing_configs']}, "
            f"max MSE harm {float(row['max_mse_harm_pct']):.6f}%"
        )
    failed = [r for r in payload["files"] if r.get("status") != "completed"]
    if failed:
        lines.extend(["", "## Failed Files", ""])
        for row in failed:
            lines.append(f"- `{row['prediction_path']}`: {row.get('blocker_reason', '')}")
    return "\n".join(lines) + "\n"


def write_batch_ledger(path: Path, payload: Dict[str, object]) -> None:
    existing = []
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
    existing = [r for r in existing if r.get("candidate_id") != payload["candidate_id"]]
    main = payload["summary"].get("main", {})
    existing.append(
        {
            "candidate_id": payload["candidate_id"],
            "main_variant": payload["main_variant"],
            "status": "completed" if payload["completed_files"] == payload["total_files"] else "failed",
            "completed_files": payload["completed_files"],
            "total_files": payload["total_files"],
            "completed_configs": payload["completed_configs"],
            "total_configs": payload["total_configs"],
            "main_mean_mse_delta_pct": main.get("mean_mse_delta_pct", ""),
            "main_improved_configs": main.get("improved_configs", ""),
            "main_beats_random_configs": main.get("beats_random_configs", ""),
            "paired_rule_win_rate": main.get("paired_rule_vs_random_win_rate", ""),
            "main_beats_matched_configs": main.get("beats_matched_smoothing_configs", ""),
            "max_mse_harm_pct": main.get("max_mse_harm_pct", ""),
            "max_mae_harm_pct": main.get("max_mae_harm_pct", ""),
            "test_threshold_leakage": payload["test_threshold_leakage"],
        }
    )
    write_csv(existing, path)


def mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return float(sum(values) / len(values)) if values else 0.0


def blocked_row(candidate_id: str, prediction_path: Path, reason: str) -> Dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "file_id": safe_stem(prediction_path),
        "source_file": str(prediction_path),
        "stress_type": "external_batch",
        "dataset": "",
        "model": "",
        "horizon": "",
        "variant": "all",
        "status": "failed",
        "mse": "",
        "mae": "",
        "mse_delta_pct_vs_no_correction": "",
        "mae_delta_pct_vs_no_correction": "",
        "correction_rate": "",
        "inference_latency_ms": "",
        "test_threshold_leakage": "",
        "blocker_reason": reason,
    }


def resolve_manifest_path(repo_root: Path, manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    local = (manifest_path.parent / path).resolve()
    if local.exists():
        return local
    return (repo_root / path).resolve()


def resolve_path(repo_root: Path, path: Path | None) -> Path:
    if path is None:
        raise ValueError("Cannot resolve a missing path.")
    return path if path.is_absolute() else repo_root / path


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def safe_stem(path: Path) -> str:
    stem = path.stem.replace(" ", "_")
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in stem)
    suffix = hashlib.md5(str(path.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{safe}_{suffix}"


if __name__ == "__main__":
    main()
