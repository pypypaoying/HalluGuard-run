"""Run the HalluGuard synthetic/stress MVP."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from correction import calibrate_thresholds
from evaluate_predictions import (
    evaluate_variant,
    flatten_metric_row,
    load_config,
    split_samples,
    write_jsonl,
    write_metrics_csv,
)
from stress import generate_synthetic_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HalluGuard MVP ablations.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test.")
    args = parser.parse_args()

    config = load_config(args.config)
    repo_root = Path(__file__).resolve().parents[2]
    result_paths = _result_paths(repo_root, config, quick=args.quick)
    for path in result_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    samples = generate_synthetic_samples(config, quick=args.quick)
    write_jsonl(samples, result_paths["predictions_jsonl"])

    val_samples = split_samples(samples, config["thresholds"].get("source_split", "val"))
    test_samples = split_samples(samples, "test")
    high_freq_cutoff_ratio = float(config["correction"].get("high_freq_cutoff_ratio", 0.5))
    quantiles = [float(q) for q in config["thresholds"]["quantiles"]]
    default_q = float(config["thresholds"]["default_quantile"])
    default_lambda_trend = float(config["correction"]["default_lambda_trend"])
    default_lambda_freq = float(config["correction"]["default_lambda_freq"])

    thresholds_by_q = {
        q: calibrate_thresholds(
            val_samples,
            quantile=q,
            high_freq_cutoff_ratio=high_freq_cutoff_ratio,
            source_split=config["thresholds"].get("source_split", "val"),
        )
        for q in quantiles
    }
    thresholds = thresholds_by_q[default_q]

    seed_base = int(config.get("seed", 7)) + (101 if args.quick else 1001)
    main_ablation = []
    for idx, variant in enumerate(config["variants"]):
        main_ablation.append(
            evaluate_variant(
                test_samples,
                thresholds,
                variant=variant,
                config=config,
                seed=seed_base + idx * 31,
                lambda_trend=default_lambda_trend,
                lambda_freq=default_lambda_freq,
            )
        )

    threshold_sensitivity = []
    for idx, q in enumerate(quantiles):
        threshold_sensitivity.append(
            evaluate_variant(
                test_samples,
                thresholds_by_q[q],
                variant="trend_frequency",
                config=config,
                seed=seed_base + 200 + idx,
                lambda_trend=default_lambda_trend,
                lambda_freq=default_lambda_freq,
            )
        )

    strength_sensitivity = []
    for idx, strength in enumerate([float(v) for v in config["correction"]["lambda_trend_values"]]):
        strength_sensitivity.append(
            evaluate_variant(
                test_samples,
                thresholds,
                variant="trend_frequency",
                config=config,
                seed=seed_base + 300 + idx,
                lambda_trend=strength,
                lambda_freq=strength,
            )
        )

    stop_checks = _stop_checks(main_ablation)
    payload = {
        "run_id": "halluguard_mvp_round1_quick" if args.quick else "halluguard_mvp_round1",
        "quick": bool(args.quick),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "threshold_source_split": config["thresholds"].get("source_split", "val"),
        "test_threshold_leakage": False,
        "sample_counts": {
            "train": len(split_samples(samples, "train")),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "thresholds": {str(q): thresholds_by_q[q].to_dict() for q in quantiles},
        "main_ablation": main_ablation,
        "threshold_sensitivity": threshold_sensitivity,
        "strength_sensitivity": strength_sensitivity,
        "stop_checks": stop_checks,
        "predictions_jsonl": str(result_paths["predictions_jsonl"]),
    }

    result_paths["metrics_json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_rows = []
    csv_rows.extend(flatten_metric_row("main_ablation", m) for m in main_ablation)
    csv_rows.extend(flatten_metric_row("threshold_sensitivity", m) for m in threshold_sensitivity)
    csv_rows.extend(flatten_metric_row("strength_sensitivity", m) for m in strength_sensitivity)
    write_metrics_csv(csv_rows, result_paths["metrics_csv"])
    result_paths["ablation_table"].write_text(_markdown_table(main_ablation), encoding="utf-8")
    result_paths["summary"].write_text(_summary_markdown(payload), encoding="utf-8")

    if not args.quick:
        _append_results_tsv(repo_root, payload)

    print(f"Wrote {result_paths['metrics_json']}")
    print(f"Wrote {result_paths['metrics_csv']}")
    print(f"Wrote {result_paths['summary']}")
    print(f"Status: {stop_checks['status']}")


def _result_paths(repo_root: Path, config: Dict, quick: bool) -> Dict[str, Path]:
    outputs = config["outputs"]
    suffix = "_quick" if quick else ""

    def path_for(key: str) -> Path:
        raw = Path(outputs[key])
        if not raw.is_absolute():
            raw = repo_root / raw
        if suffix and raw.suffix:
            raw = raw.with_name(raw.stem + suffix + raw.suffix)
        return raw

    results_dir = Path(outputs["results_dir"])
    if not results_dir.is_absolute():
        results_dir = repo_root / results_dir
    return {
        "metrics_json": path_for("metrics_json"),
        "metrics_csv": path_for("metrics_csv"),
        "ablation_table": path_for("ablation_table"),
        "summary": path_for("summary"),
        "predictions_jsonl": results_dir / ("synthetic_predictions_quick.jsonl" if quick else "synthetic_predictions.jsonl"),
    }


def _stop_checks(main_ablation: List[Dict]) -> Dict[str, object]:
    by_variant = {m["variant"]: m for m in main_ablation}
    no = by_variant["no_correction"]
    full = by_variant["trend_frequency"]
    random = by_variant["random_trigger"]
    naive = by_variant["naive_smoothing"]
    trend_only = by_variant["trend_only"]
    freq_only = by_variant["frequency_only"]

    mse_delta_vs_no = _pct(full["mse"], no["mse"])
    mae_delta_vs_no = _pct(full["mae"], no["mae"])
    hallucination_drop_pct = _drop_pct(no["hallucination_rate"], full["hallucination_rate"])
    full_mse_gain = no["mse"] - full["mse"]
    random_mse_gain = no["mse"] - random["mse"]
    rule_beats_random = full["mse"] < random["mse"] and full["hallucination_rate"] <= random["hallucination_rate"]
    random_close = (
        full_mse_gain > 0
        and random_mse_gain > 0
        and random_mse_gain >= 0.80 * full_mse_gain
    )
    mse_safe = mse_delta_vs_no <= 3.0 and mae_delta_vs_no <= 3.0
    safety_ok = full["turning_point_false_correction_rate"] <= naive["turning_point_false_correction_rate"] + 0.05
    clean_ok = full["slices"].get("clean", {}).get("mse_delta_pct_vs_original", 0.0) <= 3.0

    if full["mse"] <= no["mse"] and hallucination_drop_pct > 20.0 and rule_beats_random and mse_safe and safety_ok and clean_ok:
        status = "pass"
    elif not mse_safe or random_close or not safety_ok:
        status = "fail"
    else:
        status = "mixed"

    return {
        "status": status,
        "mse_delta_pct_vs_no_correction": mse_delta_vs_no,
        "mae_delta_pct_vs_no_correction": mae_delta_vs_no,
        "hallucination_drop_pct_vs_no_correction": hallucination_drop_pct,
        "rule_beats_random": bool(rule_beats_random),
        "random_close_to_rule": bool(random_close),
        "mse_safe_under_3pct": bool(mse_safe),
        "turning_point_safety_ok": bool(safety_ok),
        "clean_slice_ok": bool(clean_ok),
        "trend_only_mse": trend_only["mse"],
        "frequency_only_mse": freq_only["mse"],
        "dominant_component": "trend" if trend_only["mse"] < freq_only["mse"] else "frequency",
    }


def _summary_markdown(payload: Dict) -> str:
    by_variant = {m["variant"]: m for m in payload["main_ablation"]}
    no = by_variant["no_correction"]
    full = by_variant["trend_frequency"]
    random = by_variant["random_trigger"]
    naive = by_variant["naive_smoothing"]
    checks = payload["stop_checks"]
    full_slices = full["slices"]
    best_stress = sorted(
        ((name, vals["mse_delta_pct_vs_original"]) for name, vals in full_slices.items()),
        key=lambda x: x[1],
    )
    threshold_lines = "\n".join(
        f"- q={m['threshold_quantile']:.2f}: MSE={m['mse']:.6f}, hallucination={m['hallucination_rate']:.3f}, false_correction={m['turning_point_false_correction_rate']:.3f}"
        for m in payload["threshold_sensitivity"]
    )
    strength_lines = "\n".join(
        f"- lambda={m['lambda_trend']:.1f}: MSE={m['mse']:.6f}, hallucination={m['hallucination_rate']:.3f}, false_correction={m['turning_point_false_correction_rate']:.3f}"
        for m in payload["strength_sensitivity"]
    )
    decision = (
        "worth a small real-prediction pilot after turning-point diagnostics, not a broad dataset expansion"
        if checks["status"] == "pass"
        else "not ready for real predictions without another diagnostic pass"
    )
    return f"""# HalluGuard MVP Summary

## Run

- Run id: `{payload['run_id']}`
- Threshold source split: `{payload['threshold_source_split']}`
- Test threshold leakage: `{payload['test_threshold_leakage']}`
- Test samples: {payload['sample_counts']['test']}

## Headline

Default HalluGuard `trend_frequency` changed MSE from {no['mse']:.6f} to {full['mse']:.6f} ({checks['mse_delta_pct_vs_no_correction']:.2f}%). HallucinationRate changed from {no['hallucination_rate']:.3f} to {full['hallucination_rate']:.3f} ({checks['hallucination_drop_pct_vs_no_correction']:.2f}% drop). Status: **{checks['status']}**.

## Main Ablation

{_markdown_table(payload['main_ablation'])}

## Stop Checks

- Rule trigger beats random trigger: {checks['rule_beats_random']} (random MSE {random['mse']:.6f}, full MSE {full['mse']:.6f})
- MSE/MAE degradation under 3%: {checks['mse_safe_under_3pct']}
- Turning-point safety versus naive smoothing: {checks['turning_point_safety_ok']} (full {full['turning_point_false_correction_rate']:.3f}, naive {naive['turning_point_false_correction_rate']:.3f})
- Clean slice not materially harmed: {checks['clean_slice_ok']}

## Sensitivity

Threshold sensitivity:
{threshold_lines}

Correction-strength sensitivity:
{strength_lines}

## Error Slice Analysis

Most improved slices under `trend_frequency`:
{_slice_lines(best_stress)}

Turning-point slice MSE delta versus original: {full_slices.get('real_turning_point', {}).get('mse_delta_pct_vs_original', 0.0):.2f}%.

## Mechanism Read

- Dominant component by MSE: `{checks['dominant_component']}`.
- Trend-only MSE: {checks['trend_only_mse']:.6f}; frequency-only MSE: {checks['frequency_only_mse']:.6f}; full MSE: {full['mse']:.6f}.
- Random trigger uses the same unsupervised rule-trigger count but randomizes which samples are corrected, with no target access.

## Next Decision

This first round is {decision}. The result remains synthetic/stress-only evidence and should not be described as real TSF benchmark performance.
"""


def _markdown_table(metrics: List[Dict]) -> str:
    headers = [
        "variant",
        "MSE",
        "MAE",
        "HallucinationRate",
        "TrendViolationRate",
        "FreqViolationRate",
        "SpectralConsistency",
        "TPFalseCorrection",
        "LatencyMs",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for metric in metrics:
        lines.append(
            "| "
            + " | ".join(
                [
                    metric["variant"],
                    f"{metric['mse']:.6f}",
                    f"{metric['mae']:.6f}",
                    f"{metric['hallucination_rate']:.3f}",
                    f"{metric['trend_violation_rate']:.3f}",
                    f"{metric['freq_violation_rate']:.3f}",
                    f"{metric['spectral_consistency']:.3f}",
                    f"{metric['turning_point_false_correction_rate']:.3f}",
                    f"{metric['inference_latency_ms']:.4f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _slice_lines(slices: List) -> str:
    return "\n".join(f"- {name}: {delta:.2f}% MSE delta vs original" for name, delta in slices)


def _append_results_tsv(repo_root: Path, payload: Dict) -> None:
    full = next(m for m in payload["main_ablation"] if m["variant"] == "trend_frequency")
    checks = payload["stop_checks"]
    commit = _git_commit(repo_root)
    description = (
        f"round1 synthetic stress MVP; full vs no MSE delta {checks['mse_delta_pct_vs_no_correction']:.2f}%; "
        f"rule_beats_random={checks['rule_beats_random']}"
    )
    row = [
        commit,
        f"{full['mse']:.10f}",
        f"{full['mae']:.10f}",
        f"{full['hallucination_rate']:.10f}",
        f"{full['trend_violation_rate']:.10f}",
        f"{full['freq_violation_rate']:.10f}",
        f"{full['turning_point_false_correction_rate']:.10f}",
        str(checks["status"]),
        description,
    ]
    tsv_path = repo_root / "results_halluguard.tsv"
    with tsv_path.open("a", encoding="utf-8", newline="") as f:
        f.write("\t".join(row) + "\n")


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "uncommitted"


def _pct(value: float, baseline: float) -> float:
    if abs(baseline) < 1e-12:
        return 0.0
    return 100.0 * (value - baseline) / baseline


def _drop_pct(before: float, after: float) -> float:
    if abs(before) < 1e-12:
        return 0.0
    return 100.0 * (before - after) / before


if __name__ == "__main__":
    main()
