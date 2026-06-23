"""Stage 2 synthetic diagnostic hardening runner."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from correction import calibrate_thresholds
from evaluate_predictions import evaluate_variant, load_config, split_samples, write_jsonl
from run_mvp import _markdown_table
from stress import generate_synthetic_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2 multi-seed HalluGuard diagnostics.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--seeds", default="7,13,23", help="Comma-separated synthetic seeds.")
    parser.add_argument("--round-id", default="stage2_round")
    parser.add_argument("--description", default="stage2 synthetic multi-seed diagnostic")
    parser.add_argument("--append-tsv", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    config = load_config(args.config)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if len(seeds) < 3:
        raise ValueError("Stage 2 requires at least 3 seeds.")

    out_dir = repo_root / "experiments" / "halluguard" / "results" / args.round_id
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_payloads = []
    all_rows = []
    for seed in seeds:
        seed_config = deepcopy(config)
        seed_config["seed"] = seed
        payload = _run_one_seed(seed_config, seed, out_dir)
        seed_payloads.append(payload)
        all_rows.extend(_rows_for_seed(payload))

    aggregate = _aggregate(seed_payloads)
    gate = _stage2_gate(aggregate, seed_payloads)
    result = {
        "run_id": args.round_id,
        "description": args.description,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "seeds": seeds,
        "test_threshold_leakage": False,
        "seed_runs": seed_payloads,
        "aggregate": aggregate,
        "stage2_gate": gate,
    }

    (out_dir / "stage2_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_csv(all_rows, out_dir / "stage2_metrics.csv")
    (out_dir / "stage2_summary.md").write_text(_summary(result), encoding="utf-8")
    (out_dir / "stage2_ablation_mean_std.md").write_text(_mean_std_table(aggregate), encoding="utf-8")
    if args.append_tsv:
        _append_results_tsv(repo_root, result)

    print(f"Wrote {out_dir / 'stage2_metrics.json'}")
    print(f"Wrote {out_dir / 'stage2_summary.md'}")
    print(f"Stage2 status: {gate['status']}")


def _run_one_seed(config: Dict, seed: int, out_dir: Path) -> Dict:
    samples = generate_synthetic_samples(config, quick=False)
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(samples, seed_dir / "synthetic_predictions.jsonl")

    val_samples = split_samples(samples, config["thresholds"].get("source_split", "val"))
    test_samples = split_samples(samples, "test")
    high_freq_cutoff_ratio = float(config["correction"].get("high_freq_cutoff_ratio", 0.5))
    quantiles = [float(q) for q in config["thresholds"]["quantiles"]]
    default_q = float(config["thresholds"]["default_quantile"])
    lambda_trend = float(config["correction"]["default_lambda_trend"])
    lambda_freq = float(config["correction"]["default_lambda_freq"])

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
    main_ablation = [
        evaluate_variant(
            test_samples,
            thresholds,
            variant=variant,
            config=config,
            seed=seed + 1001 + idx * 31,
            lambda_trend=lambda_trend,
            lambda_freq=lambda_freq,
        )
        for idx, variant in enumerate(config["variants"])
    ]
    threshold_sensitivity = [
        evaluate_variant(
            test_samples,
            thresholds_by_q[q],
            variant="trend_frequency",
            config=config,
            seed=seed + 1201 + idx,
            lambda_trend=lambda_trend,
            lambda_freq=lambda_freq,
        )
        for idx, q in enumerate(quantiles)
    ]
    payload = {
        "seed": seed,
        "sample_counts": {
            "train": len(split_samples(samples, "train")),
            "val": len(val_samples),
            "test": len(test_samples),
        },
        "thresholds": {str(q): thresholds_by_q[q].to_dict() for q in quantiles},
        "main_ablation": main_ablation,
        "threshold_sensitivity": threshold_sensitivity,
    }
    (seed_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (seed_dir / "ablation_table.md").write_text(_markdown_table(main_ablation), encoding="utf-8")
    return payload


def _aggregate(seed_payloads: List[Dict]) -> Dict:
    variants = [m["variant"] for m in seed_payloads[0]["main_ablation"]]
    aggregate = {"main_ablation": {}, "threshold_sensitivity": {}, "seed_comparisons": []}
    for variant in variants:
        metrics = [next(m for m in p["main_ablation"] if m["variant"] == variant) for p in seed_payloads]
        aggregate["main_ablation"][variant] = _mean_std(metrics)

    quantiles = [m["threshold_quantile"] for m in seed_payloads[0]["threshold_sensitivity"]]
    for q in quantiles:
        metrics = [
            next(m for m in p["threshold_sensitivity"] if abs(m["threshold_quantile"] - q) < 1e-12)
            for p in seed_payloads
        ]
        aggregate["threshold_sensitivity"][str(q)] = _mean_std(metrics)

    for p in seed_payloads:
        no = next(m for m in p["main_ablation"] if m["variant"] == "no_correction")
        full = next(m for m in p["main_ablation"] if m["variant"] == "trend_frequency")
        random = next(m for m in p["main_ablation"] if m["variant"] == "random_trigger")
        aggregate["seed_comparisons"].append(
            {
                "seed": p["seed"],
                "full_beats_no_mse": full["mse"] < no["mse"],
                "full_beats_random_mse": full["mse"] < random["mse"],
                "hallucination_drop_pct": _drop_pct(no["hallucination_rate"], full["hallucination_rate"]),
                "clean_mse_harm_pct": _slice(full, "clean", "mse_delta_pct_vs_original"),
                "turning_point_mse_harm_pct": _slice(full, "real_turning_point", "mse_delta_pct_vs_original"),
                "high_frequency_noise_mse_delta_pct": _slice(full, "high_frequency_noise", "mse_delta_pct_vs_original"),
                "local_oscillation_mse_delta_pct": _slice(full, "local_oscillation", "mse_delta_pct_vs_original"),
            }
        )
    return aggregate


def _mean_std(metrics: List[Dict]) -> Dict:
    keys = [
        "mse",
        "mae",
        "hallucination_rate",
        "trend_violation_rate",
        "freq_violation_rate",
        "spectral_consistency",
        "turning_point_false_correction_rate",
        "correction_rate",
        "inference_latency_ms",
    ]
    out = {}
    for key in keys:
        values = np.asarray([float(m[key]) for m in metrics], dtype=np.float64)
        out[f"{key}_mean"] = float(values.mean())
        out[f"{key}_std"] = float(values.std(ddof=0))
    for stress_type in ["clean", "trend_drift", "high_frequency_noise", "local_oscillation", "real_turning_point"]:
        values = np.asarray([_slice(m, stress_type, "mse_delta_pct_vs_original") for m in metrics], dtype=np.float64)
        out[f"{stress_type}_mse_delta_pct_mean"] = float(values.mean())
        out[f"{stress_type}_mse_delta_pct_std"] = float(values.std(ddof=0))
    return out


def _stage2_gate(aggregate: Dict, seed_payloads: List[Dict]) -> Dict:
    no = aggregate["main_ablation"]["no_correction"]
    full = aggregate["main_ablation"]["trend_frequency"]
    random = aggregate["main_ablation"]["random_trigger"]
    comparisons = aggregate["seed_comparisons"]
    rule_beats_random_count = sum(1 for row in comparisons if row["full_beats_random_mse"])
    rule_beats_random_majority = rule_beats_random_count > len(comparisons) / 2
    hallucination_drop_pct = _drop_pct(no["hallucination_rate_mean"], full["hallucination_rate_mean"])
    threshold_improves = all(
        q_stats["mse_mean"] < no["mse_mean"]
        for q_stats in aggregate["threshold_sensitivity"].values()
    )
    threshold_mse_range = max(
        q_stats["mse_mean"] for q_stats in aggregate["threshold_sensitivity"].values()
    ) - min(q_stats["mse_mean"] for q_stats in aggregate["threshold_sensitivity"].values())
    checks = {
        "mean_mse_improves": full["mse_mean"] < no["mse_mean"],
        "mean_hallucination_drop_ge_20pct": hallucination_drop_pct >= 20.0,
        "rule_beats_random_majority": rule_beats_random_majority,
        "rule_beats_random_count": rule_beats_random_count,
        "n_seeds": len(comparisons),
        "turning_point_mse_harm_lt_3pct": full["real_turning_point_mse_delta_pct_mean"] < 3.0,
        "turning_point_mse_harm_lt_1pct": full["real_turning_point_mse_delta_pct_mean"] < 1.0,
        "clean_mse_harm_lt_1pct": full["clean_mse_delta_pct_mean"] < 1.0,
        "high_frequency_noise_benefit_preserved": full["high_frequency_noise_mse_delta_pct_mean"] < 0.0,
        "local_oscillation_benefit_preserved": full["local_oscillation_mse_delta_pct_mean"] < 0.0,
        "threshold_all_quantiles_improve_mse": threshold_improves,
        "threshold_mse_mean_range": float(threshold_mse_range),
        "rule_mse_mean_advantage_vs_random": random["mse_mean"] - full["mse_mean"],
        "hallucination_drop_pct": hallucination_drop_pct,
    }
    required = [
        "mean_mse_improves",
        "mean_hallucination_drop_ge_20pct",
        "rule_beats_random_majority",
        "turning_point_mse_harm_lt_3pct",
        "clean_mse_harm_lt_1pct",
        "high_frequency_noise_benefit_preserved",
        "local_oscillation_benefit_preserved",
        "threshold_all_quantiles_improve_mse",
    ]
    status = "pass" if all(checks[k] for k in required) else "fail"
    checks["status"] = status
    return checks


def _summary(result: Dict) -> str:
    agg = result["aggregate"]
    gate = result["stage2_gate"]
    no = agg["main_ablation"]["no_correction"]
    full = agg["main_ablation"]["trend_frequency"]
    random = agg["main_ablation"]["random_trigger"]
    comparisons = "\n".join(
        "- seed {seed}: full<random={full_beats_random_mse}, hallucination_drop={hallucination_drop_pct:.2f}%, clean_harm={clean_mse_harm_pct:.2f}%, tp_harm={turning_point_mse_harm_pct:.2f}%".format(**row)
        for row in agg["seed_comparisons"]
    )
    return f"""# Stage 2 Synthetic Diagnostic Summary

## Run

- Run id: `{result['run_id']}`
- Seeds: {', '.join(str(s) for s in result['seeds'])}
- Test threshold leakage: `{result['test_threshold_leakage']}`
- Status: **{gate['status']}**

## Mean/Std Headline

- no_correction MSE: {no['mse_mean']:.6f} +/- {no['mse_std']:.6f}
- trend_frequency MSE: {full['mse_mean']:.6f} +/- {full['mse_std']:.6f}
- random_trigger MSE: {random['mse_mean']:.6f} +/- {random['mse_std']:.6f}
- HallucinationRate drop: {gate['hallucination_drop_pct']:.2f}%
- Rule beats random seeds: {gate['rule_beats_random_count']} / {gate['n_seeds']}
- Clean MSE harm: {full['clean_mse_delta_pct_mean']:.2f}% +/- {full['clean_mse_delta_pct_std']:.2f}%
- Turning-point MSE harm: {full['real_turning_point_mse_delta_pct_mean']:.2f}% +/- {full['real_turning_point_mse_delta_pct_std']:.2f}%
- High-frequency-noise MSE delta: {full['high_frequency_noise_mse_delta_pct_mean']:.2f}% +/- {full['high_frequency_noise_mse_delta_pct_std']:.2f}%
- Local-oscillation MSE delta: {full['local_oscillation_mse_delta_pct_mean']:.2f}% +/- {full['local_oscillation_mse_delta_pct_std']:.2f}%

## Gate Checks

- Mean MSE improves over no correction: {gate['mean_mse_improves']}
- HallucinationRate mean drop >= 20%: {gate['mean_hallucination_drop_ge_20pct']}
- Rule trigger beats random in most seeds: {gate['rule_beats_random_majority']}
- Real turning-point MSE harm < 3%: {gate['turning_point_mse_harm_lt_3pct']}
- Real turning-point MSE harm < 1%: {gate['turning_point_mse_harm_lt_1pct']}
- Clean MSE harm < 1%: {gate['clean_mse_harm_lt_1pct']}
- Threshold sensitivity not extreme: {gate['threshold_all_quantiles_improve_mse']} (mean MSE range {gate['threshold_mse_mean_range']:.6f})

## Per-Seed Rule Vs Random

{comparisons}

## Mean/Std Ablation

{_mean_std_table(agg)}
"""


def _mean_std_table(aggregate: Dict) -> str:
    headers = ["variant", "MSE mean", "MSE std", "MAE mean", "Hallucination mean", "TP harm mean", "Clean harm mean"]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for variant, stats in aggregate["main_ablation"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    variant,
                    f"{stats['mse_mean']:.6f}",
                    f"{stats['mse_std']:.6f}",
                    f"{stats['mae_mean']:.6f}",
                    f"{stats['hallucination_rate_mean']:.3f}",
                    f"{stats['real_turning_point_mse_delta_pct_mean']:.2f}%",
                    f"{stats['clean_mse_delta_pct_mean']:.2f}%",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _rows_for_seed(payload: Dict) -> List[Dict[str, object]]:
    rows = []
    for metric in payload["main_ablation"]:
        row = {
            "table": "main_ablation",
            "seed": payload["seed"],
            "variant": metric["variant"],
            "threshold_quantile": metric["threshold_quantile"],
            "mse": metric["mse"],
            "mae": metric["mae"],
            "hallucination_rate": metric["hallucination_rate"],
            "trend_violation_rate": metric["trend_violation_rate"],
            "freq_violation_rate": metric["freq_violation_rate"],
            "turning_point_false_correction_rate": metric["turning_point_false_correction_rate"],
            "clean_mse_delta_pct": _slice(metric, "clean", "mse_delta_pct_vs_original"),
            "turning_point_mse_delta_pct": _slice(metric, "real_turning_point", "mse_delta_pct_vs_original"),
            "high_frequency_noise_mse_delta_pct": _slice(metric, "high_frequency_noise", "mse_delta_pct_vs_original"),
            "local_oscillation_mse_delta_pct": _slice(metric, "local_oscillation", "mse_delta_pct_vs_original"),
        }
        rows.append(row)
    return rows


def _write_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _append_results_tsv(repo_root: Path, result: Dict) -> None:
    full = result["aggregate"]["main_ablation"]["trend_frequency"]
    gate = result["stage2_gate"]
    status = "kept" if gate["status"] == "pass" else "reverted"
    description = (
        f"{result['run_id']}: {result['description']}; seeds={len(result['seeds'])}; "
        f"rule_beats_random={gate['rule_beats_random_count']}/{gate['n_seeds']}; "
        f"tp_harm={full['real_turning_point_mse_delta_pct_mean']:.2f}%; clean_harm={full['clean_mse_delta_pct_mean']:.2f}%"
    )
    row = [
        _git_commit(repo_root),
        f"{full['mse_mean']:.10f}",
        f"{full['mae_mean']:.10f}",
        f"{full['hallucination_rate_mean']:.10f}",
        f"{full['trend_violation_rate_mean']:.10f}",
        f"{full['freq_violation_rate_mean']:.10f}",
        f"{full['turning_point_false_correction_rate_mean']:.10f}",
        status,
        description,
    ]
    with (repo_root / "results_halluguard.tsv").open("a", encoding="utf-8", newline="") as f:
        f.write("\t".join(row) + "\n")


def _slice(metric: Dict, stress_type: str, key: str) -> float:
    return float(metric["slices"].get(stress_type, {}).get(key, 0.0))


def _drop_pct(before: float, after: float) -> float:
    if abs(before) < 1e-12:
        return 0.0
    return 100.0 * (before - after) / before


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


if __name__ == "__main__":
    main()
