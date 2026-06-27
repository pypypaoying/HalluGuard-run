#!/usr/bin/env python
"""Run Stage 6 compact mechanism validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from halluguard_lrbn_bp import load_forecast_batch_from_metrics
from halluguard_stage6_mechanism import (
    df_to_md,
    feature_schema,
    fomc_results,
    make_mechanism_sample_table,
    make_sra_predictions,
    mrc_results,
    slice_thresholds,
    split_batch,
    tae_results,
)


DEFAULT_METRICS = Path("experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv")
DEFAULT_STAGE5 = Path("experiments/halluguard/results/lrbn_sra_bp_stage5")
DEFAULT_OUTPUT = Path("experiments/halluguard/results/stage6_mechanism")


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    return str(obj)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_mrc(out: Path, result: Dict[str, Any]) -> None:
    ensure_dir(out)
    result["point"].to_csv(out / "point_residual_results.csv", index=False)
    result["quantile"].to_csv(out / "quantile_calibration.csv", index=False)
    result["abstention"].to_csv(out / "abstention_curve.csv", index=False)
    result["shrink_cap_grid"].to_csv(out / "shrink_cap_grid.csv", index=False)
    result["slice"].to_csv(out / "slice_results.csv", index=False)
    write_json(out / "bootstrap_ci.json", result["ci"])
    write_json(
        out / "selected_params.json",
        {
            "risk_threshold": result["verdict"]["selected_threshold"],
            "alpha_by_horizon": result["verdict"]["alpha_by_horizon"],
            "shrink_cap_params": result["verdict"]["shrink_cap_params"],
            "selection_source": "validation_split_only",
        },
    )
    write_json(out / "verdict.json", result["verdict"])
    lines = [
        "# Stage 6 MRC Summary",
        "",
        "## Verdict",
        "",
        f"- Safe go: `{result['verdict']['safe_go']}`",
        f"- Point pass: `{result['verdict']['point_pass']}`",
        f"- Harm pass: `{result['verdict']['harm_pass']}`",
        f"- Coverage pass: `{result['verdict']['coverage_pass']}`",
        f"- Abstention pass: `{result['verdict']['abstention_pass']}`",
        f"- Non-SRA slice pass: `{result['verdict']['non_sra_slice_pass']}`",
        f"- Test threshold leakage: `{result['verdict']['test_threshold_leakage']}`",
        "",
        "## Point Results",
        "",
        df_to_md(result["point"].sort_values("mse"), max_rows=20),
        "",
        "## Quantile Calibration",
        "",
        df_to_md(result["quantile"], max_rows=20),
        "",
        "## Slice Results",
        "",
        df_to_md(result["slice"], max_rows=30),
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_tae(out: Path, result: Dict[str, Any]) -> None:
    ensure_dir(out)
    result["candidate_table"].to_csv(out / "candidate_table.csv", index=False)
    result["candidate_table"].to_parquet(out / "candidate_table.parquet", index=False)
    result["oracle"].to_csv(out / "oracle_best_experts.csv", index=False)
    result["failure"].to_csv(out / "failure_mode_separability.csv", index=False)
    result["router"].to_csv(out / "router_ranker_results.csv", index=False)
    result["decision"].to_csv(out / "decision_eval.csv", index=False)
    write_json(out / "verdict.json", result["verdict"])
    lines = [
        "# Stage 6 TAE Summary",
        "",
        "## Verdict",
        "",
        f"- Compact go: `{result['verdict']['compact_go']}`",
        f"- Oracle gain vs LRBN: `{result['verdict']['oracle_gain_pct_vs_lrbn']:.6f}%`",
        f"- Oracle extra vs SRA-balanced: `{result['verdict']['oracle_extra_pct_vs_sra_balanced']:.6f}%`",
        f"- Router gain fraction: `{result['verdict']['router_gain_fraction']:.6f}`",
        f"- Ranker gain fraction: `{result['verdict']['ranker_gain_fraction']:.6f}`",
        f"- Top-2 hit: `{result['verdict']['router_top2_hit']:.6f}`",
        f"- Score/gain Spearman: `{result['verdict']['ranker_score_gain_spearman']:.6f}`",
        f"- Test threshold leakage: `{result['verdict']['test_threshold_leakage']}`",
        "",
        "## Decision Evaluation",
        "",
        df_to_md(result["decision"].sort_values("mse"), max_rows=20),
        "",
        "## Candidate Table",
        "",
        df_to_md(result["candidate_table"].sort_values("mse_delta_pct_vs_lrbn"), max_rows=20),
        "",
        "## Router / Ranker",
        "",
        df_to_md(result["router"], max_rows=10),
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_fomc(out: Path, result: Dict[str, Any], config: Dict[str, Any]) -> None:
    ensure_dir(out)
    result["spectral_autocorr"].to_csv(out / "spectral_autocorr.csv", index=False)
    result["adapter"].to_csv(out / "online_adapter_results.csv", index=False)
    result["conformal"].to_csv(out / "conformal_results.csv", index=False)
    write_json(out / "protocol_guard.json", result["guard"])
    write_json(out / "online_replay_config.json", config)
    write_json(out / "verdict.json", result["verdict"])
    lines = [
        "# Stage 6 FOMC Summary",
        "",
        "## Verdict",
        "",
        f"- Compact go: `{result['verdict']['compact_go']}`",
        f"- Spectral delta vs LRBN: `{result['verdict']['spectral_delta_pct_vs_lrbn']:.6f}%`",
        f"- Rolling delta vs LRBN: `{result['verdict']['rolling_delta_pct_vs_lrbn']:.6f}%`",
        f"- Spectral minus rolling: `{result['verdict']['spectral_minus_rolling_pct']:.6f}%`",
        f"- Spectral harm: `{result['verdict']['spectral_harm']:.6f}`",
        f"- Coverage gap: `{result['verdict']['coverage_gap_pp']:.6f}pp`",
        f"- Protocol guard pass: `{result['verdict']['protocol_guard_pass']}`",
        f"- Test threshold leakage: `{result['verdict']['test_threshold_leakage']}`",
        "",
        "## Online Adapter Results",
        "",
        df_to_md(result["adapter"].sort_values("mse"), max_rows=20),
        "",
        "## Spectral Autocorrelation",
        "",
        df_to_md(result["spectral_autocorr"], max_rows=20),
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_overall_summary(out: Path, mrc: Dict[str, Any], tae: Dict[str, Any], fomc: Dict[str, Any], config: Dict[str, Any]) -> None:
    verdict = {
        "stage": "stage6_mechanism",
        "mrc_go": bool(mrc["verdict"]["safe_go"]),
        "tae_go": bool(tae["verdict"]["compact_go"]),
        "fomc_go": bool(fomc["verdict"]["compact_go"]),
        "test_threshold_leakage": False,
        "decision": "promote_go_lines_only",
    }
    write_json(out / "stage6_verdict.json", verdict)
    lines = [
        "# Stage 6 Mechanism Validation Summary",
        "",
        "## Setup",
        "",
        f"- Input metrics: `{config['metrics_csv']}`",
        f"- Output directory: `{config['output_dir']}`",
        f"- Validation samples: `{config['n_val_samples']}`",
        f"- Test samples: `{config['n_test_samples']}`",
        f"- Test configs: `{config['n_test_configs']}`",
        f"- Test threshold leakage: `{verdict['test_threshold_leakage']}`",
        "",
        "## Go / No-Go",
        "",
        f"- MRC: `{verdict['mrc_go']}`",
        f"- TAE: `{verdict['tae_go']}`",
        f"- FOMC: `{verdict['fomc_go']}`",
        "",
        "## Headline Metrics",
        "",
        f"- MRC ridge-abstain MSE delta vs LRBN: `{float(mrc['point'][mrc['point']['method'].eq('MRC-ridge-abstain')]['mse_delta_pct_vs_lrbn'].iloc[0]):.6f}%`",
        f"- MRC ridge-abstain harm: `{float(mrc['point'][mrc['point']['method'].eq('MRC-ridge-abstain')]['harm_rate'].iloc[0]):.6f}`",
        f"- TAE oracle gain vs LRBN: `{tae['verdict']['oracle_gain_pct_vs_lrbn']:.6f}%`",
        f"- TAE router/ranker best gain fraction: `{max(tae['verdict']['router_gain_fraction'], tae['verdict']['ranker_gain_fraction']):.6f}`",
        f"- FOMC spectral delta vs LRBN: `{fomc['verdict']['spectral_delta_pct_vs_lrbn']:.6f}%`",
        f"- FOMC protocol guard pass: `{fomc['verdict']['protocol_guard_pass']}`",
        "",
        "## Interpretation",
        "",
        "Stage 6 is mechanism validation only. Promote only lines with `go=True`; failed lines remain useful diagnostics.",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--stage5-dir", type=Path, default=DEFAULT_STAGE5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out = ensure_dir(args.output_dir)
    batch = load_forecast_batch_from_metrics(args.metrics_csv)
    val, test = split_batch(batch)
    schema = feature_schema(val)
    thresholds = slice_thresholds(val)
    safe_params = json.loads((args.stage5_dir / "stage5_selected_safe_params.json").read_text(encoding="utf-8"))
    balanced_params = json.loads((args.stage5_dir / "stage5_selected_balanced_params.json").read_text(encoding="utf-8"))
    sra_val = make_sra_predictions(val, safe_params, balanced_params)
    sra_test = make_sra_predictions(test, safe_params, balanced_params)

    config = {
        "metrics_csv": str(args.metrics_csv),
        "stage5_dir": str(args.stage5_dir),
        "output_dir": str(args.output_dir),
        "scope": "stage6_compact_mechanism_validation",
        "datasets": sorted(test.meta["dataset"].unique().tolist()),
        "backbones": sorted(test.meta["backbone"].unique().tolist()),
        "horizons": sorted([int(x) for x in test.meta["horizon"].unique().tolist()]),
        "seeds": sorted([int(x) for x in test.meta["seed"].unique().tolist()]),
        "n_val_samples": int(len(val.meta)),
        "n_test_samples": int(len(test.meta)),
        "n_test_configs": int(test.meta.groupby(["dataset", "backbone", "horizon", "seed"]).ngroups),
        "feature_schema": schema,
        "slice_thresholds_validation_only": thresholds,
        "safe_sra_params": safe_params,
        "balanced_sra_params": balanced_params,
        "n_bootstrap": int(args.n_bootstrap),
        "seed": int(args.seed),
        "test_threshold_leakage": False,
    }
    write_json(out / "stage6_config.json", config)

    sample = make_mechanism_sample_table(test, schema, sra_test)
    sample.to_csv(out / "mechanism_sample_table.csv", index=False)

    mrc = mrc_results(val, test, sra_val, sra_test, schema, thresholds, n_bootstrap=args.n_bootstrap)
    write_mrc(out / "mrc", mrc)

    tae = tae_results(val, test, sra_val, sra_test, schema, thresholds)
    write_tae(out / "tae", tae)

    fomc_config = {
        "protocol": "chronological_replay",
        "buffer_size": 128,
        "validation_buffer": "historical_matured",
        "test_update_rule": "past test labels can be used only if local_index + horizon < current_index",
        "band_weights_source": "validation_spectral_autocorr",
        "test_threshold_leakage": False,
    }
    fomc = fomc_results(val, test)
    write_fomc(out / "fomc", fomc, fomc_config)
    write_overall_summary(out, mrc, tae, fomc, config)
    print(
        json.dumps(
            {
                "output_dir": str(out),
                "mrc_go": bool(mrc["verdict"]["safe_go"]),
                "tae_go": bool(tae["verdict"]["compact_go"]),
                "fomc_go": bool(fomc["verdict"]["compact_go"]),
                "test_threshold_leakage": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
