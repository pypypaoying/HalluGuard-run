#!/usr/bin/env python
"""Secondary Stage 4E learnable alpha adapter validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from halluguard_lrbn_bp import load_forecast_batch_from_metrics, mae_per_sample, mse_per_sample, paired_bootstrap_delta
from halluguard_stage4_bp_harm_control import boundary_features, make_boundary_delta


ALPHA_GRID = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.50], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halluguard/results/lrbn_bp_stage4_alpha"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch = load_forecast_batch_from_metrics(args.metrics_csv)
    split = batch.meta["split"].to_numpy()
    val = batch.subset(split == "val")
    test = batch.subset(split == "test")

    model, feature_info, val_rows = fit_adaptive_alpha(val, args.seed)
    test_rows, test_pred = evaluate_adaptive_alpha(test, model, feature_info, split_name="test")
    global_rows = evaluate_global_alpha(val, test)
    out = pd.concat([pd.DataFrame(val_rows), pd.DataFrame(test_rows), pd.DataFrame(global_rows)], ignore_index=True)
    out.to_csv(args.output_dir / "stage4e_learnable_alpha.csv", index=False)

    ci = paired_bootstrap_delta(mse_per_sample(test_pred, test.y_true), mse_per_sample(test.lrbn_pred, test.y_true), n_boot=args.n_bootstrap, seed=args.seed)
    (args.output_dir / "stage4e_bootstrap_ci.json").write_text(json.dumps(ci, indent=2), encoding="utf-8")
    params = {"alpha_grid": ALPHA_GRID.tolist(), "feature_columns": feature_info["columns"], "test_threshold_leakage": False}
    (args.output_dir / "stage4e_alpha_params.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    write_summary(args.output_dir, out, ci)
    print(json.dumps({"output_dir": str(args.output_dir), "adaptive_test_delta_pct": float(test_rows[0]["delta_pct_vs_lrbn"])}))


def fit_adaptive_alpha(batch, seed: int):
    delta, info = make_boundary_delta(batch, batch.lrbn_pred, alpha=1.0, anchor_mode="last", bridge_mode="linear")
    feats = boundary_features(batch, delta, info["anchor"], info["scale"])
    x_num = pd.DataFrame(
        {
            "post_lrbn_gap": feats["post_lrbn_gap"],
            "repair_ratio": feats["repair_ratio"],
            "conflict_cosine": feats["conflict_cosine"],
            "norm_ratio": feats["norm_ratio"],
            "tail_volatility": feats["tail_volatility"],
            "anchor_disagreement": feats["anchor_disagreement"],
            "horizon_norm": batch.meta["horizon"].to_numpy(float) / 720.0,
        }
    )
    enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
    cats = enc.fit_transform(batch.meta[["dataset", "backbone"]])
    x = np.concatenate([x_num.to_numpy(float), cats], axis=1)
    columns = list(x_num.columns) + [f"cat_{i}" for i in range(cats.shape[1])]
    target_alpha = oracle_alpha(batch, delta)
    reg = RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=24, random_state=seed)
    reg.fit(x, target_alpha)
    pred_alpha = np.clip(reg.predict(x), 0.0, 0.5)
    pred = batch.lrbn_pred + pred_alpha.reshape(-1, 1, 1) * delta
    rows = [summary_row("adaptive-alpha-safe-loss", "val", pred, batch, pred_alpha)]
    return reg, {"encoder": enc, "columns": columns}, rows


def evaluate_adaptive_alpha(batch, reg, feature_info, split_name: str):
    delta, info = make_boundary_delta(batch, batch.lrbn_pred, alpha=1.0, anchor_mode="last", bridge_mode="linear")
    feats = boundary_features(batch, delta, info["anchor"], info["scale"])
    x_num = pd.DataFrame(
        {
            "post_lrbn_gap": feats["post_lrbn_gap"],
            "repair_ratio": feats["repair_ratio"],
            "conflict_cosine": feats["conflict_cosine"],
            "norm_ratio": feats["norm_ratio"],
            "tail_volatility": feats["tail_volatility"],
            "anchor_disagreement": feats["anchor_disagreement"],
            "horizon_norm": batch.meta["horizon"].to_numpy(float) / 720.0,
        }
    )
    cats = feature_info["encoder"].transform(batch.meta[["dataset", "backbone"]])
    x = np.concatenate([x_num.to_numpy(float), cats], axis=1)
    alpha = np.clip(reg.predict(x), 0.0, 0.5)
    pred = batch.lrbn_pred + alpha.reshape(-1, 1, 1) * delta
    return [summary_row("adaptive-alpha-safe-loss", split_name, pred, batch, alpha)], pred


def oracle_alpha(batch, delta: np.ndarray) -> np.ndarray:
    base = mse_per_sample(batch.lrbn_pred, batch.y_true)
    objectives = []
    for alpha in ALPHA_GRID:
        pred = batch.lrbn_pred + alpha * delta
        mse = mse_per_sample(pred, batch.y_true)
        obj = mse + 2.0 * np.maximum(0.0, mse - base) + 0.01 * alpha
        objectives.append(obj)
    stack = np.vstack(objectives)
    return ALPHA_GRID[np.argmin(stack, axis=0)]


def evaluate_global_alpha(val, test) -> List[dict]:
    rows = []
    delta_val, _ = make_boundary_delta(val, val.lrbn_pred, alpha=1.0, anchor_mode="last", bridge_mode="linear")
    target = oracle_alpha(val, delta_val)
    global_alpha = float(np.median(target))
    for split_name, batch in [("val", val), ("test", test)]:
        delta, _ = make_boundary_delta(batch, batch.lrbn_pred, alpha=1.0, anchor_mode="last", bridge_mode="linear")
        alpha = np.repeat(global_alpha, len(batch.meta))
        pred = batch.lrbn_pred + global_alpha * delta
        rows.append(summary_row("global-alpha-safe-loss", split_name, pred, batch, alpha))
    return rows


def summary_row(method: str, split: str, pred: np.ndarray, batch, alpha: np.ndarray) -> Dict[str, float]:
    parent_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    method_mse = mse_per_sample(pred, batch.y_true)
    parent_mae = mae_per_sample(batch.lrbn_pred, batch.y_true)
    method_mae = mae_per_sample(pred, batch.y_true)
    delta = method_mse - parent_mse
    return {
        "method": method,
        "split": split,
        "mean_mse": float(method_mse.mean()),
        "mean_mae": float(method_mae.mean()),
        "delta_mse_vs_lrbn": float(delta.mean()),
        "delta_pct_vs_lrbn": float((method_mse.mean() - parent_mse.mean()) / (parent_mse.mean() + 1e-8) * 100.0),
        "delta_mae_pct_vs_lrbn": float((method_mae.mean() - parent_mae.mean()) / (parent_mae.mean() + 1e-8) * 100.0),
        "harm_rate_vs_lrbn": float(np.mean(delta > 1e-12)),
        "win_rate_vs_lrbn": float(np.mean(delta < 0.0)),
        "mean_alpha": float(np.mean(alpha)),
        "alpha_nonzero_rate": float(np.mean(alpha > 1e-8)),
        "test_threshold_leakage": False,
    }


def write_summary(out: Path, df: pd.DataFrame, ci: Dict[str, float]) -> None:
    lines = [
        "# Stage 4E Learnable Alpha Adapter Summary",
        "",
        "- Secondary experiment; does not replace Stage 4A-C decision.",
        "- Validation split learns alpha labels/model; test split only evaluates.",
        "",
        "## Results",
        "",
        "| Method | Split | MSE | MAE | Delta % vs LRBN | Harm | Mean alpha | Nonzero alpha |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['method']} | {r['split']} | {r['mean_mse']:.6f} | {r['mean_mae']:.6f} | "
            f"{r['delta_pct_vs_lrbn']:.6f} | {r['harm_rate_vs_lrbn']:.6f} | {r['mean_alpha']:.6f} | {r['alpha_nonzero_rate']:.6f} |"
        )
    lines.extend(["", "## Bootstrap CI", "", "```json", json.dumps(ci, indent=2), "```", ""])
    (out / "stage4e_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

