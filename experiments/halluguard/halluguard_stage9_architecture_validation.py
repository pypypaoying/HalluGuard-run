#!/usr/bin/env python
"""Stage 9 large-architecture validation for HalluGuard.

This stage is a compact mechanism screen, not a TableA method submission.  It
validates the large-architecture hypotheses from the deep research report by
reusing LRBN, SRA-BP, MRC, SafeTAE, and online spectral assets under the same
validation-only/test-only compact protocol.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from halluguard_lrbn_bp import (
    EPS,
    ForecastBatch,
    load_forecast_batch_from_metrics,
    mae_per_sample,
    mse_per_sample,
)
from halluguard_stage6_mechanism import (
    bootstrap_ci as sample_bootstrap_ci,
    calibrate_spectral_band_weights,
    feature_frame,
    feature_schema,
    fit_ridge_residual_models,
    fomc_results,
    horizons,
    mrc_cap_matrix,
    online_adapter_eval,
    safe_pct,
    slice_thresholds,
    valid_part,
    volatility_shrink_candidate,
)
from halluguard_stage7_safe_tae import (
    ExpertCandidate,
    align_frame,
    build_candidate_pool,
    build_mrc_artifacts,
    candidate_dict,
    compute_probabilities,
    expert_features,
    fit_heads,
    load_stage3_params,
    oracle_best,
    select_basic_sra_params,
    split_batch,
    stratified_inner_split,
    subset_candidates,
    write_json,
)
from halluguard_stage8_safe_tae_pareto import stage8_slice_masks, stage8_slice_thresholds


@dataclass
class Stage9Assets:
    batch: ForecastBatch
    val: ForecastBatch
    test: ForecastBatch
    val_train: ForecastBatch
    val_calib: ForecastBatch
    train_mask: np.ndarray
    calib_mask: np.ndarray
    schema: Dict[str, List[Any]]
    mrc: Dict[str, Any]
    old_val_candidates: List[ExpertCandidate]
    old_test_candidates: List[ExpertCandidate]
    old_train_candidates: List[ExpertCandidate]
    old_calib_candidates: List[ExpertCandidate]
    new_val_candidates: List[ExpertCandidate]
    new_test_candidates: List[ExpertCandidate]
    new_train_candidates: List[ExpertCandidate]
    new_calib_candidates: List[ExpertCandidate]
    expanded_val_candidates: List[ExpertCandidate]
    expanded_test_candidates: List[ExpertCandidate]
    expanded_train_candidates: List[ExpertCandidate]
    expanded_calib_candidates: List[ExpertCandidate]
    old_heads: Any
    old_calib_probs: Dict[str, Any]
    old_test_probs: Dict[str, Any]


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return str(obj)


def df_to_md(df: pd.DataFrame, max_rows: int = 24) -> str:
    if df.empty:
        return "_empty_"
    show = df.head(max_rows)
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def config_key_series(meta: pd.DataFrame) -> pd.Series:
    return (
        meta["dataset"].astype(str)
        + "/"
        + meta["backbone"].astype(str)
        + "/"
        + meta["horizon"].astype(str)
        + "/seed"
        + meta["seed"].astype(str)
    )


def instance_key_series(meta: pd.DataFrame) -> pd.Series:
    return config_key_series(meta) + "::" + meta["sample_key"].astype(str)


def metric_row(
    variant: str,
    pred: np.ndarray,
    batch: ForecastBatch,
    selected: Optional[np.ndarray] = None,
    oracle_mse: Optional[np.ndarray] = None,
    n_bootstrap: int = 0,
    seed: int = 2026,
) -> Dict[str, Any]:
    method_mse = mse_per_sample(pred, batch.y_true)
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    method_mae = mae_per_sample(pred, batch.y_true)
    base_mae = mae_per_sample(batch.lrbn_pred, batch.y_true)
    delta = method_mse - base_mse
    config_harms: List[float] = []
    config_improved: List[bool] = []
    meta = batch.meta.assign(row_index=np.arange(len(batch.meta)))
    for _, group in meta.groupby(["dataset", "backbone", "horizon", "seed"], observed=True):
        idx = group["row_index"].to_numpy(int)
        d = delta[idx]
        config_harms.append(float(np.mean(d > 1e-12)))
        config_improved.append(bool(np.mean(d) < 0.0))
    row: Dict[str, Any] = {
        "variant": variant,
        "n": int(len(batch.meta)),
        "mse": float(np.mean(method_mse)),
        "mae": float(np.mean(method_mae)),
        "lrbn_mse": float(np.mean(base_mse)),
        "lrbn_mae": float(np.mean(base_mae)),
        "mse_delta_vs_lrbn": float(np.mean(delta)),
        "mse_delta_pct_vs_lrbn": safe_pct(float(np.mean(method_mse)), float(np.mean(base_mse))),
        "mae_delta_pct_vs_lrbn": safe_pct(float(np.mean(method_mae)), float(np.mean(base_mae))),
        "harm_rate": float(np.mean(delta > 1e-12)),
        "win_rate": float(np.mean(delta < 0.0)),
        "max_config_harm": float(np.max(config_harms)) if config_harms else float(np.mean(delta > 1e-12)),
        "config_improved_ratio": float(np.mean(config_improved)) if config_improved else float("nan"),
        "improved_configs": int(np.sum(config_improved)),
        "total_configs": int(len(config_improved)),
        "test_threshold_leakage": False,
    }
    if selected is not None:
        selected = np.asarray(selected, dtype=bool)
        row["coverage"] = float(np.mean(selected))
        row["selected_count"] = int(np.sum(selected))
        row["selected_harm_rate"] = float(np.mean((delta > 1e-12)[selected])) if selected.any() else 0.0
    else:
        row["coverage"] = 0.0
        row["selected_count"] = 0
        row["selected_harm_rate"] = 0.0
    if oracle_mse is not None:
        denom = float(np.mean(base_mse - oracle_mse))
        row["oracle_gain_fraction"] = float(np.mean(base_mse - method_mse) / (denom + EPS))
    else:
        row["oracle_gain_fraction"] = float("nan")
    if n_bootstrap > 0:
        ci = sample_bootstrap_ci(delta, n_boot=n_bootstrap, seed=seed)
        row["ci95_low_delta_raw"] = ci["ci95_low"]
        row["ci95_high_delta_raw"] = ci["ci95_high"]
        row["p_bootstrap_delta_lt_zero"] = ci["p_lt_zero"]
    return row


def per_config_rows(variant: str, pred: np.ndarray, batch: ForecastBatch, selected: Optional[np.ndarray] = None) -> pd.DataFrame:
    method_mse = mse_per_sample(pred, batch.y_true)
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    method_mae = mae_per_sample(pred, batch.y_true)
    base_mae = mae_per_sample(batch.lrbn_pred, batch.y_true)
    selected = np.zeros(len(batch.meta), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    rows: List[Dict[str, Any]] = []
    meta = batch.meta.assign(row_index=np.arange(len(batch.meta)))
    for keys, group in meta.groupby(["dataset", "backbone", "horizon", "seed"], observed=True):
        idx = group["row_index"].to_numpy(int)
        delta = method_mse[idx] - base_mse[idx]
        rows.append(
            {
                "variant": variant,
                "dataset": keys[0],
                "backbone": keys[1],
                "horizon": int(keys[2]),
                "seed": int(keys[3]),
                "n": int(len(idx)),
                "mse": float(np.mean(method_mse[idx])),
                "mae": float(np.mean(method_mae[idx])),
                "mse_delta_pct_vs_lrbn": safe_pct(float(np.mean(method_mse[idx])), float(np.mean(base_mse[idx]))),
                "mae_delta_pct_vs_lrbn": safe_pct(float(np.mean(method_mae[idx])), float(np.mean(base_mae[idx]))),
                "harm_rate": float(np.mean(delta > 1e-12)),
                "win_rate": float(np.mean(delta < 0.0)),
                "coverage": float(np.mean(selected[idx])),
            }
        )
    return pd.DataFrame(rows)


def slice_rows(
    variant: str,
    pred: np.ndarray,
    batch: ForecastBatch,
    masks: Mapping[str, np.ndarray],
    selected: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    method_mse = mse_per_sample(pred, batch.y_true)
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    selected = np.zeros(len(batch.meta), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    rows: List[Dict[str, Any]] = []
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        d = method_mse[mask] - base_mse[mask]
        rows.append(
            {
                "variant": variant,
                "slice": name,
                "n": int(mask.sum()),
                "mse": float(np.mean(method_mse[mask])),
                "lrbn_mse": float(np.mean(base_mse[mask])),
                "mse_delta_pct_vs_lrbn": safe_pct(float(np.mean(method_mse[mask])), float(np.mean(base_mse[mask]))),
                "harm_rate": float(np.mean(d > 1e-12)),
                "win_rate": float(np.mean(d < 0.0)),
                "coverage": float(np.mean(selected[mask])),
            }
        )
    return pd.DataFrame(rows)


def deployable_candidates(candidates: Sequence[ExpertCandidate], include_aggressive: bool = False) -> List[ExpertCandidate]:
    return [c for c in candidates if c.name == "keep_lrbn" or c.deployable or include_aggressive]


def valid_flat_delta(batch: ForecastBatch, delta: np.ndarray, idx: int, h: int) -> np.ndarray:
    return np.asarray(delta[idx, :h, :], dtype=float).reshape(-1)


def _empty_delta_like(batch: ForecastBatch) -> np.ndarray:
    return np.zeros_like(batch.lrbn_pred, dtype=float)


def residual_median_delta(train: ForecastBatch, batch: ForecastBatch) -> np.ndarray:
    out = _empty_delta_like(batch)
    train_meta = train.meta.assign(row_index=np.arange(len(train.meta)))
    batch_meta = batch.meta.assign(row_index=np.arange(len(batch.meta)))
    residual = train.y_true - train.lrbn_pred
    horizon_fallback: Dict[int, np.ndarray] = {}
    for h, group in train_meta.groupby("horizon", observed=True):
        h = int(h)
        idx = group["row_index"].to_numpy(int)
        horizon_fallback[h] = np.nanmedian(np.stack([valid_part(residual, int(i), h) for i in idx], axis=0), axis=0)
    for keys, group in batch_meta.groupby(["dataset", "backbone", "horizon", "seed"], observed=True):
        dataset, backbone, horizon, seed = keys
        h = int(horizon)
        tr = train_meta[
            train_meta["dataset"].eq(dataset)
            & train_meta["backbone"].eq(backbone)
            & train_meta["horizon"].eq(horizon)
            & train_meta["seed"].eq(seed)
        ]
        if tr.empty:
            med = horizon_fallback.get(h, np.zeros((h, batch.lrbn_pred.shape[2]), dtype=float))
        else:
            med = np.nanmedian(np.stack([valid_part(residual, int(i), h) for i in tr["row_index"]], axis=0), axis=0)
        out[group["row_index"].to_numpy(int), :h, :] = med
    return out


def knn_residual_delta(train: ForecastBatch, batch: ForecastBatch, schema: Dict[str, List[Any]], k: int = 16) -> np.ndarray:
    out = _empty_delta_like(batch)
    x_train = feature_frame(train, schema)
    x_query = align_frame(feature_frame(batch, schema), list(x_train.columns))
    residual = train.y_true - train.lrbn_pred
    train_keys = instance_key_series(train.meta).to_numpy(str)
    query_keys = instance_key_series(batch.meta).to_numpy(str)
    train_h = horizons(train)
    query_h = horizons(batch)
    for h in sorted(set(query_h.tolist())):
        tr_idx = np.where(train_h == int(h))[0]
        q_idx = np.where(query_h == int(h))[0]
        if len(tr_idx) == 0 or len(q_idx) == 0:
            continue
        n_neighbors = min(max(1, k + 1), len(tr_idx))
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(x_train.iloc[tr_idx].to_numpy(float))
        Xq = scaler.transform(x_query.iloc[q_idx].to_numpy(float))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(Xtr)
        neigh = nn.kneighbors(Xq, return_distance=False)
        res_flat = np.stack([valid_flat_delta(train, residual, int(i), int(h)) for i in tr_idx], axis=0)
        for local_q, global_q in enumerate(q_idx):
            selected: List[int] = []
            for n_local in neigh[local_q]:
                global_train = int(tr_idx[int(n_local)])
                if train_keys[global_train] == query_keys[global_q]:
                    continue
                selected.append(int(n_local))
                if len(selected) >= k:
                    break
            if not selected:
                selected = [int(neigh[local_q][0])]
            pred = np.nanmean(res_flat[selected], axis=0).reshape(int(h), -1)
            out[int(global_q), : int(h), :] = pred
    return out


def clip_delta(batch: ForecastBatch, delta: np.ndarray, cap_mult: float) -> np.ndarray:
    cap = mrc_cap_matrix(batch)
    out = delta.copy()
    if cap.shape[1] == 1:
        cap = np.repeat(cap, out.shape[1], axis=1)
    out = np.clip(out, -cap_mult * cap, cap_mult * cap)
    return out


def select_delta_policy(
    calib: ForecastBatch,
    delta_components: Mapping[str, np.ndarray],
    target: str,
) -> Tuple[Dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray]:
    weight_grid = [
        {"ridge": 1.0, "knn": 0.0, "median": 0.0},
        {"ridge": 0.0, "knn": 1.0, "median": 0.0},
        {"ridge": 0.0, "knn": 0.0, "median": 1.0},
        {"ridge": 0.5, "knn": 0.5, "median": 0.0},
        {"ridge": 0.5, "knn": 0.0, "median": 0.5},
        {"ridge": 0.4, "knn": 0.4, "median": 0.2},
        {"ridge": 0.25, "knn": 0.5, "median": 0.25},
    ]
    shrink_grid = [0.25, 0.5, 0.75, 1.0]
    cap_grid = [0.5, 1.0, 1.5, 2.0]
    risk_quantiles = [0.50, 0.75, 0.90, 1.0]
    rows: List[Dict[str, Any]] = []
    best_score = float("inf")
    best_policy: Dict[str, Any] = {}
    best_pred = calib.lrbn_pred.copy()
    best_selected = np.zeros(len(calib.meta), dtype=bool)
    x = feature_frame(calib, feature_schema(calib))
    vol = x["context_diff_std"].to_numpy(float) + 1e-6
    for weights in weight_grid:
        raw_delta = sum(float(weights.get(name, 0.0)) * delta_components[name] for name in delta_components)
        risk = np.linalg.norm(raw_delta.reshape(len(calib.meta), -1), axis=1) / (vol * math.sqrt(raw_delta.shape[1]))
        for shrink in shrink_grid:
            for cap_mult in cap_grid:
                capped = clip_delta(calib, raw_delta * float(shrink), cap_mult=float(cap_mult))
                for rq in risk_quantiles:
                    tau = float(np.quantile(risk, rq))
                    selected = risk <= tau
                    pred = calib.lrbn_pred.copy()
                    pred[selected] = calib.lrbn_pred[selected] + capped[selected]
                    row = metric_row(f"policy_{target}", pred, calib, selected=selected)
                    row.update(
                        {
                            "target": target,
                            "weights": json.dumps(weights, sort_keys=True),
                            "shrink": float(shrink),
                            "cap_mult": float(cap_mult),
                            "risk_quantile": float(rq),
                            "risk_tau": tau,
                        }
                    )
                    feasible = (
                        row["harm_rate"] <= (0.03 if target == "safe" else 0.08)
                        and row["max_config_harm"] <= (0.10 if target == "safe" else 0.18)
                    )
                    score = float(row["mse_delta_pct_vs_lrbn"])
                    score += 100.0 * max(0.0, float(row["harm_rate"]) - (0.03 if target == "safe" else 0.08))
                    score += 100.0 * max(0.0, float(row["max_config_harm"]) - (0.10 if target == "safe" else 0.18))
                    row["calibration_feasible"] = bool(feasible)
                    row["calibration_score"] = float(score)
                    rows.append(row)
                    ranked = score if feasible else score + 1000.0
                    if ranked < best_score:
                        best_score = ranked
                        best_policy = dict(row)
                        best_policy["weights"] = weights
                        best_pred = pred
                        best_selected = selected
    return best_policy, pd.DataFrame(rows), best_pred, best_selected


def apply_delta_policy(batch: ForecastBatch, delta_components: Mapping[str, np.ndarray], policy: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    weights = policy["weights"]
    raw_delta = sum(float(weights.get(name, 0.0)) * delta_components[name] for name in delta_components)
    x = feature_frame(batch, feature_schema(batch))
    risk = np.linalg.norm(raw_delta.reshape(len(batch.meta), -1), axis=1) / (
        (x["context_diff_std"].to_numpy(float) + 1e-6) * math.sqrt(raw_delta.shape[1])
    )
    selected = risk <= float(policy["risk_tau"])
    capped = clip_delta(batch, raw_delta * float(policy["shrink"]), cap_mult=float(policy["cap_mult"]))
    pred = batch.lrbn_pred.copy()
    pred[selected] = batch.lrbn_pred[selected] + capped[selected]
    return pred, selected


def jump_aware_boundary_candidate(train: ForecastBatch, batch: ForecastBatch, sra_balanced_pred: np.ndarray) -> np.ndarray:
    schema = feature_schema(train)
    x_train = feature_frame(train, schema)
    x_batch = feature_frame(batch, schema)
    g_tau = float(x_train["boundary_gap_lrbn"].quantile(0.75))
    repair_tau = float(x_train["repair_ratio"].quantile(0.50))
    jump_tau = float(x_train["jump_support"].quantile(0.75))
    mask = (
        (x_batch["boundary_gap_lrbn"].to_numpy(float) >= g_tau)
        & (x_batch["repair_ratio"].to_numpy(float) <= repair_tau)
        & (x_batch["jump_support"].to_numpy(float) <= jump_tau)
    )
    out = batch.lrbn_pred.copy()
    out[mask] = batch.lrbn_pred[mask] + 0.75 * (sra_balanced_pred[mask] - batch.lrbn_pred[mask])
    return out


def smoothing_teacher_candidate(batch: ForecastBatch) -> np.ndarray:
    if batch.extra_preds:
        for name in ["median_smoothing", "ema_smoothing", "naive_smoothing"]:
            arr = batch.extra_preds.get(name)
            if arr is not None and np.isfinite(arr).any():
                out = arr.copy()
                missing = ~np.isfinite(out)
                out[missing] = batch.lrbn_pred[missing]
                return out
    return volatility_shrink_candidate(batch, alpha=0.35)


def build_new_candidates(
    train: ForecastBatch,
    batch: ForecastBatch,
    schema: Dict[str, List[Any]],
    old_candidates: Sequence[ExpertCandidate],
) -> Tuple[List[ExpertCandidate], Dict[str, np.ndarray]]:
    old = candidate_dict(old_candidates)
    ridge_model = fit_ridge_residual_models(train, feature_frame(train, schema), seed=2026)
    ridge_delta = ridge_model.val_pred if len(train.meta) == len(batch.meta) and train.meta["sample_key"].equals(batch.meta["sample_key"]) else None
    if ridge_delta is None:
        ridge_delta = np.zeros_like(batch.lrbn_pred)
        x_batch = align_frame(feature_frame(batch, schema), list(feature_frame(train, schema).columns))
        hs = horizons(batch)
        x_all = x_batch.to_numpy(float)
        for h, model in ridge_model.models.items():
            idx = np.where(hs == int(h))[0]
            if len(idx) == 0:
                continue
            pred = model.predict(x_all[idx])
            for local, global_idx in enumerate(idx):
                ridge_delta[int(global_idx), : int(h), :] = pred[local].reshape(int(h), -1)
    median_delta = residual_median_delta(train, batch)
    knn_delta = knn_residual_delta(train, batch, schema, k=16)
    sra_balanced = old.get("sra_balanced")
    jump = jump_aware_boundary_candidate(train, batch, sra_balanced.pred if sra_balanced is not None else batch.lrbn_pred)
    teacher = smoothing_teacher_candidate(batch)
    robust_median = batch.lrbn_pred + clip_delta(batch, 0.50 * median_delta, cap_mult=1.0)
    memory = batch.lrbn_pred + clip_delta(batch, 0.50 * knn_delta, cap_mult=1.0)
    ridge = batch.lrbn_pred + clip_delta(batch, 0.50 * ridge_delta, cap_mult=1.0)
    comps = {"ridge": ridge_delta, "knn": knn_delta, "median": median_delta}
    return (
        [
            ExpertCandidate("teacher_median_smoothing", "balanced", "teacher_smoothing", teacher, True),
            ExpertCandidate("residual_quantile_median", "safe", "residual_distribution", robust_median, True),
            ExpertCandidate("residual_memory_knn", "balanced", "retrieval_memory", memory, True),
            ExpertCandidate("residual_ridge_refit", "balanced", "residual_distribution", ridge, True),
            ExpertCandidate("jump_aware_boundary", "safe", "causal_boundary", jump, True),
        ],
        comps,
    )


def prepare_assets(
    metrics_csv: Path,
    stage5_dir: Path,
    stage3_dir: Optional[Path],
    seed: int,
) -> Stage9Assets:
    batch = load_forecast_batch_from_metrics(metrics_csv)
    val, test = split_batch(batch)
    schema = feature_schema(val)
    train_mask, calib_mask = stratified_inner_split(val.meta, seed=seed)
    val_train = val.subset(train_mask)
    val_calib = val.subset(calib_mask)
    safe_params = json.loads((stage5_dir / "stage5_selected_safe_params.json").read_text(encoding="utf-8"))
    balanced_params = json.loads((stage5_dir / "stage5_selected_balanced_params.json").read_text(encoding="utf-8"))
    basic_params = select_basic_sra_params(stage5_dir)
    stage3_params = load_stage3_params(stage3_dir)
    mrc = build_mrc_artifacts(val, test, schema)
    old_val_candidates = build_candidate_pool(
        val, safe_params, balanced_params, basic_params, stage3_params, mrc["val_delta"], mrc["val_abstain_pred"]
    )
    old_test_candidates = build_candidate_pool(
        test, safe_params, balanced_params, basic_params, stage3_params, mrc["test_delta"], mrc["test_abstain_pred"]
    )
    old_train_candidates = subset_candidates(old_val_candidates, train_mask)
    old_calib_candidates = subset_candidates(old_val_candidates, calib_mask)
    new_val_candidates, _ = build_new_candidates(val_train, val, schema, old_val_candidates)
    new_test_candidates, _ = build_new_candidates(val_train, test, schema, old_test_candidates)
    new_train_candidates = subset_candidates(new_val_candidates, train_mask)
    new_calib_candidates = subset_candidates(new_val_candidates, calib_mask)
    expanded_val_candidates = old_val_candidates + new_val_candidates
    expanded_test_candidates = old_test_candidates + new_test_candidates
    expanded_train_candidates = old_train_candidates + new_train_candidates
    expanded_calib_candidates = old_calib_candidates + new_calib_candidates
    old_heads = fit_heads(
        val_train,
        val_calib,
        old_train_candidates,
        old_calib_candidates,
        schema,
        mrc["val_delta"][train_mask],
        mrc["val_delta"][calib_mask],
    )
    old_calib_probs = compute_probabilities(val_calib, old_calib_candidates, schema, mrc["val_delta"][calib_mask], old_heads)
    old_test_probs = compute_probabilities(test, old_test_candidates, schema, mrc["test_delta"], old_heads)
    return Stage9Assets(
        batch=batch,
        val=val,
        test=test,
        val_train=val_train,
        val_calib=val_calib,
        train_mask=train_mask,
        calib_mask=calib_mask,
        schema=schema,
        mrc=mrc,
        old_val_candidates=old_val_candidates,
        old_test_candidates=old_test_candidates,
        old_train_candidates=old_train_candidates,
        old_calib_candidates=old_calib_candidates,
        new_val_candidates=new_val_candidates,
        new_test_candidates=new_test_candidates,
        new_train_candidates=new_train_candidates,
        new_calib_candidates=new_calib_candidates,
        expanded_val_candidates=expanded_val_candidates,
        expanded_test_candidates=expanded_test_candidates,
        expanded_train_candidates=expanded_train_candidates,
        expanded_calib_candidates=expanded_calib_candidates,
        old_heads=old_heads,
        old_calib_probs=old_calib_probs,
        old_test_probs=old_test_probs,
    )


def oracle_table(candidates: Sequence[ExpertCandidate], batch: ForecastBatch, name: str, n_bootstrap: int, seed: int) -> Tuple[Dict[str, Any], pd.DataFrame]:
    pred, best_mse, best_names = oracle_best(candidates, batch)
    row = metric_row(name, pred, batch, oracle_mse=best_mse, n_bootstrap=n_bootstrap, seed=seed)
    dist = pd.Series(best_names).value_counts(normalize=True).rename_axis("expert").reset_index(name="oracle_share")
    dist.insert(0, "oracle_pool", name)
    return row, dist


def run_restricted_oracle(assets: Stage9Assets, n_bootstrap: int, seed: int) -> Dict[str, Any]:
    old_deploy = deployable_candidates(assets.old_test_candidates, include_aggressive=False)
    old_all = deployable_candidates(assets.old_test_candidates, include_aggressive=True)
    expanded_deploy = deployable_candidates(assets.expanded_test_candidates, include_aggressive=False)
    expanded_all = deployable_candidates(assets.expanded_test_candidates, include_aggressive=True)
    rows: List[Dict[str, Any]] = []
    dists: List[pd.DataFrame] = []
    for name, cand in [
        ("oracle_current_deployable", old_deploy),
        ("oracle_current_all", old_all),
        ("oracle_expanded_deployable", expanded_deploy),
        ("oracle_expanded_all", expanded_all),
    ]:
        row, dist = oracle_table(cand, assets.test, name, n_bootstrap, seed)
        rows.append(row)
        dists.append(dist)
    overall = pd.DataFrame(rows)
    cur = float(overall.loc[overall["variant"].eq("oracle_current_deployable"), "mse"].iloc[0])
    exp = float(overall.loc[overall["variant"].eq("oracle_expanded_deployable"), "mse"].iloc[0])
    base = float(np.mean(mse_per_sample(assets.test.lrbn_pred, assets.test.y_true)))
    verdict = {
        "prototype": "restricted_oracle",
        "current_deployable_mse": cur,
        "expanded_deployable_mse": exp,
        "expanded_extra_delta_pct_vs_current_oracle": safe_pct(exp, cur),
        "expanded_extra_delta_pct_vs_lrbn": safe_pct(exp, base),
        "pool_redesign_promising": bool(safe_pct(exp, cur) <= -5.0),
        "test_threshold_leakage": False,
    }
    return {"overall": overall, "distribution": pd.concat(dists, ignore_index=True), "verdict": verdict}


def build_pairwise_dataset(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    mrc_prior: np.ndarray,
) -> pd.DataFrame:
    feat_by = expert_features(batch, schema, candidates, mrc_prior)
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    rows: List[pd.DataFrame] = []
    meta_cols = batch.meta[["dataset", "backbone", "horizon", "seed", "sample_id", "sample_key"]].reset_index(drop=True)
    meta_cols["row_index"] = np.arange(len(batch.meta), dtype=int)
    meta_cols["instance_id"] = instance_key_series(batch.meta).reset_index(drop=True)
    for c in candidates:
        if c.name == "keep_lrbn":
            continue
        x = feat_by[c.name].copy().reset_index(drop=True)
        cmse = mse_per_sample(c.pred, batch.y_true)
        cmae = mae_per_sample(c.pred, batch.y_true)
        x = pd.concat([meta_cols, x], axis=1)
        x["candidate"] = c.name
        x["tier"] = c.tier
        x["family"] = c.family
        x["deployable"] = bool(c.deployable)
        x["candidate_mse"] = cmse
        x["candidate_mae"] = cmae
        x["mse_delta_vs_lrbn"] = cmse - base_mse
        x["gain_label"] = x["mse_delta_vs_lrbn"] < -1e-4
        x["harm_label"] = x["mse_delta_vs_lrbn"] > 1e-4
        rows.append(x)
    return pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def feature_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "dataset",
        "backbone",
        "horizon",
        "seed",
        "sample_id",
        "sample_key",
        "row_index",
        "instance_id",
        "candidate",
        "tier",
        "family",
        "deployable",
        "candidate_mse",
        "candidate_mae",
        "mse_delta_vs_lrbn",
        "gain_label",
        "harm_label",
    }
    cols = [c for c in df.columns if c not in excluded and pd.api.types.is_numeric_dtype(df[c])]
    return cols


def binary_eval(y: np.ndarray, p: np.ndarray, name: str, split: str) -> Dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    row: Dict[str, Any] = {"model": name, "split": split, "n": int(len(y)), "positive_rate": float(np.mean(y))}
    if len(np.unique(y)) >= 2:
        row["roc_auc"] = float(roc_auc_score(y, p))
        row["pr_auc"] = float(average_precision_score(y, p))
    else:
        row["roc_auc"] = float("nan")
        row["pr_auc"] = float("nan")
    return row


def run_pairwise_distillation(assets: Stage9Assets, seed: int) -> Dict[str, Any]:
    train_df = build_pairwise_dataset(
        assets.val_train,
        assets.expanded_train_candidates,
        assets.schema,
        assets.mrc["val_delta"][assets.train_mask],
    )
    calib_df = build_pairwise_dataset(
        assets.val_calib,
        assets.expanded_calib_candidates,
        assets.schema,
        assets.mrc["val_delta"][assets.calib_mask],
    )
    test_df = build_pairwise_dataset(assets.test, assets.expanded_test_candidates, assets.schema, assets.mrc["test_delta"])
    cols = feature_columns(train_df)
    x_train = train_df[cols].to_numpy(float)
    gain_model = RandomForestClassifier(
        n_estimators=350,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    harm_model = RandomForestClassifier(
        n_estimators=350,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=seed + 17,
        n_jobs=1,
    )
    gain_model.fit(x_train, train_df["gain_label"].astype(int).to_numpy())
    harm_model.fit(x_train, train_df["harm_label"].astype(int).to_numpy())
    rows: List[Dict[str, Any]] = []
    for split, df in [("inner_calib", calib_df), ("test", test_df)]:
        x = align_frame(df[cols], cols).to_numpy(float)
        pg = gain_model.predict_proba(x)[:, 1]
        ph = harm_model.predict_proba(x)[:, 1]
        rows.append(binary_eval(df["gain_label"].astype(int).to_numpy(), pg, "stage9_global_pairwise_gain", split))
        rows.append(binary_eval(df["harm_label"].astype(int).to_numpy(), ph, "stage9_global_pairwise_harm", split))
    global_metrics = pd.DataFrame(rows)
    old = assets.old_heads.metrics.copy()
    old_summary = (
        old[old["split"].eq("inner_calib")]
        .groupby("head", observed=True)[["roc_auc", "pr_auc"]]
        .mean(numeric_only=True)
        .reset_index()
    )
    old_summary["model"] = "stage7_per_expert_heads"
    old_summary["split"] = "inner_calib"
    oracle_names = oracle_best(deployable_candidates(assets.expanded_test_candidates), assets.test)[2]
    test_scores = test_df.copy()
    x_test = align_frame(test_scores[cols], cols).to_numpy(float)
    test_scores["p_gain"] = gain_model.predict_proba(x_test)[:, 1]
    test_scores["p_harm"] = harm_model.predict_proba(x_test)[:, 1]
    test_scores["utility_score"] = test_scores["p_gain"] - test_scores["p_harm"]
    hit_rows = []
    for row_index, group in test_scores.groupby("row_index", sort=True):
        ranked = group.sort_values("utility_score", ascending=False)["candidate"].head(2).astype(str).tolist()
        oracle = str(oracle_names[int(row_index)]) if int(row_index) < len(oracle_names) else ""
        hit_rows.append(
            {
                "row_index": int(row_index),
                "instance_id": group["instance_id"].iloc[0],
                "oracle_expert": oracle,
                "top2_hit": oracle in ranked or oracle == "keep_lrbn",
            }
        )
    hit_df = pd.DataFrame(hit_rows)
    calib_gain = global_metrics[(global_metrics["model"].eq("stage9_global_pairwise_gain")) & (global_metrics["split"].eq("inner_calib"))]
    calib_harm = global_metrics[(global_metrics["model"].eq("stage9_global_pairwise_harm")) & (global_metrics["split"].eq("inner_calib"))]
    old_gain = old_summary[old_summary["head"].eq("gain")]
    old_harm = old_summary[old_summary["head"].eq("harm")]
    verdict = {
        "prototype": "pairwise_distillation",
        "global_gain_calib_roc_auc": float(calib_gain["roc_auc"].iloc[0]),
        "global_harm_calib_roc_auc": float(calib_harm["roc_auc"].iloc[0]),
        "stage7_gain_calib_roc_auc": float(old_gain["roc_auc"].iloc[0]) if not old_gain.empty else float("nan"),
        "stage7_harm_calib_roc_auc": float(old_harm["roc_auc"].iloc[0]) if not old_harm.empty else float("nan"),
        "test_top2_oracle_hit": float(hit_df["top2_hit"].mean()),
        "representation_promising": bool(
            float(calib_gain["roc_auc"].iloc[0]) >= (float(old_gain["roc_auc"].iloc[0]) if not old_gain.empty else 0.0) + 0.03
            and float(calib_harm["roc_auc"].iloc[0]) >= 0.70
            and float(hit_df["top2_hit"].mean()) >= 0.55
        ),
        "test_threshold_leakage": False,
    }
    return {
        "metrics": global_metrics,
        "stage7_head_reference": old_summary,
        "top2_hit": hit_df,
        "verdict": verdict,
        "models": {"gain": gain_model, "harm": harm_model, "columns": cols},
    }


def residual_components_from_train(
    train: ForecastBatch,
    batch: ForecastBatch,
    schema: Dict[str, List[Any]],
    seed: int,
) -> Dict[str, np.ndarray]:
    x_train = feature_frame(train, schema)
    ridge = fit_ridge_residual_models(train, x_train, seed=seed)
    x_batch = align_frame(feature_frame(batch, schema), list(x_train.columns))
    ridge_delta = _empty_delta_like(batch)
    hs = horizons(batch)
    x_all = x_batch.to_numpy(float)
    for h, model in ridge.models.items():
        idx = np.where(hs == int(h))[0]
        if len(idx) == 0:
            continue
        pred = model.predict(x_all[idx])
        for local, global_idx in enumerate(idx):
            ridge_delta[int(global_idx), : int(h), :] = pred[local].reshape(int(h), -1)
    return {
        "ridge": ridge_delta,
        "knn": knn_residual_delta(train, batch, schema, k=16),
        "median": residual_median_delta(train, batch),
    }


def interval_calibration(calib: ForecastBatch, calib_pred: np.ndarray, test: ForecastBatch, test_pred: np.ndarray) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    widths: Dict[int, float] = {}
    for h in sorted(set(horizons(calib).tolist())):
        idx = np.where(horizons(calib) == int(h))[0]
        errs = np.abs(calib.y_true[idx, : int(h), :] - calib_pred[idx, : int(h), :]).reshape(-1)
        widths[int(h)] = float(np.quantile(errs, 0.90)) if len(errs) else 0.0
    coverages: List[float] = []
    width_vals: List[float] = []
    for i, h in enumerate(horizons(test)):
        q = widths.get(int(h), 0.0)
        err = np.abs(test.y_true[i, : int(h), :] - test_pred[i, : int(h), :]).reshape(-1)
        coverages.append(float(np.mean(err <= q)) if len(err) else 0.0)
        width_vals.append(2.0 * q)
    return {
        "mean_pointwise_coverage90": float(np.mean(coverages)) if coverages else 0.0,
        "coverage_gap_pp": float(abs(np.mean(coverages) - 0.90) * 100.0) if coverages else float("nan"),
        "mean_interval_width": float(np.mean(width_vals)) if width_vals else 0.0,
        "q90_by_horizon": widths,
    }


def run_mrc_v2(assets: Stage9Assets, n_bootstrap: int, seed: int) -> Dict[str, Any]:
    calib_components = residual_components_from_train(assets.val_train, assets.val_calib, assets.schema, seed)
    test_components = residual_components_from_train(assets.val_train, assets.test, assets.schema, seed)
    rows: List[Dict[str, Any]] = []
    grids: List[pd.DataFrame] = []
    preds: Dict[str, np.ndarray] = {}
    selected_by: Dict[str, np.ndarray] = {}
    policies: Dict[str, Dict[str, Any]] = {}
    for target, variant in [("safe", "MRC-v2-multiscale-safe"), ("tradeoff", "MRC-v2-multiscale-tradeoff")]:
        policy, grid, calib_pred, calib_selected = select_delta_policy(assets.val_calib, calib_components, target)
        test_pred, test_selected = apply_delta_policy(assets.test, test_components, policy)
        row = metric_row(variant, test_pred, assets.test, selected=test_selected, n_bootstrap=n_bootstrap, seed=seed)
        interval = interval_calibration(assets.val_calib, calib_pred, assets.test, test_pred)
        row.update(interval)
        rows.append(row)
        grid["target_variant"] = variant
        grids.append(grid)
        preds[variant] = test_pred
        selected_by[variant] = test_selected
        policies[variant] = policy
    overall = pd.DataFrame(rows)
    verdict = {
        "prototype": "mrc_v2_multiscale",
        "safe_delta_pct_vs_lrbn": float(overall.loc[overall["variant"].eq("MRC-v2-multiscale-safe"), "mse_delta_pct_vs_lrbn"].iloc[0]),
        "tradeoff_delta_pct_vs_lrbn": float(overall.loc[overall["variant"].eq("MRC-v2-multiscale-tradeoff"), "mse_delta_pct_vs_lrbn"].iloc[0]),
        "safe_harm": float(overall.loc[overall["variant"].eq("MRC-v2-multiscale-safe"), "harm_rate"].iloc[0]),
        "tradeoff_harm": float(overall.loc[overall["variant"].eq("MRC-v2-multiscale-tradeoff"), "harm_rate"].iloc[0]),
        "mrc_v2_safe_pass": bool(
            overall.loc[overall["variant"].eq("MRC-v2-multiscale-safe"), "mse_delta_pct_vs_lrbn"].iloc[0] <= -2.0
            and overall.loc[overall["variant"].eq("MRC-v2-multiscale-safe"), "harm_rate"].iloc[0] <= 0.03
            and overall.loc[overall["variant"].eq("MRC-v2-multiscale-safe"), "max_config_harm"].iloc[0] <= 0.10
        ),
        "mrc_v2_tradeoff_pass": bool(
            overall.loc[overall["variant"].eq("MRC-v2-multiscale-tradeoff"), "mse_delta_pct_vs_lrbn"].iloc[0] <= -3.0
            and overall.loc[overall["variant"].eq("MRC-v2-multiscale-tradeoff"), "harm_rate"].iloc[0] <= 0.08
            and overall.loc[overall["variant"].eq("MRC-v2-multiscale-tradeoff"), "max_config_harm"].iloc[0] <= 0.18
        ),
        "test_threshold_leakage": False,
    }
    return {
        "overall": overall,
        "grid": pd.concat(grids, ignore_index=True),
        "preds": preds,
        "selected": selected_by,
        "policies": policies,
        "verdict": verdict,
    }


def run_energy_scorer(assets: Stage9Assets, n_bootstrap: int, seed: int) -> Dict[str, Any]:
    train_df = build_pairwise_dataset(
        assets.val_train,
        deployable_candidates(assets.expanded_train_candidates),
        assets.schema,
        assets.mrc["val_delta"][assets.train_mask],
    )
    calib_df = build_pairwise_dataset(
        assets.val_calib,
        deployable_candidates(assets.expanded_calib_candidates),
        assets.schema,
        assets.mrc["val_delta"][assets.calib_mask],
    )
    test_df = build_pairwise_dataset(
        assets.test,
        deployable_candidates(assets.expanded_test_candidates),
        assets.schema,
        assets.mrc["test_delta"],
    )
    cols = feature_columns(train_df)
    model = RandomForestRegressor(
        n_estimators=450,
        max_depth=10,
        min_samples_leaf=8,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(train_df[cols].to_numpy(float), train_df["mse_delta_vs_lrbn"].to_numpy(float))
    for df in [calib_df, test_df]:
        df["predicted_delta"] = model.predict(align_frame(df[cols], cols).to_numpy(float))
        df["energy_score"] = -df["predicted_delta"]
    selection_rows: List[Dict[str, Any]] = []
    best_score = float("inf")
    best_policy: Dict[str, Any] = {}
    for tau in sorted(set(np.quantile(calib_df["predicted_delta"], [0.05, 0.10, 0.20, 0.35, 0.50]).tolist() + [0.0])):
        for lam in [0.5, 0.75, 1.0]:
            pred, selected, names = apply_energy_policy(assets.val_calib, deployable_candidates(assets.expanded_calib_candidates), calib_df, tau, lam)
            row = metric_row("energy_calib", pred, assets.val_calib, selected=selected)
            row.update({"tau_predicted_delta": float(tau), "lambda": float(lam)})
            feasible = row["harm_rate"] <= 0.08 and row["max_config_harm"] <= 0.18
            score = float(row["mse_delta_pct_vs_lrbn"]) + 100.0 * max(0.0, row["harm_rate"] - 0.08) + 100.0 * max(
                0.0, row["max_config_harm"] - 0.18
            )
            row["calibration_feasible"] = bool(feasible)
            row["calibration_score"] = float(score)
            selection_rows.append(row)
            ranked = score if feasible else score + 1000.0
            if ranked < best_score:
                best_score = ranked
                best_policy = dict(row)
    test_pred, test_selected, selected_names = apply_energy_policy(
        assets.test,
        deployable_candidates(assets.expanded_test_candidates),
        test_df,
        float(best_policy["tau_predicted_delta"]),
        float(best_policy["lambda"]),
    )
    row = metric_row("Energy-feasibility-reranker", test_pred, assets.test, selected=test_selected, n_bootstrap=n_bootstrap, seed=seed)
    corr = spearmanr(-test_df["predicted_delta"].to_numpy(float), -test_df["mse_delta_vs_lrbn"].to_numpy(float), nan_policy="omit")
    row["score_gain_spearman"] = float(corr.correlation) if np.isfinite(corr.correlation) else 0.0
    selected_dist = pd.Series(selected_names).value_counts(normalize=True).rename_axis("selected_expert").reset_index(name="share")
    verdict = {
        "prototype": "energy_feasibility",
        "score_gain_spearman": float(row["score_gain_spearman"]),
        "mse_delta_pct_vs_lrbn": float(row["mse_delta_pct_vs_lrbn"]),
        "harm_rate": float(row["harm_rate"]),
        "max_config_harm": float(row["max_config_harm"]),
        "energy_promising": bool(
            row["score_gain_spearman"] >= 0.20
            and row["mse_delta_pct_vs_lrbn"] <= -2.0
            and row["harm_rate"] <= 0.08
            and row["max_config_harm"] <= 0.18
        ),
        "test_threshold_leakage": False,
    }
    return {
        "overall": pd.DataFrame([row]),
        "calibration_grid": pd.DataFrame(selection_rows),
        "candidate_scores_test": test_df,
        "selected_distribution": selected_dist,
        "policy": best_policy,
        "verdict": verdict,
        "pred": test_pred,
        "selected": test_selected,
    }


def apply_energy_policy(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    score_df: pd.DataFrame,
    tau: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_name = candidate_dict(candidates)
    out = batch.lrbn_pred.copy()
    selected = np.zeros(len(batch.meta), dtype=bool)
    names = np.array(["keep_lrbn"] * len(batch.meta), dtype=object)
    grouped = {str(k): g.copy() for k, g in score_df.groupby("instance_id", sort=False)}
    instance_keys = instance_key_series(batch.meta).reset_index(drop=True)
    for i, row in batch.meta.reset_index(drop=True).iterrows():
        key = str(instance_keys.iloc[i])
        group = grouped.get(key)
        if group is None or group.empty:
            continue
        best = group.sort_values("predicted_delta", ascending=True).iloc[0]
        if float(best["predicted_delta"]) < float(tau):
            cname = str(best["candidate"])
            cand = by_name.get(cname)
            if cand is None:
                continue
            h = int(row["horizon"])
            out[i, :h, :] = batch.lrbn_pred[i, :h, :] + float(lam) * (cand.pred[i, :h, :] - batch.lrbn_pred[i, :h, :])
            selected[i] = True
            names[i] = cname
    return out, selected, names


def run_online_spectral(assets: Stage9Assets) -> Dict[str, Any]:
    weights = calibrate_spectral_band_weights(assets.val)
    adapter_df, conformal_df, guard = online_adapter_eval(assets.val, assets.test, buffer_size=128, band_weights=weights)
    rows = adapter_df.set_index("method")
    spectral_delta = float(rows.loc["spectral_adapter", "mse_delta_pct_vs_lrbn"])
    rolling_delta = float(rows.loc["rolling_mean_residual", "mse_delta_pct_vs_lrbn"])
    spectral_harm = float(rows.loc["spectral_adapter", "harm_rate"])
    coverage_gap = abs(float(rows.loc["spectral_adapter", "mean_pointwise_coverage90"]) - 0.90) * 100.0
    verdict = {
        "prototype": "online_spectral_matured_label",
        "spectral_delta_pct_vs_lrbn": spectral_delta,
        "rolling_delta_pct_vs_lrbn": rolling_delta,
        "spectral_minus_rolling_pct": float(spectral_delta - rolling_delta),
        "spectral_harm": spectral_harm,
        "coverage_gap_pp": float(coverage_gap),
        "protocol_guard_pass": guard["n_protocol_violations"] == 0,
        "online_promising": bool(
            spectral_delta <= -1.0
            and spectral_delta <= rolling_delta - 0.5
            and spectral_harm <= 0.10
            and coverage_gap <= 5.0
            and guard["n_protocol_violations"] == 0
        ),
        "test_threshold_leakage": False,
    }
    return {"adapter": adapter_df, "conformal": conformal_df, "guard": guard, "weights": weights, "verdict": verdict}


def verify_outputs(output_dir: Path) -> pd.DataFrame:
    files = [
        "stage9_config.json",
        "prototype1_restricted_oracle.csv",
        "prototype1_oracle_distribution.csv",
        "prototype2_pairwise_metrics.csv",
        "prototype2_stage7_head_reference.csv",
        "prototype2_top2_hit.csv",
        "prototype3_mrc_v2_metrics.csv",
        "prototype3_mrc_v2_grid.csv",
        "prototype4_energy_metrics.csv",
        "prototype4_energy_grid.csv",
        "prototype4_energy_selected_distribution.csv",
        "prototype5_online_spectral_metrics.csv",
        "prototype5_online_conformal.csv",
        "stage9_overall.csv",
        "stage9_per_config.csv",
        "stage9_slice_metrics.csv",
        "stage9_verdict.json",
        "summary.md",
    ]
    rows = []
    for f in files:
        p = output_dir / f
        rows.append({"file": f, "exists": p.exists(), "bytes": int(p.stat().st_size) if p.exists() else 0})
    return pd.DataFrame(rows)


def build_summary(output_dir: Path, verdict: Dict[str, Any], overall: pd.DataFrame) -> str:
    lines = [
        "# Stage 9 Architecture Validation Summary",
        "",
        "## Verdict",
        "",
        f"- Status: `{verdict['status']}`",
        f"- Recommended directions: `{', '.join(verdict['recommended_directions']) if verdict['recommended_directions'] else 'none'}`",
        f"- Test threshold leakage: `{verdict['test_threshold_leakage']}`",
        "",
        "## Prototype Verdicts",
        "",
        "```json",
        json.dumps(verdict["prototype_verdicts"], indent=2, ensure_ascii=False, default=json_default),
        "```",
        "",
        "## Deployable / Diagnostic Metrics",
        "",
        df_to_md(
            overall[
                [
                    "variant",
                    "mse",
                    "mae",
                    "mse_delta_pct_vs_lrbn",
                    "harm_rate",
                    "max_config_harm",
                    "config_improved_ratio",
                    "coverage",
                    "oracle_gain_fraction",
                ]
            ].sort_values("mse_delta_pct_vs_lrbn"),
            max_rows=32,
        ),
        "",
        "## Interpretation",
        "",
        verdict["interpretation"],
        "",
    ]
    return "\n".join(lines)


def build_all_artifacts(
    metrics_csv: Path,
    stage5_dir: Path,
    stage6_dir: Path,
    stage7_dir: Path,
    stage8_dir: Path,
    output_dir: Path,
    stage3_dir: Optional[Path] = None,
    seed: int = 2026,
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = prepare_assets(metrics_csv, stage5_dir, stage3_dir, seed)
    write_json(
        output_dir / "stage9_config.json",
        {
            "stage": "stage9_architecture_validation",
            "metrics_csv": metrics_csv,
            "stage5_dir": stage5_dir,
            "stage6_dir": stage6_dir,
            "stage7_dir": stage7_dir,
            "stage8_dir": stage8_dir,
            "stage3_dir": stage3_dir,
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "compact_protocol": "ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192 x seed 2026",
            "test_threshold_leakage": False,
        },
    )

    oracle = run_restricted_oracle(assets, n_bootstrap, seed)
    pairwise = run_pairwise_distillation(assets, seed)
    mrc_v2 = run_mrc_v2(assets, n_bootstrap, seed)
    energy = run_energy_scorer(assets, n_bootstrap, seed)
    online = run_online_spectral(assets)

    oracle["overall"].to_csv(output_dir / "prototype1_restricted_oracle.csv", index=False)
    oracle["distribution"].to_csv(output_dir / "prototype1_oracle_distribution.csv", index=False)
    pairwise["metrics"].to_csv(output_dir / "prototype2_pairwise_metrics.csv", index=False)
    pairwise["stage7_head_reference"].to_csv(output_dir / "prototype2_stage7_head_reference.csv", index=False)
    pairwise["top2_hit"].to_csv(output_dir / "prototype2_top2_hit.csv", index=False)
    mrc_v2["overall"].to_csv(output_dir / "prototype3_mrc_v2_metrics.csv", index=False)
    mrc_v2["grid"].to_csv(output_dir / "prototype3_mrc_v2_grid.csv", index=False)
    write_json(output_dir / "prototype3_mrc_v2_policies.json", mrc_v2["policies"])
    energy["overall"].to_csv(output_dir / "prototype4_energy_metrics.csv", index=False)
    energy["calibration_grid"].to_csv(output_dir / "prototype4_energy_grid.csv", index=False)
    energy["selected_distribution"].to_csv(output_dir / "prototype4_energy_selected_distribution.csv", index=False)
    write_json(output_dir / "prototype4_energy_policy.json", energy["policy"])
    online["adapter"].to_csv(output_dir / "prototype5_online_spectral_metrics.csv", index=False)
    online["conformal"].to_csv(output_dir / "prototype5_online_conformal.csv", index=False)
    write_json(output_dir / "prototype5_online_guard.json", online["guard"])
    write_json(output_dir / "prototype5_online_band_weights.json", {"band_weights": online["weights"]})

    # Overall method table: include LRBN, key old references, oracle diagnostics, and deployable prototypes.
    old_overall_path = stage7_dir / "safe_tae_overall.csv"
    old_overall = pd.read_csv(old_overall_path) if old_overall_path.exists() else pd.DataFrame()
    old_keep = old_overall[
        old_overall["variant"].isin(["LRBN", "sra_balanced", "SafeTAE-safe", "TAE-oracle-best"])
    ].copy()
    for col in ["max_config_harm", "coverage", "oracle_gain_fraction"]:
        if col not in old_keep.columns and col == "max_config_harm" and "max_per_config_harm_rate" in old_keep.columns:
            old_keep[col] = old_keep["max_per_config_harm_rate"]
        elif col not in old_keep.columns:
            old_keep[col] = np.nan
    overall = pd.concat(
        [
            old_keep,
            oracle["overall"],
            mrc_v2["overall"],
            energy["overall"],
            online["adapter"].rename(columns={"method": "variant"}),
        ],
        ignore_index=True,
        sort=False,
    )
    overall.to_csv(output_dir / "stage9_overall.csv", index=False)

    masks = stage8_slice_masks(assets.test, assets.schema, stage8_slice_thresholds(assets.val, assets.schema))
    per_config_frames: List[pd.DataFrame] = []
    slice_frames: List[pd.DataFrame] = []
    for variant, pred, selected in [
        ("MRC-v2-multiscale-safe", mrc_v2["preds"]["MRC-v2-multiscale-safe"], mrc_v2["selected"]["MRC-v2-multiscale-safe"]),
        (
            "MRC-v2-multiscale-tradeoff",
            mrc_v2["preds"]["MRC-v2-multiscale-tradeoff"],
            mrc_v2["selected"]["MRC-v2-multiscale-tradeoff"],
        ),
        ("Energy-feasibility-reranker", energy["pred"], energy["selected"]),
    ]:
        per_config_frames.append(per_config_rows(variant, pred, assets.test, selected))
        slice_frames.append(slice_rows(variant, pred, assets.test, masks, selected))
    pd.concat(per_config_frames, ignore_index=True).to_csv(output_dir / "stage9_per_config.csv", index=False)
    pd.concat(slice_frames, ignore_index=True).to_csv(output_dir / "stage9_slice_metrics.csv", index=False)

    prototype_verdicts = {
        "restricted_oracle": oracle["verdict"],
        "pairwise_distillation": pairwise["verdict"],
        "mrc_v2_multiscale": mrc_v2["verdict"],
        "energy_feasibility": energy["verdict"],
        "online_spectral": online["verdict"],
    }
    recommended: List[str] = []
    if oracle["verdict"]["pool_redesign_promising"]:
        recommended.append("candidate_pool_redesign")
    if pairwise["verdict"]["representation_promising"]:
        recommended.append("forecastability_fingerprint_or_hierarchical_arbitration")
    if mrc_v2["verdict"]["mrc_v2_safe_pass"] or mrc_v2["verdict"]["mrc_v2_tradeoff_pass"]:
        recommended.append("multiscale_residual_distribution")
    if energy["verdict"]["energy_promising"]:
        recommended.append("energy_feasibility_reranker")
    if online["verdict"]["online_promising"]:
        recommended.append("online_spectral_calibration")

    if "candidate_pool_redesign" in recommended and not (
        mrc_v2["verdict"]["mrc_v2_safe_pass"] or energy["verdict"]["energy_promising"]
    ):
        status = "oracle_space_found_but_deployable_selector_missing"
        interpretation = (
            "The expanded candidate pool opens additional oracle space, but current deployable selectors do not safely capture enough of it. "
            "The next promising direction is a redesigned hierarchical arbitrator with a stronger candidate pool, not another threshold sweep."
        )
    elif recommended:
        status = "architecture_candidate_promising"
        interpretation = (
            "At least one deployable prototype passed its compact gate. The direction should move to a mini-extension before any TableA claim."
        )
    else:
        status = "no_large_architecture_breakthrough_yet"
        interpretation = (
            "None of the compact prototypes passed a deployable gate. The evidence narrows the next move to expert-pool redesign or stronger representations."
        )

    verdict = {
        "stage": "stage9_architecture_validation",
        "status": status,
        "recommended_directions": recommended,
        "prototype_verdicts": prototype_verdicts,
        "interpretation": interpretation,
        "test_threshold_leakage": False,
    }
    write_json(output_dir / "stage9_verdict.json", verdict)
    summary = build_summary(output_dir, verdict, overall)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    completeness = verify_outputs(output_dir)
    completeness.to_csv(output_dir / "stage9_output_completeness.csv", index=False)
    return {
        "verdict": verdict,
        "overall": overall,
        "completeness": completeness,
        "oracle": oracle,
        "pairwise": pairwise,
        "mrc_v2": mrc_v2,
        "energy": energy,
        "online": online,
    }
