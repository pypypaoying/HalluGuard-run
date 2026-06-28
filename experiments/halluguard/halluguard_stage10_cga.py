#!/usr/bin/env python
"""Stage 10 CGA compact validation.

HalluGuard-CGA tests whether a richer candidate pool plus hierarchical
family-aware arbitration can move beyond the Stage 9 oracle/deployability gap.
It is intentionally compact and validation-only calibrated.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from halluguard_lrbn_bp import ForecastBatch, mae_per_sample, mse_per_sample
from halluguard_stage6_mechanism import (
    feature_frame,
    feature_schema,
    horizons,
    mrc_cap_matrix,
    safe_pct,
    valid_part,
)
from halluguard_stage7_safe_tae import (
    ConstantProbability,
    ExpertCandidate,
    align_frame,
    array_cosine,
    candidate_dict,
    robust_nan_to_num,
    stratified_inner_split,
    subset_candidates,
    write_json,
)
from halluguard_stage8_safe_tae_pareto import stage8_slice_masks, stage8_slice_thresholds
from halluguard_stage9_architecture_validation import (
    Stage9Assets,
    deployable_candidates,
    metric_row,
    oracle_best,
    per_config_rows,
    prepare_assets,
    slice_rows,
)


NEW_FAMILIES = {"smoothing_teacher", "residual_distribution", "retrieval_memory"}
OLD_REFERENCE_FAMILIES = {"boundary", "residual", "volatility", "level", "amplitude", "ensemble", "default"}


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
        vals: List[str] = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _finite_pred(arr: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float).copy()
    mask = ~np.isfinite(out)
    if mask.any():
        out[mask] = fallback[mask]
    return out


def moving_median(y: np.ndarray, kernel: int = 5) -> np.ndarray:
    pad = max(1, kernel // 2)
    ypad = np.pad(y, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(y)
    for t in range(y.shape[1]):
        out[:, t, :] = np.nanmedian(ypad[:, t : t + kernel, :], axis=1)
    return out


def ema_smooth(y: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    for t in range(1, out.shape[1]):
        out[:, t, :] = alpha * y[:, t, :] + (1.0 - alpha) * out[:, t - 1, :]
    return out


def robust_trend_smooth(batch: ForecastBatch, blend: float = 0.5) -> np.ndarray:
    """Blend LRBN toward a robust linear trend residual-smoothed trajectory."""

    y = batch.lrbn_pred
    out = y.copy()
    for i, h in enumerate(horizons(batch)):
        h = int(h)
        yy = y[i, :h, :]
        if h <= 2:
            continue
        t = np.linspace(-1.0, 1.0, h)
        trend = np.zeros_like(yy)
        for c in range(yy.shape[1]):
            coef = np.polyfit(t, yy[:, c], deg=1)
            trend[:, c] = coef[0] * t + coef[1]
        residual = yy - trend
        smooth_resid = moving_median(residual[None, :, :], kernel=5)[0]
        teacher = trend + smooth_resid
        out[i, :h, :] = yy + blend * (teacher - yy)
    return out


def clip_delta(batch: ForecastBatch, delta: np.ndarray, cap_mult: float = 1.0) -> np.ndarray:
    cap = mrc_cap_matrix(batch)
    if cap.shape[1] == 1:
        cap = np.repeat(cap, delta.shape[1], axis=1)
    return np.clip(delta, -float(cap_mult) * cap, float(cap_mult) * cap)


def config_or_horizon_residual_quantile(
    train: ForecastBatch,
    batch: ForecastBatch,
    quantile: float,
    shrink: float,
    cap_mult: float,
) -> np.ndarray:
    """Validation-only residual quantile transfer by config with horizon fallback."""

    out = np.zeros_like(batch.lrbn_pred, dtype=float)
    residual = train.y_true - train.lrbn_pred
    train_meta = train.meta.assign(row_index=np.arange(len(train.meta)))
    batch_meta = batch.meta.assign(row_index=np.arange(len(batch.meta)))
    horizon_fallback: Dict[int, np.ndarray] = {}
    for h, group in train_meta.groupby("horizon", observed=True):
        h = int(h)
        idx = group["row_index"].to_numpy(int)
        horizon_fallback[h] = np.nanquantile(np.stack([valid_part(residual, int(i), h) for i in idx], axis=0), quantile, axis=0)
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
            delta = horizon_fallback.get(h, np.zeros((h, batch.lrbn_pred.shape[2]), dtype=float))
        else:
            idx = tr["row_index"].to_numpy(int)
            delta = np.nanquantile(np.stack([valid_part(residual, int(i), h) for i in idx], axis=0), quantile, axis=0)
        out[group["row_index"].to_numpy(int), :h, :] = delta
    return clip_delta(batch, shrink * out, cap_mult=cap_mult)


def slice_residual_median(train: ForecastBatch, batch: ForecastBatch, schema: Dict[str, List[Any]], shrink: float = 0.5) -> np.ndarray:
    """Residual median by a boundary-like slice estimated on inner train only."""

    train_feat = feature_frame(train, schema)
    batch_feat = feature_frame(batch, schema)
    tau_gap = float(train_feat["boundary_gap_lrbn"].quantile(0.75))
    tau_repair = float(train_feat["repair_ratio"].quantile(0.50))
    train_slice = (train_feat["boundary_gap_lrbn"].to_numpy(float) >= tau_gap) & (
        train_feat["repair_ratio"].to_numpy(float) <= tau_repair
    )
    batch_slice = (batch_feat["boundary_gap_lrbn"].to_numpy(float) >= tau_gap) & (
        batch_feat["repair_ratio"].to_numpy(float) <= tau_repair
    )
    residual = train.y_true - train.lrbn_pred
    med_by_flag: Dict[bool, Dict[int, np.ndarray]] = {}
    for flag in [False, True]:
        med_by_flag[flag] = {}
        idx_flag = np.where(train_slice == flag)[0]
        for h in sorted(set(horizons(batch).tolist())):
            tr = idx_flag[horizons(train)[idx_flag] == int(h)]
            if len(tr) == 0:
                tr = np.where(horizons(train) == int(h))[0]
            med_by_flag[flag][int(h)] = np.nanmedian(
                np.stack([valid_part(residual, int(i), int(h)) for i in tr], axis=0), axis=0
            )
    out = np.zeros_like(batch.lrbn_pred, dtype=float)
    for i, h in enumerate(horizons(batch)):
        out[i, : int(h), :] = med_by_flag[bool(batch_slice[i])][int(h)]
    return clip_delta(batch, shrink * out, cap_mult=1.0)


def knn_residual_delta(
    train: ForecastBatch,
    batch: ForecastBatch,
    schema: Dict[str, List[Any]],
    k: int = 9,
    weighted: bool = False,
    quantile: Optional[float] = None,
    temperature: float = 0.5,
) -> Tuple[np.ndarray, pd.DataFrame]:
    x_train = feature_frame(train, schema)
    x_query = align_frame(feature_frame(batch, schema), list(x_train.columns))
    residual = train.y_true - train.lrbn_pred
    train_instance = (
        train.meta["dataset"].astype(str)
        + "/"
        + train.meta["backbone"].astype(str)
        + "/"
        + train.meta["horizon"].astype(str)
        + "/"
        + train.meta["seed"].astype(str)
        + "::"
        + train.meta["sample_key"].astype(str)
    ).to_numpy(str)
    query_instance = (
        batch.meta["dataset"].astype(str)
        + "/"
        + batch.meta["backbone"].astype(str)
        + "/"
        + batch.meta["horizon"].astype(str)
        + "/"
        + batch.meta["seed"].astype(str)
        + "::"
        + batch.meta["sample_key"].astype(str)
    ).to_numpy(str)
    out = np.zeros_like(batch.lrbn_pred, dtype=float)
    diag_rows: List[Dict[str, Any]] = []
    for h in sorted(set(horizons(batch).tolist())):
        tr_idx = np.where(horizons(train) == int(h))[0]
        q_idx = np.where(horizons(batch) == int(h))[0]
        if len(tr_idx) == 0 or len(q_idx) == 0:
            continue
        n_neighbors = min(max(k + 1, 1), len(tr_idx))
        scaler = StandardScaler()
        xtr = scaler.fit_transform(x_train.iloc[tr_idx].to_numpy(float))
        xq = scaler.transform(x_query.iloc[q_idx].to_numpy(float))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn.fit(xtr)
        dist, neigh = nn.kneighbors(xq, return_distance=True)
        res_flat = np.stack([valid_part(residual, int(i), int(h)).reshape(-1) for i in tr_idx], axis=0)
        for local_q, global_q in enumerate(q_idx):
            selected_local: List[int] = []
            selected_dist: List[float] = []
            for local_n, d in zip(neigh[local_q], dist[local_q]):
                global_train = int(tr_idx[int(local_n)])
                if train_instance[global_train] == query_instance[global_q]:
                    continue
                selected_local.append(int(local_n))
                selected_dist.append(float(d))
                if len(selected_local) >= k:
                    break
            if not selected_local:
                selected_local = [int(neigh[local_q][0])]
                selected_dist = [float(dist[local_q][0])]
            neigh_res = res_flat[selected_local]
            if weighted:
                w = np.exp(-np.asarray(selected_dist, dtype=float) / max(float(temperature), 1e-6))
                w = w / (np.sum(w) + 1e-12)
                delta = np.sum(neigh_res * w[:, None], axis=0)
            elif quantile is not None:
                delta = np.nanquantile(neigh_res, float(quantile), axis=0)
            else:
                delta = np.nanmedian(neigh_res, axis=0)
            out[int(global_q), : int(h), :] = delta.reshape(int(h), -1)
            diag_rows.append(
                {
                    "row_index": int(global_q),
                    "horizon": int(h),
                    "k_used": int(len(selected_local)),
                    "mean_distance": float(np.mean(selected_dist)),
                    "min_distance": float(np.min(selected_dist)),
                }
            )
    return clip_delta(batch, 0.5 * out, cap_mult=1.0), pd.DataFrame(diag_rows)


def build_stage10_family_candidates(
    train: ForecastBatch,
    batch: ForecastBatch,
    schema: Dict[str, List[Any]],
) -> Tuple[List[ExpertCandidate], pd.DataFrame]:
    candidates: List[ExpertCandidate] = []

    if batch.extra_preds is not None:
        for method, family_name in [
            ("median_smoothing", "teacher_median_smoothing"),
            ("ema_smoothing", "teacher_ema_smoothing"),
            ("naive_smoothing", "teacher_naive_smoothing"),
        ]:
            arr = batch.extra_preds.get(method)
            if arr is not None:
                candidates.append(
                    ExpertCandidate(
                        family_name,
                        "balanced",
                        "smoothing_teacher",
                        _finite_pred(arr, batch.lrbn_pred),
                        True,
                    )
                )
    median = moving_median(batch.lrbn_pred, kernel=5)
    short = batch.lrbn_pred.copy()
    for i, h in enumerate(horizons(batch)):
        cut = max(1, int(h) // 4)
        short[i, cut : int(h), :] = median[i, cut : int(h), :]
    candidates.extend(
        [
            ExpertCandidate("teacher_short_preserve_mid_smooth", "safe", "smoothing_teacher", short, True),
            ExpertCandidate("teacher_robust_trend", "balanced", "smoothing_teacher", robust_trend_smooth(batch), True),
        ]
    )

    for q, name, tier in [
        (0.25, "residual_q25", "balanced"),
        (0.50, "residual_config_median", "safe"),
        (0.75, "residual_q75", "balanced"),
    ]:
        delta = config_or_horizon_residual_quantile(train, batch, q, shrink=0.5, cap_mult=1.0)
        candidates.append(ExpertCandidate(name, tier, "residual_distribution", batch.lrbn_pred + delta, True))
    slice_delta = slice_residual_median(train, batch, schema, shrink=0.5)
    candidates.append(
        ExpertCandidate("residual_slice_quantile_median", "balanced", "residual_distribution", batch.lrbn_pred + slice_delta, True)
    )
    # Stage 9 already carries the ridge residual refit candidate as a reference
    # old-pool expert. Stage 10 keeps residual generation lightweight here and
    # focuses on quantile/slice residuals plus memory retrieval.

    knn_med, diag = knn_residual_delta(train, batch, schema, k=9, weighted=False)
    knn_weight, diag_w = knn_residual_delta(train, batch, schema, k=9, weighted=True)
    candidates.extend(
        [
            ExpertCandidate("residual_memory_knn_median", "balanced", "retrieval_memory", batch.lrbn_pred + knn_med, True),
            ExpertCandidate("residual_memory_knn_weighted", "balanced", "retrieval_memory", batch.lrbn_pred + knn_weight, True),
        ]
    )
    diag["memory_candidate"] = "residual_memory_knn_median"
    diag_w["memory_candidate"] = "residual_memory_knn_weighted"
    return dedupe_candidates(candidates), pd.concat([diag, diag_w], ignore_index=True)


def dedupe_candidates(candidates: Sequence[ExpertCandidate]) -> List[ExpertCandidate]:
    out: Dict[str, ExpertCandidate] = {}
    for c in candidates:
        out[c.name] = c
    if "keep_lrbn" not in out and candidates:
        first = candidates[0]
        out["keep_lrbn"] = ExpertCandidate("keep_lrbn", "default", "default", np.zeros_like(first.pred), True)
    return list(out.values())


@dataclass
class CGACandidatePools:
    train_candidates: List[ExpertCandidate]
    calib_candidates: List[ExpertCandidate]
    val_candidates: List[ExpertCandidate]
    test_candidates: List[ExpertCandidate]
    train_memory_diag: pd.DataFrame
    calib_memory_diag: pd.DataFrame
    test_memory_diag: pd.DataFrame


def build_cga_pools(assets: Stage9Assets) -> CGACandidatePools:
    train_new, train_diag = build_stage10_family_candidates(assets.val_train, assets.val_train, assets.schema)
    calib_new, calib_diag = build_stage10_family_candidates(assets.val_train, assets.val_calib, assets.schema)
    test_new, test_diag = build_stage10_family_candidates(assets.val_train, assets.test, assets.schema)
    train = dedupe_candidates(assets.old_train_candidates + train_new)
    calib = dedupe_candidates(assets.old_calib_candidates + calib_new)
    val = dedupe_candidates(assets.old_val_candidates)
    test = dedupe_candidates(assets.old_test_candidates + test_new)
    return CGACandidatePools(train, calib, val, test, train_diag, calib_diag, test_diag)


def candidate_metadata(candidates: Sequence[ExpertCandidate], pool_name: str) -> pd.DataFrame:
    rows = []
    for c in candidates:
        rows.append(
            {
                "pool": pool_name,
                "candidate": c.name,
                "family": c.family,
                "tier": c.tier,
                "deployable": bool(c.deployable),
                "new_family": bool(c.family in NEW_FAMILIES),
            }
        )
    return pd.DataFrame(rows)


def candidate_sample_table(candidates: Sequence[ExpertCandidate], batch: ForecastBatch, split: str) -> pd.DataFrame:
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    base_mae = mae_per_sample(batch.lrbn_pred, batch.y_true)
    meta = batch.meta.reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    for c in candidates:
        cmse = mse_per_sample(c.pred, batch.y_true)
        cmae = mae_per_sample(c.pred, batch.y_true)
        corr = c.pred - batch.lrbn_pred
        corr_norm = np.linalg.norm(corr.reshape(len(meta), -1), axis=1)
        for i in range(len(meta)):
            rows.append(
                {
                    "split": split,
                    "row_index": i,
                    "dataset": meta.loc[i, "dataset"],
                    "backbone": meta.loc[i, "backbone"],
                    "horizon": int(meta.loc[i, "horizon"]),
                    "seed": int(meta.loc[i, "seed"]),
                    "sample_id": meta.loc[i, "sample_id"],
                    "sample_key": meta.loc[i, "sample_key"],
                    "candidate": c.name,
                    "family": c.family,
                    "tier": c.tier,
                    "deployable": bool(c.deployable),
                    "mse": float(cmse[i]),
                    "mae": float(cmae[i]),
                    "mse_lrbn": float(base_mse[i]),
                    "mae_lrbn": float(base_mae[i]),
                    "mse_delta_vs_lrbn": float(cmse[i] - base_mse[i]),
                    "mae_delta_vs_lrbn": float(cmae[i] - base_mae[i]),
                    "gain_label": int(cmse[i] < base_mse[i] - 1e-4),
                    "harm_label": int(cmse[i] > base_mse[i] + 1e-4),
                    "corr_norm": float(corr_norm[i]),
                }
            )
    return pd.DataFrame(rows)


def candidate_feature_frame(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    families: Sequence[str],
) -> pd.DataFrame:
    base = feature_frame(batch, schema).reset_index(drop=True)
    base_norm = np.linalg.norm(batch.lrbn_pred.reshape(len(batch.meta), -1), axis=1) + 1e-8
    frames: List[pd.DataFrame] = []
    for c in candidates:
        if c.name == "keep_lrbn" or not c.deployable:
            continue
        delta = c.pred - batch.lrbn_pred
        early = delta[:, : max(1, delta.shape[1] // 4), :]
        mid = delta[:, delta.shape[1] // 4 : 3 * delta.shape[1] // 4, :]
        late = delta[:, 3 * delta.shape[1] // 4 :, :]
        dflat = delta.reshape(len(batch.meta), -1)
        extra = pd.DataFrame(
            {
                "candidate": c.name,
                "family": c.family,
                "tier": c.tier,
                "corr_norm": np.linalg.norm(dflat, axis=1),
                "corr_norm_ratio": np.linalg.norm(dflat, axis=1) / base_norm,
                "corr_mean": np.nanmean(delta, axis=(1, 2)),
                "corr_std": np.nanstd(delta, axis=(1, 2)),
                "early_energy": np.nanmean(early**2, axis=(1, 2)),
                "mid_energy": np.nanmean(mid**2, axis=(1, 2)) if mid.size else np.zeros(len(batch.meta)),
                "late_energy": np.nanmean(late**2, axis=(1, 2)) if late.size else np.zeros(len(batch.meta)),
                "cos_with_lrbn": array_cosine(c.pred, batch.lrbn_pred),
                "tier_safe": float(c.tier == "safe"),
                "tier_balanced": float(c.tier == "balanced"),
                "tier_aggressive": float(c.tier == "aggressive"),
                "deployable": float(c.deployable),
            }
        )
        for fam in families:
            extra[f"family={fam}"] = float(c.family == fam)
        for name in [cc.name for cc in candidates if cc.name != "keep_lrbn" and cc.deployable]:
            extra[f"candidate={name}"] = float(c.name == name)
        frames.append(pd.concat([base, extra], axis=1))
    return robust_nan_to_num(pd.concat(frames, ignore_index=True))


def feature_columns(df: pd.DataFrame) -> List[str]:
    exclude = {"candidate", "family", "tier", "gain_label", "harm_label", "mse_delta_vs_lrbn"}
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def fit_head(x: pd.DataFrame, y: np.ndarray, seed: int, forest: bool = True) -> Any:
    labels = np.asarray(y, dtype=int)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return ConstantProbability(float(labels.mean()) if len(labels) else 0.0)
    if forest:
        model = RandomForestClassifier(
            n_estimators=260,
            max_depth=7,
            min_samples_leaf=12,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        )
        model.fit(x.to_numpy(float), labels)
        return model
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.7, class_weight="balanced", max_iter=1000, random_state=seed))
    model.fit(x.to_numpy(float), labels)
    return model


def predict_head(head: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(head.predict_proba(x.to_numpy(float))[:, 1], dtype=float)


def binary_metric(name: str, split: str, y: np.ndarray, p: np.ndarray, level: str) -> Dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    row: Dict[str, Any] = {
        "selector": name,
        "split": split,
        "level": level,
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)) if len(y) else 0.0,
        "mean_score": float(np.mean(p)) if len(p) else 0.0,
    }
    if len(np.unique(y)) >= 2:
        row["roc_auc"] = float(roc_auc_score(y, p))
        row["pr_auc"] = float(average_precision_score(y, p))
    else:
        row["roc_auc"] = float("nan")
        row["pr_auc"] = float("nan")
    return row


@dataclass
class CGAModels:
    leave_head: Any
    family_gain_heads: Dict[str, Any]
    family_harm_heads: Dict[str, Any]
    candidate_gain_head: Any
    candidate_harm_head: Any
    sample_columns: List[str]
    family_columns: List[str]
    candidate_columns: List[str]
    families: List[str]
    candidate_names: List[str]


def family_table(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    families: Sequence[str],
) -> pd.DataFrame:
    base = feature_frame(batch, schema).reset_index(drop=True)
    cand_losses = {c.name: mse_per_sample(c.pred, batch.y_true) for c in candidates}
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    rows: List[pd.DataFrame] = []
    for fam in families:
        fam_cands = [c for c in candidates if c.family == fam and c.name != "keep_lrbn" and c.deployable]
        if not fam_cands:
            continue
        losses = np.stack([cand_losses[c.name] for c in fam_cands], axis=1)
        best = np.min(losses, axis=1)
        deltas = np.stack([(c.pred - batch.lrbn_pred).reshape(len(batch.meta), -1) for c in fam_cands], axis=1)
        stats = pd.DataFrame(
            {
                "family": fam,
                "family_candidate_count": float(len(fam_cands)),
                "family_corr_norm_mean": np.mean(np.linalg.norm(deltas, axis=2), axis=1),
                "family_corr_norm_min": np.min(np.linalg.norm(deltas, axis=2), axis=1),
                "family_corr_norm_max": np.max(np.linalg.norm(deltas, axis=2), axis=1),
                "family_gain_label": (best < base_mse - 1e-4).astype(int),
                "family_harm_label": (best > base_mse + 1e-4).astype(int),
            }
        )
        for f in families:
            stats[f"family={f}"] = float(fam == f)
        rows.append(pd.concat([base, stats], axis=1))
    return robust_nan_to_num(pd.concat(rows, ignore_index=True))


def fit_cga_models(
    train_batch: ForecastBatch,
    calib_batch: ForecastBatch,
    train_candidates: Sequence[ExpertCandidate],
    calib_candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    seed: int,
) -> Tuple[CGAModels, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    families = sorted({c.family for c in train_candidates if c.name != "keep_lrbn" and c.deployable})
    candidate_names = sorted({c.name for c in train_candidates if c.name != "keep_lrbn" and c.deployable})
    train_sample_x = feature_frame(train_batch, schema)
    calib_sample_x = align_frame(feature_frame(calib_batch, schema), list(train_sample_x.columns))
    train_base = mse_per_sample(train_batch.lrbn_pred, train_batch.y_true)
    train_losses = np.stack([mse_per_sample(c.pred, train_batch.y_true) for c in train_candidates if c.name != "keep_lrbn" and c.deployable], axis=1)
    leave_label = (np.min(train_losses, axis=1) < train_base - 1e-4).astype(int)
    leave_head = fit_head(train_sample_x, leave_label, seed=seed, forest=False)

    train_family = family_table(train_batch, train_candidates, schema, families)
    calib_family = family_table(calib_batch, calib_candidates, schema, families)
    family_cols = feature_columns(train_family)
    family_gain_heads: Dict[str, Any] = {}
    family_harm_heads: Dict[str, Any] = {}
    metric_rows: List[Dict[str, Any]] = []
    for fam in families:
        train_rows = train_family[train_family["family"].eq(fam)]
        calib_rows = calib_family[calib_family["family"].eq(fam)]
        x_train = train_rows[family_cols]
        x_calib = align_frame(calib_rows[family_cols], family_cols)
        gain = fit_head(x_train, train_rows["family_gain_label"].to_numpy(int), seed=seed + len(fam), forest=False)
        harm = fit_head(x_train, train_rows["family_harm_label"].to_numpy(int), seed=seed + 113 + len(fam), forest=False)
        family_gain_heads[fam] = gain
        family_harm_heads[fam] = harm
        metric_rows.append(binary_metric(f"family_gain::{fam}", "inner_calib", calib_rows["family_gain_label"].to_numpy(int), predict_head(gain, x_calib), "family"))
        metric_rows.append(binary_metric(f"family_harm::{fam}", "inner_calib", calib_rows["family_harm_label"].to_numpy(int), predict_head(harm, x_calib), "family"))

    train_cand = candidate_feature_frame(train_batch, train_candidates, schema, families)
    calib_cand = candidate_feature_frame(calib_batch, calib_candidates, schema, families)
    train_cand_table = candidate_sample_table([c for c in train_candidates if c.name != "keep_lrbn" and c.deployable], train_batch, "inner_train")
    calib_cand_table = candidate_sample_table([c for c in calib_candidates if c.name != "keep_lrbn" and c.deployable], calib_batch, "inner_calib")
    train_cand = pd.concat([train_cand.reset_index(drop=True), train_cand_table[["gain_label", "harm_label", "mse_delta_vs_lrbn"]].reset_index(drop=True)], axis=1)
    calib_cand = pd.concat([calib_cand.reset_index(drop=True), calib_cand_table[["gain_label", "harm_label", "mse_delta_vs_lrbn"]].reset_index(drop=True)], axis=1)
    candidate_cols = feature_columns(train_cand)
    gain_head = fit_head(train_cand[candidate_cols], train_cand["gain_label"].to_numpy(int), seed=seed + 211, forest=False)
    harm_head = fit_head(train_cand[candidate_cols], train_cand["harm_label"].to_numpy(int), seed=seed + 311, forest=False)
    metric_rows.append(
        binary_metric(
            "candidate_gain_global",
            "inner_calib",
            calib_cand["gain_label"].to_numpy(int),
            predict_head(gain_head, align_frame(calib_cand[candidate_cols], candidate_cols)),
            "candidate",
        )
    )
    metric_rows.append(
        binary_metric(
            "candidate_harm_global",
            "inner_calib",
            calib_cand["harm_label"].to_numpy(int),
            predict_head(harm_head, align_frame(calib_cand[candidate_cols], candidate_cols)),
            "candidate",
        )
    )
    leave_calib = (np.min(np.stack([mse_per_sample(c.pred, calib_batch.y_true) for c in calib_candidates if c.name != "keep_lrbn" and c.deployable], axis=1), axis=1) < mse_per_sample(calib_batch.lrbn_pred, calib_batch.y_true) - 1e-4).astype(int)
    metric_rows.append(binary_metric("leave_lrbn", "inner_calib", leave_calib, predict_head(leave_head, calib_sample_x), "sample"))
    models = CGAModels(
        leave_head=leave_head,
        family_gain_heads=family_gain_heads,
        family_harm_heads=family_harm_heads,
        candidate_gain_head=gain_head,
        candidate_harm_head=harm_head,
        sample_columns=list(train_sample_x.columns),
        family_columns=family_cols,
        candidate_columns=candidate_cols,
        families=families,
        candidate_names=candidate_names,
    )
    return models, pd.DataFrame(metric_rows), train_cand, calib_cand


@dataclass
class Policy:
    variant: str
    tau_leave: float
    tau_family_gain: float
    tau_family_harm: float
    tau_candidate_gain: float
    tau_candidate_harm: float
    beta_harm: float
    lambda_existing: float
    lambda_smoothing: float
    lambda_residual: float
    lambda_memory: float

    def lambda_for_family(self, family: str) -> float:
        if family == "smoothing_teacher":
            return self.lambda_smoothing
        if family == "residual_distribution":
            return self.lambda_residual
        if family == "retrieval_memory":
            return self.lambda_memory
        return self.lambda_existing


def policy_grid(variant: str) -> Iterable[Policy]:
    # Compact validation grid: enough to test the CGA mechanism without
    # spending the run budget on redundant threshold combinations.
    tau_leave = [0.55, 0.75]
    tau_family_gain = [0.55]
    tau_family_harm = [0.15]
    tau_candidate_gain = [0.55]
    tau_candidate_harm = [0.15]
    beta_harm = [2.0]
    lambda_profiles = [
        (0.50, 0.50, 0.25, 0.25),
        (0.75, 0.75, 0.50, 0.50),
        (1.00, 1.00, 0.75, 0.75),
    ]
    for tl in tau_leave:
        for tfg in tau_family_gain:
            for tfh in tau_family_harm:
                for tcg in tau_candidate_gain:
                    for tch in tau_candidate_harm:
                        for beta in beta_harm:
                            for le, ls, lr, lm in lambda_profiles:
                                yield Policy(variant, tl, tfg, tfh, tcg, tch, beta, le, ls, lr, lm)


@dataclass
class ScoreBundle:
    leave_score: np.ndarray
    cand_scores: pd.DataFrame
    fam_scores: pd.DataFrame
    cand_by: Dict[str, ExpertCandidate]


def prepare_score_bundle(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    models: CGAModels,
) -> ScoreBundle:
    sample_x = align_frame(feature_frame(batch, schema), models.sample_columns)
    leave_score = predict_head(models.leave_head, sample_x)
    return ScoreBundle(
        leave_score=leave_score,
        cand_scores=scored_candidates(batch, candidates, schema, models),
        fam_scores=family_scores(batch, candidates, schema, models),
        cand_by={c.name: c for c in candidates},
    )


def scored_candidates(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    models: CGAModels,
) -> pd.DataFrame:
    rows = candidate_feature_frame(batch, candidates, schema, models.families)
    rows["row_index"] = np.tile(np.arange(len(batch.meta)), len([c for c in candidates if c.name != "keep_lrbn" and c.deployable]))
    candidate_names = []
    families = []
    for c in candidates:
        if c.name == "keep_lrbn" or not c.deployable:
            continue
        candidate_names.extend([c.name] * len(batch.meta))
        families.extend([c.family] * len(batch.meta))
    rows["candidate"] = candidate_names
    rows["family"] = families
    x = align_frame(rows[models.candidate_columns], models.candidate_columns)
    rows["p_candidate_gain"] = predict_head(models.candidate_gain_head, x)
    rows["p_candidate_harm"] = predict_head(models.candidate_harm_head, x)
    return rows


def family_scores(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    models: CGAModels,
) -> pd.DataFrame:
    rows = family_table(batch, candidates, schema, models.families)
    x = align_frame(rows[models.family_columns], models.family_columns)
    rows["p_family_gain"] = 0.0
    rows["p_family_harm"] = 1.0
    for fam in models.families:
        mask = rows["family"].eq(fam).to_numpy()
        if mask.any():
            xx = align_frame(rows.loc[mask, models.family_columns], models.family_columns)
            rows.loc[mask, "p_family_gain"] = predict_head(models.family_gain_heads[fam], xx)
            rows.loc[mask, "p_family_harm"] = predict_head(models.family_harm_heads[fam], xx)
    rows["row_index"] = np.tile(np.arange(len(batch.meta)), len(models.families))
    return rows


def apply_policy(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    models: CGAModels,
    policy: Policy,
) -> Tuple[np.ndarray, pd.DataFrame]:
    return apply_policy_precomputed(batch, policy, prepare_score_bundle(batch, candidates, schema, models))


def apply_policy_precomputed(
    batch: ForecastBatch,
    policy: Policy,
    scores: ScoreBundle,
) -> Tuple[np.ndarray, pd.DataFrame]:
    pred = batch.lrbn_pred.copy()
    decision_rows: List[Dict[str, Any]] = []
    for i in range(len(batch.meta)):
        reason = "leave_below_threshold"
        selected = "keep_lrbn"
        family = "default"
        lam = 0.0
        score = 0.0
        if scores.leave_score[i] >= policy.tau_leave:
            fam_i = scores.fam_scores[scores.fam_scores["row_index"].eq(i)].copy()
            eligible_families = fam_i[
                (fam_i["p_family_gain"] >= policy.tau_family_gain) & (fam_i["p_family_harm"] <= policy.tau_family_harm)
            ]["family"].astype(str).tolist()
            if eligible_families:
                cand_i = scores.cand_scores[scores.cand_scores["row_index"].eq(i)].copy()
                cand_i = cand_i[cand_i["family"].isin(eligible_families)]
                cand_i = cand_i[
                    (cand_i["p_candidate_gain"] >= policy.tau_candidate_gain)
                    & (cand_i["p_candidate_harm"] <= policy.tau_candidate_harm)
                ].copy()
                if not cand_i.empty:
                    fam_lookup = fam_i.set_index("family")[["p_family_gain", "p_family_harm"]].to_dict("index")
                    cand_i["utility"] = cand_i["p_candidate_gain"] - policy.beta_harm * cand_i["p_candidate_harm"]
                    cand_i["utility"] += 0.25 * cand_i["family"].map(lambda f: fam_lookup.get(f, {}).get("p_family_gain", 0.0))
                    cand_i["utility"] -= 0.25 * cand_i["family"].map(lambda f: fam_lookup.get(f, {}).get("p_family_harm", 1.0))
                    best = cand_i.sort_values("utility", ascending=False).iloc[0]
                    selected = str(best["candidate"])
                    family = str(best["family"])
                    lam = policy.lambda_for_family(family)
                    cand = scores.cand_by[selected]
                    pred[i] = batch.lrbn_pred[i] + lam * (cand.pred[i] - batch.lrbn_pred[i])
                    reason = "selected"
                    score = float(best["utility"])
                else:
                    reason = "no_candidate_eligible"
            else:
                reason = "no_family_eligible"
        decision_rows.append(
            {
                "row_index": i,
                "selected_candidate": selected,
                "selected_family": family,
                "selected": selected != "keep_lrbn",
                "lambda": float(lam),
                "leave_score": float(scores.leave_score[i]),
                "utility_score": float(score),
                "reason": reason,
            }
        )
    return pred, pd.DataFrame(decision_rows)


def calibrate_policy(
    variant: str,
    calib: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    models: CGAModels,
    n_bootstrap: int,
    seed: int,
) -> Tuple[Policy, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    oracle_mse = oracle_best(deployable_candidates(candidates), calib)[1]
    best_policy: Optional[Policy] = None
    best_score = float("inf")
    score_bundle = prepare_score_bundle(calib, candidates, schema, models)
    for policy in policy_grid(variant):
        pred, decisions = apply_policy_precomputed(calib, policy, score_bundle)
        row = metric_row(
            variant,
            pred,
            calib,
            selected=decisions["selected"].to_numpy(bool),
            oracle_mse=oracle_mse,
            n_bootstrap=0,
            seed=seed,
        )
        row.update(asdict(policy))
        safe = variant == "Safe-CGA"
        harm_cap = 0.03 if safe else 0.08
        config_harm_cap = 0.10 if safe else 0.18
        feasible = row["harm_rate"] <= harm_cap and row["max_config_harm"] <= config_harm_cap
        row["calibration_feasible"] = bool(feasible)
        score = float(row["mse_delta_pct_vs_lrbn"])
        score += 200.0 * max(0.0, float(row["harm_rate"]) - harm_cap)
        score += 100.0 * max(0.0, float(row["max_config_harm"]) - config_harm_cap)
        score += 10.0 * max(0.0, 0.10 - float(row["coverage"]))
        row["calibration_score"] = float(score)
        rows.append(row)
        ranked = score if feasible else score + 1000.0
        if ranked < best_score:
            best_score = ranked
            best_policy = policy
    assert best_policy is not None
    return best_policy, pd.DataFrame(rows)


def topk_metrics(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    models: CGAModels,
    k: int = 2,
) -> Dict[str, Any]:
    deploy = deployable_candidates(candidates)
    cand_scores = scored_candidates(batch, deploy, schema, models)
    fam_scores = family_scores(batch, deploy, schema, models)
    cand_losses = {c.name: mse_per_sample(c.pred, batch.y_true) for c in deploy if c.name != "keep_lrbn"}
    names = list(cand_losses.keys())
    if not names:
        return {}
    loss_mat = np.stack([cand_losses[n] for n in names], axis=1)
    best_idx = np.argmin(loss_mat, axis=1)
    oracle_candidate = np.asarray([names[j] for j in best_idx], dtype=object)
    cand_to_family = {c.name: c.family for c in deploy}
    oracle_family = np.asarray([cand_to_family[n] for n in oracle_candidate], dtype=object)
    cand_hit = []
    fam_hit = []
    fam_top1 = []
    for i in range(len(batch.meta)):
        ctop = (
            cand_scores[cand_scores["row_index"].eq(i)]
            .assign(utility=lambda d: d["p_candidate_gain"] - 2.0 * d["p_candidate_harm"])
            .sort_values("utility", ascending=False)["candidate"]
            .astype(str)
            .head(k)
            .tolist()
        )
        ftop = (
            fam_scores[fam_scores["row_index"].eq(i)]
            .assign(utility=lambda d: d["p_family_gain"] - 2.0 * d["p_family_harm"])
            .sort_values("utility", ascending=False)["family"]
            .astype(str)
            .head(k)
            .tolist()
        )
        cand_hit.append(oracle_candidate[i] in ctop)
        fam_hit.append(oracle_family[i] in ftop)
        fam_top1.append(oracle_family[i] == (ftop[0] if ftop else ""))
    return {
        "candidate_top2_hit": float(np.mean(cand_hit)),
        "family_top2_hit": float(np.mean(fam_hit)),
        "family_top1_hit": float(np.mean(fam_top1)),
        "family_minus_candidate_top2_pp": float(100.0 * (np.mean(fam_hit) - np.mean(cand_hit))),
    }


def selector_test_metrics(
    train_metrics: pd.DataFrame,
    test: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    models: CGAModels,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    families = models.families
    test_family = family_table(test, candidates, schema, families)
    fam_rows: List[Dict[str, Any]] = []
    for fam in families:
        rows = test_family[test_family["family"].eq(fam)]
        if rows.empty:
            continue
        x = align_frame(rows[models.family_columns], models.family_columns)
        fam_rows.append(binary_metric(f"family_gain::{fam}", "test", rows["family_gain_label"].to_numpy(int), predict_head(models.family_gain_heads[fam], x), "family"))
        fam_rows.append(binary_metric(f"family_harm::{fam}", "test", rows["family_harm_label"].to_numpy(int), predict_head(models.family_harm_heads[fam], x), "family"))
    test_cand_features = candidate_feature_frame(test, candidates, schema, families)
    test_cand_table = candidate_sample_table([c for c in candidates if c.name != "keep_lrbn" and c.deployable], test, "test")
    test_cand_features = pd.concat(
        [test_cand_features.reset_index(drop=True), test_cand_table[["gain_label", "harm_label", "mse_delta_vs_lrbn"]].reset_index(drop=True)],
        axis=1,
    )
    x_cand = align_frame(test_cand_features[models.candidate_columns], models.candidate_columns)
    cand_rows = [
        binary_metric("candidate_gain_global", "test", test_cand_features["gain_label"].to_numpy(int), predict_head(models.candidate_gain_head, x_cand), "candidate"),
        binary_metric("candidate_harm_global", "test", test_cand_features["harm_label"].to_numpy(int), predict_head(models.candidate_harm_head, x_cand), "candidate"),
    ]
    sample_x = align_frame(feature_frame(test, schema), models.sample_columns)
    leave_label = (
        np.min(
            np.stack([mse_per_sample(c.pred, test.y_true) for c in candidates if c.name != "keep_lrbn" and c.deployable], axis=1),
            axis=1,
        )
        < mse_per_sample(test.lrbn_pred, test.y_true) - 1e-4
    ).astype(int)
    leave_row = binary_metric("leave_lrbn", "test", leave_label, predict_head(models.leave_head, sample_x), "sample")
    metrics = pd.concat([train_metrics, pd.DataFrame(fam_rows + cand_rows + [leave_row])], ignore_index=True)
    topk = pd.DataFrame([topk_metrics(test, candidates, schema, models)])
    return metrics, topk, test_cand_table


def selection_distribution(decisions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    for variant, df in decisions.items():
        vc = df["selected_candidate"].value_counts(normalize=True).rename_axis("selected_candidate").reset_index(name="share")
        vc.insert(0, "variant", variant)
        rows.append(vc)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def oracle_distribution(
    candidates: Sequence[ExpertCandidate],
    batch: ForecastBatch,
    pool_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    deploy = deployable_candidates(candidates)
    _, _, best_names = oracle_best(deploy, batch)
    cand_to_family = {c.name: c.family for c in deploy}
    cand = pd.Series(best_names).value_counts(normalize=True).rename_axis("candidate").reset_index(name="oracle_share")
    cand.insert(0, "oracle_pool", pool_name)
    fam = pd.Series([cand_to_family[str(n)] for n in best_names]).value_counts(normalize=True).rename_axis("family").reset_index(name="oracle_share")
    fam.insert(0, "oracle_pool", pool_name)
    return cand, fam


def family_oracle_slice_rows(
    candidates: Sequence[ExpertCandidate],
    batch: ForecastBatch,
    masks: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    base = mse_per_sample(batch.lrbn_pred, batch.y_true)
    for fam in sorted({c.family for c in candidates if c.name != "keep_lrbn" and c.deployable}):
        fam_cands = [c for c in candidates if c.family == fam and c.deployable]
        if not fam_cands:
            continue
        pred, _, _ = oracle_best(fam_cands, batch)
        mse = mse_per_sample(pred, batch.y_true)
        for name, mask in masks.items():
            if not np.asarray(mask).any():
                continue
            rows.append(
                {
                    "family_oracle": fam,
                    "slice": name,
                    "n": int(np.asarray(mask).sum()),
                    "mse": float(np.mean(mse[mask])),
                    "lrbn_mse": float(np.mean(base[mask])),
                    "mse_delta_pct_vs_lrbn": safe_pct(float(np.mean(mse[mask])), float(np.mean(base[mask]))),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_delta_json(rows: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for _, row in rows.iterrows():
        name = str(row.get("variant", row.get("method", "unknown")))
        out[name] = {
            "ci95_low_delta_raw": float(row.get("ci95_low_delta_raw", np.nan)),
            "ci95_high_delta_raw": float(row.get("ci95_high_delta_raw", np.nan)),
            "p_bootstrap_delta_lt_zero": float(row.get("p_bootstrap_delta_lt_zero", np.nan)),
        }
    return out


def failure_cases(batch: ForecastBatch, preds: Mapping[str, np.ndarray], decisions: Mapping[str, pd.DataFrame], top_n: int = 80) -> pd.DataFrame:
    base = mse_per_sample(batch.lrbn_pred, batch.y_true)
    frames = []
    meta = batch.meta.reset_index(drop=True)
    for variant, pred in preds.items():
        mse = mse_per_sample(pred, batch.y_true)
        delta = mse - base
        df = meta.copy()
        df["variant"] = variant
        df["mse_lrbn"] = base
        df["mse_variant"] = mse
        df["mse_delta_vs_lrbn"] = delta
        dec = decisions.get(variant)
        if dec is not None:
            df = pd.concat([df, dec[["selected_candidate", "selected_family", "lambda", "reason"]].reset_index(drop=True)], axis=1)
        frames.append(df.sort_values("mse_delta_vs_lrbn", ascending=False).head(top_n))
    return pd.concat(frames, ignore_index=True)


def build_summary(output_dir: Path, verdict: Mapping[str, Any], overall: pd.DataFrame, selector_topk: pd.DataFrame) -> str:
    head = [
        "# Stage 10 CGA Compact Validation",
        "",
        f"Status: `{verdict['status']}`.",
        "",
        "## Headline Metrics",
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
                    "test_threshold_leakage",
                ]
            ],
            max_rows=24,
        ),
        "",
        "## Selector Top-k",
        "",
        df_to_md(selector_topk, max_rows=8),
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(verdict, indent=2, ensure_ascii=False, default=json_default),
        "```",
        "",
        "## Artifacts",
        "",
        f"Output directory: `{output_dir}`",
    ]
    return "\n".join(head)


def verify_outputs(output_dir: Path) -> pd.DataFrame:
    required = [
        "stage10_config.json",
        "stage10_candidate_pool.csv",
        "stage10_candidate_metadata.csv",
        "stage10_oracle_old_vs_new.csv",
        "stage10_oracle_family_distribution.csv",
        "stage10_candidate_oracle_distribution.csv",
        "stage10_family_selector_metrics.csv",
        "stage10_candidate_selector_metrics.csv",
        "stage10_policy_grid_safe.csv",
        "stage10_policy_grid_balanced.csv",
        "stage10_overall.csv",
        "stage10_per_config.csv",
        "stage10_slice_metrics.csv",
        "stage10_selection_distribution.csv",
        "stage10_memory_retrieval_diagnostics.csv",
        "stage10_bootstrap_ci.json",
        "stage10_failure_cases.csv",
        "stage10_verdict.json",
        "summary.md",
    ]
    return pd.DataFrame(
        [
            {
                "artifact": name,
                "exists": bool((output_dir / name).exists()),
                "bytes": int((output_dir / name).stat().st_size) if (output_dir / name).exists() else 0,
            }
            for name in required
        ]
    )


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
    pools = build_cga_pools(assets)
    write_json(
        output_dir / "stage10_config.json",
        {
            "stage": "stage10_cga",
            "metrics_csv": metrics_csv,
            "stage5_dir": stage5_dir,
            "stage6_dir": stage6_dir,
            "stage7_dir": stage7_dir,
            "stage8_dir": stage8_dir,
            "stage3_dir": stage3_dir,
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "compact_protocol": "ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192 x seed 2026",
            "calibration": "validation inner-train/inner-calib only",
            "test_threshold_leakage": False,
        },
    )

    candidate_pool = pd.concat(
        [
            candidate_sample_table(deployable_candidates(pools.calib_candidates), assets.val_calib, "inner_calib"),
            candidate_sample_table(deployable_candidates(pools.test_candidates), assets.test, "test"),
        ],
        ignore_index=True,
    )
    metadata = pd.concat(
        [
            candidate_metadata(pools.train_candidates, "inner_train"),
            candidate_metadata(pools.calib_candidates, "inner_calib"),
            candidate_metadata(pools.test_candidates, "test"),
        ],
        ignore_index=True,
    ).drop_duplicates()
    candidate_pool.to_csv(output_dir / "stage10_candidate_pool.csv", index=False)
    metadata.to_csv(output_dir / "stage10_candidate_metadata.csv", index=False)

    old_deploy = deployable_candidates(assets.old_test_candidates)
    stage9_expanded = deployable_candidates(assets.expanded_test_candidates)
    stage10_deploy = deployable_candidates(pools.test_candidates)
    oracle_rows: List[Dict[str, Any]] = []
    cand_dists: List[pd.DataFrame] = []
    fam_dists: List[pd.DataFrame] = []
    for name, cand in [
        ("oracle_old_deployable", old_deploy),
        ("oracle_stage9_expanded", stage9_expanded),
        ("oracle_stage10_cga_full", stage10_deploy),
    ]:
        pred, oracle_mse, _ = oracle_best(cand, assets.test)
        row = metric_row(name, pred, assets.test, oracle_mse=oracle_mse, n_bootstrap=n_bootstrap, seed=seed)
        row["variant"] = name
        oracle_rows.append(row)
        cd, fd = oracle_distribution(cand, assets.test, name)
        cand_dists.append(cd)
        fam_dists.append(fd)
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(output_dir / "stage10_oracle_old_vs_new.csv", index=False)
    pd.concat(cand_dists, ignore_index=True).to_csv(output_dir / "stage10_candidate_oracle_distribution.csv", index=False)
    family_dist = pd.concat(fam_dists, ignore_index=True)
    family_dist.to_csv(output_dir / "stage10_oracle_family_distribution.csv", index=False)

    models, selector_calib_metrics, train_cand_features, calib_cand_features = fit_cga_models(
        assets.val_train,
        assets.val_calib,
        pools.train_candidates,
        pools.calib_candidates,
        assets.schema,
        seed=seed,
    )
    selector_metrics, topk_df, test_cand_table = selector_test_metrics(
        selector_calib_metrics,
        assets.test,
        pools.test_candidates,
        assets.schema,
        models,
    )
    family_selector_metrics = selector_metrics[selector_metrics["level"].isin(["family", "sample"])].copy()
    candidate_selector_metrics = selector_metrics[selector_metrics["level"].eq("candidate")].copy()
    family_selector_metrics.to_csv(output_dir / "stage10_family_selector_metrics.csv", index=False)
    candidate_selector_metrics.to_csv(output_dir / "stage10_candidate_selector_metrics.csv", index=False)

    safe_policy, safe_grid = calibrate_policy(
        "Safe-CGA",
        assets.val_calib,
        pools.calib_candidates,
        assets.schema,
        models,
        n_bootstrap=0,
        seed=seed,
    )
    balanced_policy, balanced_grid = calibrate_policy(
        "Balanced-CGA",
        assets.val_calib,
        pools.calib_candidates,
        assets.schema,
        models,
        n_bootstrap=0,
        seed=seed,
    )
    safe_grid.to_csv(output_dir / "stage10_policy_grid_safe.csv", index=False)
    balanced_grid.to_csv(output_dir / "stage10_policy_grid_balanced.csv", index=False)

    safe_pred, safe_decisions = apply_policy(assets.test, pools.test_candidates, assets.schema, models, safe_policy)
    balanced_pred, balanced_decisions = apply_policy(assets.test, pools.test_candidates, assets.schema, models, balanced_policy)
    full_oracle_pred, full_oracle_mse, _ = oracle_best(stage10_deploy, assets.test)
    preds = {
        "LRBN": assets.test.lrbn_pred,
        "SRA-BP-balanced": candidate_dict(assets.old_test_candidates).get("sra_balanced", assets.old_test_candidates[0]).pred,
        "oracle_stage10_cga_full": full_oracle_pred,
        "Safe-CGA": safe_pred,
        "Balanced-CGA": balanced_pred,
    }
    decisions = {
        "Safe-CGA": safe_decisions,
        "Balanced-CGA": balanced_decisions,
    }
    overall_rows: List[Dict[str, Any]] = []
    for variant, pred in preds.items():
        selected = None
        if variant in decisions:
            selected = decisions[variant]["selected"].to_numpy(bool)
        oracle_mse = full_oracle_mse if variant in {"Safe-CGA", "Balanced-CGA", "oracle_stage10_cga_full"} else None
        row = metric_row(variant, pred, assets.test, selected=selected, oracle_mse=oracle_mse, n_bootstrap=n_bootstrap, seed=seed)
        row["variant"] = variant
        if variant == "Safe-CGA":
            row.update({f"policy_{k}": v for k, v in asdict(safe_policy).items()})
        if variant == "Balanced-CGA":
            row.update({f"policy_{k}": v for k, v in asdict(balanced_policy).items()})
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(output_dir / "stage10_overall.csv", index=False)

    masks = stage8_slice_masks(assets.test, assets.schema, stage8_slice_thresholds(assets.val, assets.schema))
    per_config = pd.concat(
        [per_config_rows(variant, pred, assets.test, decisions.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    per_config.to_csv(output_dir / "stage10_per_config.csv", index=False)
    slice_df = pd.concat(
        [slice_rows(variant, pred, assets.test, masks, decisions.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    family_slice = family_oracle_slice_rows(stage10_deploy, assets.test, masks)
    family_slice.insert(0, "variant", "family_oracle")
    slice_out = pd.concat([slice_df, family_slice], ignore_index=True, sort=False)
    slice_out.to_csv(output_dir / "stage10_slice_metrics.csv", index=False)
    selection_distribution(decisions).to_csv(output_dir / "stage10_selection_distribution.csv", index=False)

    mem_diag = pd.concat(
        [
            pools.train_memory_diag.assign(split="inner_train"),
            pools.calib_memory_diag.assign(split="inner_calib"),
            pools.test_memory_diag.assign(split="test"),
        ],
        ignore_index=True,
    )
    mem_gain = test_cand_table[test_cand_table["family"].eq("retrieval_memory")].groupby("candidate", observed=True).agg(
        n=("mse_delta_vs_lrbn", "size"),
        mean_mse_delta=("mse_delta_vs_lrbn", "mean"),
        win_rate=("gain_label", "mean"),
        harm_rate=("harm_label", "mean"),
    )
    mem_diag_summary = mem_gain.reset_index()
    if not mem_diag.empty:
        mem_diag = mem_diag.merge(mem_diag_summary, left_on="memory_candidate", right_on="candidate", how="left")
    mem_diag.to_csv(output_dir / "stage10_memory_retrieval_diagnostics.csv", index=False)
    write_json(output_dir / "stage10_bootstrap_ci.json", bootstrap_delta_json(overall))
    failure_cases(assets.test, {"Safe-CGA": safe_pred, "Balanced-CGA": balanced_pred}, decisions).to_csv(
        output_dir / "stage10_failure_cases.csv", index=False
    )

    old_mse = float(oracle_df.loc[oracle_df["variant"].eq("oracle_old_deployable"), "mse"].iloc[0])
    stage10_mse = float(oracle_df.loc[oracle_df["variant"].eq("oracle_stage10_cga_full"), "mse"].iloc[0])
    oracle_improvement = safe_pct(stage10_mse, old_mse)
    stage10_family_dist = family_dist[family_dist["oracle_pool"].eq("oracle_stage10_cga_full")]
    new_family_share = float(stage10_family_dist[stage10_family_dist["family"].isin(NEW_FAMILIES)]["oracle_share"].sum())
    topk = topk_df.iloc[0].to_dict() if not topk_df.empty else {}
    non_boundary = slice_out[(slice_out["variant"].eq("family_oracle")) & (slice_out["slice"].eq("non_boundary"))]
    new_family_non_boundary_improves = bool(
        not non_boundary[non_boundary["family_oracle"].isin(NEW_FAMILIES) & (non_boundary["mse_delta_pct_vs_lrbn"] < 0.0)].empty
    )

    safe_row = overall[overall["variant"].eq("Safe-CGA")].iloc[0].to_dict()
    balanced_row = overall[overall["variant"].eq("Balanced-CGA")].iloc[0].to_dict()
    safe_pass = bool(
        safe_row["mse_delta_pct_vs_lrbn"] <= -2.2
        and safe_row["harm_rate"] <= 0.03
        and safe_row["max_config_harm"] <= 0.10
        and safe_row["config_improved_ratio"] >= 0.75
        and safe_row["oracle_gain_fraction"] >= 0.15
        and safe_row.get("ci95_high_delta_raw", 1.0) < 0.0
    )
    balanced_pass = bool(
        balanced_row["mse_delta_pct_vs_lrbn"] <= -3.0
        and balanced_row["harm_rate"] <= 0.08
        and balanced_row["max_config_harm"] <= 0.18
        and balanced_row["config_improved_ratio"] >= 0.75
        and balanced_row["oracle_gain_fraction"] >= 0.20
    )
    mechanism_pass = bool(
        oracle_improvement <= -5.0
        and new_family_share >= 0.30
        and float(topk.get("family_minus_candidate_top2_pp", -999.0)) >= 15.0
        and new_family_non_boundary_improves
    )
    if safe_pass or balanced_pass:
        status = "deployable_cga_candidate_passed_compact"
    elif mechanism_pass:
        status = "mechanism_pass_selector_still_insufficient"
    elif oracle_improvement <= -5.0:
        status = "oracle_space_expanded_but_arbitration_failed"
    else:
        status = "candidate_generation_insufficient"
    verdict = {
        "stage": "stage10_cga",
        "status": status,
        "mechanism_pass": mechanism_pass,
        "safe_cga_pass": safe_pass,
        "balanced_cga_pass": balanced_pass,
        "oracle_improvement_pct_vs_old_deployable": oracle_improvement,
        "new_family_oracle_share": new_family_share,
        "family_top2_hit": float(topk.get("family_top2_hit", np.nan)),
        "candidate_top2_hit": float(topk.get("candidate_top2_hit", np.nan)),
        "family_minus_candidate_top2_pp": float(topk.get("family_minus_candidate_top2_pp", np.nan)),
        "new_family_non_boundary_improves": new_family_non_boundary_improves,
        "safe_cga": {
            "mse_delta_pct_vs_lrbn": float(safe_row["mse_delta_pct_vs_lrbn"]),
            "harm_rate": float(safe_row["harm_rate"]),
            "max_config_harm": float(safe_row["max_config_harm"]),
            "config_improved_ratio": float(safe_row["config_improved_ratio"]),
            "oracle_gain_fraction": float(safe_row["oracle_gain_fraction"]),
        },
        "balanced_cga": {
            "mse_delta_pct_vs_lrbn": float(balanced_row["mse_delta_pct_vs_lrbn"]),
            "harm_rate": float(balanced_row["harm_rate"]),
            "max_config_harm": float(balanced_row["max_config_harm"]),
            "config_improved_ratio": float(balanced_row["config_improved_ratio"]),
            "oracle_gain_fraction": float(balanced_row["oracle_gain_fraction"]),
        },
        "safe_policy": asdict(safe_policy),
        "balanced_policy": asdict(balanced_policy),
        "test_threshold_leakage": False,
    }
    write_json(output_dir / "stage10_verdict.json", verdict)
    summary = build_summary(output_dir, verdict, overall, topk_df)
    (output_dir / "summary.md").write_text(summary, encoding="utf-8")
    completeness = verify_outputs(output_dir)
    completeness.to_csv(output_dir / "stage10_output_completeness.csv", index=False)
    return {
        "verdict": verdict,
        "overall": overall,
        "selector_metrics": selector_metrics,
        "topk": topk_df,
        "completeness": completeness,
    }
