#!/usr/bin/env python
"""Stage 15 endogenous low-harm editor validation.

This stage implements the first-round compact plan from
`deep-research-report (5).md`: keep LRBN and the Stage 10 source families
fixed, then test bounded residual editors instead of discrete selectors.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from halluguard_lrbn_bp import ForecastBatch, mae_per_sample, mse_per_sample
from halluguard_stage6_mechanism import feature_frame, horizons, safe_pct, valid_part
from halluguard_stage7_safe_tae import ExpertCandidate, align_frame, robust_nan_to_num, write_json
from halluguard_stage8_safe_tae_pareto import stage8_slice_masks, stage8_slice_thresholds
from halluguard_stage9_architecture_validation import (
    deployable_candidates,
    metric_row,
    oracle_best,
    per_config_rows,
    prepare_assets,
    slice_rows,
)
from halluguard_stage10_cga import (
    build_cga_pools,
    candidate_feature_frame,
    candidate_metadata,
    df_to_md,
    json_default,
)


STAGE15_VARIANTS = [
    "H1 Residual Atom Simplex Editor",
    "H3 Any-Quantile Residual Envelope",
    "H5 Local-Global Decoupled Sparse Editor",
    "H2 Prototype Codebook Local Editor",
    "H4 Retrieval-Conditioned Residual Adapter",
]


@dataclass(frozen=True)
class EditorPolicy:
    variant: str
    rho: float = 0.50
    group_lasso: float = 0.0
    tau_score: float = 0.0
    tau_harm: float = 0.35
    beta_harm: float = 1.5
    residual_cap: float = 0.35
    temp: float = 0.35
    shrink: float = 0.50
    width_coef: float = 1.0
    center_mode: str = "q50"
    lowpass_kernel: int = 5
    local_shrink: float = 0.25
    global_shrink: float = 0.25
    mask_quantile: float = 0.85
    patch_len: int = 16
    stride: int = 8
    codebook_size: int = 16
    retrieval_k: int = 9


@dataclass
class PatchCodebook:
    patch_len: int
    stride: int
    codebook_size: int
    centers: np.ndarray
    train_dist_q: Dict[float, float]


@dataclass
class RetrievalMemory:
    nn: Any
    columns: List[str]
    mean: np.ndarray
    std: np.ndarray
    residual: np.ndarray
    train_dist_median: float


class ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value, dtype=float)


class ConstantProbability:
    def __init__(self, p: float):
        self.p = float(np.clip(p, 0.0, 1.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([np.full(len(x), 1.0 - self.p), np.full(len(x), self.p)])


@dataclass
class AtomModels:
    utility: Any
    harm: Any
    columns: List[str]
    candidate_names: List[str]
    candidate_families: Dict[str, str]


@dataclass
class AtomScores:
    rows: pd.DataFrame
    candidate_by: Dict[str, ExpertCandidate]


def safe_feature_columns(df: pd.DataFrame) -> List[str]:
    blocked_exact = {"candidate", "family", "tier", "sample_id", "sample_key", "split", "row_index"}
    blocked_tokens = ["label", "target", "mse", "mae", "delta", "oracle", "best", "true"]
    cols: List[str] = []
    for col in df.columns:
        low = col.lower()
        if col in blocked_exact or any(tok in low for tok in blocked_tokens):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def candidate_list(candidates: Sequence[ExpertCandidate]) -> List[ExpertCandidate]:
    return [c for c in candidates if c.name != "keep_lrbn" and c.deployable and c.family not in {"raw", "default"}]


def fit_regressor(x: pd.DataFrame, y: np.ndarray, seed: int) -> Any:
    yy = np.asarray(y, dtype=float)
    if len(yy) == 0 or float(np.nanstd(yy)) < 1e-12:
        return ConstantRegressor(float(np.nanmean(yy)) if len(yy) else 0.0)
    model = RandomForestRegressor(n_estimators=180, max_depth=8, min_samples_leaf=8, random_state=seed, n_jobs=1)
    model.fit(x.to_numpy(float), np.nan_to_num(yy, nan=0.0, posinf=0.0, neginf=0.0))
    return model


def fit_classifier(x: pd.DataFrame, y: np.ndarray, seed: int) -> Any:
    yy = np.asarray(y, dtype=int)
    if len(yy) == 0 or len(np.unique(yy)) < 2:
        return ConstantProbability(float(yy.mean()) if len(yy) else 0.0)
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=7,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x.to_numpy(float), yy)
    return model


def candidate_delta(batch: ForecastBatch, candidates: Sequence[ExpertCandidate]) -> Tuple[List[ExpertCandidate], np.ndarray]:
    cands = candidate_list(candidates)
    base = mse_per_sample(batch.lrbn_pred, batch.y_true)
    delta = np.stack([mse_per_sample(c.pred, batch.y_true) - base for c in cands], axis=1)
    return cands, delta


def fit_atom_models(train: ForecastBatch, candidates: Sequence[ExpertCandidate], schema: Dict[str, List[Any]], seed: int) -> AtomModels:
    cands, delta = candidate_delta(train, candidates)
    families = sorted({c.family for c in cands})
    feats = candidate_feature_frame(train, cands, schema, families).reset_index(drop=True)
    cols = safe_feature_columns(feats)
    flat_delta = np.concatenate([delta[:, j] for j in range(delta.shape[1])], axis=0)
    utility = fit_regressor(feats[cols], -flat_delta, seed + 101)
    harm = fit_classifier(feats[cols], flat_delta > 1e-4, seed + 102)
    return AtomModels(
        utility=utility,
        harm=harm,
        columns=cols,
        candidate_names=[c.name for c in cands],
        candidate_families={c.name: c.family for c in cands},
    )


def prepare_atom_scores(batch: ForecastBatch, candidates: Sequence[ExpertCandidate], schema: Dict[str, List[Any]], models: AtomModels) -> AtomScores:
    cands = [c for c in candidate_list(candidates) if c.name in models.candidate_names]
    families = sorted({c.family for c in cands})
    feats = candidate_feature_frame(batch, cands, schema, families).reset_index(drop=True)
    x = align_frame(feats[models.columns], models.columns)
    rows = feats.copy()
    rows["row_index"] = np.tile(np.arange(len(batch.meta)), len(cands))
    rows["pred_utility"] = np.asarray(models.utility.predict(x.to_numpy(float)), dtype=float)
    rows["p_harm"] = np.asarray(models.harm.predict_proba(x.to_numpy(float))[:, 1], dtype=float)
    return AtomScores(rows=rows, candidate_by={c.name: c for c in cands})


def scale_matrix(batch: ForecastBatch) -> np.ndarray:
    scale = np.nanstd(batch.context, axis=(1, 2))
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    return scale.reshape(-1, 1, 1)


def clip_delta(delta: np.ndarray, batch: ForecastBatch, cap: float) -> np.ndarray:
    return np.clip(delta, -float(cap) * scale_matrix(batch), float(cap) * scale_matrix(batch))


def moving_average(arr: np.ndarray, kernel: int) -> np.ndarray:
    k = max(1, int(kernel))
    if k <= 1:
        return arr.copy()
    pad = k // 2
    padded = np.pad(arr, ((0, 0), (pad, pad), (0, 0)), mode="edge")
    out = np.zeros_like(arr)
    for t in range(arr.shape[1]):
        out[:, t, :] = np.nanmean(padded[:, t : t + k, :], axis=1)
    return np.where(np.isfinite(out), out, 0.0)


def softmax(x: np.ndarray, temp: float) -> np.ndarray:
    if len(x) == 0:
        return x
    z = x / max(float(temp), 1e-6)
    z = z - np.nanmax(z)
    e = np.exp(z)
    return e / (np.nansum(e) + 1e-12)


def apply_residual_atom_simplex(batch: ForecastBatch, scores: AtomScores, policy: EditorPolicy) -> Tuple[np.ndarray, pd.DataFrame]:
    pred = batch.lrbn_pred.copy()
    decisions: List[Dict[str, Any]] = []
    by_row = {int(k): v for k, v in scores.rows.groupby("row_index", observed=True)}
    for i in range(len(batch.meta)):
        rows = by_row.get(i, pd.DataFrame()).copy()
        rows["score"] = rows["pred_utility"] - policy.beta_harm * rows["p_harm"] - policy.group_lasso
        rows = rows[(rows["score"] >= policy.tau_score) & (rows["p_harm"] <= policy.tau_harm)].sort_values("score", ascending=False)
        selected = False
        if not rows.empty:
            weights = softmax(rows["score"].to_numpy(float), policy.temp)
            delta = np.zeros_like(batch.lrbn_pred[i])
            used: List[str] = []
            for w, (_, row) in zip(weights, rows.iterrows()):
                cname = str(row["candidate"])
                cand = scores.candidate_by.get(cname)
                if cand is None:
                    continue
                delta += float(policy.rho) * float(w) * (cand.pred[i] - batch.lrbn_pred[i])
                used.append(cname)
            delta = clip_delta(delta[None, ...], batch.subset(np.arange(len(batch.meta)) == i), policy.residual_cap)[0]
            pred[i] = batch.lrbn_pred[i] + delta
            selected = bool(used)
            action = ",".join(used[:4]) + ("..." if len(used) > 4 else "")
            score = float(rows["score"].mean())
        else:
            action = "keep_lrbn"
            score = 0.0
        decisions.append({"row_index": i, "selected": selected, "selected_action": action, "accept_score": score})
    return pred, pd.DataFrame(decisions)


def residual_quantiles(
    train: ForecastBatch,
    batch: ForecastBatch,
    qs: Sequence[float] = (0.10, 0.25, 0.50, 0.75, 0.90),
) -> Dict[float, np.ndarray]:
    residual = train.y_true - train.lrbn_pred
    out = {float(q): np.zeros_like(batch.lrbn_pred, dtype=float) for q in qs}
    train_meta = train.meta.assign(row_index=np.arange(len(train.meta)))
    batch_meta = batch.meta.assign(row_index=np.arange(len(batch.meta)))
    fallback: Dict[int, Dict[float, np.ndarray]] = {}
    for h, group in train_meta.groupby("horizon", observed=True):
        h = int(h)
        idx = group["row_index"].to_numpy(int)
        stack = np.stack([valid_part(residual, int(i), h) for i in idx], axis=0)
        fallback[h] = {float(q): np.nanquantile(stack, float(q), axis=0) for q in qs}
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
            qmap = fallback[h]
        else:
            idx = tr["row_index"].to_numpy(int)
            stack = np.stack([valid_part(residual, int(i), h) for i in idx], axis=0)
            qmap = {float(q): np.nanquantile(stack, float(q), axis=0) for q in qs}
        batch_idx = group["row_index"].to_numpy(int)
        for q in qs:
            out[float(q)][batch_idx, :h, :] = qmap[float(q)]
    return {q: np.where(np.isfinite(arr), arr, 0.0) for q, arr in out.items()}


def quantile_center(qd: Mapping[float, np.ndarray], mode: str) -> np.ndarray:
    if mode == "trimmed_mean":
        return (qd[0.25] + qd[0.50] + qd[0.75]) / 3.0
    if mode == "mid_iqr":
        return (qd[0.25] + qd[0.75]) / 2.0
    return qd[0.50]


def apply_any_quantile(batch: ForecastBatch, qd: Mapping[float, np.ndarray], policy: EditorPolicy) -> Tuple[np.ndarray, pd.DataFrame]:
    center = quantile_center(qd, policy.center_mode)
    width = np.abs(qd[0.90] - qd[0.10])
    width_score = np.nanmean(width / (scale_matrix(batch) + 1e-8), axis=(1, 2))
    shrink = policy.shrink / (1.0 + policy.width_coef * width_score)
    delta = center * shrink[:, None, None]
    delta = clip_delta(delta, batch, policy.residual_cap)
    pred = batch.lrbn_pred + delta
    selected = np.sqrt(np.nanmean(delta**2, axis=(1, 2))) / (scale_matrix(batch).reshape(-1) + 1e-8) > 0.01
    decisions = pd.DataFrame(
        {
            "row_index": np.arange(len(batch.meta)),
            "selected": selected,
            "selected_action": np.where(selected, "quantile_envelope", "keep_lrbn"),
            "accept_score": 1.0 / (1.0 + width_score),
        }
    )
    return pred, decisions


def temporal_boundary_score(batch: ForecastBatch) -> np.ndarray:
    p = np.asarray(batch.lrbn_pred, dtype=float)
    c = np.asarray(batch.context, dtype=float)
    score = np.zeros_like(p)
    score[:, 0, :] = np.abs(p[:, 0, :] - c[:, -1, :])
    if p.shape[1] > 1:
        score[:, 1:, :] += np.abs(np.diff(p, axis=1))
    if p.shape[1] > 2:
        score[:, 2:, :] += np.abs(np.diff(p, n=2, axis=1))
    return score / (scale_matrix(batch) + 1e-8)


def local_atom_residual(batch: ForecastBatch, candidates: Sequence[ExpertCandidate]) -> np.ndarray:
    fams = {"boundary", "residual_distribution", "smoothing_teacher"}
    arrs = [c.pred - batch.lrbn_pred for c in candidate_list(candidates) if c.family in fams]
    if not arrs:
        return np.zeros_like(batch.lrbn_pred)
    return np.nanmedian(np.stack(arrs, axis=0), axis=0)


def _patch_starts(horizon: int, patch_len: int, stride: int) -> List[int]:
    h = int(horizon)
    p = int(patch_len)
    s = max(1, int(stride))
    if h <= p:
        return [0]
    starts = list(range(0, h - p + 1, s))
    if starts[-1] != h - p:
        starts.append(h - p)
    return starts


def fit_patch_codebooks(
    train: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    patch_lens: Sequence[int] = (8, 16, 24),
    strides: Sequence[int] = (4, 8),
    codebook_sizes: Sequence[int] = (16,),
    seed: int = 2026,
    max_patches: int = 20000,
) -> Dict[Tuple[int, int, int], PatchCodebook]:
    """Fit residual patch prototypes from validation inner-train winners.

    Candidate residuals are normalized by each sample's context scale before
    clustering, then rescaled at application time. The target is used only on
    validation inner-train to decide which candidate patches are worth learning.
    """

    cands = candidate_list(candidates)
    if not cands:
        return {}
    base = mse_per_sample(train.lrbn_pred, train.y_true)
    cand_mse = {c.name: mse_per_sample(c.pred, train.y_true) for c in cands}
    scales = scale_matrix(train).reshape(-1)
    hvec = horizons(train)
    rng = np.random.default_rng(seed)
    out: Dict[Tuple[int, int, int], PatchCodebook] = {}
    for patch_len in patch_lens:
        for stride in strides:
            patches: List[np.ndarray] = []
            for c in cands:
                if c.family not in {"boundary", "residual_distribution", "smoothing_teacher", "retrieval_memory"}:
                    continue
                delta = c.pred - train.lrbn_pred
                improved = cand_mse[c.name] < base - 1e-6
                idxs = np.where(improved)[0]
                if len(idxs) == 0:
                    continue
                for i in idxs:
                    h = int(hvec[i])
                    if h <= 0:
                        continue
                    for start in _patch_starts(h, patch_len, stride):
                        end = min(h, start + patch_len)
                        if end - start != patch_len:
                            continue
                        patch = delta[i, start:end, :] / (scales[i] + 1e-8)
                        if np.all(np.isfinite(patch)):
                            patches.append(patch.reshape(-1))
            if len(patches) < 4:
                continue
            x = np.vstack(patches)
            if len(x) > max_patches:
                keep = rng.choice(len(x), size=max_patches, replace=False)
                x = x[keep]
            for codebook_size in codebook_sizes:
                k = min(int(codebook_size), max(2, len(x)))
                km = MiniBatchKMeans(
                    n_clusters=k,
                    random_state=seed + patch_len * 17 + stride * 31 + k,
                    batch_size=min(1024, max(16, len(x))),
                    n_init=5,
                    max_iter=150,
                )
                labels = km.fit_predict(x)
                centers = np.asarray(km.cluster_centers_, dtype=float)
                dist = np.sqrt(np.nanmean((x - centers[labels]) ** 2, axis=1))
                qmap = {float(q): float(np.nanquantile(dist, q)) for q in [0.25, 0.50, 0.75, 0.90]}
                out[(int(patch_len), int(stride), int(codebook_size))] = PatchCodebook(
                    patch_len=int(patch_len),
                    stride=int(stride),
                    codebook_size=int(codebook_size),
                    centers=centers,
                    train_dist_q=qmap,
                )
    return out


def apply_prototype_codebook(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    codebooks: Mapping[Tuple[int, int, int], PatchCodebook],
    mask_threshold: float,
    policy: EditorPolicy,
) -> Tuple[np.ndarray, pd.DataFrame]:
    key = (int(policy.patch_len), int(policy.stride), int(policy.codebook_size))
    codebook = codebooks.get(key)
    if codebook is None or len(codebook.centers) == 0:
        pred = batch.lrbn_pred.copy()
        return pred, pd.DataFrame(
            {
                "row_index": np.arange(len(batch.meta)),
                "selected": np.zeros(len(batch.meta), dtype=bool),
                "selected_action": "keep_lrbn",
                "accept_score": 0.0,
            }
        )
    source = local_atom_residual(batch, candidates)
    scales = scale_matrix(batch).reshape(-1)
    boundary = temporal_boundary_score(batch)
    hvec = horizons(batch)
    delta_sum = np.zeros_like(batch.lrbn_pred)
    weight_sum = np.zeros_like(batch.lrbn_pred)
    selected = np.zeros(len(batch.meta), dtype=bool)
    scores = np.zeros(len(batch.meta), dtype=float)
    dist_cut = codebook.train_dist_q.get(float(policy.tau_score), codebook.train_dist_q.get(0.75, float("inf")))
    for i, h in enumerate(hvec):
        h = int(h)
        sample_scores: List[float] = []
        for start in _patch_starts(h, codebook.patch_len, codebook.stride):
            end = min(h, start + codebook.patch_len)
            if end - start != codebook.patch_len:
                continue
            if float(np.nanmean(boundary[i, start:end, :])) < float(mask_threshold):
                continue
            q = source[i, start:end, :] / (scales[i] + 1e-8)
            if not np.all(np.isfinite(q)):
                continue
            qflat = q.reshape(1, -1)
            d = np.sqrt(np.nanmean((codebook.centers - qflat) ** 2, axis=1))
            j = int(np.nanargmin(d))
            if float(d[j]) > float(dist_cut):
                continue
            center = codebook.centers[j].reshape(codebook.patch_len, batch.lrbn_pred.shape[2])
            patch_delta = float(policy.shrink) * center * (scales[i] + 1e-8)
            delta_sum[i, start:end, :] += patch_delta
            weight_sum[i, start:end, :] += 1.0
            sample_scores.append(float(1.0 / (1.0 + d[j] / (dist_cut + 1e-8))))
        if sample_scores:
            selected[i] = True
            scores[i] = float(np.mean(sample_scores))
    delta = np.divide(delta_sum, np.maximum(weight_sum, 1.0))
    delta = clip_delta(delta, batch, policy.residual_cap)
    decisions = pd.DataFrame(
        {
            "row_index": np.arange(len(batch.meta)),
            "selected": selected,
            "selected_action": np.where(selected, "prototype_codebook", "keep_lrbn"),
            "accept_score": scores,
        }
    )
    return batch.lrbn_pred + delta, decisions


def fit_retrieval_memory(train: ForecastBatch, schema: Dict[str, List[Any]]) -> Dict[int, RetrievalMemory]:
    memories: Dict[int, RetrievalMemory] = {}
    residual = train.y_true - train.lrbn_pred
    meta = train.meta.assign(row_index=np.arange(len(train.meta)))
    for horizon, group in meta.groupby("horizon", observed=True):
        idx = group["row_index"].to_numpy(int)
        sub = train.subset(np.isin(np.arange(len(train.meta)), idx))
        feats = feature_frame(sub, schema).reset_index(drop=True)
        cols = safe_feature_columns(feats)
        if not cols or len(sub.meta) < 4:
            continue
        x = robust_nan_to_num(feats[cols]).to_numpy(float)
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
        xz = np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
        k = min(20, len(xz))
        nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        nn.fit(xz)
        dist, _ = nn.kneighbors(xz, n_neighbors=min(k, len(xz)))
        train_dist_median = float(np.nanmedian(dist[:, 1:])) if dist.shape[1] > 1 else 1.0
        memories[int(horizon)] = RetrievalMemory(
            nn=nn,
            columns=cols,
            mean=mean,
            std=std,
            residual=residual[idx],
            train_dist_median=max(train_dist_median, 1e-6),
        )
    return memories


def apply_retrieval_adapter(
    batch: ForecastBatch,
    schema: Dict[str, List[Any]],
    memories: Mapping[int, RetrievalMemory],
    policy: EditorPolicy,
) -> Tuple[np.ndarray, pd.DataFrame]:
    if not memories:
        pred = batch.lrbn_pred.copy()
        return pred, pd.DataFrame(
            {
                "row_index": np.arange(len(batch.meta)),
                "selected": np.zeros(len(batch.meta), dtype=bool),
                "selected_action": "keep_lrbn",
                "accept_score": 0.0,
            }
        )
    delta = np.zeros_like(batch.lrbn_pred)
    selected = np.zeros(len(batch.meta), dtype=bool)
    scores = np.zeros(len(batch.meta), dtype=float)
    meta = batch.meta.assign(row_index=np.arange(len(batch.meta)))
    fallback = next(iter(memories.values()))
    for horizon, group in meta.groupby("horizon", observed=True):
        mem = memories.get(int(horizon), fallback)
        idx = group["row_index"].to_numpy(int)
        sub = batch.subset(np.isin(np.arange(len(batch.meta)), idx))
        feats = feature_frame(sub, schema).reset_index(drop=True)
        x = align_frame(feats, mem.columns)[mem.columns].to_numpy(float)
        xz = np.nan_to_num((x - mem.mean) / mem.std, nan=0.0, posinf=0.0, neginf=0.0)
        k = min(int(policy.retrieval_k), len(mem.residual))
        dist, nn_idx = mem.nn.kneighbors(xz, n_neighbors=k)
        conf = 1.0 / (1.0 + np.nanmean(dist, axis=1) / (mem.train_dist_median + 1e-8))
        weights = np.exp(-dist / (mem.train_dist_median + 1e-8))
        weights = weights / (np.sum(weights, axis=1, keepdims=True) + 1e-12)
        retrieved = np.nansum(weights[:, :, None, None] * mem.residual[nn_idx], axis=1)
        active = conf >= float(policy.tau_score)
        delta[idx] = float(policy.shrink) * conf[:, None, None] * retrieved * active[:, None, None].astype(float)
        selected[idx] = active
        scores[idx] = conf
    delta = clip_delta(delta, batch, policy.residual_cap)
    decisions = pd.DataFrame(
        {
            "row_index": np.arange(len(batch.meta)),
            "selected": selected,
            "selected_action": np.where(selected, "retrieval_residual_adapter", "keep_lrbn"),
            "accept_score": scores,
        }
    )
    return batch.lrbn_pred + delta, decisions


def apply_local_global_sparse(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    qd: Mapping[float, np.ndarray],
    mask_threshold: float,
    policy: EditorPolicy,
) -> Tuple[np.ndarray, pd.DataFrame]:
    center = quantile_center(qd, "q50")
    global_delta = policy.global_shrink * moving_average(center, policy.lowpass_kernel)
    atom = local_atom_residual(batch, candidates)
    local_raw = atom - moving_average(atom, policy.lowpass_kernel)
    mask = temporal_boundary_score(batch) >= float(mask_threshold)
    local_delta = policy.local_shrink * mask.astype(float) * local_raw
    delta = clip_delta(global_delta + local_delta, batch, policy.residual_cap)
    pred = batch.lrbn_pred + delta
    selected = np.sqrt(np.nanmean(delta**2, axis=(1, 2))) / (scale_matrix(batch).reshape(-1) + 1e-8) > 0.01
    decisions = pd.DataFrame(
        {
            "row_index": np.arange(len(batch.meta)),
            "selected": selected,
            "selected_action": np.where(selected, "local_global_sparse", "keep_lrbn"),
            "accept_score": np.nanmean(mask, axis=(1, 2)),
        }
    )
    return pred, decisions


def editor_grid(variant: str) -> Iterable[EditorPolicy]:
    if variant == "H1 Residual Atom Simplex Editor":
        for rho in [0.25, 0.50, 0.75]:
            for gl in [0.0, 1e-3, 1e-2]:
                for tau_harm in [0.20, 0.35]:
                    for cap in [0.20, 0.35]:
                        yield EditorPolicy(variant=variant, rho=rho, group_lasso=gl, tau_harm=tau_harm, residual_cap=cap)
    elif variant == "H3 Any-Quantile Residual Envelope":
        for shrink in [0.25, 0.50, 0.75]:
            for width_coef in [0.0, 1.0, 2.0]:
                for mode in ["q50", "trimmed_mean", "mid_iqr"]:
                    for cap in [0.20, 0.35]:
                        yield EditorPolicy(variant=variant, shrink=shrink, width_coef=width_coef, center_mode=mode, residual_cap=cap)
    elif variant == "H5 Local-Global Decoupled Sparse Editor":
        for mq in [0.80, 0.90, 0.95]:
            for kernel in [3, 5]:
                for local in [0.10, 0.25, 0.50]:
                    for glob in [0.10, 0.25, 0.50]:
                        yield EditorPolicy(variant=variant, mask_quantile=mq, lowpass_kernel=kernel, local_shrink=local, global_shrink=glob, residual_cap=0.35)
    elif variant == "H2 Prototype Codebook Local Editor":
        for patch_len in [8, 16, 24]:
            for stride in [4, 8]:
                for shrink in [0.25, 0.50]:
                    for mq in [0.80, 0.90]:
                        for tau in [0.25, 0.50, 0.75]:
                            for cap in [0.20, 0.35]:
                                yield EditorPolicy(
                                    variant=variant,
                                    patch_len=patch_len,
                                    stride=stride,
                                    codebook_size=16,
                                    shrink=shrink,
                                    mask_quantile=mq,
                                    tau_score=tau,
                                    residual_cap=cap,
                                )
    elif variant == "H4 Retrieval-Conditioned Residual Adapter":
        for k in [5, 9, 15]:
            for shrink in [0.25, 0.50, 0.75]:
                for tau in [0.20, 0.35, 0.50]:
                    for cap in [0.20, 0.35]:
                        yield EditorPolicy(variant=variant, retrieval_k=k, shrink=shrink, tau_score=tau, residual_cap=cap)
    else:
        raise ValueError(f"unknown variant {variant}")


def nontriviality_metrics(pred: np.ndarray, batch: ForecastBatch, patch_len: int = 16) -> Dict[str, float]:
    delta = pred - batch.lrbn_pred
    scale = scale_matrix(batch).reshape(-1)
    sample_rms = np.sqrt(np.nanmean(delta**2, axis=(1, 2))) / (scale + 1e-8)
    patch_flags: List[bool] = []
    hs = horizons(batch)
    for i, h in enumerate(hs):
        h = int(h)
        step = max(1, patch_len)
        for start in range(0, h, step):
            end = min(h, start + patch_len)
            if end <= start:
                continue
            rms = float(np.sqrt(np.nanmean(delta[i, start:end, :] ** 2)) / (scale[i] + 1e-8))
            patch_flags.append(rms > 0.01)
    base_energy = float(np.nanmean(batch.lrbn_pred**2)) + 1e-12
    return {
        "lrbn_equiv_rate": float(np.mean(sample_rms <= 0.01)),
        "active_patch_ratio": float(np.mean(patch_flags)) if patch_flags else 0.0,
        "edit_energy_ratio": float(np.nanmean(delta**2) / base_energy),
        "mean_delta_norm": float(np.nanmean(sample_rms)),
    }


def evaluate_editor(
    variant: str,
    pred: np.ndarray,
    batch: ForecastBatch,
    decisions: Optional[pd.DataFrame],
    oracle_mse: Optional[np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    selected = decisions["selected"].to_numpy(bool) if decisions is not None else None
    row = metric_row(variant, pred, batch, selected=selected, oracle_mse=oracle_mse, n_bootstrap=n_bootstrap, seed=seed)
    row.update(nontriviality_metrics(pred, batch))
    if decisions is not None:
        row["mean_accept_score"] = float(decisions["accept_score"].mean())
    else:
        row["mean_accept_score"] = float("nan")
    return row


def calibration_score(row: Mapping[str, Any]) -> float:
    score = float(row["mse_delta_pct_vs_lrbn"])
    score += 180.0 * max(0.0, float(row["harm_rate"]) - 0.10)
    score += 150.0 * max(0.0, float(row["max_config_harm"]) - 0.18)
    score += 40.0 * max(0.0, float(row["lrbn_equiv_rate"]) - 0.80)
    score += 25.0 * max(0.0, 0.08 - float(row["active_patch_ratio"]))
    score += 30.0 * max(0.0, 0.08 - float(row.get("oracle_gain_fraction", 0.0)))
    return float(score)


def apply_policy(
    variant: str,
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    policy: EditorPolicy,
    atom_scores: Optional[AtomScores],
    qd: Mapping[float, np.ndarray],
    mask_thresholds: Mapping[float, float],
    codebooks: Optional[Mapping[Tuple[int, int, int], PatchCodebook]] = None,
    retrieval_memory: Optional[Mapping[int, RetrievalMemory]] = None,
    schema: Optional[Dict[str, List[Any]]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    if variant == "H1 Residual Atom Simplex Editor":
        assert atom_scores is not None
        return apply_residual_atom_simplex(batch, atom_scores, policy)
    if variant == "H3 Any-Quantile Residual Envelope":
        return apply_any_quantile(batch, qd, policy)
    if variant == "H5 Local-Global Decoupled Sparse Editor":
        threshold = mask_thresholds.get(float(policy.mask_quantile))
        if threshold is None:
            threshold = float(np.nanquantile(temporal_boundary_score(batch).reshape(-1), policy.mask_quantile))
        return apply_local_global_sparse(batch, candidates, qd, threshold, policy)
    if variant == "H2 Prototype Codebook Local Editor":
        threshold = mask_thresholds.get(float(policy.mask_quantile))
        if threshold is None:
            threshold = float(np.nanquantile(temporal_boundary_score(batch).reshape(-1), policy.mask_quantile))
        return apply_prototype_codebook(batch, candidates, codebooks or {}, threshold, policy)
    if variant == "H4 Retrieval-Conditioned Residual Adapter":
        if schema is None:
            raise ValueError("schema is required for H4 Retrieval-Conditioned Residual Adapter")
        return apply_retrieval_adapter(batch, schema, retrieval_memory or {}, policy)
    raise ValueError(f"unknown variant {variant}")


def calibrate_variant(
    variant: str,
    calib: ForecastBatch,
    calib_candidates: Sequence[ExpertCandidate],
    atom_scores: Optional[AtomScores],
    qd: Mapping[float, np.ndarray],
    mask_thresholds: Mapping[float, float],
    codebooks: Optional[Mapping[Tuple[int, int, int], PatchCodebook]],
    retrieval_memory: Optional[Mapping[int, RetrievalMemory]],
    schema: Dict[str, List[Any]],
    oracle_mse: np.ndarray,
    seed: int,
) -> Tuple[EditorPolicy, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best_policy: Optional[EditorPolicy] = None
    best_score = float("inf")
    for policy in editor_grid(variant):
        pred, decisions = apply_policy(
            variant,
            calib,
            calib_candidates,
            policy,
            atom_scores,
            qd,
            mask_thresholds,
            codebooks=codebooks,
            retrieval_memory=retrieval_memory,
            schema=schema,
        )
        row = evaluate_editor(variant, pred, calib, decisions, oracle_mse, n_bootstrap=0, seed=seed)
        row.update(asdict(policy))
        row["calibration_score"] = calibration_score(row)
        rows.append(row)
        if float(row["calibration_score"]) < best_score:
            best_score = float(row["calibration_score"])
            best_policy = policy
    assert best_policy is not None
    return best_policy, pd.DataFrame(rows)


def known_harmed_delta(per_config: pd.DataFrame, variant: str) -> float:
    row = per_config[
        per_config["variant"].eq(variant)
        & per_config["dataset"].eq("ETTm1")
        & per_config["backbone"].eq("DLinear")
        & per_config["horizon"].eq(192)
    ]
    return float(row["mse_delta_pct_vs_lrbn"].iloc[0]) if not row.empty else float("nan")


def slice_value(slice_df: pd.DataFrame, variant: str, name: str) -> float:
    row = slice_df[slice_df["variant"].eq(variant) & slice_df["slice"].eq(name)]
    return float(row["mse_delta_pct_vs_lrbn"].iloc[0]) if not row.empty else float("nan")


def gate_table(overall: pd.DataFrame, per_config: pd.DataFrame, slice_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in overall[overall["variant"].isin(STAGE15_VARIANTS)].iterrows():
        variant = str(row["variant"])
        q4 = slice_value(slice_df, variant, "q4_boundary")
        known = known_harmed_delta(per_config, variant)
        base = {
            "variant": variant,
            "mse_delta_pct_vs_lrbn": float(row["mse_delta_pct_vs_lrbn"]),
            "harm_rate": float(row["harm_rate"]),
            "max_config_harm": float(row["max_config_harm"]),
            "oracle_gain_fraction": float(row.get("oracle_gain_fraction", np.nan)),
            "lrbn_equiv_rate": float(row["lrbn_equiv_rate"]),
            "active_patch_ratio": float(row["active_patch_ratio"]),
            "edit_energy_ratio": float(row["edit_energy_ratio"]),
            "q4_boundary_delta_pct": q4,
            "known_harmed_config_delta_pct": known,
            "bootstrap_high_delta_raw": float(row.get("ci95_high_delta_raw", np.nan)),
        }
        safe = (
            base["mse_delta_pct_vs_lrbn"] <= -1.8
            and base["harm_rate"] <= 0.02
            and base["max_config_harm"] <= 0.08
            and base["bootstrap_high_delta_raw"] < 0
            and base["lrbn_equiv_rate"] < 0.80
            and base["active_patch_ratio"] >= 0.08
            and base["q4_boundary_delta_pct"] <= 0.0
            and base["known_harmed_config_delta_pct"] <= 0.5
        )
        tradeoff = (
            base["mse_delta_pct_vs_lrbn"] <= -2.6
            and base["harm_rate"] <= 0.10
            and base["max_config_harm"] <= 0.18
            and base["bootstrap_high_delta_raw"] < 0
            and base["lrbn_equiv_rate"] < 0.70
            and base["active_patch_ratio"] >= 0.12
            and base["q4_boundary_delta_pct"] <= 0.0
            and base["known_harmed_config_delta_pct"] <= 0.5
        )
        mechanism = base["oracle_gain_fraction"] >= 0.08 and base["q4_boundary_delta_pct"] <= 0.0 and base["known_harmed_config_delta_pct"] <= 0.5
        base["safe_gate_pass"] = bool(safe)
        base["tradeoff_gate_pass"] = bool(tradeoff)
        base["mechanism_gate_pass"] = bool(mechanism)
        base["compact_gate_pass"] = bool(safe or tradeoff)
        rows.append(base)
    return pd.DataFrame(rows)


def add_reference_rows(overall: pd.DataFrame, stage7_dir: Path, stage14_dir: Path) -> pd.DataFrame:
    frames = [overall]
    stage7_overall = stage7_dir / "safe_tae_overall.csv"
    if stage7_overall.exists():
        s7 = pd.read_csv(stage7_overall)
        row = s7[s7["variant"].eq("SafeTAE-safe")].head(1).copy()
        if not row.empty:
            for col in overall.columns:
                if col not in row.columns:
                    row[col] = np.nan
            row = row[overall.columns]
            row["variant"] = "SafeTAE-safe (Stage7 table)"
            frames.append(row)
    stage14_overall = stage14_dir / "stage14_overall.csv"
    if stage14_overall.exists():
        s14 = pd.read_csv(stage14_overall)
        row = s14[s14["variant"].eq("FamilyMix Selector")].head(1).copy()
        if not row.empty:
            for col in overall.columns:
                if col not in row.columns:
                    row[col] = np.nan
            row = row[overall.columns]
            row["variant"] = "Stage14 FamilyMix Selector"
            frames.append(row)
    return pd.concat(frames, ignore_index=True)


def build_summary(output_dir: Path, verdict: Mapping[str, Any], overall: pd.DataFrame, gates: pd.DataFrame) -> str:
    cols = [
        "variant",
        "mse",
        "mae",
        "mse_delta_pct_vs_lrbn",
        "harm_rate",
        "max_config_harm",
        "coverage",
        "oracle_gain_fraction",
        "lrbn_equiv_rate",
        "active_patch_ratio",
        "edit_energy_ratio",
        "ci95_high_delta_raw",
    ]
    show = [c for c in cols if c in overall.columns]
    return "\n".join(
        [
            "# Stage 15 Endogenous Low-Harm Editors",
            "",
            f"Status: `{verdict['status']}`.",
            "",
            "## Overall",
            "",
            df_to_md(overall[show], max_rows=32),
            "",
            "## Gate Table",
            "",
            df_to_md(gates, max_rows=16),
            "",
            "## Verdict",
            "",
            "```json",
            json.dumps(verdict, ensure_ascii=False, indent=2, default=json_default),
            "```",
            "",
            f"Output directory: `{output_dir}`",
        ]
    )


def verify_outputs(output_dir: Path) -> pd.DataFrame:
    required = [
        "stage15_config.json",
        "stage15_candidate_metadata.csv",
        "stage15_calibration_grid.csv",
        "stage15_policies.json",
        "stage15_overall.csv",
        "stage15_per_config.csv",
        "stage15_slice_metrics.csv",
        "stage15_gate_table.csv",
        "stage15_verdict.json",
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
    stage7_dir: Path,
    stage14_dir: Path,
    output_dir: Path,
    stage3_dir: Optional[Path] = None,
    seed: int = 2026,
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[stage15-editors] preparing assets", flush=True)
    assets = prepare_assets(metrics_csv, stage5_dir, stage3_dir, seed)
    pools = build_cga_pools(assets)
    candidate_metadata(pools.test_candidates, "test").to_csv(output_dir / "stage15_candidate_metadata.csv", index=False)
    write_json(
        output_dir / "stage15_config.json",
        {
            "stage": "stage15_endogenous_editors",
            "source_plan": "deep-research-report (5).md",
            "compact_protocol": "ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192 x seed 2026",
            "metrics_csv": metrics_csv,
            "stage5_dir": stage5_dir,
            "stage7_dir": stage7_dir,
            "stage14_dir": stage14_dir,
            "stage3_dir": stage3_dir,
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "test_threshold_leakage": False,
        },
    )
    print("[stage15-editors] fitting atom utility/harm heads", flush=True)
    atom_models = fit_atom_models(assets.val_train, pools.train_candidates, assets.schema, seed)
    calib_atom_scores = prepare_atom_scores(assets.val_calib, pools.calib_candidates, assets.schema, atom_models)
    test_atom_scores = prepare_atom_scores(assets.test, pools.test_candidates, assets.schema, atom_models)
    print("[stage15-editors] fitting residual quantile envelopes", flush=True)
    calib_q = residual_quantiles(assets.val_train, assets.val_calib)
    test_q = residual_quantiles(assets.val_train, assets.test)
    print("[stage15-editors] fitting reserve patch codebooks and retrieval memory", flush=True)
    patch_codebooks = fit_patch_codebooks(assets.val_train, pools.train_candidates, seed=seed)
    retrieval_memory = fit_retrieval_memory(assets.val_train, assets.schema)
    mask_thresholds = {
        q: float(np.nanquantile(temporal_boundary_score(assets.val_calib).reshape(-1), q))
        for q in [0.80, 0.90, 0.95]
    }
    calib_oracle_mse = oracle_best(deployable_candidates(pools.calib_candidates), assets.val_calib)[1]
    oracle_pred, oracle_mse, _ = oracle_best(deployable_candidates(pools.test_candidates), assets.test)

    preds: Dict[str, np.ndarray] = {
        "LRBN": assets.test.lrbn_pred,
        "SRA-BP-safe": next((c.pred for c in assets.old_test_candidates if c.name == "sra_safe"), assets.test.lrbn_pred),
        "SRA-BP-balanced": next((c.pred for c in assets.old_test_candidates if c.name == "sra_balanced"), assets.test.lrbn_pred),
        "oracle_stage15_cga_full": oracle_pred,
    }
    decisions_by: Dict[str, pd.DataFrame] = {}
    policies: Dict[str, Any] = {}
    grid_frames: List[pd.DataFrame] = []

    print("[stage15-editors] calibrating first-round editors", flush=True)
    for variant in STAGE15_VARIANTS:
        print(f"[stage15-editors] calibrating {variant}", flush=True)
        policy, grid = calibrate_variant(
            variant,
            assets.val_calib,
            pools.calib_candidates,
            calib_atom_scores if variant == "H1 Residual Atom Simplex Editor" else None,
            calib_q,
            mask_thresholds,
            patch_codebooks,
            retrieval_memory,
            assets.schema,
            calib_oracle_mse,
            seed,
        )
        grid["target_variant"] = variant
        grid_frames.append(grid)
        policies[variant] = asdict(policy)
        pred, decisions = apply_policy(
            variant,
            assets.test,
            pools.test_candidates,
            policy,
            test_atom_scores if variant == "H1 Residual Atom Simplex Editor" else None,
            test_q,
            mask_thresholds,
            codebooks=patch_codebooks,
            retrieval_memory=retrieval_memory,
            schema=assets.schema,
        )
        preds[variant] = pred
        decisions_by[variant] = decisions

    print("[stage15-editors] evaluating", flush=True)
    overall_rows = []
    for variant, pred in preds.items():
        decisions = decisions_by.get(variant)
        overall_rows.append(evaluate_editor(variant, pred, assets.test, decisions, oracle_mse if variant != "LRBN" else None, n_bootstrap, seed))
    overall = pd.DataFrame(overall_rows)
    overall_with_refs = add_reference_rows(overall, stage7_dir, stage14_dir)
    overall_with_refs.to_csv(output_dir / "stage15_overall.csv", index=False)
    pd.concat(grid_frames, ignore_index=True).to_csv(output_dir / "stage15_calibration_grid.csv", index=False)
    write_json(output_dir / "stage15_policies.json", policies)

    per_config = pd.concat(
        [per_config_rows(variant, pred, assets.test, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    per_config.to_csv(output_dir / "stage15_per_config.csv", index=False)
    masks = stage8_slice_masks(assets.test, assets.schema, stage8_slice_thresholds(assets.val, assets.schema))
    slice_df = pd.concat(
        [slice_rows(variant, pred, assets.test, masks, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    slice_df.to_csv(output_dir / "stage15_slice_metrics.csv", index=False)
    gates = gate_table(overall, per_config, slice_df)
    gates.to_csv(output_dir / "stage15_gate_table.csv", index=False)
    deployable = overall[overall["variant"].isin(STAGE15_VARIANTS)].copy()
    best = deployable.sort_values(["mse_delta_pct_vs_lrbn", "max_config_harm", "harm_rate"], ascending=[True, True, True]).iloc[0]
    passed = gates[gates["compact_gate_pass"]]
    verdict = {
        "stage": "stage15_endogenous_editors",
        "status": "compact_pass_ready_for_mini_extension" if not passed.empty else "compact_failed_stop_before_mini_extension",
        "compact_pass": bool(not passed.empty),
        "passed_variants": passed["variant"].astype(str).tolist(),
        "best_variant": str(best["variant"]),
        "best_mse": float(best["mse"]),
        "best_mae": float(best["mae"]),
        "best_mse_delta_pct_vs_lrbn": float(best["mse_delta_pct_vs_lrbn"]),
        "best_harm_rate": float(best["harm_rate"]),
        "best_max_config_harm": float(best["max_config_harm"]),
        "best_oracle_gain_fraction": float(best.get("oracle_gain_fraction", np.nan)),
        "test_threshold_leakage": False,
        "stop_reason": None,
    }
    if passed.empty:
        verdict["stop_reason"] = "no endogenous editor passed compact safe/tradeoff gates"
    write_json(output_dir / "stage15_verdict.json", verdict)
    (output_dir / "summary.md").write_text(build_summary(output_dir, verdict, overall_with_refs, gates), encoding="utf-8")
    completeness = verify_outputs(output_dir)
    completeness.to_csv(output_dir / "stage15_output_completeness.csv", index=False)
    print(f"[stage15-editors] done in {time.time() - start:.1f}s", flush=True)
    return {
        "verdict": verdict,
        "overall": overall_with_refs,
        "per_config": per_config,
        "slice": slice_df,
        "gates": gates,
        "completeness": completeness,
    }
