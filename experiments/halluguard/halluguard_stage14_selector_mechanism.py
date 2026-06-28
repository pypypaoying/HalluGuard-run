#!/usr/bin/env python
"""Stage 14 selector-mechanism validation.

This stage keeps the CGA candidate families fixed and tests whether learned
family-level selectors can safely convert the oracle space into deployable
low-harm gains. All fitted selector heads use validation inner-train only.
All decision thresholds are chosen on validation inner-calib only.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors

from halluguard_lrbn_bp import ForecastBatch, mae_per_sample, mse_per_sample
from halluguard_stage6_mechanism import feature_frame, safe_pct
from halluguard_stage7_safe_tae import ConstantProbability, ExpertCandidate, align_frame, robust_nan_to_num, write_json
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
    Policy as Stage10Policy,
    apply_policy_precomputed as apply_stage10_policy_precomputed,
    build_cga_pools,
    candidate_feature_frame,
    candidate_metadata,
    candidate_sample_table,
    df_to_md,
    json_default,
    policy_grid as stage10_policy_grid,
)


STAGE14_VARIANTS = [
    "FamilyMix Selector",
    "Two-stage Cost-Sensitive Router",
    "ListSafe Top-k Selector",
    "Retrieval-Prior Selector",
    "Bayes-Abstain Selector",
]


@dataclass(frozen=True)
class SelectorPolicy:
    variant: str
    tau_leave: float = 0.50
    tau_select: float = 0.00
    tau_harm: float = 0.35
    beta_harm: float = 1.5
    lam: float = 0.35
    residual_cap: float = 0.35
    top_k_family: int = 1
    top_k_candidate: int = 2
    mix_temp: float = 0.35
    max_coverage: float = 1.0
    uncertainty_beta: float = 0.50
    retrieval_weight: float = 0.50


class ConstantRegressor:
    def __init__(self, value: float):
        self.value = float(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value, dtype=float)


@dataclass
class SelectorModels:
    leave_head: Any
    family_best_head: Any
    family_gain_head: Any
    family_harm_head: Any
    candidate_best_head: Any
    candidate_gain_head: Any
    candidate_harm_head: Any
    utility_regressor: Any
    sample_columns: List[str]
    family_columns: List[str]
    candidate_columns: List[str]
    families: List[str]
    candidate_names: List[str]
    retrieval_nn: Any
    retrieval_family_probs: np.ndarray
    retrieval_feature_columns: List[str]


@dataclass
class SelectorScores:
    leave_score: np.ndarray
    family_scores: pd.DataFrame
    candidate_scores: pd.DataFrame
    candidate_by: Dict[str, ExpertCandidate]
    retrieval_prior: np.ndarray


def _candidate_list(candidates: Sequence[ExpertCandidate]) -> List[ExpertCandidate]:
    return [c for c in candidates if c.name != "keep_lrbn" and c.deployable]


def _family_list(candidates: Sequence[ExpertCandidate]) -> List[str]:
    return sorted({c.family for c in _candidate_list(candidates)})


def _safe_feature_columns(df: pd.DataFrame) -> List[str]:
    blocked_tokens = [
        "label",
        "target",
        "mse",
        "mae",
        "delta",
        "oracle",
        "best",
        "true",
        "gain_amount",
        "harm_amount",
    ]
    blocked_exact = {
        "candidate",
        "family",
        "tier",
        "sample_id",
        "sample_key",
        "split",
        "row_index",
    }
    cols: List[str] = []
    for col in df.columns:
        low = col.lower()
        if col in blocked_exact or any(tok in low for tok in blocked_tokens):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _fit_binary(x: pd.DataFrame, y: np.ndarray, seed: int) -> Any:
    labels = np.asarray(y, dtype=int)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return ConstantProbability(float(labels.mean()) if len(labels) else 0.0)
    model = RandomForestClassifier(
        n_estimators=180,
        max_depth=7,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x.to_numpy(float), labels)
    return model


def _predict_binary(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(x.to_numpy(float))[:, 1], dtype=float)


def _fit_regressor(x: pd.DataFrame, y: np.ndarray, seed: int) -> Any:
    target = np.asarray(y, dtype=float)
    if len(target) == 0 or float(np.nanstd(target)) < 1e-12:
        return ConstantRegressor(float(np.nanmean(target)) if len(target) else 0.0)
    model = RandomForestRegressor(
        n_estimators=180,
        max_depth=8,
        min_samples_leaf=8,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(x.to_numpy(float), np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0))
    return model


def _predict_regressor(model: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(x.to_numpy(float)), dtype=float)


def _predict_regressor_std(model: Any, x: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "estimators_"):
        return np.zeros(len(x), dtype=float)
    xx = x.to_numpy(float)
    preds = np.stack([tree.predict(xx) for tree in model.estimators_], axis=0)
    return np.asarray(np.nanstd(preds, axis=0), dtype=float)


def _scale(batch: ForecastBatch) -> np.ndarray:
    s = np.nanstd(batch.context, axis=(1, 2))
    s = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return s.reshape(-1, 1, 1)


def _clip_delta(delta: np.ndarray, batch: ForecastBatch, cap: float) -> np.ndarray:
    bound = max(float(cap), 1e-6) * _scale(batch)
    return np.clip(delta, -bound, bound)


def candidate_delta_matrix(batch: ForecastBatch, candidates: Sequence[ExpertCandidate]) -> Tuple[List[ExpertCandidate], np.ndarray]:
    cands = _candidate_list(candidates)
    base = mse_per_sample(batch.lrbn_pred, batch.y_true)
    losses = np.stack([mse_per_sample(c.pred, batch.y_true) for c in cands], axis=1) if cands else np.zeros((len(batch.meta), 0))
    return cands, losses - base[:, None]


def family_delta_matrix(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    families: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, List[int]], List[ExpertCandidate]]:
    cands, delta = candidate_delta_matrix(batch, candidates)
    fam_to_idx: Dict[str, List[int]] = {fam: [] for fam in families}
    for j, c in enumerate(cands):
        fam_to_idx.setdefault(c.family, []).append(j)
    out = np.full((len(batch.meta), len(families)), np.inf, dtype=float)
    for f_idx, fam in enumerate(families):
        idx = fam_to_idx.get(fam, [])
        if idx:
            out[:, f_idx] = np.nanmin(delta[:, idx], axis=1)
    return out, fam_to_idx, cands


def family_feature_frame_target_free(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    families: Sequence[str],
) -> pd.DataFrame:
    base = feature_frame(batch, schema).reset_index(drop=True)
    rows: List[pd.DataFrame] = []
    base_norm = np.linalg.norm(batch.lrbn_pred.reshape(len(batch.meta), -1), axis=1) + 1e-8
    for fam in families:
        fam_cands = [c for c in _candidate_list(candidates) if c.family == fam]
        if fam_cands:
            deltas = np.stack([(c.pred - batch.lrbn_pred).reshape(len(batch.meta), -1) for c in fam_cands], axis=1)
            norms = np.linalg.norm(deltas, axis=2)
            pred_stack = np.stack([c.pred for c in fam_cands], axis=0)
            residual = pred_stack - batch.lrbn_pred[None, ...]
            early = residual[:, :, : max(1, residual.shape[2] // 4), :]
            late = residual[:, :, 3 * residual.shape[2] // 4 :, :]
            stats = pd.DataFrame(
                {
                    "family": fam,
                    "family_candidate_count": float(len(fam_cands)),
                    "family_corr_norm_mean": np.nanmean(norms, axis=1),
                    "family_corr_norm_min": np.nanmin(norms, axis=1),
                    "family_corr_norm_max": np.nanmax(norms, axis=1),
                    "family_corr_norm_std": np.nanstd(norms, axis=1),
                    "family_corr_norm_ratio": np.nanmean(norms, axis=1) / base_norm,
                    "family_corr_mean": np.nanmean(residual, axis=(0, 2, 3)),
                    "family_corr_std": np.nanstd(residual, axis=(0, 2, 3)),
                    "family_early_energy": np.nanmean(early**2, axis=(0, 2, 3)),
                    "family_late_energy": np.nanmean(late**2, axis=(0, 2, 3)),
                }
            )
        else:
            stats = pd.DataFrame(
                {
                    "family": fam,
                    "family_candidate_count": 0.0,
                    "family_corr_norm_mean": np.zeros(len(batch.meta)),
                    "family_corr_norm_min": np.zeros(len(batch.meta)),
                    "family_corr_norm_max": np.zeros(len(batch.meta)),
                    "family_corr_norm_std": np.zeros(len(batch.meta)),
                    "family_corr_norm_ratio": np.zeros(len(batch.meta)),
                    "family_corr_mean": np.zeros(len(batch.meta)),
                    "family_corr_std": np.zeros(len(batch.meta)),
                    "family_early_energy": np.zeros(len(batch.meta)),
                    "family_late_energy": np.zeros(len(batch.meta)),
                }
            )
        for f in families:
            stats[f"family={f}"] = float(fam == f)
        rows.append(pd.concat([base, stats], axis=1))
    df = pd.concat(rows, ignore_index=True)
    df["row_index"] = np.tile(np.arange(len(batch.meta)), len(families))
    return robust_nan_to_num(df)


def add_family_labels(df: pd.DataFrame, batch: ForecastBatch, candidates: Sequence[ExpertCandidate], families: Sequence[str]) -> pd.DataFrame:
    fam_delta, _, _ = family_delta_matrix(batch, candidates, families)
    safe = fam_delta <= 0.0
    target = np.full(len(batch.meta), -1, dtype=int)
    for i in range(len(batch.meta)):
        if safe[i].any():
            masked = np.where(safe[i], fam_delta[i], np.inf)
            target[i] = int(np.nanargmin(masked))
    out = df.copy()
    flat_delta = np.concatenate([fam_delta[:, j] for j in range(len(families))])
    out["family_best_delta"] = flat_delta
    out["family_gain_label"] = flat_delta < -1e-4
    out["family_harm_label"] = flat_delta > 1e-4
    labels = []
    for j in range(len(families)):
        labels.extend((target == j).astype(int).tolist())
    out["family_target_label"] = labels
    return out


def add_candidate_labels(df: pd.DataFrame, batch: ForecastBatch, candidates: Sequence[ExpertCandidate]) -> pd.DataFrame:
    cands, delta = candidate_delta_matrix(batch, candidates)
    target = np.full(len(batch.meta), -1, dtype=int)
    safe = delta <= 0.0
    for i in range(len(batch.meta)):
        if safe[i].any():
            masked = np.where(safe[i], delta[i], np.inf)
            target[i] = int(np.nanargmin(masked))
    out = df.copy()
    flat_delta = np.concatenate([delta[:, j] for j in range(len(cands))]) if cands else np.asarray([], dtype=float)
    out["candidate_delta"] = flat_delta
    out["candidate_gain_label"] = flat_delta < -1e-4
    out["candidate_harm_label"] = flat_delta > 1e-4
    labels = []
    for j in range(len(cands)):
        labels.extend((target == j).astype(int).tolist())
    out["candidate_target_label"] = labels
    return out


def train_oracle_family_probs(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    families: Sequence[str],
    feature_columns: Sequence[str],
) -> Tuple[Any, np.ndarray, List[str]]:
    x = align_frame(feature_frame(batch, schema), feature_columns)
    fam_delta, _, _ = family_delta_matrix(batch, candidates, families)
    probs = np.zeros((len(batch.meta), len(families)), dtype=float)
    safe = fam_delta <= 0.0
    for i in range(len(batch.meta)):
        if safe[i].any():
            masked = np.where(safe[i], fam_delta[i], np.inf)
            probs[i, int(np.nanargmin(masked))] = 1.0
    nn = NearestNeighbors(n_neighbors=min(15, max(1, len(x))), metric="euclidean")
    nn.fit(x.to_numpy(float))
    return nn, probs, list(feature_columns)


def fit_selector_models(
    train: ForecastBatch,
    train_candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    seed: int,
) -> Tuple[SelectorModels, pd.DataFrame]:
    families = _family_list(train_candidates)
    cands = _candidate_list(train_candidates)
    sample_x = robust_nan_to_num(feature_frame(train, schema))
    sample_cols = _safe_feature_columns(sample_x)
    _, cand_delta = candidate_delta_matrix(train, train_candidates)
    leave_label = (np.nanmin(cand_delta, axis=1) < -1e-4).astype(int) if cand_delta.size else np.zeros(len(train.meta), dtype=int)

    family_x = add_family_labels(family_feature_frame_target_free(train, train_candidates, schema, families), train, train_candidates, families)
    family_cols = _safe_feature_columns(family_x)
    cand_x = add_candidate_labels(candidate_feature_frame(train, train_candidates, schema, families), train, train_candidates)
    cand_cols = _safe_feature_columns(cand_x)

    leave_head = _fit_binary(sample_x[sample_cols], leave_label, seed)
    family_best = _fit_binary(family_x[family_cols], family_x["family_target_label"].to_numpy(int), seed + 11)
    family_gain = _fit_binary(family_x[family_cols], family_x["family_gain_label"].to_numpy(int), seed + 12)
    family_harm = _fit_binary(family_x[family_cols], family_x["family_harm_label"].to_numpy(int), seed + 13)
    cand_best = _fit_binary(cand_x[cand_cols], cand_x["candidate_target_label"].to_numpy(int), seed + 21)
    cand_gain = _fit_binary(cand_x[cand_cols], cand_x["candidate_gain_label"].to_numpy(int), seed + 22)
    cand_harm = _fit_binary(cand_x[cand_cols], cand_x["candidate_harm_label"].to_numpy(int), seed + 23)
    utility = _fit_regressor(cand_x[cand_cols], -cand_x["candidate_delta"].to_numpy(float), seed + 31)
    nn, fam_probs, retrieval_cols = train_oracle_family_probs(train, train_candidates, schema, families, sample_cols)

    metric_rows: List[Dict[str, Any]] = []
    for name, labels, pred, level in [
        ("leave_lrbn", leave_label, _predict_binary(leave_head, sample_x[sample_cols]), "sample"),
        ("family_best", family_x["family_target_label"].to_numpy(int), _predict_binary(family_best, family_x[family_cols]), "family"),
        ("family_gain", family_x["family_gain_label"].to_numpy(int), _predict_binary(family_gain, family_x[family_cols]), "family"),
        ("family_harm", family_x["family_harm_label"].to_numpy(int), _predict_binary(family_harm, family_x[family_cols]), "family"),
        ("candidate_best", cand_x["candidate_target_label"].to_numpy(int), _predict_binary(cand_best, cand_x[cand_cols]), "candidate"),
        ("candidate_gain", cand_x["candidate_gain_label"].to_numpy(int), _predict_binary(cand_gain, cand_x[cand_cols]), "candidate"),
        ("candidate_harm", cand_x["candidate_harm_label"].to_numpy(int), _predict_binary(cand_harm, cand_x[cand_cols]), "candidate"),
    ]:
        row: Dict[str, Any] = {"selector": name, "split": "inner_train", "level": level, "n": int(len(labels)), "positive_rate": float(np.mean(labels))}
        if len(np.unique(labels)) >= 2:
            row["roc_auc"] = float(roc_auc_score(labels, pred))
            row["pr_auc"] = float(average_precision_score(labels, pred))
        else:
            row["roc_auc"] = float("nan")
            row["pr_auc"] = float("nan")
        metric_rows.append(row)

    models = SelectorModels(
        leave_head=leave_head,
        family_best_head=family_best,
        family_gain_head=family_gain,
        family_harm_head=family_harm,
        candidate_best_head=cand_best,
        candidate_gain_head=cand_gain,
        candidate_harm_head=cand_harm,
        utility_regressor=utility,
        sample_columns=sample_cols,
        family_columns=family_cols,
        candidate_columns=cand_cols,
        families=families,
        candidate_names=[c.name for c in cands],
        retrieval_nn=nn,
        retrieval_family_probs=fam_probs,
        retrieval_feature_columns=retrieval_cols,
    )
    return models, pd.DataFrame(metric_rows)


def prepare_selector_scores(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    models: SelectorModels,
) -> SelectorScores:
    sample_x = align_frame(feature_frame(batch, schema), models.sample_columns)
    leave_score = _predict_binary(models.leave_head, sample_x)
    fam = family_feature_frame_target_free(batch, candidates, schema, models.families)
    fam_x = align_frame(fam[models.family_columns], models.family_columns)
    fam["p_family_best"] = _predict_binary(models.family_best_head, fam_x)
    fam["p_family_gain"] = _predict_binary(models.family_gain_head, fam_x)
    fam["p_family_harm"] = _predict_binary(models.family_harm_head, fam_x)
    cand = candidate_feature_frame(batch, candidates, schema, models.families)
    cand_x = align_frame(cand[models.candidate_columns], models.candidate_columns)
    cand["p_candidate_best"] = _predict_binary(models.candidate_best_head, cand_x)
    cand["p_candidate_gain"] = _predict_binary(models.candidate_gain_head, cand_x)
    cand["p_candidate_harm"] = _predict_binary(models.candidate_harm_head, cand_x)
    cand["pred_utility"] = _predict_regressor(models.utility_regressor, cand_x)
    cand["utility_std"] = _predict_regressor_std(models.utility_regressor, cand_x)
    cand["row_index"] = np.tile(np.arange(len(batch.meta)), len(models.candidate_names))
    sample_r = align_frame(feature_frame(batch, schema), models.retrieval_feature_columns)
    neigh = models.retrieval_nn.kneighbors(sample_r.to_numpy(float), return_distance=False)
    prior = models.retrieval_family_probs[neigh].mean(axis=1)
    return SelectorScores(
        leave_score=leave_score,
        family_scores=fam,
        candidate_scores=cand,
        candidate_by={c.name: c for c in _candidate_list(candidates)},
        retrieval_prior=prior,
    )


def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    if len(x) == 0:
        return x
    scale = max(float(temp), 1e-6)
    z = x / scale
    z = z - np.nanmax(z)
    e = np.exp(z)
    return e / (np.nansum(e) + 1e-12)


def _family_score(row: pd.Series, policy: SelectorPolicy, prior: float = 0.0) -> float:
    return float(
        0.55 * row.get("p_family_best", 0.0)
        + 0.35 * row.get("p_family_gain", 0.0)
        - policy.beta_harm * row.get("p_family_harm", 1.0)
        + policy.retrieval_weight * prior
    )


def _candidate_score(row: pd.Series, policy: SelectorPolicy, bayes: bool = False) -> float:
    score = (
        0.45 * row.get("p_candidate_best", 0.0)
        + 0.25 * row.get("p_candidate_gain", 0.0)
        + 0.30 * row.get("pred_utility", 0.0)
        - policy.beta_harm * row.get("p_candidate_harm", 1.0)
    )
    if bayes:
        score -= policy.uncertainty_beta * row.get("utility_std", 0.0)
    return float(score)


def _apply_coverage_cap(pred: np.ndarray, batch: ForecastBatch, decisions: pd.DataFrame, policy: SelectorPolicy) -> Tuple[np.ndarray, pd.DataFrame]:
    if policy.max_coverage >= 0.999:
        return pred, decisions
    selected = decisions["selected"].to_numpy(bool)
    cap_n = int(np.floor(float(policy.max_coverage) * len(selected)))
    cap_n = max(0, min(cap_n, int(selected.sum())))
    if int(selected.sum()) <= cap_n:
        return pred, decisions
    scores = decisions["accept_score"].to_numpy(float)
    selected_idx = np.where(selected)[0]
    keep_idx = selected_idx[np.argsort(scores[selected_idx])[::-1][:cap_n]]
    keep = np.zeros(len(selected), dtype=bool)
    keep[keep_idx] = True
    drop = selected & ~keep
    pred = pred.copy()
    pred[drop] = batch.lrbn_pred[drop]
    out = decisions.copy()
    out.loc[drop, "selected"] = False
    out.loc[drop, "selected_action"] = "coverage_cap_fallback"
    return pred, out


def apply_selector_variant(batch: ForecastBatch, scores: SelectorScores, policy: SelectorPolicy) -> Tuple[np.ndarray, pd.DataFrame]:
    pred = batch.lrbn_pred.copy()
    decisions: List[Dict[str, Any]] = []
    cand_rows_by = {int(k): v for k, v in scores.candidate_scores.groupby("row_index", observed=True)}
    fam_rows_by = {int(k): v for k, v in scores.family_scores.groupby("row_index", observed=True)}
    fam_to_pos = {f: j for j, f in enumerate(policy_families(scores))}

    for i in range(len(batch.meta)):
        selected = False
        action = "keep_lrbn"
        accept_score = 0.0
        expected_harm = 0.0
        lam = 0.0
        if scores.leave_score[i] >= policy.tau_leave:
            fam_rows = fam_rows_by.get(i, pd.DataFrame()).copy()
            if not fam_rows.empty:
                fam_rows["prior"] = fam_rows["family"].map(lambda f: scores.retrieval_prior[i, fam_to_pos.get(str(f), 0)] if scores.retrieval_prior.size else 0.0)
                if policy.variant == "Retrieval-Prior Selector":
                    fam_rows["family_score"] = fam_rows.apply(lambda r: _family_score(r, policy, float(r["prior"])), axis=1)
                else:
                    fam_rows["family_score"] = fam_rows.apply(lambda r: _family_score(r, policy, 0.0), axis=1)
                fam_rows = fam_rows[(fam_rows["p_family_harm"] <= policy.tau_harm) & (fam_rows["family_score"] >= policy.tau_select)]
                fam_rows = fam_rows.sort_values("family_score", ascending=False).head(policy.top_k_family)
                cand_rows = cand_rows_by.get(i, pd.DataFrame()).copy()
                if not fam_rows.empty and not cand_rows.empty:
                    fams = fam_rows["family"].astype(str).tolist()
                    cand_rows = cand_rows[cand_rows["family"].isin(fams)].copy()
                    if policy.variant == "Bayes-Abstain Selector":
                        cand_rows["selector_score"] = cand_rows.apply(lambda r: _candidate_score(r, policy, bayes=True), axis=1)
                    elif policy.variant == "ListSafe Top-k Selector":
                        cand_rows["selector_score"] = cand_rows.apply(lambda r: _candidate_score(r, policy, bayes=False), axis=1)
                        cand_rows["selector_score"] += 0.15 * cand_rows["family"].map(fam_rows.set_index("family")["family_score"].to_dict())
                    elif policy.variant == "Two-stage Cost-Sensitive Router":
                        best_fam = str(fam_rows.iloc[0]["family"])
                        cand_rows = cand_rows[cand_rows["family"].eq(best_fam)].copy()
                        cand_rows["selector_score"] = cand_rows.apply(lambda r: _candidate_score(r, policy, bayes=False), axis=1)
                    else:
                        cand_rows["selector_score"] = cand_rows.apply(lambda r: _candidate_score(r, policy, bayes=False), axis=1)
                    cand_rows = cand_rows[
                        (cand_rows["p_candidate_harm"] <= policy.tau_harm)
                        & (cand_rows["selector_score"] >= policy.tau_select)
                    ].sort_values("selector_score", ascending=False)
                    if policy.variant == "FamilyMix Selector":
                        cand_rows = cand_rows.groupby("family", observed=True).head(policy.top_k_candidate)
                    else:
                        cand_rows = cand_rows.head(policy.top_k_candidate)
                    if not cand_rows.empty:
                        values = cand_rows["selector_score"].to_numpy(float)
                        weights = _softmax(values, policy.mix_temp)
                        delta = np.zeros_like(batch.lrbn_pred[i])
                        used: List[str] = []
                        for w, (_, row) in zip(weights, cand_rows.iterrows()):
                            cname = str(row["candidate"])
                            cand = scores.candidate_by.get(cname)
                            if cand is None:
                                continue
                            delta += float(w) * (cand.pred[i] - batch.lrbn_pred[i])
                            used.append(cname)
                        clipped = _clip_delta(delta[None, ...], batch.subset(np.arange(len(batch.meta)) == i), policy.residual_cap)[0]
                        pred[i] = batch.lrbn_pred[i] + policy.lam * clipped
                        selected = bool(used)
                        action = ",".join(used) if used else "keep_lrbn"
                        accept_score = float(np.nanmean(values))
                        expected_harm = float(np.nanmean(cand_rows["p_candidate_harm"].to_numpy(float)))
                        lam = float(policy.lam)
        decisions.append(
            {
                "row_index": i,
                "selected": selected,
                "selected_action": action if selected else "keep_lrbn",
                "accept_score": accept_score,
                "expected_harm": expected_harm,
                "lambda": lam,
                "leave_score": float(scores.leave_score[i]),
            }
        )
    dec = pd.DataFrame(decisions)
    return _apply_coverage_cap(pred, batch, dec, policy)


def policy_families(scores: SelectorScores) -> List[str]:
    return sorted(scores.family_scores["family"].astype(str).unique().tolist())


def selector_policy_grid(variant: str) -> Iterable[SelectorPolicy]:
    if variant == "FamilyMix Selector":
        for tau_harm in [0.15, 0.25, 0.35]:
            for tau in [-0.10, 0.00, 0.10]:
                for lam in [0.20, 0.35, 0.50]:
                    for cap in [0.25, 0.50, 1.00]:
                        yield SelectorPolicy(variant, tau_leave=0.45, tau_select=tau, tau_harm=tau_harm, beta_harm=1.5, lam=lam, top_k_family=2, top_k_candidate=2, max_coverage=cap)
    elif variant == "Two-stage Cost-Sensitive Router":
        for tau_harm in [0.10, 0.20, 0.30]:
            for tau in [-0.05, 0.05, 0.15]:
                for lam in [0.25, 0.40, 0.60]:
                    for beta in [1.5, 2.5]:
                        yield SelectorPolicy(variant, tau_leave=0.50, tau_select=tau, tau_harm=tau_harm, beta_harm=beta, lam=lam, top_k_family=1, top_k_candidate=1)
    elif variant == "ListSafe Top-k Selector":
        for tau_harm in [0.15, 0.25, 0.35]:
            for tau in [-0.10, 0.00, 0.10]:
                for lam in [0.20, 0.35, 0.50]:
                    for cap in [0.25, 0.50, 1.00]:
                        yield SelectorPolicy(variant, tau_leave=0.45, tau_select=tau, tau_harm=tau_harm, beta_harm=1.8, lam=lam, top_k_family=2, top_k_candidate=3, max_coverage=cap)
    elif variant == "Retrieval-Prior Selector":
        for w in [0.25, 0.50, 0.85]:
            for tau_harm in [0.15, 0.25, 0.35]:
                for tau in [-0.05, 0.05]:
                    for lam in [0.25, 0.40]:
                        yield SelectorPolicy(variant, tau_leave=0.45, tau_select=tau, tau_harm=tau_harm, beta_harm=1.5, lam=lam, top_k_family=2, top_k_candidate=2, retrieval_weight=w, max_coverage=0.50)
    elif variant == "Bayes-Abstain Selector":
        for ub in [0.25, 0.50, 1.00]:
            for tau_harm in [0.10, 0.20, 0.30]:
                for tau in [0.00, 0.10, 0.20]:
                    for lam in [0.20, 0.35, 0.50]:
                        yield SelectorPolicy(variant, tau_leave=0.50, tau_select=tau, tau_harm=tau_harm, beta_harm=1.5, lam=lam, top_k_family=2, top_k_candidate=2, uncertainty_beta=ub, max_coverage=0.50)
    else:
        raise ValueError(f"unknown selector variant {variant}")


def evaluate_selector_variant(
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
    if decisions is not None:
        base = mse_per_sample(batch.lrbn_pred, batch.y_true)
        method = mse_per_sample(pred, batch.y_true)
        sel = decisions["selected"].to_numpy(bool)
        row["accept_precision"] = float(np.mean(method[sel] < base[sel])) if sel.any() else 0.0
        row["selected_nonharm_rate"] = float(np.mean(method[sel] <= base[sel] + 1e-12)) if sel.any() else 0.0
        row["mean_expected_harm"] = float(decisions["expected_harm"].mean())
        row["mean_accept_score"] = float(decisions["accept_score"].mean())
    else:
        row["accept_precision"] = float("nan")
        row["selected_nonharm_rate"] = float("nan")
        row["mean_expected_harm"] = float("nan")
        row["mean_accept_score"] = float("nan")
    return row


def calibration_score(row: Mapping[str, Any]) -> float:
    selected_nonharm = float(row.get("selected_nonharm_rate", 0.0))
    score = float(row["mse_delta_pct_vs_lrbn"])
    score += 180.0 * max(0.0, float(row["harm_rate"]) - 0.10)
    score += 150.0 * max(0.0, float(row["max_config_harm"]) - 0.18)
    score += 90.0 * max(0.0, 0.85 - selected_nonharm)
    score += 20.0 * max(0.0, 0.04 - float(row.get("coverage", 0.0)))
    score += 20.0 * max(0.0, 0.08 - float(row.get("oracle_gain_fraction", 0.0)))
    return float(score)


def calibrate_selector_variant(
    variant: str,
    calib: ForecastBatch,
    scores: SelectorScores,
    oracle_mse: np.ndarray,
    seed: int,
) -> Tuple[SelectorPolicy, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best_policy: Optional[SelectorPolicy] = None
    best_score = float("inf")
    for policy in selector_policy_grid(variant):
        pred, decisions = apply_selector_variant(calib, scores, policy)
        row = evaluate_selector_variant(variant, pred, calib, decisions, oracle_mse, n_bootstrap=0, seed=seed)
        row.update(asdict(policy))
        row["calibration_score"] = calibration_score(row)
        row["calibration_feasible_safe"] = bool(
            row["harm_rate"] <= 0.03 and row["max_config_harm"] <= 0.10 and row["selected_nonharm_rate"] >= 0.90
        )
        row["calibration_feasible_balanced"] = bool(
            row["harm_rate"] <= 0.10 and row["max_config_harm"] <= 0.18 and row["selected_nonharm_rate"] >= 0.85
        )
        rows.append(row)
        rank = float(row["calibration_score"])
        if row["calibration_feasible_balanced"]:
            rank -= 100.0
        if row["calibration_feasible_safe"]:
            rank -= 50.0
        if rank < best_score:
            best_score = rank
            best_policy = policy
    assert best_policy is not None
    return best_policy, pd.DataFrame(rows)


def calibrate_stage10_hard(
    calib: ForecastBatch,
    scores: SelectorScores,
    oracle_mse: np.ndarray,
    seed: int,
) -> Tuple[Stage10Policy, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best: Optional[Stage10Policy] = None
    best_score = float("inf")
    converted = stage10_like_scores(scores)
    for policy in stage10_policy_grid("Balanced-CGA"):
        pred, decisions = apply_stage10_policy_precomputed(calib, policy, converted)
        dec_eval = decisions.rename(columns={"selected_candidate": "selected_action", "utility_score": "accept_score"}).copy()
        dec_eval["expected_harm"] = 0.0
        row = evaluate_selector_variant("Stage10 hard selector", pred, calib, dec_eval, oracle_mse, 0, seed)
        row.update(asdict(policy))
        row["calibration_score"] = calibration_score(row)
        rows.append(row)
        if float(row["calibration_score"]) < best_score:
            best_score = float(row["calibration_score"])
            best = policy
    assert best is not None
    return best, pd.DataFrame(rows)


@dataclass
class Stage10LikeScoreBundle:
    leave_score: np.ndarray
    cand_scores: pd.DataFrame
    fam_scores: pd.DataFrame
    cand_by: Dict[str, ExpertCandidate]


def stage10_like_scores(scores: SelectorScores) -> Stage10LikeScoreBundle:
    fam = scores.family_scores.copy()
    cand = scores.candidate_scores.copy()
    fam["p_family_gain"] = 0.5 * fam["p_family_gain"] + 0.5 * fam["p_family_best"]
    cand["p_candidate_gain"] = 0.5 * cand["p_candidate_gain"] + 0.5 * cand["p_candidate_best"]
    return Stage10LikeScoreBundle(scores.leave_score, cand, fam, scores.candidate_by)


def rank_metrics_for_variant(
    variant: str,
    scores: SelectorScores,
    candidates: Sequence[ExpertCandidate],
    batch: ForecastBatch,
    policy: SelectorPolicy,
    k: int = 2,
) -> Dict[str, Any]:
    cands, cand_delta = candidate_delta_matrix(batch, candidates)
    fams = policy_families(scores)
    fam_delta, _, _ = family_delta_matrix(batch, candidates, fams)
    oracle_cand_idx = np.nanargmin(cand_delta, axis=1)
    oracle_cand = np.asarray([cands[j].name for j in oracle_cand_idx], dtype=object)
    oracle_fam_idx = np.nanargmin(fam_delta, axis=1)
    oracle_fam = np.asarray([fams[j] for j in oracle_fam_idx], dtype=object)
    cand_hit: List[bool] = []
    fam_hit: List[bool] = []
    fam_top1: List[bool] = []
    for i in range(len(batch.meta)):
        fam_i = scores.family_scores[scores.family_scores["row_index"].eq(i)].copy()
        fam_to_pos = {f: j for j, f in enumerate(fams)}
        fam_i["prior"] = fam_i["family"].map(lambda f: scores.retrieval_prior[i, fam_to_pos.get(str(f), 0)] if scores.retrieval_prior.size else 0.0)
        fam_i["rank_score"] = fam_i.apply(
            lambda r: _family_score(r, policy, float(r["prior"]) if variant == "Retrieval-Prior Selector" else 0.0),
            axis=1,
        )
        top_f = fam_i.sort_values("rank_score", ascending=False)["family"].astype(str).head(k).tolist()
        cand_i = scores.candidate_scores[scores.candidate_scores["row_index"].eq(i)].copy()
        cand_i["rank_score"] = cand_i.apply(lambda r: _candidate_score(r, policy, bayes=variant == "Bayes-Abstain Selector"), axis=1)
        top_c = cand_i.sort_values("rank_score", ascending=False)["candidate"].astype(str).head(k).tolist()
        fam_hit.append(str(oracle_fam[i]) in top_f)
        fam_top1.append(str(oracle_fam[i]) == (top_f[0] if top_f else ""))
        cand_hit.append(str(oracle_cand[i]) in top_c)
    return {
        "variant": variant,
        "family_top2_hit": float(np.mean(fam_hit)),
        "family_top1_hit": float(np.mean(fam_top1)),
        "candidate_top2_hit": float(np.mean(cand_hit)),
    }


def selection_distribution(decisions: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for variant, df in decisions.items():
        vc = df["selected_action"].value_counts(normalize=True).rename_axis("selected_action").reset_index(name="share")
        vc.insert(0, "variant", variant)
        frames.append(vc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def gate_table(overall: pd.DataFrame, topk: pd.DataFrame) -> pd.DataFrame:
    topk_by = topk.set_index("variant").to_dict("index") if not topk.empty else {}
    rows: List[Dict[str, Any]] = []
    for _, row in overall[overall["variant"].isin(STAGE14_VARIANTS)].iterrows():
        variant = str(row["variant"])
        tk = topk_by.get(variant, {})
        base = {
            "variant": variant,
            "mse_delta_pct_vs_lrbn": float(row["mse_delta_pct_vs_lrbn"]),
            "harm_rate": float(row["harm_rate"]),
            "max_config_harm": float(row["max_config_harm"]),
            "selected_nonharm_rate": float(row.get("selected_nonharm_rate", np.nan)),
            "family_top2_hit": float(tk.get("family_top2_hit", np.nan)),
            "candidate_top2_hit": float(tk.get("candidate_top2_hit", np.nan)),
            "oracle_gain_fraction": float(row.get("oracle_gain_fraction", np.nan)),
            "coverage": float(row.get("coverage", np.nan)),
            "bootstrap_high_delta_raw": float(row.get("ci95_high_delta_raw", np.nan)),
        }
        safe_pass = (
            base["mse_delta_pct_vs_lrbn"] <= -1.8
            and base["harm_rate"] <= 0.03
            and base["max_config_harm"] <= 0.10
            and base["selected_nonharm_rate"] >= 0.90
            and base["family_top2_hit"] >= 0.70
            and base["oracle_gain_fraction"] >= 0.08
            and base["bootstrap_high_delta_raw"] < 0
        )
        balanced_pass = (
            base["mse_delta_pct_vs_lrbn"] <= -2.7
            and base["harm_rate"] <= 0.10
            and base["max_config_harm"] <= 0.18
            and base["selected_nonharm_rate"] >= 0.85
            and base["family_top2_hit"] >= 0.75
            and base["candidate_top2_hit"] >= 0.20
            and base["oracle_gain_fraction"] >= 0.12
            and base["bootstrap_high_delta_raw"] < 0
        )
        base["safe_gate_pass"] = bool(safe_pass)
        base["balanced_gate_pass"] = bool(balanced_pass)
        base["compact_gate_pass"] = bool(safe_pass or balanced_pass)
        rows.append(base)
    return pd.DataFrame(rows)


def verify_outputs(output_dir: Path) -> pd.DataFrame:
    required = [
        "stage14_config.json",
        "stage14_candidate_metadata.csv",
        "stage14_selector_train_metrics.csv",
        "stage14_calibration_grid.csv",
        "stage14_policies.json",
        "stage14_overall.csv",
        "stage14_per_config.csv",
        "stage14_slice_metrics.csv",
        "stage14_selection_distribution.csv",
        "stage14_topk_metrics.csv",
        "stage14_gate_table.csv",
        "stage14_verdict.json",
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


def build_summary(output_dir: Path, verdict: Mapping[str, Any], overall: pd.DataFrame, gates: pd.DataFrame, topk: pd.DataFrame) -> str:
    cols = [
        "variant",
        "mse",
        "mae",
        "mse_delta_pct_vs_lrbn",
        "harm_rate",
        "max_config_harm",
        "coverage",
        "selected_nonharm_rate",
        "oracle_gain_fraction",
        "ci95_high_delta_raw",
    ]
    show_cols = [c for c in cols if c in overall.columns]
    return "\n".join(
        [
            "# Stage 14 Selector Mechanism Validation",
            "",
            f"Status: `{verdict['status']}`.",
            "",
            "## Overall",
            "",
            df_to_md(overall[show_cols], max_rows=32),
            "",
            "## Selector Top-k",
            "",
            df_to_md(topk, max_rows=32),
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
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    start = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[stage14-selector] preparing assets", flush=True)
    assets = prepare_assets(metrics_csv, stage5_dir, stage3_dir, seed)
    pools = build_cga_pools(assets)
    candidate_metadata(pools.test_candidates, "test").to_csv(output_dir / "stage14_candidate_metadata.csv", index=False)
    write_json(
        output_dir / "stage14_config.json",
        {
            "stage": "stage14_selector_mechanism",
            "source_plan": "deep-research-report (4).md",
            "compact_protocol": "ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192 x seed 2026",
            "metrics_csv": metrics_csv,
            "stage5_dir": stage5_dir,
            "stage6_dir": stage6_dir,
            "stage7_dir": stage7_dir,
            "stage8_dir": stage8_dir,
            "stage3_dir": stage3_dir,
            "seed": seed,
            "n_bootstrap": n_bootstrap,
            "calibration": "selector heads fit on validation inner-train; policy thresholds fit on validation inner-calib only",
            "test_threshold_leakage": False,
        },
    )
    print("[stage14-selector] fitting selector heads on inner-train", flush=True)
    models, train_metrics = fit_selector_models(assets.val_train, pools.train_candidates, assets.schema, seed)
    train_metrics.to_csv(output_dir / "stage14_selector_train_metrics.csv", index=False)
    print("[stage14-selector] preparing calibration/test scores", flush=True)
    calib_scores = prepare_selector_scores(assets.val_calib, pools.calib_candidates, assets.schema, models)
    test_scores = prepare_selector_scores(assets.test, pools.test_candidates, assets.schema, models)
    calib_oracle_mse = oracle_best(deployable_candidates(pools.calib_candidates), assets.val_calib)[1]
    oracle_pred, oracle_mse, _ = oracle_best(deployable_candidates(pools.test_candidates), assets.test)

    policies: Dict[str, Any] = {}
    grid_frames: List[pd.DataFrame] = []
    preds: Dict[str, np.ndarray] = {
        "LRBN": assets.test.lrbn_pred,
        "SRA-BP-balanced": next((c.pred for c in assets.old_test_candidates if c.name == "sra_balanced"), assets.test.lrbn_pred),
        "oracle_stage14_cga_full": oracle_pred,
    }
    decisions_by: Dict[str, pd.DataFrame] = {}

    print("[stage14-selector] calibrating Stage10 hard selector control", flush=True)
    hard_policy, hard_grid = calibrate_stage10_hard(assets.val_calib, calib_scores, calib_oracle_mse, seed)
    hard_grid["target_variant"] = "Stage10 hard selector"
    grid_frames.append(hard_grid)
    policies["Stage10 hard selector"] = asdict(hard_policy)
    hard_pred, hard_dec = apply_stage10_policy_precomputed(assets.test, hard_policy, stage10_like_scores(test_scores))
    hard_dec = hard_dec.rename(columns={"selected_candidate": "selected_action", "utility_score": "accept_score"})
    if "expected_harm" not in hard_dec.columns:
        hard_dec["expected_harm"] = 0.0
    preds["Stage10 hard selector"] = hard_pred
    decisions_by["Stage10 hard selector"] = hard_dec

    topk_rows: List[Dict[str, Any]] = []
    print("[stage14-selector] calibrating selector variants", flush=True)
    for variant in STAGE14_VARIANTS:
        print(f"[stage14-selector] calibrating {variant}", flush=True)
        policy, grid = calibrate_selector_variant(variant, assets.val_calib, calib_scores, calib_oracle_mse, seed)
        grid["target_variant"] = variant
        grid_frames.append(grid)
        policies[variant] = asdict(policy)
        pred, dec = apply_selector_variant(assets.test, test_scores, policy)
        preds[variant] = pred
        decisions_by[variant] = dec
        topk_rows.append(rank_metrics_for_variant(variant, test_scores, pools.test_candidates, assets.test, policy, k=2))

    topk = pd.DataFrame(topk_rows)
    topk.to_csv(output_dir / "stage14_topk_metrics.csv", index=False)
    pd.concat(grid_frames, ignore_index=True).to_csv(output_dir / "stage14_calibration_grid.csv", index=False)
    write_json(output_dir / "stage14_policies.json", policies)

    print("[stage14-selector] evaluating", flush=True)
    overall_rows: List[Dict[str, Any]] = []
    for variant, pred in preds.items():
        decisions = decisions_by.get(variant)
        overall_rows.append(evaluate_selector_variant(variant, pred, assets.test, decisions, oracle_mse if variant != "LRBN" else None, n_bootstrap, seed))
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(output_dir / "stage14_overall.csv", index=False)

    per_config = pd.concat(
        [per_config_rows(variant, pred, assets.test, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    per_config.to_csv(output_dir / "stage14_per_config.csv", index=False)
    masks = stage8_slice_masks(assets.test, assets.schema, stage8_slice_thresholds(assets.val, assets.schema))
    slice_df = pd.concat(
        [slice_rows(variant, pred, assets.test, masks, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    slice_df.to_csv(output_dir / "stage14_slice_metrics.csv", index=False)
    selection_distribution(decisions_by).to_csv(output_dir / "stage14_selection_distribution.csv", index=False)
    gates = gate_table(overall, topk)
    gates.to_csv(output_dir / "stage14_gate_table.csv", index=False)

    deployable = overall[overall["variant"].isin(STAGE14_VARIANTS)].copy()
    best = deployable.sort_values(["mse_delta_pct_vs_lrbn", "max_config_harm", "harm_rate"], ascending=[True, True, True]).iloc[0]
    passed = gates[gates["compact_gate_pass"]]
    verdict = {
        "stage": "stage14_selector_mechanism",
        "status": "compact_pass_ready_for_mini_extension" if not passed.empty else "compact_failed_stop_before_mini_extension",
        "compact_pass": bool(not passed.empty),
        "passed_variants": passed["variant"].astype(str).tolist(),
        "best_variant": str(best["variant"]),
        "best_mse": float(best["mse"]),
        "best_mae": float(best["mae"]),
        "best_mse_delta_pct_vs_lrbn": float(best["mse_delta_pct_vs_lrbn"]),
        "best_harm_rate": float(best["harm_rate"]),
        "best_max_config_harm": float(best["max_config_harm"]),
        "best_selected_nonharm_rate": float(best.get("selected_nonharm_rate", np.nan)),
        "best_oracle_gain_fraction": float(best.get("oracle_gain_fraction", np.nan)),
        "test_threshold_leakage": False,
        "stop_reason": None,
    }
    if passed.empty:
        verdict["stop_reason"] = "no selector candidate passed compact selected-subset safety/balanced gates"
    write_json(output_dir / "stage14_verdict.json", verdict)
    (output_dir / "summary.md").write_text(build_summary(output_dir, verdict, overall, gates, topk), encoding="utf-8")
    completeness = verify_outputs(output_dir)
    completeness.to_csv(output_dir / "stage14_output_completeness.csv", index=False)
    print(f"[stage14-selector] done in {time.time() - start:.1f}s", flush=True)
    return {
        "verdict": verdict,
        "overall": overall,
        "per_config": per_config,
        "slice": slice_df,
        "gates": gates,
        "topk": topk,
        "completeness": completeness,
    }
