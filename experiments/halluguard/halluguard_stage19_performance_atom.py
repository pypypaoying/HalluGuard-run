#!/usr/bin/env python
"""Stage19 Residual Quantile Atom and Non-Boundary Shape Atom validation.

Stage19 follows the Stage18 conclusion: use the CGA/TAE oracle only to create
family-specific residual targets, then test whether deployable continuous
coefficient adapters can safely improve `SRA-BP-balanced`.

All fitted pieces use validation inner-train / inner-calib only:

- inner-train: fit atom bases and coefficient models
- inner-calib: select shrink, norm cap, coverage, and segment policies
- test: final reporting only
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from halluguard_lrbn_bp import ForecastBatch, mae_per_sample, mse_per_sample
from halluguard_stage6_mechanism import bootstrap_ci, feature_frame, horizons, safe_pct, valid_part
from halluguard_stage7_safe_tae import ExpertCandidate, align_frame, candidate_dict, write_json
from halluguard_stage8_safe_tae_pareto import stage8_slice_masks, stage8_slice_thresholds
from halluguard_stage9_architecture_validation import deployable_candidates, prepare_assets
from halluguard_stage10_cga import build_cga_pools, df_to_md, json_default
from halluguard_stage15_endogenous_editors import scale_matrix
from halluguard_stage18_performance_atom_diagnosis import MAIN_PARENT, parent_predictions


RQA_FAMILY = "residual_distribution"
NBSA_FAMILY = "smoothing_teacher"
PARENT = MAIN_PARENT


@dataclass(frozen=True)
class Stage19Config:
    seed: int = 2026
    bootstrap: int = 2000
    output_dir: str = "experiments/halluguard/results/stage19_performance_atom_validation"
    n_pca_basis: int = 5
    n_dct_basis: int = 8
    ridge_alpha: float = 5.0
    shrink_grid: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00)
    cap_grid: Tuple[float, ...] = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.35)
    coverage_grid: Tuple[float, ...] = (1.0, 0.85, 0.70, 0.50, 0.35, 0.20)
    safe_harm: float = 0.03
    safe_max_config_harm: float = 0.10
    tradeoff_harm: float = 0.08
    tradeoff_max_config_harm: float = 0.18


@dataclass
class SplitBundle:
    batch: ForecastBatch
    old_candidates: List[ExpertCandidate]
    cga_candidates: List[ExpertCandidate]


@dataclass
class FamilyTarget:
    target_norm: np.ndarray
    selected_candidate: np.ndarray
    selected_family: np.ndarray
    selected_gain: np.ndarray
    selected_non_parent: np.ndarray


@dataclass
class BasisModel:
    variant: str
    family: str
    basis_type: str
    basis: np.ndarray
    model: Any
    feature_columns: List[str]
    train_coef: np.ndarray
    test_target_coef: Optional[np.ndarray] = None
    policy: Optional[Dict[str, Any]] = None
    calib_delta_norm_raw: Optional[np.ndarray] = None
    test_delta_norm_raw: Optional[np.ndarray] = None
    calib_score_vector: Optional[np.ndarray] = None
    test_score_vector: Optional[np.ndarray] = None


def finite(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def valid_mask(batch: ForecastBatch) -> np.ndarray:
    mask = np.zeros_like(batch.lrbn_pred, dtype=bool)
    for i, h in enumerate(horizons(batch)):
        mask[i, : int(h), :] = True
    return mask


def scaled_delta(batch: ForecastBatch, parent: np.ndarray, pred: np.ndarray) -> np.ndarray:
    delta = (np.asarray(pred, dtype=float) - np.asarray(parent, dtype=float)) / (scale_matrix(batch) + 1e-8)
    delta[~valid_mask(batch)] = 0.0
    return finite(delta)


def flatten_delta(delta: np.ndarray) -> np.ndarray:
    return finite(delta.reshape(delta.shape[0], -1))


def unflatten_delta(x: np.ndarray, template: np.ndarray) -> np.ndarray:
    return finite(np.asarray(x, dtype=float).reshape((x.shape[0],) + template.shape[1:]))


def family_candidates(candidates: Sequence[ExpertCandidate], family: str) -> List[ExpertCandidate]:
    return [c for c in deployable_candidates(candidates) if c.family == family and c.name != "keep_lrbn"]


def family_oracle_target(batch: ForecastBatch, old: Sequence[ExpertCandidate], candidates: Sequence[ExpertCandidate], family: str) -> FamilyTarget:
    parent = parent_predictions(batch, old)[PARENT]
    fam = family_candidates(candidates, family)
    n = len(batch.meta)
    if not fam:
        return FamilyTarget(
            target_norm=np.zeros_like(batch.lrbn_pred),
            selected_candidate=np.asarray(["parent"] * n, dtype=object),
            selected_family=np.asarray(["parent"] * n, dtype=object),
            selected_gain=np.zeros(n, dtype=float),
            selected_non_parent=np.zeros(n, dtype=bool),
        )
    losses = [mse_per_sample(parent, batch.y_true)] + [mse_per_sample(c.pred, batch.y_true) for c in fam]
    names = np.asarray(["parent"] + [c.name for c in fam], dtype=object)
    families = np.asarray(["parent"] + [c.family for c in fam], dtype=object)
    stack = np.stack(losses, axis=1)
    best = np.argmin(stack, axis=1)
    best_pred = parent.copy()
    for j, c in enumerate(fam, start=1):
        m = best == j
        if m.any():
            best_pred[m] = c.pred[m]
    parent_loss = losses[0]
    target = scaled_delta(batch, parent, best_pred)
    return FamilyTarget(
        target_norm=target,
        selected_candidate=names[best],
        selected_family=families[best],
        selected_gain=stack[np.arange(n), best] - parent_loss,
        selected_non_parent=best != 0,
    )


def base_features(batch: ForecastBatch, schema: Mapping[str, List[Any]], columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    df = feature_frame(batch, schema).reset_index(drop=True)
    # Keep model/dataset labels out of the default adapter; horizon and target-free
    # geometry stay available.
    for col in list(df.columns):
        if col.startswith("dataset=") or col.startswith("backbone=") or col.startswith("model="):
            df = df.drop(columns=[col])
    if columns is not None:
        df = align_frame(df, list(columns))
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def pca_basis(x: np.ndarray, n_basis: int, seed: int) -> Tuple[np.ndarray, pd.DataFrame]:
    n = min(max(1, n_basis), x.shape[0], x.shape[1])
    pca = PCA(n_components=n, random_state=seed)
    pca.fit(x)
    rows = []
    cum = 0.0
    for i, evr in enumerate(pca.explained_variance_ratio_, start=1):
        cum += float(evr)
        rows.append({"basis_type": "pca", "component": i, "explained_variance_ratio": float(evr), "cumulative_evr": cum})
    return finite(pca.components_), pd.DataFrame(rows)


def dct_basis(template: np.ndarray, n_basis: int) -> Tuple[np.ndarray, pd.DataFrame]:
    _, t_max, channels = template.shape
    rows: List[np.ndarray] = []
    meta = []
    t = np.arange(t_max, dtype=float)
    for k in range(n_basis):
        wave = np.cos(math.pi * (t + 0.5) * k / max(1, t_max))
        wave = wave / (np.linalg.norm(wave) + 1e-8)
        arr = np.zeros((t_max, channels), dtype=float)
        arr[:, :] = wave[:, None] / math.sqrt(max(1, channels))
        rows.append(arr.reshape(-1))
        meta.append({"basis_type": "dct", "component": k + 1, "explained_variance_ratio": np.nan, "cumulative_evr": np.nan})
    basis = finite(np.vstack(rows))
    return basis, pd.DataFrame(meta)


def project_coefficients(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    # Use least squares projection because DCT vectors are masked/padded and PCA
    # components may be fewer than the effective target rank.
    coef, *_ = np.linalg.lstsq(basis.T, x.T, rcond=None)
    return finite(coef.T)


def coeff_to_delta(coef: np.ndarray, basis: np.ndarray, template: np.ndarray) -> np.ndarray:
    flat = finite(coef @ basis)
    return unflatten_delta(flat, template)


def fit_basis_model(
    variant: str,
    family: str,
    basis_type: str,
    train: SplitBundle,
    calib: SplitBundle,
    test: SplitBundle,
    schema: Mapping[str, List[Any]],
    target_train: FamilyTarget,
    target_test: FamilyTarget,
    cfg: Stage19Config,
    n_basis: int,
) -> Tuple[BasisModel, pd.DataFrame]:
    x_train_delta = flatten_delta(target_train.target_norm)
    if basis_type == "pca":
        basis, basis_report = pca_basis(x_train_delta, n_basis, cfg.seed)
    elif basis_type == "dct":
        basis, basis_report = dct_basis(train.batch.lrbn_pred, n_basis)
    else:
        raise ValueError(f"Unknown basis_type={basis_type}")
    coef_train = project_coefficients(x_train_delta, basis)
    features_train = base_features(train.batch, schema)
    model = make_pipeline(StandardScaler(), Ridge(alpha=cfg.ridge_alpha))
    model.fit(features_train, coef_train)
    columns = list(features_train.columns)
    test_coef_target = project_coefficients(flatten_delta(target_test.target_norm), basis)
    basis_report.insert(0, "variant", variant)
    basis_report.insert(1, "family", family)
    return (
        BasisModel(
            variant=variant,
            family=family,
            basis_type=basis_type,
            basis=basis,
            model=model,
            feature_columns=columns,
            train_coef=coef_train,
            test_target_coef=test_coef_target,
        ),
        basis_report,
    )


def predict_raw_delta(model: BasisModel, batch: ForecastBatch, schema: Mapping[str, List[Any]]) -> Tuple[np.ndarray, np.ndarray]:
    feat = base_features(batch, schema, model.feature_columns)
    coef = finite(np.asarray(model.model.predict(feat), dtype=float))
    delta = coeff_to_delta(coef, model.basis, batch.lrbn_pred)
    delta[~valid_mask(batch)] = 0.0
    score = np.linalg.norm(delta.reshape(delta.shape[0], -1), axis=1)
    return delta, score


def candidate_delta_matrix(batch: ForecastBatch, old: Sequence[ExpertCandidate], candidates: Sequence[ExpertCandidate], family: str) -> Tuple[List[str], np.ndarray]:
    parent = parent_predictions(batch, old)[PARENT]
    fam = family_candidates(candidates, family)
    names = [c.name for c in fam]
    if not fam:
        return names, np.zeros((0,) + batch.lrbn_pred.shape, dtype=float)
    arr = np.stack([scaled_delta(batch, parent, c.pred) for c in fam], axis=0)
    return names, arr


def fit_quantile_head(
    train: SplitBundle,
    calib: SplitBundle,
    test: SplitBundle,
    schema: Mapping[str, List[Any]],
    target_train: FamilyTarget,
    cfg: Stage19Config,
) -> BasisModel:
    names, train_deltas = candidate_delta_matrix(train.batch, train.old_candidates, train.cga_candidates, RQA_FAMILY)
    if len(names) == 0:
        basis = np.zeros((1, train.batch.lrbn_pred.reshape(len(train.batch.meta), -1).shape[1]), dtype=float)
        coef_train = np.zeros((len(train.batch.meta), 1), dtype=float)
    else:
        y = np.zeros((len(train.batch.meta), len(names)), dtype=float)
        for j, name in enumerate(names):
            y[target_train.selected_candidate == name, j] = 1.0
        coef_train = y
        basis = flatten_delta(np.transpose(train_deltas, (1, 0, 2, 3)).reshape(len(train.batch.meta), -1))
        # This placeholder is not used for reconstruction; raw prediction uses
        # split-specific candidate deltas because quantile candidates are not a
        # fixed geometric basis.
        basis = np.eye(max(1, len(names)), dtype=float)
    features_train = base_features(train.batch, schema)
    model = make_pipeline(StandardScaler(), Ridge(alpha=max(1.0, cfg.ridge_alpha)))
    model.fit(features_train, coef_train)
    return BasisModel(
        variant="RQA-QuantileHead",
        family=RQA_FAMILY,
        basis_type="quantile_head",
        basis=basis,
        model=model,
        feature_columns=list(features_train.columns),
        train_coef=coef_train,
        test_target_coef=None,
    )


def predict_quantile_head_delta(
    model: BasisModel,
    split: SplitBundle,
    schema: Mapping[str, List[Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    names, deltas = candidate_delta_matrix(split.batch, split.old_candidates, split.cga_candidates, RQA_FAMILY)
    if len(names) == 0:
        out = np.zeros_like(split.batch.lrbn_pred)
        return out, np.zeros(len(split.batch.meta), dtype=float)
    feat = base_features(split.batch, schema, model.feature_columns)
    weights = finite(np.asarray(model.model.predict(feat), dtype=float))
    weights = np.clip(weights, 0.0, 1.0)
    weight_sum = weights.sum(axis=1, keepdims=True)
    weights = np.where(weight_sum > 1.0, weights / (weight_sum + 1e-8), weights)
    out = np.einsum("nk,kntc->ntc", weights, deltas)
    out[~valid_mask(split.batch)] = 0.0
    score = np.linalg.norm(out.reshape(out.shape[0], -1), axis=1)
    return finite(out), finite(score)


def segment_mask(batch: ForecastBatch, mode: str) -> np.ndarray:
    mask = np.zeros_like(batch.lrbn_pred, dtype=float)
    for i, h0 in enumerate(horizons(batch)):
        h = int(h0)
        if mode == "early":
            s, e = 0, max(1, h // 3)
        elif mode == "mid":
            s, e = max(1, h // 3), max(2, 2 * h // 3)
        elif mode == "late":
            s, e = max(2, 2 * h // 3), h
        else:
            s, e = 0, h
        mask[i, s:e, :] = 1.0
    return mask


def apply_policy(
    batch: ForecastBatch,
    parent: np.ndarray,
    delta_norm_raw: np.ndarray,
    policy: Mapping[str, Any],
    score_vector: Optional[np.ndarray] = None,
    sample_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    shrink = float(policy.get("shrink", 0.0))
    cap_ratio = float(policy.get("cap_ratio", 0.10))
    coverage = float(policy.get("coverage", 1.0))
    segment = str(policy.get("segment", "all"))
    scores = np.asarray(score_vector if score_vector is not None else np.linalg.norm(delta_norm_raw.reshape(len(batch.meta), -1), axis=1), dtype=float)
    selected = np.ones(len(batch.meta), dtype=bool)
    if coverage < 0.999:
        tau = float(policy.get("score_tau", np.nanquantile(scores, max(0.0, min(1.0, 1.0 - coverage)))))
        selected &= scores >= tau
    if sample_mask is not None:
        selected &= np.asarray(sample_mask, dtype=bool)
    delta_norm = shrink * delta_norm_raw.copy()
    delta_norm *= segment_mask(batch, segment)
    delta_norm[~selected] = 0.0
    delta_norm[~valid_mask(batch)] = 0.0
    delta = delta_norm * (scale_matrix(batch) + 1e-8)
    # Per-sample norm cap in observed units.
    flat = delta.reshape(len(batch.meta), -1)
    for i, h0 in enumerate(horizons(batch)):
        h = int(h0)
        scale = float(scale_matrix(batch)[i, 0, 0] + 1e-8)
        max_norm = cap_ratio * math.sqrt(max(1, h * batch.lrbn_pred.shape[2])) * scale
        norm = float(np.linalg.norm(flat[i]))
        if norm > max_norm > 0:
            flat[i] *= max_norm / (norm + 1e-8)
    delta = flat.reshape(delta.shape)
    pred = np.asarray(parent, dtype=float) + delta
    return finite(pred), selected


def config_harm(delta: np.ndarray, batch: ForecastBatch) -> Tuple[float, int, int]:
    meta = batch.meta.assign(row_index=np.arange(len(batch.meta)))
    harms: List[float] = []
    improved: List[bool] = []
    for _, group in meta.groupby(["dataset", "backbone", "horizon", "seed"], observed=True):
        idx = group["row_index"].to_numpy(int)
        d = np.asarray(delta, dtype=float)[idx]
        harms.append(float(np.mean(d > 1e-12)))
        improved.append(bool(np.mean(d) < 0.0))
    return (
        float(np.max(harms)) if harms else float(np.mean(delta > 1e-12)),
        int(np.sum(improved)),
        int(len(improved)),
    )


def metric_vs_parent(
    variant: str,
    pred: np.ndarray,
    batch: ForecastBatch,
    parent_pred: np.ndarray,
    cfg: Stage19Config,
    selected: Optional[np.ndarray] = None,
    family: str = "",
    status: str = "completed",
) -> Dict[str, Any]:
    method_mse = mse_per_sample(pred, batch.y_true)
    parent_mse = mse_per_sample(parent_pred, batch.y_true)
    method_mae = mae_per_sample(pred, batch.y_true)
    parent_mae = mae_per_sample(parent_pred, batch.y_true)
    delta = method_mse - parent_mse
    max_harm, improved, total = config_harm(delta, batch)
    ci = bootstrap_ci(delta, n_boot=cfg.bootstrap, seed=cfg.seed) if cfg.bootstrap > 0 else {}
    selected_mask = np.ones(len(batch.meta), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    row: Dict[str, Any] = {
        "variant": variant,
        "family": family,
        "status": status,
        "n": int(len(batch.meta)),
        "mse": float(np.mean(method_mse)),
        "mae": float(np.mean(method_mae)),
        "parent": PARENT,
        "parent_mse": float(np.mean(parent_mse)),
        "parent_mae": float(np.mean(parent_mae)),
        "mse_delta_vs_sra": float(np.mean(delta)),
        "mse_delta_pct_vs_sra": safe_pct(float(np.mean(method_mse)), float(np.mean(parent_mse))),
        "mae_delta_pct_vs_sra": safe_pct(float(np.mean(method_mae)), float(np.mean(parent_mae))),
        "harm_rate_vs_sra": float(np.mean(delta > 1e-12)),
        "win_rate_vs_sra": float(np.mean(delta < 0.0)),
        "max_config_harm": max_harm,
        "improved_configs": improved,
        "total_configs": total,
        "coverage": float(np.mean(selected_mask)),
        "selected_count": int(np.sum(selected_mask)),
        "selected_harm_rate": float(np.mean((delta > 1e-12)[selected_mask])) if selected_mask.any() else 0.0,
        "test_threshold_leakage": False,
    }
    if ci:
        row.update(
            {
                "ci95_low_delta_raw": ci["ci95_low"],
                "ci95_high_delta_raw": ci["ci95_high"],
                "p_bootstrap_delta_lt_zero": ci["p_lt_zero"],
            }
        )
    return row


def calibration_score(row: Mapping[str, Any], target: str) -> float:
    harm_limit = 0.03 if target == "safe" else 0.08
    cfg_limit = 0.10 if target == "safe" else 0.18
    score = float(row["mse_delta_pct_vs_sra"])
    score += 120.0 * max(0.0, float(row["harm_rate_vs_sra"]) - harm_limit)
    score += 90.0 * max(0.0, float(row["max_config_harm"]) - cfg_limit)
    score += 4.0 * max(0.0, 0.02 - float(row["coverage"]))
    return score


def calibrate_delta_policy(
    variant: str,
    family: str,
    calib: SplitBundle,
    parent_calib: np.ndarray,
    delta_norm_raw: np.ndarray,
    score_vector: np.ndarray,
    cfg: Stage19Config,
    target: str = "tradeoff",
    sample_mask: Optional[np.ndarray] = None,
    segments: Sequence[str] = ("all",),
    shrink_grid: Optional[Sequence[float]] = None,
    cap_grid: Optional[Sequence[float]] = None,
    coverage_grid: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    rows = []
    best: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    shrink_values = tuple(cfg.shrink_grid if shrink_grid is None else shrink_grid)
    cap_values = tuple(cfg.cap_grid if cap_grid is None else cap_grid)
    coverage_values = tuple(cfg.coverage_grid if coverage_grid is None else coverage_grid)
    for segment in segments:
        for shrink in shrink_values:
            for cap in cap_values:
                for coverage in coverage_values:
                    tau = float(np.nanquantile(score_vector, max(0.0, min(1.0, 1.0 - coverage)))) if coverage < 0.999 else float("-inf")
                    policy = {"shrink": shrink, "cap_ratio": cap, "coverage": coverage, "score_tau": tau, "segment": segment}
                    pred, selected = apply_policy(calib.batch, parent_calib, delta_norm_raw, policy, score_vector, sample_mask)
                    row = metric_vs_parent(variant, pred, calib.batch, parent_calib, Stage19Config(seed=cfg.seed, bootstrap=0), selected, family)
                    row.update(policy)
                    row["calibration_score"] = calibration_score(row, target)
                    rows.append(row)
                    if float(row["calibration_score"]) < best_score:
                        best_score = float(row["calibration_score"])
                        best = dict(policy)
                        best["calibration_score"] = best_score
    assert best is not None
    return best, pd.DataFrame(rows)


def residual_alignment(delta_pred: np.ndarray, batch: ForecastBatch, parent: np.ndarray, selected: np.ndarray) -> Dict[str, float]:
    residual = (batch.y_true - parent) / (scale_matrix(batch) + 1e-8)
    delta = delta_pred / (scale_matrix(batch) + 1e-8)
    mask = np.asarray(selected, dtype=bool) & (np.linalg.norm(delta.reshape(len(batch.meta), -1), axis=1) > 1e-12)
    if not mask.any():
        return {"A_gt1_rate": 0.0, "mean_A": 0.0, "mean_cosine_with_residual": 0.0, "delta_norm_ratio": 0.0}
    d = delta.reshape(len(batch.meta), -1)[mask]
    r = residual.reshape(len(batch.meta), -1)[mask]
    denom = np.sum(d * d, axis=1) + 1e-8
    a = 2.0 * np.sum(d * r, axis=1) / denom
    cos = np.sum(d * r, axis=1) / ((np.linalg.norm(d, axis=1) * np.linalg.norm(r, axis=1)) + 1e-8)
    ratio = np.linalg.norm(d, axis=1) / (np.linalg.norm(r, axis=1) + 1e-8)
    return {
        "A_gt1_rate": float(np.mean(a > 1.0)),
        "mean_A": float(np.mean(a)),
        "mean_cosine_with_residual": float(np.mean(cos)),
        "delta_norm_ratio": float(np.mean(ratio)),
    }


def coefficient_fit_metrics(model: BasisModel, test_raw_coef: np.ndarray) -> Dict[str, Any]:
    out: Dict[str, Any] = {"variant": model.variant, "family": model.family, "basis_type": model.basis_type}
    target = model.test_target_coef
    if target is None or target.shape != test_raw_coef.shape or target.size == 0:
        out.update({"coefficient_r2": np.nan, "coefficient_sign_accuracy": np.nan})
        return out
    flat_target = target.reshape(-1)
    flat_pred = test_raw_coef.reshape(-1)
    if np.nanstd(flat_target) > 1e-12:
        out["coefficient_r2"] = float(r2_score(flat_target, flat_pred))
    else:
        out["coefficient_r2"] = np.nan
    nz = np.abs(flat_target) > 1e-8
    out["coefficient_sign_accuracy"] = float(np.mean(np.sign(flat_target[nz]) == np.sign(flat_pred[nz]))) if np.any(nz) else np.nan
    return out


def per_config_rows(variant: str, pred: np.ndarray, batch: ForecastBatch, parent: np.ndarray) -> pd.DataFrame:
    mse = mse_per_sample(pred, batch.y_true)
    parent_mse = mse_per_sample(parent, batch.y_true)
    mae = mae_per_sample(pred, batch.y_true)
    parent_mae = mae_per_sample(parent, batch.y_true)
    df = batch.meta.reset_index(drop=True).copy()
    df["variant"] = variant
    df["mse"] = mse
    df["mae"] = mae
    df["parent_mse"] = parent_mse
    df["parent_mae"] = parent_mae
    grouped = df.groupby(["variant", "dataset", "backbone", "horizon", "seed"], observed=True).agg(
        n=("mse", "size"),
        mse=("mse", "mean"),
        mae=("mae", "mean"),
        parent_mse=("parent_mse", "mean"),
        parent_mae=("parent_mae", "mean"),
    )
    out = grouped.reset_index()
    out["mse_delta_pct_vs_sra"] = [safe_pct(m, p) for m, p in zip(out["mse"], out["parent_mse"])]
    out["mae_delta_pct_vs_sra"] = [safe_pct(m, p) for m, p in zip(out["mae"], out["parent_mae"])]
    out["test_threshold_leakage"] = False
    return out


def slice_metric_rows(variant: str, pred: np.ndarray, batch: ForecastBatch, parent: np.ndarray, masks: Mapping[str, np.ndarray]) -> pd.DataFrame:
    method = mse_per_sample(pred, batch.y_true)
    base = mse_per_sample(parent, batch.y_true)
    rows = []
    all_masks = {"overall": np.ones(len(batch.meta), dtype=bool), **{k: np.asarray(v, dtype=bool) for k, v in masks.items()}}
    for name, mask in all_masks.items():
        if not mask.any():
            continue
        rows.append(
            {
                "variant": variant,
                "slice": name,
                "n": int(mask.sum()),
                "mse": float(np.mean(method[mask])),
                "parent_mse": float(np.mean(base[mask])),
                "mse_delta_pct_vs_sra": safe_pct(float(np.mean(method[mask])), float(np.mean(base[mask]))),
                "harm_rate_vs_sra": float(np.mean((method - base)[mask] > 1e-12)),
                "test_threshold_leakage": False,
            }
        )
    return pd.DataFrame(rows)


def segment_metric_rows(variant: str, pred: np.ndarray, batch: ForecastBatch, parent: np.ndarray) -> pd.DataFrame:
    rows = []
    for seg in ["early", "mid", "late"]:
        method_vals = []
        parent_vals = []
        for i, h0 in enumerate(horizons(batch)):
            h = int(h0)
            if seg == "early":
                s, e = 0, max(1, h // 3)
            elif seg == "mid":
                s, e = max(1, h // 3), max(2, 2 * h // 3)
            else:
                s, e = max(2, 2 * h // 3), h
            method_vals.append(float(np.mean((pred[i, s:e, :] - batch.y_true[i, s:e, :]) ** 2)))
            parent_vals.append(float(np.mean((parent[i, s:e, :] - batch.y_true[i, s:e, :]) ** 2)))
        m = np.asarray(method_vals)
        p = np.asarray(parent_vals)
        rows.append(
            {
                "variant": variant,
                "segment": seg,
                "mse": float(np.mean(m)),
                "parent_mse": float(np.mean(p)),
                "mse_delta_pct_vs_sra": safe_pct(float(np.mean(m)), float(np.mean(p))),
                "harm_rate_vs_sra": float(np.mean((m - p) > 1e-12)),
                "test_threshold_leakage": False,
            }
        )
    return pd.DataFrame(rows)


def variant_gate(row: Mapping[str, Any], slice_df: pd.DataFrame, best_single_delta: Optional[float] = None) -> Tuple[bool, bool, Dict[str, Any]]:
    name = str(row["variant"])
    mse_delta = float(row["mse_delta_pct_vs_sra"])
    harm = float(row["harm_rate_vs_sra"])
    max_harm = float(row["max_config_harm"])
    ci_high = float(row.get("ci95_high_delta_raw", 1.0))
    boundary = slice_df[(slice_df["variant"].eq(name)) & (slice_df["slice"].eq("q4_boundary"))]
    non_boundary = slice_df[(slice_df["variant"].eq(name)) & (slice_df["slice"].eq("non_boundary"))]
    boundary_delta = float(boundary["mse_delta_pct_vs_sra"].iloc[0]) if not boundary.empty else np.nan
    non_boundary_delta = float(non_boundary["mse_delta_pct_vs_sra"].iloc[0]) if not non_boundary.empty else np.nan
    detail = {"boundary_delta": boundary_delta, "non_boundary_delta": non_boundary_delta}
    if name.startswith("RQA"):
        safe = mse_delta <= -0.5 and harm <= 0.03 and max_harm <= 0.10 and ci_high < 0.0
        tradeoff = mse_delta <= -1.2 and harm <= 0.08 and max_harm <= 0.18
    elif name.startswith("NBSA"):
        safe = mse_delta <= -0.4 and harm <= 0.03 and (not np.isfinite(boundary_delta) or boundary_delta <= 0.3)
        tradeoff = mse_delta <= -1.0 and harm <= 0.08 and np.isfinite(non_boundary_delta) and non_boundary_delta <= -2.0
    elif name.startswith("RQA+NBSA"):
        combo_extra = 0.0 if best_single_delta is None else best_single_delta - mse_delta
        safe = mse_delta <= -0.8 and harm <= 0.04 and max_harm <= 0.12
        tradeoff = mse_delta <= -1.8 and harm <= 0.10 and max_harm <= 0.20 and combo_extra >= 0.3
        detail["combo_gain_over_best_single_pp"] = combo_extra
    else:
        safe = tradeoff = False
    return bool(safe), bool(tradeoff), detail


def write_required_summary(
    output_dir: Path,
    verdict: Mapping[str, Any],
    metrics: pd.DataFrame,
    basis: pd.DataFrame,
    coeffs: pd.DataFrame,
    complement: pd.DataFrame,
) -> str:
    return "\n".join(
        [
            "# Stage19 Residual Quantile / Non-Boundary Shape Atom Validation",
            "",
            f"Status: `{verdict['status']}`.",
            "",
            "## Compact Variant Metrics",
            "",
            df_to_md(
                metrics[
                    [
                        "variant",
                        "family",
                        "mse",
                        "mae",
                        "mse_delta_pct_vs_sra",
                        "harm_rate_vs_sra",
                        "max_config_harm",
                        "coverage",
                        "safe_gate",
                        "tradeoff_gate",
                        "test_threshold_leakage",
                    ]
                ].sort_values("mse_delta_pct_vs_sra"),
                max_rows=24,
            ),
            "",
            "## Basis Report",
            "",
            df_to_md(basis.head(24), max_rows=24),
            "",
            "## Coefficient Fit",
            "",
            df_to_md(coeffs, max_rows=24),
            "",
            "## Complementarity",
            "",
            df_to_md(complement, max_rows=24),
            "",
            "## Verdict",
            "",
            "```json",
            json.dumps(verdict, indent=2, ensure_ascii=False, default=json_default),
            "```",
            "",
            f"Output directory: `{output_dir}`",
        ]
    )


def verify_outputs(output_dir: Path) -> pd.DataFrame:
    required = [
        "stage19_config.json",
        "family_oracle_targets.csv",
        "atom_basis_report.csv",
        "coefficient_fit_report.csv",
        "calibration_grid.csv",
        "compact_variant_metrics.csv",
        "compact_per_config.csv",
        "compact_slice_metrics.csv",
        "compact_segment_metrics.csv",
        "complementarity_report.csv",
        "bootstrap_ci.json",
        "stage19_verdict.json",
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
    output_dir: Path,
    stage3_dir: Optional[Path] = None,
    cfg: Optional[Stage19Config] = None,
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    start = time.time()
    cfg = cfg or Stage19Config(bootstrap=n_bootstrap)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[stage19-atoms] preparing assets", flush=True)
    assets = prepare_assets(metrics_csv, stage5_dir, stage3_dir, cfg.seed)
    pools = build_cga_pools(assets)
    splits: Dict[str, SplitBundle] = {
        "inner_train": SplitBundle(assets.val_train, assets.old_train_candidates, pools.train_candidates),
        "inner_calib": SplitBundle(assets.val_calib, assets.old_calib_candidates, pools.calib_candidates),
        "test": SplitBundle(assets.test, assets.old_test_candidates, pools.test_candidates),
    }
    parent = {name: parent_predictions(sp.batch, sp.old_candidates)[PARENT] for name, sp in splits.items()}
    write_json(
        output_dir / "stage19_config.json",
        {
            "stage": "stage19_performance_atom_validation",
            "source_plan": "halluguard_stage19_performance_atom_validation_doc.md",
            "metrics_csv": metrics_csv,
            "stage5_dir": stage5_dir,
            "stage3_dir": stage3_dir,
            "config": asdict(cfg),
            "compact_protocol": "ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192 x seed 2026",
            "parent": PARENT,
            "calibration": "inner-train fits bases/models; inner-calib chooses policies; test only evaluates",
            "test_threshold_leakage": False,
        },
    )

    print("[stage19-atoms] building family oracle targets", flush=True)
    targets: Dict[Tuple[str, str], FamilyTarget] = {}
    target_rows: List[pd.DataFrame] = []
    for split_name, sp in splits.items():
        for family in [RQA_FAMILY, NBSA_FAMILY]:
            tgt = family_oracle_target(sp.batch, sp.old_candidates, sp.cga_candidates, family)
            targets[(split_name, family)] = tgt
            meta = sp.batch.meta.reset_index(drop=True).copy()
            meta["split_eval"] = split_name
            meta["family"] = family
            meta["selected_candidate"] = tgt.selected_candidate
            meta["selected_non_parent"] = tgt.selected_non_parent
            meta["selected_gain_vs_sra"] = tgt.selected_gain
            meta["target_norm"] = np.linalg.norm(tgt.target_norm.reshape(len(meta), -1), axis=1)
            target_rows.append(meta)
    family_oracle_targets = pd.concat(target_rows, ignore_index=True)
    family_oracle_targets.to_csv(output_dir / "family_oracle_targets.csv", index=False)

    masks_calib = stage8_slice_masks(assets.val_calib, assets.schema, stage8_slice_thresholds(assets.val, assets.schema))
    masks_test = stage8_slice_masks(assets.test, assets.schema, stage8_slice_thresholds(assets.val, assets.schema))
    non_boundary_calib = np.asarray(masks_calib.get("non_boundary", np.ones(len(assets.val_calib.meta), dtype=bool)), dtype=bool)
    non_boundary_test = np.asarray(masks_test.get("non_boundary", np.ones(len(assets.test.meta), dtype=bool)), dtype=bool)

    print("[stage19-atoms] fitting continuous atom adapters", flush=True)
    models: Dict[str, BasisModel] = {}
    basis_reports: List[pd.DataFrame] = []
    coeff_rows: List[Dict[str, Any]] = []
    calibration_grids: List[pd.DataFrame] = []
    preds: Dict[str, np.ndarray] = {
        "LRBN": assets.test.lrbn_pred,
        "SRA-BP-safe": candidate_dict(assets.old_test_candidates).get("sra_safe", ExpertCandidate("fallback", "safe", "boundary", assets.test.lrbn_pred, True)).pred,
        "SRA-BP-balanced": parent["test"],
    }
    selected_by_variant: Dict[str, np.ndarray] = {k: np.ones(len(assets.test.meta), dtype=bool) for k in preds}
    raw_delta_by_variant: Dict[str, np.ndarray] = {}

    fit_specs = [
        ("RQA-PCA-Coef", RQA_FAMILY, "pca", cfg.n_pca_basis, "tradeoff", ("all",), None, None, None),
        ("RQA-DCT-Coef", RQA_FAMILY, "dct", cfg.n_dct_basis, "tradeoff", ("all",), None, None, None),
        (
            "RQA-HarmAwareCoef",
            RQA_FAMILY,
            "pca",
            cfg.n_pca_basis,
            "safe",
            ("all",),
            (0.0, 0.05, 0.10, 0.20, 0.35, 0.50),
            (0.03, 0.05, 0.08, 0.10, 0.15),
            (1.0, 0.85, 0.70, 0.50, 0.35, 0.20),
        ),
        ("NBSA-DCT-Shape", NBSA_FAMILY, "dct", cfg.n_dct_basis, "tradeoff", ("all",), None, None, None),
        ("NBSA-RoughnessAdapter", NBSA_FAMILY, "dct", cfg.n_dct_basis, "tradeoff", ("all",), None, None, (1.0, 0.70, 0.50, 0.35, 0.20)),
        ("NBSA-NonBoundaryOnly", NBSA_FAMILY, "dct", cfg.n_dct_basis, "safe", ("all",), None, None, (1.0, 0.85, 0.70, 0.50)),
        ("NBSA-LocalShapeEnvelope", NBSA_FAMILY, "dct", cfg.n_dct_basis, "tradeoff", ("early", "mid", "late"), None, None, None),
    ]

    for variant, family, basis_type, n_basis, target, segments, shrinks, caps, coverages in fit_specs:
        bm, report = fit_basis_model(
            variant,
            family,
            basis_type,
            splits["inner_train"],
            splits["inner_calib"],
            splits["test"],
            assets.schema,
            targets[("inner_train", family)],
            targets[("test", family)],
            cfg,
            n_basis,
        )
        calib_delta, calib_score = predict_raw_delta(bm, assets.val_calib, assets.schema)
        test_delta, test_score = predict_raw_delta(bm, assets.test, assets.schema)
        bm.calib_delta_norm_raw = calib_delta
        bm.test_delta_norm_raw = test_delta
        bm.calib_score_vector = calib_score
        bm.test_score_vector = test_score
        sample_mask_calib = non_boundary_calib if variant == "NBSA-NonBoundaryOnly" else None
        sample_mask_test = non_boundary_test if variant == "NBSA-NonBoundaryOnly" else None
        policy, grid = calibrate_delta_policy(
            variant,
            family,
            splits["inner_calib"],
            parent["inner_calib"],
            calib_delta,
            calib_score,
            cfg,
            target=target,
            sample_mask=sample_mask_calib,
            segments=segments,
            shrink_grid=shrinks,
            cap_grid=caps,
            coverage_grid=coverages,
        )
        bm.policy = policy
        pred, selected = apply_policy(assets.test, parent["test"], test_delta, policy, test_score, sample_mask_test)
        preds[variant] = pred
        selected_by_variant[variant] = selected
        raw_delta_by_variant[variant] = test_delta
        models[variant] = bm
        grid["variant"] = variant
        calibration_grids.append(grid)
        basis_reports.append(report)
        test_feat = base_features(assets.test, assets.schema, bm.feature_columns)
        test_coef_pred = finite(np.asarray(bm.model.predict(test_feat), dtype=float))
        coef_row = coefficient_fit_metrics(bm, test_coef_pred)
        coef_row.update({f"policy_{k}": v for k, v in policy.items()})
        coeff_rows.append(coef_row)

    qh = fit_quantile_head(splits["inner_train"], splits["inner_calib"], splits["test"], assets.schema, targets[("inner_train", RQA_FAMILY)], cfg)
    q_calib_delta, q_calib_score = predict_quantile_head_delta(qh, splits["inner_calib"], assets.schema)
    q_test_delta, q_test_score = predict_quantile_head_delta(qh, splits["test"], assets.schema)
    q_policy, q_grid = calibrate_delta_policy(
        "RQA-QuantileHead",
        RQA_FAMILY,
        splits["inner_calib"],
        parent["inner_calib"],
        q_calib_delta,
        q_calib_score,
        cfg,
        target="tradeoff",
    )
    q_pred, q_selected = apply_policy(assets.test, parent["test"], q_test_delta, q_policy, q_test_score)
    preds["RQA-QuantileHead"] = q_pred
    selected_by_variant["RQA-QuantileHead"] = q_selected
    raw_delta_by_variant["RQA-QuantileHead"] = q_test_delta
    q_grid["variant"] = "RQA-QuantileHead"
    calibration_grids.append(q_grid)
    coeff_rows.append({"variant": "RQA-QuantileHead", "family": RQA_FAMILY, "basis_type": "quantile_head", **{f"policy_{k}": v for k, v in q_policy.items()}, "coefficient_r2": np.nan, "coefficient_sign_accuracy": np.nan})

    # Combination: use the best validation policy family representatives, then
    # recalibrate a two-atom residual composition on inner-calib.
    calib_grid = pd.concat(calibration_grids, ignore_index=True)
    best_rqa_name = (
        calib_grid[calib_grid["variant"].str.startswith("RQA")].groupby("variant", observed=True)["calibration_score"].min().sort_values().index[0]
    )
    best_nbsa_name = (
        calib_grid[calib_grid["variant"].str.startswith("NBSA")].groupby("variant", observed=True)["calibration_score"].min().sort_values().index[0]
    )
    rqa_calib_delta = q_calib_delta if best_rqa_name == "RQA-QuantileHead" else models[best_rqa_name].calib_delta_norm_raw
    rqa_test_delta = q_test_delta if best_rqa_name == "RQA-QuantileHead" else models[best_rqa_name].test_delta_norm_raw
    nbsa_calib_delta = models[best_nbsa_name].calib_delta_norm_raw
    nbsa_test_delta = models[best_nbsa_name].test_delta_norm_raw
    combo_rows = []
    combo_best: Optional[Dict[str, Any]] = None
    combo_best_score = float("inf")
    for srqa in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75):
        for snbsa in (0.0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75):
            for cap in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
                policy = {"shrink": 1.0, "cap_ratio": cap, "coverage": 1.0, "score_tau": float("-inf"), "segment": "all", "rqa_shrink": srqa, "nbsa_shrink": snbsa}
                delta = srqa * rqa_calib_delta + snbsa * nbsa_calib_delta
                pred, selected = apply_policy(assets.val_calib, parent["inner_calib"], delta, policy, np.linalg.norm(delta.reshape(len(assets.val_calib.meta), -1), axis=1))
                row = metric_vs_parent("RQA+NBSA", pred, assets.val_calib, parent["inner_calib"], Stage19Config(seed=cfg.seed, bootstrap=0), selected, "combined")
                row.update(policy)
                row["calibration_score"] = calibration_score(row, "tradeoff")
                row["best_rqa"] = best_rqa_name
                row["best_nbsa"] = best_nbsa_name
                combo_rows.append(row)
                if float(row["calibration_score"]) < combo_best_score:
                    combo_best_score = float(row["calibration_score"])
                    combo_best = dict(policy)
                    combo_best["calibration_score"] = combo_best_score
                    combo_best["best_rqa"] = best_rqa_name
                    combo_best["best_nbsa"] = best_nbsa_name
    assert combo_best is not None
    combo_delta_test = float(combo_best["rqa_shrink"]) * rqa_test_delta + float(combo_best["nbsa_shrink"]) * nbsa_test_delta
    combo_pred, combo_selected = apply_policy(
        assets.test,
        parent["test"],
        combo_delta_test,
        combo_best,
        np.linalg.norm(combo_delta_test.reshape(len(assets.test.meta), -1), axis=1),
    )
    preds["RQA+NBSA"] = combo_pred
    selected_by_variant["RQA+NBSA"] = combo_selected
    raw_delta_by_variant["RQA+NBSA"] = combo_delta_test
    calibration_grids.append(pd.DataFrame(combo_rows))
    coeff_rows.append({"variant": "RQA+NBSA", "family": "combined", "basis_type": "composed", **{f"policy_{k}": v for k, v in combo_best.items()}, "coefficient_r2": np.nan, "coefficient_sign_accuracy": np.nan})

    print("[stage19-atoms] evaluating compact variants", flush=True)
    metric_rows: List[Dict[str, Any]] = []
    per_config_frames: List[pd.DataFrame] = []
    slice_frames: List[pd.DataFrame] = []
    segment_frames: List[pd.DataFrame] = []
    for variant, pred in preds.items():
        family = "reference"
        if variant.startswith("RQA") and variant != "RQA+NBSA":
            family = RQA_FAMILY
        elif variant.startswith("NBSA"):
            family = NBSA_FAMILY
        elif variant == "RQA+NBSA":
            family = "combined"
        row = metric_vs_parent(variant, pred, assets.test, parent["test"], cfg, selected_by_variant.get(variant), family)
        if variant in raw_delta_by_variant:
            actual_delta = pred - parent["test"]
            row.update(residual_alignment(actual_delta, assets.test, parent["test"], selected_by_variant[variant]))
        else:
            row.update({"A_gt1_rate": np.nan, "mean_A": np.nan, "mean_cosine_with_residual": np.nan, "delta_norm_ratio": np.nan})
        metric_rows.append(row)
        per_config_frames.append(per_config_rows(variant, pred, assets.test, parent["test"]))
        slice_frames.append(slice_metric_rows(variant, pred, assets.test, parent["test"], masks_test))
        segment_frames.append(segment_metric_rows(variant, pred, assets.test, parent["test"]))

    metrics = pd.DataFrame(metric_rows)
    per_config = pd.concat(per_config_frames, ignore_index=True)
    slice_df = pd.concat(slice_frames, ignore_index=True)
    segment_df = pd.concat(segment_frames, ignore_index=True)

    adapter_metrics = metrics[metrics["variant"].str.startswith(("RQA", "NBSA")) & ~metrics["variant"].isin(["RQA+NBSA"])].copy()
    best_single_delta = float(adapter_metrics["mse_delta_pct_vs_sra"].min()) if not adapter_metrics.empty else None
    gate_rows = []
    for i, row in metrics.iterrows():
        safe, tradeoff, detail = variant_gate(row, slice_df, best_single_delta)
        metrics.loc[i, "safe_gate"] = safe
        metrics.loc[i, "tradeoff_gate"] = tradeoff
        for k, v in detail.items():
            metrics.loc[i, k] = v
        gate_rows.append({"variant": row["variant"], "safe_gate": safe, "tradeoff_gate": tradeoff, **detail})

    basis_report = pd.concat(basis_reports, ignore_index=True) if basis_reports else pd.DataFrame()
    coeff_report = pd.DataFrame(coeff_rows)
    calibration_grid = pd.concat(calibration_grids, ignore_index=True)

    # Complementarity diagnostics.
    parent_mse = mse_per_sample(parent["test"], assets.test.y_true)
    best_rqa_variant = metrics[metrics["variant"].str.startswith("RQA") & ~metrics["variant"].eq("RQA+NBSA")].sort_values("mse_delta_pct_vs_sra").head(1)["variant"].iloc[0]
    best_nbsa_variant = metrics[metrics["variant"].str.startswith("NBSA")].sort_values("mse_delta_pct_vs_sra").head(1)["variant"].iloc[0]
    rqa_gain = mse_per_sample(preds[best_rqa_variant], assets.test.y_true) < parent_mse
    nbsa_gain = mse_per_sample(preds[best_nbsa_variant], assets.test.y_true) < parent_mse
    combo_gain = mse_per_sample(preds["RQA+NBSA"], assets.test.y_true) < parent_mse
    complement = pd.DataFrame(
        [
            {
                "best_rqa_variant": best_rqa_variant,
                "best_nbsa_variant": best_nbsa_variant,
                "rqa_gain_rate": float(np.mean(rqa_gain)),
                "nbsa_gain_rate": float(np.mean(nbsa_gain)),
                "overlap_gain_rate": float(np.mean(rqa_gain & nbsa_gain)),
                "rqa_only_gain_rate": float(np.mean(rqa_gain & ~nbsa_gain)),
                "nbsa_only_gain_rate": float(np.mean(nbsa_gain & ~rqa_gain)),
                "combo_gain_rate": float(np.mean(combo_gain)),
                "combo_gain_over_best_single_pp": float(best_single_delta - metrics.loc[metrics["variant"].eq("RQA+NBSA"), "mse_delta_pct_vs_sra"].iloc[0]) if best_single_delta is not None else np.nan,
                "test_threshold_leakage": False,
            }
        ]
    )

    metrics.to_csv(output_dir / "compact_variant_metrics.csv", index=False)
    per_config.to_csv(output_dir / "compact_per_config.csv", index=False)
    slice_df.to_csv(output_dir / "compact_slice_metrics.csv", index=False)
    segment_df.to_csv(output_dir / "compact_segment_metrics.csv", index=False)
    basis_report.to_csv(output_dir / "atom_basis_report.csv", index=False)
    coeff_report.to_csv(output_dir / "coefficient_fit_report.csv", index=False)
    calibration_grid.to_csv(output_dir / "calibration_grid.csv", index=False)
    complement.to_csv(output_dir / "complementarity_report.csv", index=False)

    boot = {
        str(row["variant"]): {
            "ci95_low_delta_raw": float(row.get("ci95_low_delta_raw", np.nan)),
            "ci95_high_delta_raw": float(row.get("ci95_high_delta_raw", np.nan)),
            "p_bootstrap_delta_lt_zero": float(row.get("p_bootstrap_delta_lt_zero", np.nan)),
        }
        for _, row in metrics.iterrows()
    }
    write_json(output_dir / "bootstrap_ci.json", boot)

    rqa_pass = bool(metrics[metrics["variant"].str.startswith("RQA") & ~metrics["variant"].eq("RQA+NBSA")][["safe_gate", "tradeoff_gate"]].any().any())
    nbsa_pass = bool(metrics[metrics["variant"].str.startswith("NBSA")][["safe_gate", "tradeoff_gate"]].any().any())
    combo_row = metrics[metrics["variant"].eq("RQA+NBSA")]
    combo_pass = bool(not combo_row.empty and (bool(combo_row["safe_gate"].iloc[0]) or bool(combo_row["tradeoff_gate"].iloc[0])))
    best_row = metrics[metrics["variant"].str.startswith(("RQA", "NBSA"))].sort_values("mse_delta_pct_vs_sra").head(1).to_dict("records")
    no_leak = bool(not metrics["test_threshold_leakage"].astype(bool).any())
    if combo_pass:
        status = "combined_atom_compact_pass_ready_for_mini_extension"
        recommendation = "Promote RQA+NBSA to mini-extension with the recorded validation-only policy."
    elif rqa_pass and nbsa_pass:
        status = "both_atoms_individually_pass_combination_not_ready"
        recommendation = "Keep both atom lines but redesign composition before mini-extension."
    elif rqa_pass:
        status = "rqa_compact_pass_nbsa_failed"
        recommendation = "Promote SRA-BP + Residual Quantile Atom as the next parent candidate."
    elif nbsa_pass:
        status = "nbsa_compact_pass_rqa_failed"
        recommendation = "Promote SRA-BP + Non-Boundary Shape Atom as the next parent candidate."
    else:
        status = "compact_failed_stop_performance_atom_route"
        recommendation = "Do not promote Stage19 adapters; current continuous atom application does not beat SRA-BP-balanced safely."

    verdict = {
        "stage": "stage19_performance_atom_validation",
        "status": status,
        "compact_protocol_completed": True,
        "mini_extension_ran": False,
        "reason_no_mini_extension": "Compact gate did not pass." if not (rqa_pass or nbsa_pass or combo_pass) else "Mini-extension deferred to next stage runner after compact promotion.",
        "rqa_pass": rqa_pass,
        "nbsa_pass": nbsa_pass,
        "combo_pass": combo_pass,
        "best_variant": best_row[0] if best_row else {},
        "best_rqa_variant": best_rqa_variant,
        "best_nbsa_variant": best_nbsa_variant,
        "best_single_delta_pct_vs_sra": best_single_delta,
        "combo_delta_pct_vs_sra": float(combo_row["mse_delta_pct_vs_sra"].iloc[0]) if not combo_row.empty else np.nan,
        "test_threshold_leakage": not no_leak,
        "recommendation": recommendation,
        "runtime_seconds": float(time.time() - start),
    }
    write_json(output_dir / "stage19_verdict.json", verdict)
    (output_dir / "summary.md").write_text(write_required_summary(output_dir, verdict, metrics, basis_report, coeff_report, complement), encoding="utf-8")
    completeness = verify_outputs(output_dir)
    completeness.to_csv(output_dir / "stage19_output_completeness.csv", index=False)
    print(f"[stage19-atoms] done in {time.time() - start:.1f}s", flush=True)
    return {
        "verdict": verdict,
        "metrics": metrics,
        "per_config": per_config,
        "slice": slice_df,
        "segment": segment_df,
        "basis": basis_report,
        "coefficients": coeff_report,
        "complementarity": complement,
        "completeness": completeness,
    }
