#!/usr/bin/env python
"""Stage 6 mechanism validation utilities for HalluGuard.

The functions here are deliberately lightweight. Stage 6 is a compact
mechanism check over existing forecast assets, not a final TableA method.
All trainable choices are fit on validation only; test is evaluation only.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from halluguard_lrbn_bp import EPS, ForecastBatch, mae_per_sample, mse_per_sample
from halluguard_sra_bp import apply_sra_bp, compute_sra_features
from halluguard_stage4_bp_harm_control import apply_candidate


def horizons(batch: ForecastBatch) -> np.ndarray:
    return batch.meta["horizon"].to_numpy(int)


def split_batch(batch: ForecastBatch) -> Tuple[ForecastBatch, ForecastBatch]:
    split = batch.meta["split"].to_numpy(str)
    return batch.subset(split == "val"), batch.subset(split == "test")


def safe_pct(method_mean: float, base_mean: float) -> float:
    return float((method_mean - base_mean) / (base_mean + EPS) * 100.0)


def valid_part(arr: np.ndarray, idx: int, h: Optional[int] = None) -> np.ndarray:
    if h is None:
        h = int(arr.shape[1])
    return np.asarray(arr[idx, :h, :], dtype=float)


def valid_flat(arr: np.ndarray, idx: int, h: int) -> np.ndarray:
    return valid_part(arr, idx, h).reshape(-1)


def finite_mean(x: np.ndarray, default: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return default
    return float(np.nanmean(x))


def finite_std(x: np.ndarray, default: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).any():
        return default
    return float(np.nanstd(x))


def finite_norm(x: np.ndarray, default: float = 0.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return default
    return float(np.linalg.norm(x))


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, seed: int = 2026) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan"), "p_lt_zero": float("nan")}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        boots[i] = float(np.mean(values[idx]))
    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "p_lt_zero": float(np.mean(boots < 0.0)),
    }


def metric_summary(
    name: str,
    pred: np.ndarray,
    batch: ForecastBatch,
    baseline: Optional[np.ndarray] = None,
    selected: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    baseline = batch.lrbn_pred if baseline is None else baseline
    method_mse = mse_per_sample(pred, batch.y_true)
    base_mse = mse_per_sample(baseline, batch.y_true)
    method_mae = mae_per_sample(pred, batch.y_true)
    base_mae = mae_per_sample(baseline, batch.y_true)
    delta = method_mse - base_mse
    wins = delta < 0.0
    harms = delta > 1e-12
    mean_win = float((-delta[wins]).mean()) if wins.any() else 0.0
    mean_loss = float(delta[harms].mean()) if harms.any() else 0.0
    row: Dict[str, Any] = {
        "method": name,
        "n": int(len(batch.meta)),
        "mse": float(np.mean(method_mse)),
        "mae": float(np.mean(method_mae)),
        "lrbn_mse": float(np.mean(base_mse)),
        "lrbn_mae": float(np.mean(base_mae)),
        "mse_delta_vs_lrbn": float(np.mean(delta)),
        "mse_delta_pct_vs_lrbn": safe_pct(float(np.mean(method_mse)), float(np.mean(base_mse))),
        "mae_delta_pct_vs_lrbn": safe_pct(float(np.mean(method_mae)), float(np.mean(base_mae))),
        "harm_rate": float(np.mean(harms)),
        "win_rate": float(np.mean(wins)),
        "mean_win_size": mean_win,
        "mean_loss_size": mean_loss,
        "win_loss_ratio": float(mean_win / (mean_loss + EPS)),
        "top5_loss_contribution": top_loss_contribution(delta, top_frac=0.05),
    }
    if selected is not None:
        selected = np.asarray(selected, dtype=bool)
        row["coverage"] = float(selected.mean())
        row["selected_count"] = int(selected.sum())
        row["selected_harm_rate"] = float(np.mean(harms[selected])) if selected.any() else 0.0
    else:
        row["coverage"] = 0.0
        row["selected_count"] = 0
        row["selected_harm_rate"] = 0.0
    return row


def top_loss_contribution(delta: np.ndarray, top_frac: float = 0.05) -> float:
    losses = np.asarray(delta, dtype=float)
    losses = np.maximum(losses, 0.0)
    total = float(losses.sum())
    if total <= 0:
        return 0.0
    k = max(1, int(math.ceil(len(losses) * top_frac)))
    return float(np.sort(losses)[-k:].sum() / total)


def parse_sample_index(sample_id: Any) -> int:
    m = re.search(r"(\d+)$", str(sample_id))
    return int(m.group(1)) if m else 0


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_sra_predictions(
    batch: ForecastBatch,
    safe_params: Dict[str, Any],
    balanced_params: Dict[str, Any],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    safe_pred, safe_aux = apply_sra_bp(batch.context, batch.raw_pred, batch.lrbn_pred, horizons(batch), safe_params)
    balanced_pred, balanced_aux = apply_sra_bp(
        batch.context, batch.raw_pred, batch.lrbn_pred, horizons(batch), balanced_params
    )
    return {
        "SRA-BP-safe": (safe_pred, safe_aux["strength"]),
        "SRA-BP-balanced": (balanced_pred, balanced_aux["strength"]),
    }


def feature_schema(val: ForecastBatch) -> Dict[str, List[Any]]:
    return {
        "datasets": sorted(val.meta["dataset"].astype(str).unique().tolist()),
        "backbones": sorted(val.meta["backbone"].astype(str).unique().tolist()),
        "horizons": sorted([int(x) for x in val.meta["horizon"].unique().tolist()]),
    }


def feature_frame(batch: ForecastBatch, schema: Dict[str, List[Any]]) -> pd.DataFrame:
    feats = compute_sra_features(batch.context, batch.raw_pred, batch.lrbn_pred)
    hs = horizons(batch)
    rows: List[Dict[str, float]] = []
    for i, h in enumerate(hs):
        x = valid_part(batch.context, i, batch.context.shape[1])
        raw = valid_part(batch.raw_pred, i, h)
        lrbn = valid_part(batch.lrbn_pred, i, h)
        dx = np.diff(x, axis=0)
        draw = np.diff(raw, axis=0) if h > 1 else np.zeros_like(raw[:1])
        dl = np.diff(lrbn, axis=0) if h > 1 else np.zeros_like(lrbn[:1])
        row: Dict[str, float] = {
            "boundary_gap_lrbn": float(feats["g_l"][i]),
            "boundary_gap_raw": float(feats["g_raw"][i]),
            "repair_ratio": float(feats["repair_ratio"][i]),
            "jump_support": float(feats["jump_support"][i]),
            "trend_support": float(feats["trend_support"][i]),
            "vol_support": float(feats["vol_support"][i]),
            "smooth_support": float(feats["smooth_support"][i]),
            "context_mean": finite_mean(x),
            "context_std": finite_std(x),
            "context_diff_mean": finite_mean(dx),
            "context_diff_std": finite_std(dx),
            "context_last": finite_mean(x[-1:]),
            "context_range": float(np.nanmax(x) - np.nanmin(x)) if np.isfinite(x).any() else 0.0,
            "raw_mean": finite_mean(raw),
            "raw_std": finite_std(raw),
            "raw_diff_std": finite_std(draw),
            "lrbn_mean": finite_mean(lrbn),
            "lrbn_std": finite_std(lrbn),
            "lrbn_diff_std": finite_std(dl),
            "raw_lrbn_mean_gap": finite_mean(raw - lrbn),
            "raw_lrbn_norm_gap": finite_norm(raw - lrbn) / math.sqrt(max(1, raw.size)),
            "lrbn_context_level_gap": finite_mean(lrbn[: min(8, h)] - x[-1:]),
            "roughness_ratio": finite_std(dl) / (finite_std(dx) + 1e-6),
            "pred_context_var_ratio": (finite_std(lrbn) + 1e-6) / (finite_std(x) + 1e-6),
            "horizon": float(h),
            "horizon_norm": float(h) / 720.0,
        }
        meta = batch.meta.iloc[i]
        for ds in schema["datasets"]:
            row[f"dataset={ds}"] = 1.0 if str(meta["dataset"]) == str(ds) else 0.0
        for bb in schema["backbones"]:
            row[f"backbone={bb}"] = 1.0 if str(meta["backbone"]) == str(bb) else 0.0
        for hh in schema["horizons"]:
            row[f"horizon={hh}"] = 1.0 if int(meta["horizon"]) == int(hh) else 0.0
        rows.append(row)
    return pd.DataFrame(rows).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def make_mechanism_sample_table(
    batch: ForecastBatch,
    schema: Dict[str, List[Any]],
    sra: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    feat = feature_frame(batch, schema)
    out = batch.meta.reset_index(drop=True).copy()
    out["sample_order"] = out["sample_id"].apply(parse_sample_index)
    out["mse_raw"] = mse_per_sample(batch.raw_pred, batch.y_true)
    out["mse_lrbn"] = mse_per_sample(batch.lrbn_pred, batch.y_true)
    out["mae_lrbn"] = mae_per_sample(batch.lrbn_pred, batch.y_true)
    for name, (pred, strength) in sra.items():
        key = name.replace("-", "_").lower()
        out[f"mse_{key}"] = mse_per_sample(pred, batch.y_true)
        out[f"strength_{key}"] = strength
    return pd.concat([out, feat], axis=1)


def slice_thresholds(val: ForecastBatch) -> Dict[str, float]:
    feats = compute_sra_features(val.context, val.raw_pred, val.lrbn_pred, val.y_true)
    residual_norm = np.array(
        [finite_norm(valid_flat(val.y_true - val.lrbn_pred, i, int(h))) for i, h in enumerate(horizons(val))],
        dtype=float,
    )
    ctx_vol = feature_frame(val, feature_schema(val))["context_diff_std"].to_numpy(float)
    return {
        "g_l_q25": float(np.quantile(feats["g_l"], 0.25)),
        "g_l_q75": float(np.quantile(feats["g_l"], 0.75)),
        "repair_low": 0.3,
        "repair_high": 0.7,
        "context_vol_q75": float(np.quantile(ctx_vol, 0.75)),
        "residual_norm_q75": float(np.quantile(residual_norm, 0.75)),
    }


def slice_masks(batch: ForecastBatch, thresholds: Dict[str, float], schema: Dict[str, List[Any]]) -> Dict[str, np.ndarray]:
    feats = compute_sra_features(batch.context, batch.raw_pred, batch.lrbn_pred, batch.y_true)
    ff = feature_frame(batch, schema)
    residual_norm = np.array(
        [finite_norm(valid_flat(batch.y_true - batch.lrbn_pred, i, int(h))) for i, h in enumerate(horizons(batch))],
        dtype=float,
    )
    high_gap = feats["g_l"] >= thresholds["g_l_q75"]
    low_gap = feats["g_l"] <= thresholds["g_l_q25"]
    low_repair = feats["repair_ratio"] <= thresholds["repair_low"]
    high_repair = feats["repair_ratio"] >= thresholds["repair_high"]
    return {
        "overall": np.ones(len(batch.meta), dtype=bool),
        "high_gap_low_repair": high_gap & low_repair,
        "low_gap_high_repair": low_gap & high_repair,
        "non_boundary": ~(high_gap & low_repair),
        "high_volatility": ff["context_diff_std"].to_numpy(float) >= thresholds["context_vol_q75"],
        "large_residual_norm": residual_norm >= thresholds["residual_norm_q75"],
        "q4_boundary": high_gap,
    }


def slice_result_rows(
    batch: ForecastBatch,
    methods: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]],
    thresholds: Dict[str, float],
    schema: Dict[str, List[Any]],
) -> pd.DataFrame:
    masks = slice_masks(batch, thresholds, schema)
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    rows = []
    for method, (pred, strength) in methods.items():
        mm = mse_per_sample(pred, batch.y_true)
        delta = mm - base_mse
        selected = np.zeros(len(batch.meta), dtype=bool) if strength is None else np.asarray(strength) > 1e-8
        for slice_name, mask in masks.items():
            if not mask.any():
                continue
            rows.append(
                {
                    "method": method,
                    "slice": slice_name,
                    "n": int(mask.sum()),
                    "mse": float(np.mean(mm[mask])),
                    "lrbn_mse": float(np.mean(base_mse[mask])),
                    "mse_delta_pct_vs_lrbn": safe_pct(float(np.mean(mm[mask])), float(np.mean(base_mse[mask]))),
                    "mean_delta": float(np.mean(delta[mask])),
                    "harm_rate": float(np.mean(delta[mask] > 1e-12)),
                    "win_rate": float(np.mean(delta[mask] < 0.0)),
                    "coverage": float(np.mean(selected[mask])),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# MRC
# ---------------------------------------------------------------------------


@dataclass
class RidgeResidualModels:
    models: Dict[int, Any]
    alpha_by_horizon: Dict[int, float]
    cv_pred: np.ndarray
    val_pred: np.ndarray


def _empty_delta_like(batch: ForecastBatch) -> np.ndarray:
    return np.zeros_like(batch.lrbn_pred, dtype=float)


def fit_ridge_residual_models(
    val: ForecastBatch,
    x_val: pd.DataFrame,
    alphas: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 100.0),
    seed: int = 2026,
) -> RidgeResidualModels:
    hs = horizons(val)
    cv_delta = _empty_delta_like(val)
    val_delta = _empty_delta_like(val)
    models: Dict[int, Any] = {}
    alpha_by_horizon: Dict[int, float] = {}
    X_all = x_val.to_numpy(float)
    for h in sorted(set(hs.tolist())):
        idx = np.where(hs == h)[0]
        X = X_all[idx]
        Y = np.stack([valid_flat(val.y_true - val.lrbn_pred, int(i), int(h)) for i in idx], axis=0)
        splits = min(4, len(idx))
        best_alpha = float(alphas[0])
        best_mse = float("inf")
        best_cv = np.zeros_like(Y)
        for alpha in alphas:
            pred = np.zeros_like(Y)
            kf = KFold(n_splits=splits, shuffle=True, random_state=seed)
            for train_idx, hold_idx in kf.split(X):
                model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
                model.fit(X[train_idx], Y[train_idx])
                pred[hold_idx] = model.predict(X[hold_idx])
            mse = float(np.mean((pred - Y) ** 2))
            if mse < best_mse:
                best_mse = mse
                best_alpha = float(alpha)
                best_cv = pred
        model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
        model.fit(X, Y)
        full_pred = model.predict(X)
        for local, global_idx in enumerate(idx):
            cv_delta[global_idx, :h, :] = best_cv[local].reshape(h, -1)
            val_delta[global_idx, :h, :] = full_pred[local].reshape(h, -1)
        models[int(h)] = model
        alpha_by_horizon[int(h)] = best_alpha
    return RidgeResidualModels(models=models, alpha_by_horizon=alpha_by_horizon, cv_pred=cv_delta, val_pred=val_delta)


def predict_ridge_residual(models: RidgeResidualModels, batch: ForecastBatch, x: pd.DataFrame) -> np.ndarray:
    hs = horizons(batch)
    out = _empty_delta_like(batch)
    X_all = x.to_numpy(float)
    for h, model in models.models.items():
        idx = np.where(hs == int(h))[0]
        if len(idx) == 0:
            continue
        pred = model.predict(X_all[idx])
        for local, global_idx in enumerate(idx):
            out[global_idx, :h, :] = pred[local].reshape(int(h), -1)
    return out


def mean_residual_delta(val: ForecastBatch, batch: ForecastBatch) -> np.ndarray:
    out = _empty_delta_like(batch)
    val_hs = horizons(val)
    batch_hs = horizons(batch)
    for h in sorted(set(batch_hs.tolist())):
        val_idx = np.where(val_hs == h)[0]
        batch_idx = np.where(batch_hs == h)[0]
        if len(val_idx) == 0:
            continue
        residuals = np.stack([valid_part(val.y_true - val.lrbn_pred, int(i), int(h)) for i in val_idx], axis=0)
        mean_res = np.nanmean(residuals, axis=0)
        out[batch_idx, :h, :] = mean_res
    return out


def mrc_cap_matrix(batch: ForecastBatch) -> np.ndarray:
    x = batch.context
    dx = np.diff(x[:, -min(32, x.shape[1]) :, :], axis=1)
    if dx.shape[1] == 0:
        scale = np.ones((len(batch.meta), 1, x.shape[2]), dtype=float)
    else:
        med = np.nanmedian(dx, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(dx - med), axis=1, keepdims=True)
        std = np.nanstd(dx, axis=1, keepdims=True)
        scale = np.where(1.4826 * mad > 1e-6, 1.4826 * mad, std)
    scale = np.maximum(scale, 1e-6)
    return np.repeat(scale, batch.lrbn_pred.shape[1], axis=1)


def apply_mrc_shrink_cap(batch: ForecastBatch, delta: np.ndarray, shrink: float, cap_mult: float) -> np.ndarray:
    cap = cap_mult * mrc_cap_matrix(batch)
    return np.clip(delta * shrink, -cap, cap)


def select_mrc_shrink_cap(
    val: ForecastBatch,
    cv_delta: np.ndarray,
    max_harm: float = 0.10,
) -> Tuple[Dict[str, float], pd.DataFrame, np.ndarray]:
    rows = []
    best_score = float("inf")
    best_params = {"shrink": 0.0, "cap_mult": 0.0}
    best_delta = np.zeros_like(cv_delta)
    for shrink in [0.01, 0.03, 0.05, 0.10, 0.20, 0.40, 0.70, 1.00]:
        for cap_mult in [0.10, 0.25, 0.50, 1.00, 2.00]:
            cand_delta = apply_mrc_shrink_cap(val, cv_delta, shrink=shrink, cap_mult=cap_mult)
            pred = val.lrbn_pred + cand_delta
            row = metric_summary("MRC-ridge-residual", pred, val)
            row["shrink"] = float(shrink)
            row["cap_mult"] = float(cap_mult)
            row["feasible"] = bool(row["harm_rate"] <= max_harm)
            # Select on validation only. Penalize unsafe harm sharply before MSE.
            score = float(row["mse_delta_pct_vs_lrbn"]) + 100.0 * max(0.0, float(row["harm_rate"]) - max_harm)
            rows.append(row)
            if score < best_score:
                best_score = score
                best_params = {"shrink": float(shrink), "cap_mult": float(cap_mult), "selection_score": score}
                best_delta = cand_delta
    grid = pd.DataFrame(rows)
    feasible = grid[grid["feasible"]].copy()
    if not feasible.empty:
        chosen = feasible.sort_values(["mse_delta_pct_vs_lrbn", "harm_rate", "shrink"]).iloc[0]
        best_params = {
            "shrink": float(chosen["shrink"]),
            "cap_mult": float(chosen["cap_mult"]),
            "selection_score": float(chosen["mse_delta_pct_vs_lrbn"]),
        }
        best_delta = apply_mrc_shrink_cap(val, cv_delta, shrink=best_params["shrink"], cap_mult=best_params["cap_mult"])
    return best_params, grid, best_delta


def train_harm_model(x_val: pd.DataFrame, harm_label: np.ndarray) -> Any:
    if len(set(harm_label.astype(int).tolist())) < 2:
        return {"constant": float(np.mean(harm_label))}
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=2026),
    )
    model.fit(x_val.to_numpy(float), harm_label.astype(int))
    return model


def predict_harm_risk(model: Any, x: pd.DataFrame) -> np.ndarray:
    if isinstance(model, dict):
        return np.full(len(x), float(model["constant"]), dtype=float)
    return model.predict_proba(x.to_numpy(float))[:, 1]


def select_abstention_threshold(
    val: ForecastBatch,
    cv_delta: np.ndarray,
    risk: np.ndarray,
    max_harm: float = 0.05,
) -> Tuple[float, pd.DataFrame]:
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    base_mse = mse_per_sample(val.lrbn_pred, val.y_true)
    for tau in thresholds:
        selected = risk <= tau
        pred = val.lrbn_pred + cv_delta * selected.reshape(-1, 1, 1)
        row = metric_summary("MRC-ridge-abstain", pred, val, selected=selected)
        row["risk_threshold"] = float(tau)
        row["feasible"] = bool(row["harm_rate"] <= max_harm)
        row["selected_base_mse"] = float(np.mean(base_mse[selected])) if selected.any() else 0.0
        rows.append(row)
    curve = pd.DataFrame(rows)
    feasible = curve[curve["feasible"]].copy()
    if feasible.empty:
        best = curve.sort_values(["mse_delta_pct_vs_lrbn", "harm_rate"]).iloc[0]
    else:
        best = feasible.sort_values(["mse_delta_pct_vs_lrbn", "harm_rate"]).iloc[0]
    return float(best["risk_threshold"]), curve


def valid_abs_errors(batch: ForecastBatch, delta: np.ndarray) -> np.ndarray:
    vals: List[np.ndarray] = []
    hs = horizons(batch)
    residual = batch.y_true - batch.lrbn_pred - delta
    for i, h in enumerate(hs):
        vals.append(np.abs(valid_flat(residual, i, int(h))))
    return np.concatenate(vals) if vals else np.array([], dtype=float)


def interval_calibration_rows(
    val: ForecastBatch,
    test: ForecastBatch,
    val_deltas: Dict[str, np.ndarray],
    test_deltas: Dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for method, vdelta in val_deltas.items():
        tdelta = test_deltas[method]
        val_abs = valid_abs_errors(val, vdelta)
        test_abs = valid_abs_errors(test, tdelta)
        for target in [0.80, 0.90]:
            q = float(np.quantile(val_abs, target))
            coverage = float(np.mean(test_abs <= q))
            rows.append(
                {
                    "method": method,
                    "target_coverage": target,
                    "val_abs_error_quantile": q,
                    "test_coverage": coverage,
                    "coverage_gap_pp": float(abs(coverage - target) * 100.0),
                    "interval_width": float(2.0 * q),
                }
            )
    return pd.DataFrame(rows)


def mrc_results(
    val: ForecastBatch,
    test: ForecastBatch,
    sra_val: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sra_test: Dict[str, Tuple[np.ndarray, np.ndarray]],
    schema: Dict[str, List[Any]],
    thresholds: Dict[str, float],
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    x_val = feature_frame(val, schema)
    x_test = feature_frame(test, schema)
    ridge = fit_ridge_residual_models(val, x_val)
    raw_test_delta = predict_ridge_residual(ridge, test, x_test)
    shrink_cap_params, shrink_cap_grid, val_ridge_delta = select_mrc_shrink_cap(val, ridge.cv_pred)
    test_delta = apply_mrc_shrink_cap(
        test,
        raw_test_delta,
        shrink=shrink_cap_params["shrink"],
        cap_mult=shrink_cap_params["cap_mult"],
    )
    val_mean_delta = mean_residual_delta(val, val)
    test_mean_delta = mean_residual_delta(val, test)
    val_point = val.lrbn_pred + val_ridge_delta
    test_point = test.lrbn_pred + test_delta
    val_mean = val.lrbn_pred + val_mean_delta
    test_mean = test.lrbn_pred + test_mean_delta

    harm_label = mse_per_sample(val_point, val.y_true) > mse_per_sample(val.lrbn_pred, val.y_true) + 1e-12
    harm_model = train_harm_model(x_val, harm_label)
    val_risk = predict_harm_risk(harm_model, x_val)
    tau, abstain_curve = select_abstention_threshold(val, val_ridge_delta, val_risk)
    test_risk = predict_harm_risk(harm_model, x_test)
    test_selected = test_risk <= tau
    val_selected = val_risk <= tau
    val_abstain = val.lrbn_pred + val_ridge_delta * val_selected.reshape(-1, 1, 1)
    test_abstain = test.lrbn_pred + test_delta * test_selected.reshape(-1, 1, 1)

    point_rows = [
        metric_summary("LRBN", test.lrbn_pred, test),
        metric_summary("MRC-mean-residual", test_mean, test),
        metric_summary("MRC-ridge-residual", test_point, test),
        metric_summary("MRC-ridge-abstain", test_abstain, test, selected=test_selected),
    ]
    for name, (pred, strength) in sra_test.items():
        point_rows.append(metric_summary(name, pred, test, selected=strength > 1e-8))
    point_df = pd.DataFrame(point_rows)
    quantile_df = interval_calibration_rows(
        val,
        test,
        {
            "LRBN-zero": _empty_delta_like(val),
            "MRC-mean-residual": val_mean_delta,
            "MRC-ridge-residual": val_ridge_delta,
        },
        {
            "LRBN-zero": _empty_delta_like(test),
            "MRC-mean-residual": test_mean_delta,
            "MRC-ridge-residual": test_delta,
        },
    )
    slice_df = slice_result_rows(
        test,
        {
            "MRC-ridge-residual": (test_point, np.ones(len(test.meta))),
            "MRC-ridge-abstain": (test_abstain, test_selected.astype(float)),
            "SRA-BP-safe": sra_test["SRA-BP-safe"],
            "SRA-BP-balanced": sra_test["SRA-BP-balanced"],
        },
        thresholds,
        schema,
    )
    ci = {
        "MRC-ridge-residual": bootstrap_ci(
            mse_per_sample(test_point, test.y_true) - mse_per_sample(test.lrbn_pred, test.y_true), n_boot=n_bootstrap
        ),
        "MRC-ridge-abstain": bootstrap_ci(
            mse_per_sample(test_abstain, test.y_true) - mse_per_sample(test.lrbn_pred, test.y_true), n_boot=n_bootstrap
        ),
    }
    point = point_df.set_index("method")
    ridge_harm = float(point.loc["MRC-ridge-residual", "harm_rate"])
    abstain_harm = float(point.loc["MRC-ridge-abstain", "harm_rate"])
    harm_drop = (ridge_harm - abstain_harm) / (ridge_harm + EPS)
    coverage_gap_max = float(quantile_df[quantile_df["method"].eq("MRC-ridge-residual")]["coverage_gap_pp"].max())
    non_sra = slice_df[(slice_df["method"].eq("MRC-ridge-abstain")) & (slice_df["slice"].eq("non_boundary"))]
    non_sra_improved = bool(len(non_sra) and float(non_sra.iloc[0]["mse_delta_pct_vs_lrbn"]) < 0.0)
    verdict = {
        "point_pass": bool(float(point.loc["MRC-ridge-abstain", "mse_delta_pct_vs_lrbn"]) <= -1.0),
        "harm_pass": bool(abstain_harm <= 0.05 or ridge_harm <= 0.10),
        "coverage_pass": bool(coverage_gap_max <= 5.0),
        "abstention_pass": bool(harm_drop >= 0.30),
        "non_sra_slice_pass": non_sra_improved,
        "safe_go": False,
        "test_threshold_leakage": False,
        "selected_threshold": tau,
        "alpha_by_horizon": ridge.alpha_by_horizon,
        "shrink_cap_params": shrink_cap_params,
    }
    verdict["safe_go"] = bool(
        verdict["point_pass"]
        and verdict["harm_pass"]
        and verdict["coverage_pass"]
        and verdict["abstention_pass"]
        and verdict["non_sra_slice_pass"]
    )
    return {
        "point": point_df,
        "quantile": quantile_df,
        "abstention": abstain_curve,
        "shrink_cap_grid": shrink_cap_grid,
        "slice": slice_df,
        "ci": ci,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# TAE
# ---------------------------------------------------------------------------


def ema_smooth(pred: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    out = pred.copy()
    for i, h in enumerate(horizons_from_pred(pred)):
        for t in range(1, h):
            out[i, t, :] = alpha * pred[i, t, :] + (1.0 - alpha) * out[i, t - 1, :]
    return out


def horizons_from_pred(pred: np.ndarray) -> np.ndarray:
    hs = []
    for i in range(pred.shape[0]):
        finite = np.isfinite(pred[i, :, 0])
        hs.append(int(finite.sum()) if finite.any() else pred.shape[1])
    return np.asarray(hs, dtype=int)


def level_bias_candidate(batch: ForecastBatch, alpha: float = 0.5, k_value: str = "H_div_4") -> np.ndarray:
    out = batch.lrbn_pred.copy()
    hs = horizons(batch)
    x = batch.context
    dx = np.diff(x[:, -min(16, x.shape[1]) :, :], axis=1)
    slope = np.nanmean(dx, axis=1, keepdims=True) if dx.shape[1] else np.zeros((len(x), 1, x.shape[2]))
    anchor_next = x[:, -1:, :] + slope
    bias = anchor_next - batch.lrbn_pred[:, :1, :]
    for i, h in enumerate(hs):
        k = max(4, h // 4) if k_value == "H_div_4" else int(k_value)
        w = np.maximum(0.0, 1.0 - np.arange(h) / float(max(k, 1))).reshape(h, 1)
        out[i, :h, :] = batch.lrbn_pred[i, :h, :] + alpha * bias[i] * w
    return out


def phase_shift_candidate(batch: ForecastBatch, shift: int = 1, blend: float = 0.5) -> np.ndarray:
    out = batch.lrbn_pred.copy()
    hs = horizons(batch)
    for i, h in enumerate(hs):
        y = batch.lrbn_pred[i, :h, :]
        shifted = np.roll(y, shift=shift, axis=0)
        if shift > 0:
            shifted[:shift, :] = y[:1, :]
        elif shift < 0:
            shifted[shift:, :] = y[-1:, :]
        out[i, :h, :] = (1.0 - blend) * y + blend * shifted
    return out


def amplitude_candidate(batch: ForecastBatch, scale: float = 0.9) -> np.ndarray:
    out = batch.lrbn_pred.copy()
    for i, h in enumerate(horizons(batch)):
        y = batch.lrbn_pred[i, :h, :]
        center = np.nanmean(y, axis=0, keepdims=True)
        out[i, :h, :] = center + scale * (y - center)
    return out


def volatility_shrink_candidate(batch: ForecastBatch, alpha: float = 0.4) -> np.ndarray:
    out = batch.lrbn_pred.copy()
    for i, h in enumerate(horizons(batch)):
        y = batch.lrbn_pred[i, :h, :]
        ema = y.copy()
        for t in range(1, h):
            ema[t, :] = alpha * y[t, :] + (1.0 - alpha) * ema[t - 1, :]
        out[i, :h, :] = ema
    return out


def build_tae_candidates(
    batch: ForecastBatch,
    sra: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, np.ndarray]:
    candidates: Dict[str, np.ndarray] = {
        "keep_lrbn": batch.lrbn_pred.copy(),
        "raw": batch.raw_pred.copy(),
        "sra_safe": sra["SRA-BP-safe"][0],
        "sra_balanced": sra["SRA-BP-balanced"][0],
        "bp_always": apply_candidate(
            batch,
            {"method": "LRBN-BP-always", "alpha": 0.5, "anchor_mode": "last", "bridge_mode": "linear"},
        ).pred,
        "level_bias": level_bias_candidate(batch, alpha=0.5),
        "level_bias_light": level_bias_candidate(batch, alpha=0.25),
        "phase_shift_plus": phase_shift_candidate(batch, shift=1, blend=0.5),
        "phase_shift_minus": phase_shift_candidate(batch, shift=-1, blend=0.5),
        "amplitude_shrink": amplitude_candidate(batch, scale=0.85),
        "amplitude_expand": amplitude_candidate(batch, scale=1.10),
        "volatility_shrink": volatility_shrink_candidate(batch, alpha=0.35),
    }
    stack = np.stack([candidates[k] for k in candidates], axis=0)
    candidates["ensemble_median"] = np.nanmedian(stack, axis=0)
    return candidates


def candidate_feature_tensor(batch: ForecastBatch, base_x: pd.DataFrame, candidates: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
    base = base_x.to_numpy(float)
    names = list(candidates.keys())
    hs = horizons(batch)
    out = np.zeros((len(batch.meta), len(names), base.shape[1] + 8), dtype=float)
    for j, name in enumerate(names):
        pred = candidates[name]
        for i, h in enumerate(hs):
            y = valid_part(pred, i, int(h))
            l = valid_part(batch.lrbn_pred, i, int(h))
            d = y - l
            dy = np.diff(y, axis=0) if h > 1 else np.zeros_like(y[:1])
            dl = np.diff(l, axis=0) if h > 1 else np.zeros_like(l[:1])
            extra = np.array(
                [
                    finite_mean(d),
                    finite_std(d),
                    finite_norm(d) / math.sqrt(max(1, d.size)),
                    finite_mean(y[: min(8, h)] - batch.context[i, -1:, :]),
                    finite_std(dy),
                    finite_std(dy) / (finite_std(dl) + 1e-6),
                    finite_mean(y),
                    finite_std(y),
                ],
                dtype=float,
            )
            out[i, j, :] = np.concatenate([base[i], extra])
    return out, names


def oracle_best(candidates: Dict[str, np.ndarray], batch: ForecastBatch) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    names = list(candidates.keys())
    mse_stack = np.stack([mse_per_sample(candidates[n], batch.y_true) for n in names], axis=1)
    best_idx = np.argmin(mse_stack, axis=1)
    best_mse = mse_stack[np.arange(len(batch.meta)), best_idx]
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    gain = base_mse - best_mse
    return names, best_idx, best_mse, gain


def failure_mode_labels(batch: ForecastBatch, schema: Dict[str, List[Any]], thresholds: Dict[str, float]) -> np.ndarray:
    feats = compute_sra_features(batch.context, batch.raw_pred, batch.lrbn_pred, batch.y_true)
    labels = []
    hs = horizons(batch)
    for i, h in enumerate(hs):
        residual = valid_part(batch.y_true - batch.lrbn_pred, i, int(h))
        pred = valid_part(batch.lrbn_pred, i, int(h))
        res_mean = abs(finite_mean(residual))
        res_std = finite_std(residual)
        res_diff_std = finite_std(np.diff(residual, axis=0)) if h > 1 else 0.0
        pred_diff_std = finite_std(np.diff(pred, axis=0)) if h > 1 else 0.0
        if feats["g_l"][i] >= thresholds["g_l_q75"] and feats["repair_ratio"][i] <= thresholds["repair_low"]:
            labels.append("boundary")
        elif res_mean > 0.75 * (res_std + 1e-6):
            labels.append("level")
        elif res_diff_std > 1.5 * (pred_diff_std + 1e-6):
            labels.append("volatility")
        elif feats["jump_support"][i] >= 0.5:
            labels.append("turn_or_phase")
        else:
            labels.append("mixed_or_none")
    return np.asarray(labels, dtype=object)


def tae_results(
    val: ForecastBatch,
    test: ForecastBatch,
    sra_val: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sra_test: Dict[str, Tuple[np.ndarray, np.ndarray]],
    schema: Dict[str, List[Any]],
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    x_val = feature_frame(val, schema)
    x_test = feature_frame(test, schema)
    cand_val = build_tae_candidates(val, sra_val)
    cand_test = build_tae_candidates(test, sra_test)
    names, oracle_idx_val, _, oracle_gain_val = oracle_best(cand_val, val)
    _, oracle_idx_test, oracle_mse_test, oracle_gain_test = oracle_best(cand_test, test)

    router = RandomForestClassifier(n_estimators=200, min_samples_leaf=8, random_state=2026, class_weight="balanced")
    router.fit(x_val.to_numpy(float), oracle_idx_val)
    proba = router.predict_proba(x_test.to_numpy(float))
    classes = router.classes_.astype(int)
    top_order = np.argsort(-proba, axis=1)
    top1 = classes[top_order[:, 0]]
    top2_hit = np.array([oracle_idx_test[i] in classes[top_order[i, : min(2, len(classes))]] for i in range(len(test.meta))])
    router_pred = select_candidate_by_index(cand_test, names, top1)

    cand_feat_val, cand_names = candidate_feature_tensor(val, x_val, cand_val)
    cand_feat_test, _ = candidate_feature_tensor(test, x_test, cand_test)
    val_mse_stack = np.stack([mse_per_sample(cand_val[n], val.y_true) for n in cand_names], axis=1)
    val_gain_stack = mse_per_sample(val.lrbn_pred, val.y_true).reshape(-1, 1) - val_mse_stack
    ranker = RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=2026)
    ranker.fit(cand_feat_val.reshape(-1, cand_feat_val.shape[-1]), val_gain_stack.reshape(-1))
    score_test = ranker.predict(cand_feat_test.reshape(-1, cand_feat_test.shape[-1])).reshape(len(test.meta), len(cand_names))
    rank_idx = np.argmax(score_test, axis=1)
    ranker_pred = select_candidate_by_index(cand_test, cand_names, rank_idx)
    test_mse_stack = np.stack([mse_per_sample(cand_test[n], test.y_true) for n in cand_names], axis=1)
    test_gain_stack = mse_per_sample(test.lrbn_pred, test.y_true).reshape(-1, 1) - test_mse_stack
    spearman = spearman_safe(score_test.reshape(-1), test_gain_stack.reshape(-1))

    labels_val = failure_mode_labels(val, schema, thresholds)
    labels_test = failure_mode_labels(test, schema, thresholds)
    fm_clf = RandomForestClassifier(n_estimators=200, min_samples_leaf=8, random_state=2026, class_weight="balanced")
    fm_clf.fit(x_val.to_numpy(float), labels_val)
    labels_pred = fm_clf.predict(x_test.to_numpy(float))

    decision_rows = [
        metric_summary("LRBN", test.lrbn_pred, test),
        metric_summary("TAE-oracle-best", reconstruct_oracle_pred(cand_test, names, oracle_idx_test), test),
        metric_summary("TAE-router", router_pred, test),
        metric_summary("TAE-ranker", ranker_pred, test),
    ]
    for name in ["sra_safe", "sra_balanced", "bp_always", "level_bias", "amplitude_shrink", "volatility_shrink", "ensemble_median"]:
        decision_rows.append(metric_summary(name, cand_test[name], test))
    decision_df = pd.DataFrame(decision_rows)
    lrbn_mse = float(decision_df[decision_df["method"].eq("LRBN")]["mse"].iloc[0])
    sra_bal_mse = float(decision_df[decision_df["method"].eq("sra_balanced")]["mse"].iloc[0])
    oracle_mse = float(np.mean(oracle_mse_test))
    router_mse = float(decision_df[decision_df["method"].eq("TAE-router")]["mse"].iloc[0])
    ranker_mse = float(decision_df[decision_df["method"].eq("TAE-ranker")]["mse"].iloc[0])
    oracle_gain_pct = safe_pct(oracle_mse, lrbn_mse)
    oracle_extra_vs_sra_pct = safe_pct(oracle_mse, sra_bal_mse)
    router_gain_fraction = (lrbn_mse - router_mse) / (lrbn_mse - oracle_mse + EPS)
    ranker_gain_fraction = (lrbn_mse - ranker_mse) / (lrbn_mse - oracle_mse + EPS)
    oracle_counts = pd.Series([names[i] for i in oracle_idx_test]).value_counts().rename_axis("candidate").reset_index(name="oracle_count")
    oracle_counts["oracle_rate"] = oracle_counts["oracle_count"] / len(test.meta)
    candidate_rows = []
    for j, name in enumerate(cand_names):
        mm = test_mse_stack[:, j]
        base = mse_per_sample(test.lrbn_pred, test.y_true)
        candidate_rows.append(
            {
                "candidate": name,
                "mse": float(np.mean(mm)),
                "mse_delta_pct_vs_lrbn": safe_pct(float(np.mean(mm)), float(np.mean(base))),
                "win_rate": float(np.mean(mm < base)),
                "harm_rate": float(np.mean(mm > base + 1e-12)),
                "oracle_count": int((oracle_idx_test == j).sum()),
            }
        )
    candidate_df = pd.DataFrame(candidate_rows)
    non_boundary_effective = candidate_df[
        candidate_df["candidate"].isin(["level_bias", "level_bias_light", "amplitude_shrink", "amplitude_expand", "volatility_shrink", "phase_shift_plus", "phase_shift_minus"])
        & (candidate_df["mse_delta_pct_vs_lrbn"] < 0.0)
    ]
    verdict = {
        "oracle_gain_pct_vs_lrbn": oracle_gain_pct,
        "oracle_extra_pct_vs_sra_balanced": oracle_extra_vs_sra_pct,
        "router_gain_fraction": float(router_gain_fraction),
        "ranker_gain_fraction": float(ranker_gain_fraction),
        "router_top1_accuracy": float(accuracy_score(oracle_idx_test, top1)),
        "router_top2_hit": float(np.mean(top2_hit)),
        "ranker_score_gain_spearman": float(spearman),
        "router_harm_rate": float(decision_df[decision_df["method"].eq("TAE-router")]["harm_rate"].iloc[0]),
        "ranker_harm_rate": float(decision_df[decision_df["method"].eq("TAE-ranker")]["harm_rate"].iloc[0]),
        "non_boundary_effective_candidates": int(len(non_boundary_effective)),
        "compact_go": False,
        "test_threshold_leakage": False,
    }
    verdict["compact_go"] = bool(
        verdict["oracle_gain_pct_vs_lrbn"] <= -4.0
        and verdict["oracle_extra_pct_vs_sra_balanced"] <= -1.0
        and max(verdict["router_gain_fraction"], verdict["ranker_gain_fraction"]) >= 0.40
        and verdict["router_top2_hit"] >= 0.65
        and min(verdict["router_harm_rate"], verdict["ranker_harm_rate"]) <= 0.15
        and verdict["non_boundary_effective_candidates"] >= 2
    )
    failure_df = pd.DataFrame(
        [
            {
                "split": "test",
                "failure_mode_accuracy": float(accuracy_score(labels_test, labels_pred)),
                "failure_mode_macro_f1": float(f1_score(labels_test, labels_pred, average="macro")),
                "n_classes_test": int(len(set(labels_test.tolist()))),
            }
        ]
    )
    router_df = pd.DataFrame(
        [
            {
                "model": "router_classifier",
                "top1_accuracy": verdict["router_top1_accuracy"],
                "top2_hit": verdict["router_top2_hit"],
                "selected_mse_delta_pct_vs_lrbn": float(
                    decision_df[decision_df["method"].eq("TAE-router")]["mse_delta_pct_vs_lrbn"].iloc[0]
                ),
                "selected_harm": verdict["router_harm_rate"],
            },
            {
                "model": "candidate_ranker",
                "score_gain_spearman": verdict["ranker_score_gain_spearman"],
                "selected_mse_delta_pct_vs_lrbn": float(
                    decision_df[decision_df["method"].eq("TAE-ranker")]["mse_delta_pct_vs_lrbn"].iloc[0]
                ),
                "selected_harm": verdict["ranker_harm_rate"],
            },
        ]
    )
    oracle_df = pd.DataFrame(
        [
            {
                "metric": "oracle_best",
                "lrbn_mse": lrbn_mse,
                "sra_balanced_mse": sra_bal_mse,
                "oracle_mse": oracle_mse,
                "oracle_gain_pct_vs_lrbn": oracle_gain_pct,
                "oracle_extra_pct_vs_sra_balanced": oracle_extra_vs_sra_pct,
                "mean_oracle_gain": float(np.mean(oracle_gain_test)),
            }
        ]
    )
    return {
        "candidate_table": candidate_df.merge(oracle_counts, how="left", on="candidate").fillna({"oracle_count_y": 0, "oracle_rate": 0.0}),
        "oracle": oracle_df,
        "failure": failure_df,
        "router": router_df,
        "decision": decision_df,
        "verdict": verdict,
    }


def select_candidate_by_index(candidates: Dict[str, np.ndarray], names: List[str], idx: np.ndarray) -> np.ndarray:
    first = next(iter(candidates.values()))
    out = np.full_like(first, np.nan)
    for j, name in enumerate(names):
        mask = idx == j
        if mask.any():
            out[mask] = candidates[name][mask]
    return out


def reconstruct_oracle_pred(candidates: Dict[str, np.ndarray], names: List[str], idx: np.ndarray) -> np.ndarray:
    return select_candidate_by_index(candidates, names, idx)


def spearman_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return 0.0
    r = spearmanr(a[mask], b[mask]).correlation
    return 0.0 if r is None or not np.isfinite(r) else float(r)


# ---------------------------------------------------------------------------
# FOMC
# ---------------------------------------------------------------------------


def band_masks(n_freq: int, n_bands: int = 4) -> List[np.ndarray]:
    idx = np.arange(n_freq)
    splits = np.array_split(idx, n_bands)
    return [np.isin(idx, s) for s in splits]


def spectral_band_autocorr(batch: ForecastBatch) -> pd.DataFrame:
    rows = []
    for keys, g in batch.meta.assign(row_index=np.arange(len(batch.meta))).groupby(["dataset", "backbone", "horizon", "seed"], observed=True):
        order = g.sort_values("sample_id", key=lambda s: s.map(parse_sample_index))
        indices = order["row_index"].to_numpy(int)
        h = int(keys[2])
        residuals = np.stack([valid_part(batch.y_true - batch.lrbn_pred, int(i), h)[:, 0] for i in indices], axis=0)
        fft = np.fft.rfft(residuals, axis=1)
        energy = np.abs(fft) ** 2
        for b, mask in enumerate(band_masks(energy.shape[1], n_bands=4)):
            series = energy[:, mask].mean(axis=1)
            if len(series) > 2 and np.std(series[:-1]) > 1e-12 and np.std(series[1:]) > 1e-12:
                corr = float(np.corrcoef(series[:-1], series[1:])[0, 1])
            else:
                corr = 0.0
            rows.append(
                {
                    "dataset": keys[0],
                    "backbone": keys[1],
                    "horizon": int(keys[2]),
                    "seed": int(keys[3]),
                    "band": int(b),
                    "lag1_autocorr": corr,
                    "mean_energy": float(np.mean(series)),
                }
            )
    return pd.DataFrame(rows)


def residual_buffer_correction(buffer: np.ndarray, h: int, method: str, band_weights: Optional[np.ndarray] = None) -> np.ndarray:
    if buffer.size == 0:
        return np.zeros((h, 1), dtype=float)
    residuals = buffer[:, :h, :]
    if method == "rolling_mean_residual":
        return np.nanmean(residuals, axis=0)
    if method == "time_ema_residual":
        weights = np.exp(np.linspace(-2.0, 0.0, residuals.shape[0])).reshape(-1, 1, 1)
        weights = weights / weights.sum()
        return np.nansum(residuals * weights, axis=0)
    if method == "spectral_adapter":
        fft = np.fft.rfft(residuals[:, :, 0], axis=1)
        mean_fft = np.nanmean(fft, axis=0)
        if band_weights is not None:
            masks = band_masks(len(mean_fft), n_bands=len(band_weights))
            w = np.ones(len(mean_fft), dtype=float)
            for i, mask in enumerate(masks):
                w[mask] = band_weights[i]
            mean_fft = mean_fft * w
        corr = np.fft.irfft(mean_fft, n=h).reshape(h, 1)
        return corr
    raise ValueError(f"Unknown online correction method: {method}")


def calibrate_spectral_band_weights(val: ForecastBatch) -> np.ndarray:
    ac = spectral_band_autocorr(val)
    if ac.empty:
        return np.ones(4, dtype=float)
    by_band = ac.groupby("band", observed=True)["lag1_autocorr"].mean()
    weights = np.array([max(0.0, float(by_band.get(i, 0.0))) for i in range(4)], dtype=float)
    if weights.max() <= 1e-8:
        return np.zeros(4, dtype=float)
    return weights / max(1.0, weights.max())


def online_adapter_eval(
    val: ForecastBatch,
    test: ForecastBatch,
    buffer_size: int = 128,
    band_weights: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    methods = ["no_update", "rolling_mean_residual", "time_ema_residual", "spectral_adapter"]
    pred_by_method = {m: test.lrbn_pred.copy() for m in methods}
    q90_by_method: Dict[str, List[float]] = {m: [] for m in methods}
    coverage_by_method: Dict[str, List[float]] = {m: [] for m in methods}
    protocol_violations: List[Tuple[str, int, int]] = []

    test_with_idx = test.meta.assign(row_index=np.arange(len(test.meta)))
    val_with_idx = val.meta.assign(row_index=np.arange(len(val.meta)))
    for keys, gtest in test_with_idx.groupby(["dataset", "backbone", "horizon", "seed"], observed=True):
        dataset, backbone, horizon, seed = keys
        h = int(horizon)
        gval = val_with_idx[
            val_with_idx["dataset"].eq(dataset)
            & val_with_idx["backbone"].eq(backbone)
            & val_with_idx["horizon"].eq(horizon)
            & val_with_idx["seed"].eq(seed)
        ].sort_values("sample_id", key=lambda s: s.map(parse_sample_index))
        gtest = gtest.sort_values("sample_id", key=lambda s: s.map(parse_sample_index))
        buffer_indices = gval["row_index"].to_numpy(int)[-buffer_size:]
        buffer_res = np.stack([valid_part(val.y_true - val.lrbn_pred, int(i), h) for i in buffer_indices], axis=0)
        matured_test_residuals: List[np.ndarray] = []
        for local_t, (_, row) in enumerate(gtest.iterrows()):
            idx = int(row["row_index"])
            event_time = local_t
            # Only test labels with label_time < current event_time can mature.
            matured = [r for j, r in matured_test_residuals if j + h < event_time]
            if matured:
                mature_stack = np.stack(matured, axis=0)
                current_buffer = np.concatenate([buffer_res, mature_stack], axis=0)[-buffer_size:]
            else:
                current_buffer = buffer_res
            for method in methods:
                if method == "no_update":
                    correction = np.zeros((h, 1), dtype=float)
                else:
                    correction = residual_buffer_correction(current_buffer, h, method, band_weights)
                pred_by_method[method][idx, :h, :] = test.lrbn_pred[idx, :h, :] + correction
                abs_scores = np.abs(current_buffer[:, :h, :]).reshape(-1)
                q90 = float(np.quantile(abs_scores, 0.90)) if len(abs_scores) else 0.0
                q90_by_method[method].append(q90)
                err = np.abs(valid_part(test.y_true - pred_by_method[method], idx, h)).reshape(-1)
                coverage_by_method[method].append(float(np.mean(err <= q90)))
            matured_test_residuals.append((local_t, valid_part(test.y_true - test.lrbn_pred, idx, h)))

    rows = []
    for method, pred in pred_by_method.items():
        row = metric_summary(method, pred, test)
        row["buffer_size"] = int(buffer_size)
        row["mean_q90_width"] = float(2.0 * np.mean(q90_by_method[method])) if q90_by_method[method] else 0.0
        row["mean_pointwise_coverage90"] = float(np.mean(coverage_by_method[method])) if coverage_by_method[method] else 0.0
        rows.append(row)
    adapter_df = pd.DataFrame(rows)
    conformal_df = adapter_df[
        ["method", "buffer_size", "mean_pointwise_coverage90", "mean_q90_width", "mse_delta_pct_vs_lrbn", "harm_rate"]
    ].copy()
    guard = {
        "protocol_violations": protocol_violations,
        "n_protocol_violations": len(protocol_violations),
        "test_threshold_leakage": False,
        "label_time_rule": "test labels are added only if local_index + horizon < current_event_index; validation buffer is treated as historical matured data",
    }
    return adapter_df, conformal_df, guard


def fomc_results(val: ForecastBatch, test: ForecastBatch) -> Dict[str, Any]:
    ac = spectral_band_autocorr(val)
    weights = calibrate_spectral_band_weights(val)
    adapter_df, conformal_df, guard = online_adapter_eval(val, test, buffer_size=128, band_weights=weights)
    rows = adapter_df.set_index("method")
    spectral_delta = float(rows.loc["spectral_adapter", "mse_delta_pct_vs_lrbn"])
    rolling_delta = float(rows.loc["rolling_mean_residual", "mse_delta_pct_vs_lrbn"])
    spectral_harm = float(rows.loc["spectral_adapter", "harm_rate"])
    coverage_gap = abs(float(rows.loc["spectral_adapter", "mean_pointwise_coverage90"]) - 0.90) * 100.0
    verdict = {
        "spectral_delta_pct_vs_lrbn": spectral_delta,
        "rolling_delta_pct_vs_lrbn": rolling_delta,
        "spectral_minus_rolling_pct": float(spectral_delta - rolling_delta),
        "spectral_harm": spectral_harm,
        "coverage_gap_pp": float(coverage_gap),
        "mean_spectral_autocorr": float(ac["lag1_autocorr"].mean()) if not ac.empty else 0.0,
        "band_weights": [float(x) for x in weights],
        "protocol_guard_pass": guard["n_protocol_violations"] == 0,
        "compact_go": False,
        "test_threshold_leakage": False,
    }
    verdict["compact_go"] = bool(
        spectral_delta <= -1.0
        and spectral_delta <= rolling_delta - 0.5
        and spectral_harm <= 0.10
        and coverage_gap <= 5.0
        and verdict["protocol_guard_pass"]
    )
    return {
        "spectral_autocorr": ac,
        "adapter": adapter_df,
        "conformal": conformal_df,
        "guard": guard,
        "verdict": verdict,
    }


def df_to_md(df: pd.DataFrame, max_rows: int = 20) -> str:
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
