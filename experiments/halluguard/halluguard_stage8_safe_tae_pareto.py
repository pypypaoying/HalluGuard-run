#!/usr/bin/env python
"""Stage 8 Safe-TAE Pareto expansion validation.

This module reuses the Stage 7 Safe-TAE expert pool and pairwise heads, then
tests whether expert-specific thresholds, expert-specific residual strengths,
mechanism slices, and config-risk veto can recover more oracle value while
remaining validation-only and low-harm.
"""

from __future__ import annotations

import copy
import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from halluguard_lrbn_bp import ForecastBatch, load_forecast_batch_from_metrics, mae_per_sample, mse_per_sample
from halluguard_stage6_mechanism import feature_frame, feature_schema, safe_pct, slice_thresholds
from halluguard_stage7_safe_tae import (
    ExpertCandidate,
    build_candidate_pool,
    build_mrc_artifacts,
    candidate_dict,
    candidate_metric_rows,
    compute_probabilities,
    df_to_md,
    fit_heads,
    load_stage3_params,
    oracle_best,
    select_basic_sra_params,
    split_batch,
    stratified_inner_split,
    subset_candidates,
    write_json,
)

EPS = 1e-12


@dataclass
class ParetoPolicy:
    variant: str
    target: str
    default_tau_gain: float
    default_tau_harm: float
    default_lambda: float
    beta_harm: float = 1.0
    tau_gain: Dict[str, float] = field(default_factory=dict)
    tau_harm: Dict[str, float] = field(default_factory=dict)
    lambdas: Dict[str, float] = field(default_factory=dict)
    allowed_slices: Dict[str, List[str]] = field(default_factory=dict)
    blocked_configs: Dict[str, List[str]] = field(default_factory=dict)
    allow_aggressive: bool = False

    def tg(self, expert: str) -> float:
        return float(self.tau_gain.get(expert, self.default_tau_gain))

    def th(self, expert: str) -> float:
        return float(self.tau_harm.get(expert, self.default_tau_harm))

    def lam(self, expert: str) -> float:
        return float(self.lambdas.get(expert, self.default_lambda))


def config_keys(meta: pd.DataFrame) -> pd.Series:
    return meta["dataset"].astype(str) + "/" + meta["backbone"].astype(str) + "/" + meta["horizon"].astype(str)


def policy_copy(policy: ParetoPolicy, variant: Optional[str] = None, target: Optional[str] = None) -> ParetoPolicy:
    out = copy.deepcopy(policy)
    if variant is not None:
        out.variant = variant
    if target is not None:
        out.target = target
    return out


def is_enabled(candidate: ExpertCandidate, policy: ParetoPolicy) -> bool:
    return bool(candidate.deployable or policy.allow_aggressive)


def stage8_slice_thresholds(val: ForecastBatch, schema: Dict[str, List[Any]]) -> Dict[str, float]:
    x = feature_frame(val, schema)
    q = slice_thresholds(val)
    q.update(
        {
            "context_diff_std_q75": float(x["context_diff_std"].quantile(0.75)),
            "pred_context_var_ratio_q25": float(x["pred_context_var_ratio"].quantile(0.25)),
            "pred_context_var_ratio_q75": float(x["pred_context_var_ratio"].quantile(0.75)),
            "boundary_gap_q25": float(x["boundary_gap_lrbn"].quantile(0.25)),
            "repair_q75": float(x["repair_ratio"].quantile(0.75)),
        }
    )
    return q


def primary_slice_labels(batch: ForecastBatch, schema: Dict[str, List[Any]], q: Mapping[str, float]) -> pd.Series:
    x = feature_frame(batch, schema)
    labels = np.array(["default"] * len(batch.meta), dtype=object)
    high_gap = x["boundary_gap_lrbn"] >= float(q["g_l_q75"])
    low_repair = x["repair_ratio"] <= float(q["repair_low"])
    low_gap = x["boundary_gap_lrbn"] <= float(q["boundary_gap_q25"])
    high_repair = x["repair_ratio"] >= float(q["repair_q75"])
    high_vol = x["context_diff_std"] >= float(q["context_diff_std_q75"])
    amp = x["pred_context_var_ratio"]
    amp_mismatch = (amp <= float(q["pred_context_var_ratio_q25"])) | (amp >= float(q["pred_context_var_ratio_q75"]))
    known = (
        batch.meta["dataset"].astype(str).eq("ETTm1")
        & batch.meta["backbone"].astype(str).eq("DLinear")
        & batch.meta["horizon"].astype(int).eq(192)
    )

    labels[high_gap] = "q4_boundary"
    labels[high_gap & low_repair] = "high_gap_low_repair"
    labels[low_gap & high_repair] = "low_gap_high_repair"
    labels[high_vol] = "high_volatility"
    labels[amp_mismatch] = "amplitude_mismatch"
    labels[~high_gap & ~high_vol & ~amp_mismatch] = "non_boundary"
    labels[known] = "known_harmed_config"
    return pd.Series(labels, index=batch.meta.index, name="mechanism_slice")


def stage8_slice_masks(batch: ForecastBatch, schema: Dict[str, List[Any]], q: Mapping[str, float]) -> Dict[str, np.ndarray]:
    x = feature_frame(batch, schema)
    high_gap = x["boundary_gap_lrbn"] >= float(q["g_l_q75"])
    low_repair = x["repair_ratio"] <= float(q["repair_low"])
    low_gap = x["boundary_gap_lrbn"] <= float(q["boundary_gap_q25"])
    high_repair = x["repair_ratio"] >= float(q["repair_q75"])
    high_vol = x["context_diff_std"] >= float(q["context_diff_std_q75"])
    amp = x["pred_context_var_ratio"]
    amp_mismatch = (amp <= float(q["pred_context_var_ratio_q25"])) | (amp >= float(q["pred_context_var_ratio_q75"]))
    known = (
        batch.meta["dataset"].astype(str).eq("ETTm1")
        & batch.meta["backbone"].astype(str).eq("DLinear")
        & batch.meta["horizon"].astype(int).eq(192)
    ).to_numpy(bool)
    return {
        "overall": np.ones(len(batch.meta), dtype=bool),
        "high_gap_low_repair": (high_gap & low_repair).to_numpy(bool),
        "q4_boundary": high_gap.to_numpy(bool),
        "low_gap_high_repair": (low_gap & high_repair).to_numpy(bool),
        "non_boundary": (~high_gap).to_numpy(bool),
        "high_volatility": high_vol.to_numpy(bool),
        "amplitude_mismatch": amp_mismatch.to_numpy(bool),
        "known_harmed_config": known,
    }


def metric_row(
    variant: str,
    pred: np.ndarray,
    batch: ForecastBatch,
    selected: Optional[np.ndarray] = None,
    oracle_mse: Optional[np.ndarray] = None,
    ci: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    method_mse = mse_per_sample(pred, batch.y_true)
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    method_mae = mae_per_sample(pred, batch.y_true)
    base_mae = mae_per_sample(batch.lrbn_pred, batch.y_true)
    delta = method_mse - base_mse
    wins = delta < 0.0
    harms = delta > 1e-12
    meta = batch.meta.assign(row_index=np.arange(len(batch.meta)))
    config_harms = []
    config_improved = []
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
        "harm_rate": float(np.mean(harms)),
        "win_rate": float(np.mean(wins)),
        "max_config_harm": float(np.max(config_harms)) if config_harms else float(np.mean(harms)),
        "config_improved_ratio": float(np.mean(config_improved)) if config_improved else float("nan"),
        "improved_configs": int(np.sum(config_improved)),
        "total_configs": int(len(config_improved)),
        "test_threshold_leakage": False,
    }
    if selected is not None:
        selected = np.asarray(selected, dtype=bool)
        row["coverage"] = float(np.mean(selected))
        row["selected_count"] = int(np.sum(selected))
        row["selected_harm_rate"] = float(np.mean(harms[selected])) if selected.any() else 0.0
    else:
        row["coverage"] = 0.0
        row["selected_count"] = 0
        row["selected_harm_rate"] = 0.0
    if oracle_mse is not None:
        denom = float(np.mean(base_mse - oracle_mse))
        row["oracle_gain_fraction"] = float(np.mean(base_mse - method_mse) / (denom + EPS))
    else:
        row["oracle_gain_fraction"] = float("nan")
    if ci is not None:
        row["ci95_low_mse_delta"] = ci["ci_low"]
        row["ci95_high_mse_delta"] = ci["ci_high"]
    return row


def bootstrap_ci(pred: np.ndarray, batch: ForecastBatch, n_bootstrap: int, seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    delta = mse_per_sample(pred, batch.y_true) - mse_per_sample(batch.lrbn_pred, batch.y_true)
    vals = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(delta), size=len(delta))
        vals[i] = float(np.mean(delta[idx]))
    return {"mean": float(np.mean(delta)), "ci_low": float(np.quantile(vals, 0.025)), "ci_high": float(np.quantile(vals, 0.975))}


def per_config_rows(variant: str, pred: np.ndarray, selected: np.ndarray, batch: ForecastBatch) -> List[Dict[str, Any]]:
    method_mse = mse_per_sample(pred, batch.y_true)
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    method_mae = mae_per_sample(pred, batch.y_true)
    base_mae = mae_per_sample(batch.lrbn_pred, batch.y_true)
    meta = batch.meta.assign(row_index=np.arange(len(batch.meta)))
    rows: List[Dict[str, Any]] = []
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
                "config_key": f"{keys[0]}/{keys[1]}/{int(keys[2])}",
                "n": int(len(idx)),
                "mse": float(np.mean(method_mse[idx])),
                "mae": float(np.mean(method_mae[idx])),
                "mse_delta_pct_vs_lrbn": safe_pct(float(np.mean(method_mse[idx])), float(np.mean(base_mse[idx]))),
                "mae_delta_pct_vs_lrbn": safe_pct(float(np.mean(method_mae[idx])), float(np.mean(base_mae[idx]))),
                "harm_rate": float(np.mean(delta > 1e-12)),
                "win_rate": float(np.mean(delta < 0.0)),
                "coverage": float(np.mean(np.asarray(selected, dtype=bool)[idx])) if selected is not None else 0.0,
            }
        )
    return rows


def slice_rows(
    variant: str,
    pred: np.ndarray,
    selected: np.ndarray,
    batch: ForecastBatch,
    masks: Mapping[str, np.ndarray],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    method_mse = mse_per_sample(pred, batch.y_true)
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        delta = method_mse[mask] - base_mse[mask]
        rows.append(
            {
                "variant": variant,
                "slice": name,
                "n": int(mask.sum()),
                "mse": float(np.mean(method_mse[mask])),
                "lrbn_mse": float(np.mean(base_mse[mask])),
                "mse_delta_pct_vs_lrbn": safe_pct(float(np.mean(method_mse[mask])), float(np.mean(base_mse[mask]))),
                "harm_rate": float(np.mean(delta > 1e-12)),
                "win_rate": float(np.mean(delta < 0.0)),
                "coverage": float(np.mean(np.asarray(selected, dtype=bool)[mask])) if selected is not None else 0.0,
            }
        )
    return rows


def policy_score(metrics: Mapping[str, float], target: str, low_gap_harm: float = 0.0) -> float:
    if target == "safe":
        return (
            float(metrics["mse_delta_pct_vs_lrbn"])
            + 100.0 * max(0.0, float(metrics["harm_rate"]) - 0.03)
            + 100.0 * max(0.0, float(metrics["max_config_harm"]) - 0.10)
            + 30.0 * max(0.0, low_gap_harm - 0.02)
        )
    if target == "balanced":
        return (
            float(metrics["mse_delta_pct_vs_lrbn"])
            + 50.0 * max(0.0, float(metrics["harm_rate"]) - 0.10)
            + 50.0 * max(0.0, float(metrics["max_config_harm"]) - 0.20)
        )
    return float(metrics["mse_delta_pct_vs_lrbn"]) + 25.0 * max(0.0, float(metrics["harm_rate"]) - 0.15)


def feasible(metrics: Mapping[str, float], target: str, low_gap_harm: float = 0.0) -> bool:
    if target == "safe":
        return bool(metrics["harm_rate"] <= 0.03 and metrics["max_config_harm"] <= 0.10 and low_gap_harm <= 0.02)
    if target == "balanced":
        return bool(metrics["harm_rate"] <= 0.10 and metrics["max_config_harm"] <= 0.20)
    return bool(metrics["harm_rate"] <= 0.15 and metrics["max_config_harm"] <= 0.30)


def config_allowed(expert: str, meta: pd.DataFrame, policy: ParetoPolicy) -> np.ndarray:
    blocked = set(policy.blocked_configs.get(expert, []))
    if not blocked:
        return np.ones(len(meta), dtype=bool)
    return ~config_keys(meta).isin(blocked).to_numpy(bool)


def slice_allowed(expert: str, slice_labels: pd.Series, policy: ParetoPolicy) -> np.ndarray:
    allowed = policy.allowed_slices.get(expert)
    if not allowed:
        return np.ones(len(slice_labels), dtype=bool)
    return slice_labels.isin(allowed).to_numpy(bool)


def apply_policy(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    probs: Mapping[str, Any],
    policy: ParetoPolicy,
    slice_labels: Optional[pd.Series] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    n = len(batch.meta)
    out = batch.lrbn_pred.copy()
    selected = np.array(["keep_lrbn"] * n, dtype=object)
    selected_lambda = np.zeros(n, dtype=float)
    selected_score = np.full(n, -np.inf, dtype=float)
    selected_tier = np.array(["default"] * n, dtype=object)
    for c in candidates:
        if c.name == "keep_lrbn" or not is_enabled(c, policy) or c.name not in probs["p_gain"]:
            continue
        p_gain = np.asarray(probs["p_gain"][c.name], dtype=float)
        p_harm = np.asarray(probs["p_harm"][c.name], dtype=float)
        eligible = (p_gain >= policy.tg(c.name)) & (p_harm <= policy.th(c.name))
        if slice_labels is not None:
            eligible &= slice_allowed(c.name, slice_labels, policy)
        eligible &= config_allowed(c.name, batch.meta, policy)
        utility = p_gain - policy.beta_harm * p_harm
        utility[~eligible] = -np.inf
        take = utility > selected_score
        if take.any():
            lam = policy.lam(c.name)
            out[take] = batch.lrbn_pred[take] + lam * (c.pred[take] - batch.lrbn_pred[take])
            selected[take] = c.name
            selected_lambda[take] = lam
            selected_score[take] = utility[take]
            selected_tier[take] = c.tier
    decisions = batch.meta[["dataset", "backbone", "horizon", "seed", "sample_id"]].reset_index(drop=True).copy()
    decisions["variant"] = policy.variant
    decisions["selected_expert"] = selected
    decisions["selected_tier"] = selected_tier
    decisions["selected_lambda"] = selected_lambda
    decisions["selected_score"] = np.where(np.isfinite(selected_score), selected_score, 0.0)
    return out, decisions


def evaluate_policy_on_calib(
    policy: ParetoPolicy,
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    probs: Mapping[str, Any],
    oracle_mse: np.ndarray,
    slice_labels: Optional[pd.Series],
    masks: Mapping[str, np.ndarray],
) -> Tuple[Dict[str, Any], np.ndarray, pd.DataFrame]:
    pred, decisions = apply_policy(batch, candidates, probs, policy, slice_labels)
    row = metric_row(
        policy.variant,
        pred,
        batch,
        selected=decisions["selected_expert"].to_numpy(str) != "keep_lrbn",
        oracle_mse=oracle_mse,
    )
    sl = pd.DataFrame(slice_rows(policy.variant, pred, decisions["selected_expert"].to_numpy(str) != "keep_lrbn", batch, masks))
    low = sl[sl["slice"].eq("low_gap_high_repair")]
    low_harm = float(low["harm_rate"].iloc[0]) if not low.empty else 0.0
    row["low_gap_high_repair_harm"] = low_harm
    row["calibration_score"] = policy_score(row, policy.target, low_gap_harm=low_harm)
    row["calibration_feasible"] = feasible(row, policy.target, low_gap_harm=low_harm)
    return row, pred, decisions


def deployable_experts(candidates: Sequence[ExpertCandidate], allow_aggressive: bool = False) -> List[str]:
    return [c.name for c in candidates if c.name != "keep_lrbn" and (c.deployable or allow_aggressive)]


def initial_policy(variant: str, target: str, candidates: Sequence[ExpertCandidate], allow_aggressive: bool = False) -> ParetoPolicy:
    if target == "safe":
        return ParetoPolicy(variant, target, default_tau_gain=0.50, default_tau_harm=0.20, default_lambda=0.50, beta_harm=1.0, allow_aggressive=allow_aggressive)
    if target == "balanced":
        return ParetoPolicy(variant, target, default_tau_gain=0.45, default_tau_harm=0.20, default_lambda=0.75, beta_harm=1.0, allow_aggressive=allow_aggressive)
    return ParetoPolicy(variant, target, default_tau_gain=0.40, default_tau_harm=0.25, default_lambda=0.25, beta_harm=1.0, allow_aggressive=True)


def lambda_from_tier(candidate: ExpertCandidate, target: str) -> float:
    if target == "safe":
        return {"safe": 0.50, "balanced": 0.25, "aggressive": 0.05}.get(candidate.tier, 0.0)
    if target == "balanced":
        return {"safe": 0.75, "balanced": 0.50, "aggressive": 0.05}.get(candidate.tier, 0.0)
    return {"safe": 0.75, "balanced": 0.75, "aggressive": 0.10}.get(candidate.tier, 0.0)


def calibrate_global_policy(
    variant: str,
    target: str,
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    probs: Mapping[str, Any],
    oracle_mse: np.ndarray,
    slice_labels: pd.Series,
    masks: Mapping[str, np.ndarray],
    allow_aggressive: bool = False,
) -> Tuple[ParetoPolicy, pd.DataFrame]:
    if target == "safe":
        tg_grid = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        th_grid = [0.08, 0.10, 0.12, 0.16, 0.20]
        lambda_grid = [0.25, 0.50, 0.75]
    elif target == "balanced":
        tg_grid = [0.40, 0.45, 0.50, 0.55, 0.60]
        th_grid = [0.10, 0.12, 0.16, 0.20, 0.25]
        lambda_grid = [0.25, 0.50, 0.75, 1.00]
    else:
        tg_grid = [0.35, 0.40, 0.45, 0.50]
        th_grid = [0.16, 0.20, 0.25, 0.30]
        lambda_grid = [0.10, 0.25, 0.50]
    rows: List[Dict[str, Any]] = []
    best_policy: Optional[ParetoPolicy] = None
    best_score = float("inf")
    by_name = candidate_dict(candidates)
    for tg, th, lam in itertools.product(tg_grid, th_grid, lambda_grid):
        policy = ParetoPolicy(variant, target, tg, th, lam, allow_aggressive=allow_aggressive)
        policy.lambdas = {name: min(lam, lambda_from_tier(by_name[name], target)) for name in deployable_experts(candidates, allow_aggressive)}
        row, _, _ = evaluate_policy_on_calib(policy, batch, candidates, probs, oracle_mse, slice_labels, masks)
        row.update({"stage": "global_grid", "default_tau_gain": tg, "default_tau_harm": th, "default_lambda": lam})
        rows.append(row)
        score = float(row["calibration_score"]) + (0.0 if row["calibration_feasible"] else 1000.0)
        if score < best_score:
            best_score = score
            best_policy = policy_copy(policy)
    if best_policy is None:
        best_policy = initial_policy(variant, target, candidates, allow_aggressive)
    return best_policy, pd.DataFrame(rows)


def coordinate_search(
    base_policy: ParetoPolicy,
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    probs: Mapping[str, Any],
    oracle_mse: np.ndarray,
    slice_labels: pd.Series,
    masks: Mapping[str, np.ndarray],
    tune_thresholds: bool,
    tune_lambdas: bool,
) -> Tuple[ParetoPolicy, pd.DataFrame]:
    policy = policy_copy(base_policy)
    rows: List[Dict[str, Any]] = []
    experts = deployable_experts(candidates, policy.allow_aggressive)
    by_name = candidate_dict(candidates)
    for expert in experts:
        current_row, _, _ = evaluate_policy_on_calib(policy, batch, candidates, probs, oracle_mse, slice_labels, masks)
        best_score = float(current_row["calibration_score"]) + (0.0 if current_row["calibration_feasible"] else 1000.0)
        best_tuple = (policy.tg(expert), policy.th(expert), policy.lam(expert))
        tg_values = [policy.tg(expert)] if not tune_thresholds else [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        th_values = [policy.th(expert)] if not tune_thresholds else [0.04, 0.08, 0.12, 0.16, 0.20, 0.25]
        max_lam = lambda_from_tier(by_name[expert], policy.target)
        lam_values = [policy.lam(expert)] if not tune_lambdas else sorted(set([0.10, 0.25, 0.50, 0.75, 1.00, max_lam]))
        for tg, th, lam in itertools.product(tg_values, th_values, lam_values):
            lam = float(min(lam, max_lam))
            trial = policy_copy(policy)
            trial.tau_gain[expert] = float(tg)
            trial.tau_harm[expert] = float(th)
            trial.lambdas[expert] = lam
            row, _, _ = evaluate_policy_on_calib(trial, batch, candidates, probs, oracle_mse, slice_labels, masks)
            row.update({"stage": "coordinate_search", "tuned_expert": expert, "expert_tau_gain": tg, "expert_tau_harm": th, "expert_lambda": lam})
            rows.append(row)
            score = float(row["calibration_score"]) + (0.0 if row["calibration_feasible"] else 1000.0)
            if score < best_score:
                best_score = score
                best_tuple = (float(tg), float(th), lam)
        policy.tau_gain[expert], policy.tau_harm[expert], policy.lambdas[expert] = best_tuple
    return policy, pd.DataFrame(rows)


def slice_allowed_map(candidates: Sequence[ExpertCandidate]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for c in candidates:
        if c.name == "keep_lrbn":
            continue
        if c.family == "boundary":
            out[c.name] = ["high_gap_low_repair", "q4_boundary"]
        elif c.family == "volatility":
            out[c.name] = ["high_volatility", "non_boundary", "default"]
        elif c.family == "amplitude":
            out[c.name] = ["amplitude_mismatch", "non_boundary", "default"]
        elif c.family == "residual":
            out[c.name] = ["non_boundary", "high_volatility", "amplitude_mismatch", "default"]
        elif c.family == "level":
            out[c.name] = ["high_gap_low_repair", "q4_boundary", "amplitude_mismatch"]
        elif c.family == "ensemble":
            out[c.name] = ["high_gap_low_repair", "q4_boundary", "high_volatility", "amplitude_mismatch", "non_boundary"]
    return out


def config_veto_map(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    target: str,
    allow_aggressive: bool = False,
) -> Dict[str, List[str]]:
    harm_limit = 0.10 if target == "safe" else 0.20
    out: Dict[str, List[str]] = {}
    base = mse_per_sample(batch.lrbn_pred, batch.y_true)
    keys = config_keys(batch.meta)
    for c in candidates:
        if c.name == "keep_lrbn" or not (c.deployable or allow_aggressive):
            continue
        cmse = mse_per_sample(c.pred, batch.y_true)
        blocked: List[str] = []
        for key, idx in batch.meta.groupby(keys).groups.items():
            idx = list(idx)
            delta = cmse[idx] - base[idx]
            if float(np.mean(delta)) > 0.0 or float(np.mean(delta > 1e-12)) > harm_limit:
                blocked.append(str(key))
        if blocked:
            out[c.name] = blocked
    return out


def evaluate_variant(
    policy: ParetoPolicy,
    test: ForecastBatch,
    test_candidates: Sequence[ExpertCandidate],
    test_probs: Mapping[str, Any],
    oracle_mse: np.ndarray,
    slice_labels: pd.Series,
    masks: Mapping[str, np.ndarray],
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    pred, decisions = apply_policy(test, test_candidates, test_probs, policy, slice_labels)
    selected = decisions["selected_expert"].to_numpy(str) != "keep_lrbn"
    row = metric_row(policy.variant, pred, test, selected=selected, oracle_mse=oracle_mse, ci=bootstrap_ci(pred, test, n_bootstrap, seed))
    return {"row": row, "pred": pred, "decisions": decisions}


def selection_distribution(variant: str, decisions: pd.DataFrame) -> pd.DataFrame:
    out = decisions["selected_expert"].value_counts().rename_axis("expert").reset_index(name="count")
    out["variant"] = variant
    out["coverage"] = out["count"] / max(1, len(decisions))
    return out


def policy_table(policy: ParetoPolicy, candidates: Sequence[ExpertCandidate]) -> pd.DataFrame:
    rows = []
    by_name = candidate_dict(candidates)
    for expert in deployable_experts(candidates, policy.allow_aggressive):
        c = by_name[expert]
        rows.append(
            {
                "variant": policy.variant,
                "target": policy.target,
                "expert": expert,
                "tier": c.tier,
                "family": c.family,
                "tau_gain": policy.tg(expert),
                "tau_harm": policy.th(expert),
                "lambda": policy.lam(expert),
                "allowed_slices": ",".join(policy.allowed_slices.get(expert, [])),
                "blocked_configs": ",".join(policy.blocked_configs.get(expert, [])),
                "allow_aggressive": bool(policy.allow_aggressive),
            }
        )
    return pd.DataFrame(rows)


def keep_lrbn_recoverable_rows(
    variant: str,
    decisions: pd.DataFrame,
    oracle_names: np.ndarray,
    oracle_mse: np.ndarray,
    batch: ForecastBatch,
) -> Dict[str, Any]:
    keep = decisions["selected_expert"].to_numpy(str) == "keep_lrbn"
    base = mse_per_sample(batch.lrbn_pred, batch.y_true)
    recoverable = keep & (oracle_mse < base - 1e-12)
    return {
        "variant": variant,
        "keep_count": int(np.sum(keep)),
        "keep_rate": float(np.mean(keep)),
        "recoverable_keep_count": int(np.sum(recoverable)),
        "recoverable_keep_rate_among_keep": float(np.mean(recoverable[keep])) if keep.any() else 0.0,
        "mean_oracle_gain_kept": float(np.mean(base[recoverable] - oracle_mse[recoverable])) if recoverable.any() else 0.0,
        "top_oracle_expert_among_recoverable": pd.Series(oracle_names[recoverable]).mode().iloc[0] if recoverable.any() else "",
    }


def no_mrc_feature_ablation(
    normal_metrics: pd.DataFrame,
    no_mrc_metrics: pd.DataFrame,
) -> pd.DataFrame:
    a = normal_metrics.copy()
    b = no_mrc_metrics.copy()
    a["feature_set"] = "mrc_features"
    b["feature_set"] = "no_mrc_features"
    combo = pd.concat([a, b], ignore_index=True)
    rows = []
    for keys, group in combo.groupby(["head", "split"], observed=True):
        normal = group[group["feature_set"].eq("mrc_features")]
        nomrc = group[group["feature_set"].eq("no_mrc_features")]
        if normal.empty or nomrc.empty:
            continue
        rows.append(
            {
                "head": keys[0],
                "split": keys[1],
                "normal_roc_auc": float(normal["roc_auc"].mean()),
                "no_mrc_roc_auc": float(nomrc["roc_auc"].mean()),
                "delta_roc_auc": float(normal["roc_auc"].mean() - nomrc["roc_auc"].mean()),
                "normal_pr_auc": float(normal["pr_auc"].mean()),
                "no_mrc_pr_auc": float(nomrc["pr_auc"].mean()),
                "delta_pr_auc": float(normal["pr_auc"].mean() - nomrc["pr_auc"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_all_artifacts(
    metrics_csv: Path,
    stage5_dir: Path,
    stage6_dir: Path,
    stage7_dir: Path,
    output_dir: Path,
    stage3_dir: Optional[Path] = None,
    n_bootstrap: int = 2000,
    seed: int = 2026,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch = load_forecast_batch_from_metrics(metrics_csv)
    val, test = split_batch(batch)
    schema = feature_schema(val)
    q = stage8_slice_thresholds(val, schema)
    train_mask, calib_mask = stratified_inner_split(val.meta, seed=seed)
    val_train = val.subset(train_mask)
    val_calib = val.subset(calib_mask)

    safe_params = json.loads((stage5_dir / "stage5_selected_safe_params.json").read_text(encoding="utf-8"))
    balanced_params = json.loads((stage5_dir / "stage5_selected_balanced_params.json").read_text(encoding="utf-8"))
    basic_params = select_basic_sra_params(stage5_dir)
    stage3_params = load_stage3_params(stage3_dir)
    mrc = build_mrc_artifacts(val, test, schema)
    val_candidates = build_candidate_pool(val, safe_params, balanced_params, basic_params, stage3_params, mrc["val_delta"], mrc["val_abstain_pred"])
    test_candidates = build_candidate_pool(test, safe_params, balanced_params, basic_params, stage3_params, mrc["test_delta"], mrc["test_abstain_pred"])
    train_candidates = subset_candidates(val_candidates, train_mask)
    calib_candidates = subset_candidates(val_candidates, calib_mask)
    train_mrc = mrc["val_delta"][train_mask]
    calib_mrc = mrc["val_delta"][calib_mask]

    heads = fit_heads(val_train, val_calib, train_candidates, calib_candidates, schema, train_mrc, calib_mrc)
    heads_no_mrc = fit_heads(val_train, val_calib, train_candidates, calib_candidates, schema, np.zeros_like(train_mrc), np.zeros_like(calib_mrc))
    calib_probs = compute_probabilities(val_calib, calib_candidates, schema, calib_mrc, heads)
    test_probs = compute_probabilities(test, test_candidates, schema, mrc["test_delta"], heads)
    calib_probs_no_mrc = compute_probabilities(val_calib, calib_candidates, schema, np.zeros_like(calib_mrc), heads_no_mrc)
    test_probs_no_mrc = compute_probabilities(test, test_candidates, schema, np.zeros_like(mrc["test_delta"]), heads_no_mrc)

    train_slices = primary_slice_labels(val_train, schema, q)
    calib_slices = primary_slice_labels(val_calib, schema, q)
    test_slices = primary_slice_labels(test, schema, q)
    calib_masks = stage8_slice_masks(val_calib, schema, q)
    test_masks = stage8_slice_masks(test, schema, q)
    _, calib_oracle_mse, _ = oracle_best(calib_candidates, val_calib)
    oracle_pred, test_oracle_mse, test_oracle_names = oracle_best(test_candidates, test)

    policies: Dict[str, ParetoPolicy] = {}
    grids_safe: List[pd.DataFrame] = []
    grids_balanced: List[pd.DataFrame] = []

    base_safe, grid = calibrate_global_policy("SafeTAE-mrc-features", "safe", val_calib, calib_candidates, calib_probs, calib_oracle_mse, calib_slices, calib_masks)
    grids_safe.append(grid)
    base_bal, grid = calibrate_global_policy("SafeTAE-balanced-global", "balanced", val_calib, calib_candidates, calib_probs, calib_oracle_mse, calib_slices, calib_masks)
    grids_balanced.append(grid)

    no_mrc_policy, grid = calibrate_global_policy("SafeTAE-no-mrc-features", "safe", val_calib, calib_candidates, calib_probs_no_mrc, calib_oracle_mse, calib_slices, calib_masks)
    grids_safe.append(grid)
    policies["SafeTAE-mrc-features"] = base_safe
    policies["SafeTAE-no-mrc-features"] = no_mrc_policy

    p_thr, grid_thr = coordinate_search(policy_copy(base_safe, "SafeTAE-expert-thresholds"), val_calib, calib_candidates, calib_probs, calib_oracle_mse, calib_slices, calib_masks, tune_thresholds=True, tune_lambdas=False)
    p_lam, grid_lam = coordinate_search(policy_copy(base_safe, "SafeTAE-expert-lambda"), val_calib, calib_candidates, calib_probs, calib_oracle_mse, calib_slices, calib_masks, tune_thresholds=False, tune_lambdas=True)
    grids_safe.extend([grid_thr, grid_lam])
    policies[p_thr.variant] = p_thr
    policies[p_lam.variant] = p_lam

    p_slice = policy_copy(p_thr, "SafeTAE-slice-aware")
    p_slice.allowed_slices = slice_allowed_map(calib_candidates)
    p_slice, grid_slice = coordinate_search(p_slice, val_calib, calib_candidates, calib_probs, calib_oracle_mse, calib_slices, calib_masks, tune_thresholds=True, tune_lambdas=True)
    grids_safe.append(grid_slice)
    policies[p_slice.variant] = p_slice

    p_veto = policy_copy(p_slice, "SafeTAE-config-veto")
    p_veto.blocked_configs = config_veto_map(val_calib, calib_candidates, "safe")
    p_veto, grid_veto = coordinate_search(p_veto, val_calib, calib_candidates, calib_probs, calib_oracle_mse, calib_slices, calib_masks, tune_thresholds=True, tune_lambdas=True)
    grids_safe.append(grid_veto)
    policies[p_veto.variant] = p_veto

    p_safe = policy_copy(p_veto, "SafeTAE-pareto-safe", "safe")
    p_safe, grid_safe_pareto = coordinate_search(p_safe, val_calib, calib_candidates, calib_probs, calib_oracle_mse, calib_slices, calib_masks, tune_thresholds=True, tune_lambdas=True)
    grids_safe.append(grid_safe_pareto)
    policies[p_safe.variant] = p_safe

    p_bal, grid_bal_base = coordinate_search(policy_copy(base_bal, "SafeTAE-pareto-balanced", "balanced"), val_calib, calib_candidates, calib_probs, calib_oracle_mse, calib_slices, calib_masks, tune_thresholds=True, tune_lambdas=True)
    p_bal.allowed_slices = slice_allowed_map(calib_candidates)
    p_bal.blocked_configs = config_veto_map(val_calib, calib_candidates, "balanced")
    p_bal, grid_bal_final = coordinate_search(p_bal, val_calib, calib_candidates, calib_probs, calib_oracle_mse, calib_slices, calib_masks, tune_thresholds=True, tune_lambdas=True)
    grids_balanced.extend([grid_bal_base, grid_bal_final])
    policies[p_bal.variant] = p_bal

    p_aggr, grid_aggr = calibrate_global_policy("SafeTAE-pareto-aggressive-diagnostic", "aggressive_diagnostic", val_calib, calib_candidates, calib_probs, calib_oracle_mse, calib_slices, calib_masks, allow_aggressive=True)
    p_aggr.allow_aggressive = True
    p_aggr, grid_aggr_coord = coordinate_search(p_aggr, val_calib, calib_candidates, calib_probs, calib_oracle_mse, calib_slices, calib_masks, tune_thresholds=True, tune_lambdas=True)
    grids_balanced.extend([grid_aggr, grid_aggr_coord])
    policies[p_aggr.variant] = p_aggr

    overall_rows: List[Dict[str, Any]] = [
        metric_row("LRBN", test.lrbn_pred, test, oracle_mse=test_oracle_mse, ci=bootstrap_ci(test.lrbn_pred, test, n_bootstrap, seed)),
        metric_row("TAE-oracle-best", oracle_pred, test, oracle_mse=test_oracle_mse, ci=bootstrap_ci(oracle_pred, test, n_bootstrap, seed)),
    ]
    by_test = candidate_dict(test_candidates)
    baseline_names = {
        "SRA-BP-safe": "sra_safe",
        "SRA-BP-balanced": "sra_balanced",
        "MRC-ridge-abstain": "mrc_ridge_abstain",
    }
    variant_predictions: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[pd.DataFrame]]] = {}
    for label, cname in baseline_names.items():
        pred = by_test[cname].pred
        overall_rows.append(metric_row(label, pred, test, oracle_mse=test_oracle_mse, ci=bootstrap_ci(pred, test, n_bootstrap, seed)))
        variant_predictions[label] = (pred, np.zeros(len(test.meta), dtype=bool), None)

    stage7_overall = pd.read_csv(stage7_dir / "safe_tae_overall.csv")
    for ref in ["SafeTAE-safe", "TAE-router-stage6", "TAE-ranker-stage6"]:
        rows = stage7_overall[stage7_overall["variant"].eq(ref)]
        if not rows.empty:
            row = rows.iloc[0].to_dict()
            row["variant"] = ref
            if "max_config_harm" not in row or pd.isna(row.get("max_config_harm", np.nan)):
                row["max_config_harm"] = row.get("max_per_config_harm_rate", np.nan)
            row["reference_only"] = True
            row["test_threshold_leakage"] = False
            overall_rows.append(row)

    decisions_all: List[pd.DataFrame] = []
    policy_tables: List[pd.DataFrame] = []
    for name, policy in policies.items():
        probs = test_probs_no_mrc if name == "SafeTAE-no-mrc-features" else test_probs
        result = evaluate_variant(policy, test, test_candidates, probs, test_oracle_mse, test_slices, test_masks, n_bootstrap, seed)
        overall_rows.append(result["row"])
        selected_mask = result["decisions"]["selected_expert"].to_numpy(str) != "keep_lrbn"
        variant_predictions[name] = (result["pred"], selected_mask, result["decisions"])
        decisions_all.append(result["decisions"])
        policy_tables.append(policy_table(policy, test_candidates))

    overall = pd.DataFrame(overall_rows)
    per_config_frames: List[pd.DataFrame] = []
    slice_frames: List[pd.DataFrame] = []
    dist_frames: List[pd.DataFrame] = []
    recover_rows: List[Dict[str, Any]] = []
    oracle_rows: List[Dict[str, Any]] = []
    for name, (pred, selected, decisions) in variant_predictions.items():
        per_config_frames.append(pd.DataFrame(per_config_rows(name, pred, selected, test)))
        slice_frames.append(pd.DataFrame(slice_rows(name, pred, selected, test, test_masks)))
        if decisions is not None:
            dist_frames.append(selection_distribution(name, decisions))
            recover_rows.append(keep_lrbn_recoverable_rows(name, decisions, test_oracle_names, test_oracle_mse, test))
        row = overall[overall["variant"].eq(name)]
        if not row.empty:
            oracle_rows.append(
                {
                    "variant": name,
                    "mse": float(row.iloc[0]["mse"]),
                    "mse_delta_pct_vs_lrbn": float(row.iloc[0]["mse_delta_pct_vs_lrbn"]),
                    "oracle_gain_fraction": float(row.iloc[0].get("oracle_gain_fraction", np.nan)),
                }
            )

    candidate_pool = pd.DataFrame(candidate_metric_rows(val_candidates, val, "val") + candidate_metric_rows(test_candidates, test, "test"))
    pairwise_metrics = pd.concat(
        [
            heads.metrics.assign(feature_set="mrc_features"),
            heads_no_mrc.metrics.assign(feature_set="no_mrc_features"),
        ],
        ignore_index=True,
    )
    policy_grid_safe = pd.concat([g for g in grids_safe if not g.empty], ignore_index=True)
    policy_grid_balanced = pd.concat([g for g in grids_balanced if not g.empty], ignore_index=True)
    per_config = pd.concat(per_config_frames, ignore_index=True)
    slice_metrics = pd.concat(slice_frames, ignore_index=True)
    stage7_pc = stage7_dir / "safe_tae_per_config.csv"
    if stage7_pc.exists():
        ref_pc = pd.read_csv(stage7_pc)
        ref_pc = ref_pc[ref_pc["variant"].isin(["SafeTAE-safe"])]
        if not ref_pc.empty:
            if "config_key" not in ref_pc.columns:
                ref_pc["config_key"] = ref_pc["dataset"].astype(str) + "/" + ref_pc["backbone"].astype(str) + "/" + ref_pc["horizon"].astype(str)
            per_config = pd.concat([per_config, ref_pc], ignore_index=True, sort=False)
    stage7_sl = stage7_dir / "safe_tae_slices.csv"
    if stage7_sl.exists():
        ref_sl = pd.read_csv(stage7_sl)
        ref_sl = ref_sl[ref_sl["variant"].isin(["SafeTAE-safe"])]
        if not ref_sl.empty:
            slice_metrics = pd.concat([slice_metrics, ref_sl], ignore_index=True, sort=False)
    distribution = pd.concat(dist_frames, ignore_index=True) if dist_frames else pd.DataFrame()
    lambda_table = pd.concat(policy_tables, ignore_index=True) if policy_tables else pd.DataFrame()
    oracle_capture = pd.DataFrame(oracle_rows)
    recover = pd.DataFrame(recover_rows)
    harmed = per_config.sort_values("mse_delta_pct_vs_lrbn", ascending=False)
    mrc_ablation = no_mrc_feature_ablation(heads.metrics, heads_no_mrc.metrics)

    bootstrap_map = {
        str(row["variant"]): {
            "ci95_low_mse_delta": row.get("ci95_low_mse_delta"),
            "ci95_high_mse_delta": row.get("ci95_high_mse_delta"),
        }
        for _, row in overall.iterrows()
        if "ci95_low_mse_delta" in overall.columns
    }

    def overall_row(name: str) -> pd.Series:
        rows = overall[overall["variant"].eq(name)]
        if rows.empty:
            raise KeyError(name)
        return rows.iloc[0]

    def slice_value(variant: str, sl: str, col: str, default: float = np.nan) -> float:
        rows = slice_metrics[(slice_metrics["variant"].eq(variant)) & (slice_metrics["slice"].eq(sl))]
        if rows.empty:
            return float(default)
        return float(rows.iloc[0][col])

    stage7_safe = overall_row("SafeTAE-safe")
    sra_bal = overall_row("SRA-BP-balanced")
    p_safe_row = overall_row("SafeTAE-pareto-safe")
    p_bal_row = overall_row("SafeTAE-pareto-balanced")
    safe_delta_gate = min(-2.0, float(stage7_safe["mse_delta_pct_vs_lrbn"]) - 0.30)
    safe_pass = bool(
        float(p_safe_row["mse_delta_pct_vs_lrbn"]) <= safe_delta_gate
        and float(p_safe_row["harm_rate"]) <= 0.03
        and float(p_safe_row["max_config_harm"]) <= 0.10
        and float(p_safe_row["oracle_gain_fraction"]) >= 0.13
        and float(p_safe_row["ci95_high_mse_delta"]) < 0.0
        and float(p_safe_row["config_improved_ratio"]) >= 0.625
        and slice_value("SafeTAE-pareto-safe", "low_gap_high_repair", "harm_rate", 1.0) <= 0.02
    )
    balanced_pass = bool(
        float(p_bal_row["mse_delta_pct_vs_lrbn"]) < float(sra_bal["mse_delta_pct_vs_lrbn"])
        and float(p_bal_row["mse_delta_pct_vs_lrbn"]) <= -2.8
        and float(p_bal_row["harm_rate"]) <= 0.10
        and float(p_bal_row["max_config_harm"]) <= 0.20
        and float(p_bal_row["oracle_gain_fraction"]) >= 0.16
        and float(p_bal_row["ci95_high_mse_delta"]) < 0.0
        and float(p_bal_row["config_improved_ratio"]) >= 0.625
        and slice_value("SafeTAE-pareto-balanced", "high_gap_low_repair", "mse_delta_pct_vs_lrbn") < slice_value("SafeTAE-safe", "high_gap_low_repair", "mse_delta_pct_vs_lrbn")
    )
    expert_threshold_row = overall_row("SafeTAE-expert-thresholds")
    expert_lambda_row = overall_row("SafeTAE-expert-lambda")
    slice_row = overall_row("SafeTAE-slice-aware")
    veto_row = overall_row("SafeTAE-config-veto")
    mrc_row = overall_row("SafeTAE-mrc-features")
    nomrc_row = overall_row("SafeTAE-no-mrc-features")
    verdict = {
        "stage": "stage8_safe_tae_pareto",
        "test_threshold_leakage": False,
        "safe_pass": safe_pass,
        "balanced_pass": balanced_pass,
        "status": "pareto_expansion_supported" if safe_pass and balanced_pass else ("safe_expansion_only" if safe_pass else "conservative_limit_confirmed"),
        "next": "run_mini_extension" if safe_pass and balanced_pass else ("keep_as_safe_ablation_and_refine_balanced_objective" if safe_pass else "stop_safe_tae_expansion_or_redesign_expert_pool"),
        "h1_expert_specific_releases_coverage": bool(float(expert_threshold_row["coverage"]) > float(stage7_safe.get("coverage", 0.0)) and float(expert_threshold_row["harm_rate"]) <= float(stage7_safe["harm_rate"]) + 0.02),
        "h2_expert_thresholds_help": bool(float(expert_threshold_row["oracle_gain_fraction"]) > float(stage7_safe["oracle_gain_fraction"]) and float(expert_threshold_row["harm_rate"]) <= 0.10),
        "h3_expert_lambda_help": bool(float(expert_lambda_row["mse_delta_pct_vs_lrbn"]) < float(stage7_safe["mse_delta_pct_vs_lrbn"]) and float(expert_lambda_row["harm_rate"]) <= 0.10),
        "h4_slice_aware_help": bool(float(slice_row["mse_delta_pct_vs_lrbn"]) < float(stage7_safe["mse_delta_pct_vs_lrbn"]) and slice_value("SafeTAE-slice-aware", "low_gap_high_repair", "harm_rate", 1.0) <= 0.02),
        "h5_config_veto_help": bool(slice_value("SafeTAE-config-veto", "known_harmed_config", "mse_delta_pct_vs_lrbn", 999.0) <= 0.0),
        "h6_mrc_features_help": bool(float(mrc_row["mse_delta_pct_vs_lrbn"]) <= float(nomrc_row["mse_delta_pct_vs_lrbn"]) and float(mrc_row["harm_rate"]) <= float(nomrc_row["harm_rate"]) + 0.01),
        "h7_new_pareto_frontier": bool(safe_pass or balanced_pass),
        "safe_delta_pct_vs_lrbn": float(p_safe_row["mse_delta_pct_vs_lrbn"]),
        "balanced_delta_pct_vs_lrbn": float(p_bal_row["mse_delta_pct_vs_lrbn"]),
        "stage7_safe_delta_pct_vs_lrbn": float(stage7_safe["mse_delta_pct_vs_lrbn"]),
        "sra_balanced_delta_pct_vs_lrbn": float(sra_bal["mse_delta_pct_vs_lrbn"]),
    }

    config = {
        "metrics_csv": str(metrics_csv),
        "stage5_dir": str(stage5_dir),
        "stage6_dir": str(stage6_dir),
        "stage7_dir": str(stage7_dir),
        "stage3_dir": str(stage3_dir) if stage3_dir else None,
        "output_dir": str(output_dir),
        "scope": "stage8_safe_tae_pareto_compact_validation",
        "datasets": sorted(test.meta["dataset"].astype(str).unique().tolist()),
        "backbones": sorted(test.meta["backbone"].astype(str).unique().tolist()),
        "horizons": sorted([int(x) for x in test.meta["horizon"].unique().tolist()]),
        "n_val_samples": int(len(val.meta)),
        "n_inner_train_samples": int(len(val_train.meta)),
        "n_inner_calib_samples": int(len(val_calib.meta)),
        "n_test_samples": int(len(test.meta)),
        "n_test_configs": int(test.meta.groupby(["dataset", "backbone", "horizon", "seed"], observed=True).ngroups),
        "slice_thresholds_validation_only": q,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "test_threshold_leakage": False,
    }

    write_json(output_dir / "stage8_config.json", config)
    candidate_pool.to_csv(output_dir / "stage8_candidate_pool.csv", index=False)
    pairwise_metrics.to_csv(output_dir / "stage8_pairwise_head_metrics.csv", index=False)
    policy_grid_safe.to_csv(output_dir / "stage8_policy_grid_safe.csv", index=False)
    policy_grid_balanced.to_csv(output_dir / "stage8_policy_grid_balanced.csv", index=False)
    overall.to_csv(output_dir / "stage8_overall.csv", index=False)
    per_config.to_csv(output_dir / "stage8_per_config.csv", index=False)
    slice_metrics.to_csv(output_dir / "stage8_slice_metrics.csv", index=False)
    distribution.to_csv(output_dir / "stage8_expert_selection_distribution.csv", index=False)
    lambda_table.to_csv(output_dir / "stage8_expert_lambda_table.csv", index=False)
    oracle_capture.to_csv(output_dir / "stage8_oracle_capture.csv", index=False)
    recover.to_csv(output_dir / "stage8_keep_lrbn_recoverable_analysis.csv", index=False)
    harmed.to_csv(output_dir / "stage8_harmed_config_analysis.csv", index=False)
    mrc_ablation.to_csv(output_dir / "stage8_mrc_feature_ablation.csv", index=False)
    write_json(output_dir / "stage8_bootstrap_ci.json", bootstrap_map)
    write_json(output_dir / "stage8_selected_policies.json", {k: asdict(v) for k, v in policies.items()})
    write_json(output_dir / "stage8_verdict.json", verdict)

    show_cols = ["variant", "mse", "mae", "mse_delta_pct_vs_lrbn", "harm_rate", "max_config_harm", "coverage", "oracle_gain_fraction", "ci95_high_mse_delta"]
    summary = [
        "# Stage 8 Safe-TAE Pareto Summary",
        "",
        "## Verdict",
        "",
        f"- Status: `{verdict['status']}`",
        f"- Safe pass: `{verdict['safe_pass']}`",
        f"- Balanced pass: `{verdict['balanced_pass']}`",
        f"- Test threshold leakage: `{verdict['test_threshold_leakage']}`",
        "",
        "## Overall",
        "",
        df_to_md(overall[[c for c in show_cols if c in overall.columns]].sort_values("mse")),
        "",
        "## Mechanism Flags",
        "",
        "```json",
        json.dumps(verdict, indent=2, ensure_ascii=False),
        "```",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {"overall": overall, "per_config": per_config, "slices": slice_metrics, "verdict": verdict}
