#!/usr/bin/env python
"""Stage 13 compact validation for CGA decision-geometry alternatives.

This stage tests architecture-level alternatives from deep-research-report (3):
residual-prior convex mixing, time-step gated hybrid editing,
selection-conditional conformal family editing, retrieval local residual editing,
and conservative challenger comparison.
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

from halluguard_lrbn_bp import ForecastBatch, mae_per_sample, mse_per_sample
from halluguard_stage6_mechanism import safe_pct
from halluguard_stage7_safe_tae import ExpertCandidate, candidate_dict, write_json
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
    ScoreBundle,
    build_cga_pools,
    candidate_metadata,
    df_to_md,
    fit_cga_models,
    json_default,
    prepare_score_bundle,
    topk_metrics,
)


FAMILIES = ["residual_distribution", "smoothing_teacher", "retrieval_memory", "local_boundary"]
STAGE13_VARIANTS = [
    "Residual-Prior Convex Mixer",
    "Time-Step Gated Hybrid Editor",
    "Selection-Conditional Conformal Family Editor",
    "Retrieval-Augmented Local Residual Editor",
    "Conservative Challenger Comparator",
]


@dataclass(frozen=True)
class GeometryPolicy:
    variant: str
    k_max: int = 2
    tau_leave: float = 0.55
    tau_utility: float = 0.0
    beta_harm: float = 2.0
    base_floor: float = 0.75
    delta_cap: float = 0.30
    lam: float = 0.35
    temporal_q: float = 0.75
    local_lam: float = 0.25
    risk_tau: float = 0.08
    deadzone: float = 0.0
    agreement_tau: float = 0.35
    margin: float = 0.05


@dataclass
class PreparedGeometry:
    family_pred: Dict[str, np.ndarray]
    retrieval_disagreement: np.ndarray
    temporal_score: np.ndarray
    temporal_thresholds: Dict[float, float]
    scores: ScoreBundle


def _finite(a: np.ndarray, fill: np.ndarray) -> np.ndarray:
    out = np.asarray(a, dtype=float).copy()
    return np.where(np.isfinite(out), out, fill)


def _scale(batch: ForecastBatch) -> np.ndarray:
    s = np.nanstd(batch.context, axis=(1, 2))
    s = np.where(np.isfinite(s) & (s > 1e-6), s, 1.0)
    return s.reshape(-1, 1, 1)


def clip_delta(delta: np.ndarray, batch: ForecastBatch, cap: float) -> np.ndarray:
    bound = max(float(cap), 1e-6) * _scale(batch)
    return np.clip(delta, -bound, bound)


def _candidate_stack(candidates: Sequence[ExpertCandidate], names_or_families: Sequence[str], by_family: bool) -> List[np.ndarray]:
    arrs: List[np.ndarray] = []
    keys = set(names_or_families)
    for c in candidates:
        if c.name == "keep_lrbn" or not c.deployable:
            continue
        if (c.family in keys) if by_family else (c.name in keys):
            arrs.append(c.pred)
    return arrs


def build_family_priors(batch: ForecastBatch, candidates: Sequence[ExpertCandidate]) -> Dict[str, np.ndarray]:
    priors: Dict[str, np.ndarray] = {}
    for family in ["residual_distribution", "smoothing_teacher", "retrieval_memory"]:
        arrs = _candidate_stack(candidates, [family], by_family=True)
        if arrs:
            priors[family] = _finite(np.nanmedian(np.stack(arrs, axis=0), axis=0), batch.lrbn_pred)
    local_arrs = _candidate_stack(candidates, ["boundary", "causal_boundary", "local_boundary"], by_family=True)
    if local_arrs:
        priors["local_boundary"] = _finite(np.nanmedian(np.stack(local_arrs, axis=0), axis=0), batch.lrbn_pred)
    else:
        priors["local_boundary"] = priors.get("residual_distribution", batch.lrbn_pred.copy())
    return priors


def retrieval_disagreement(batch: ForecastBatch, candidates: Sequence[ExpertCandidate]) -> np.ndarray:
    arrs = _candidate_stack(candidates, ["retrieval_memory"], by_family=True)
    if len(arrs) < 2:
        return np.ones(len(batch.meta), dtype=float)
    stack = np.stack(arrs, axis=0)
    disagreement = np.nanmean(np.nanstd(stack, axis=0), axis=(1, 2))
    return disagreement / (_scale(batch).reshape(-1) + 1e-8)


def temporal_boundary_score(batch: ForecastBatch) -> np.ndarray:
    p = np.asarray(batch.lrbn_pred, dtype=float)
    c = np.asarray(batch.context, dtype=float)
    n, h, ch = p.shape
    score = np.zeros((n, h, ch), dtype=float)
    score[:, 0, :] = np.abs(p[:, 0, :] - c[:, -1, :])
    if h > 1:
        d1 = np.abs(np.diff(p, axis=1))
        score[:, 1:, :] += d1
    if h > 2:
        d2 = np.abs(np.diff(p, n=2, axis=1))
        score[:, 2:, :] += d2
    return score / (_scale(batch) + 1e-8)


def temporal_threshold_map(score: np.ndarray, quantiles: Sequence[float]) -> Dict[float, float]:
    flat = score.reshape(-1)
    return {float(q): float(np.nanquantile(flat, float(q))) for q in quantiles}


def prepare_geometry(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    models: Any,
    score_cache: Optional[ScoreBundle] = None,
    temporal_thresholds: Optional[Dict[float, float]] = None,
) -> PreparedGeometry:
    score = temporal_boundary_score(batch)
    return PreparedGeometry(
        family_pred=build_family_priors(batch, candidates),
        retrieval_disagreement=retrieval_disagreement(batch, candidates),
        temporal_score=score,
        temporal_thresholds=temporal_thresholds or temporal_threshold_map(score, [0.5, 0.65, 0.75, 0.85, 0.9]),
        scores=score_cache or prepare_score_bundle(batch, candidates, schema, models),
    )


def family_score_map(scores: ScoreBundle, row_index: int) -> Dict[str, Tuple[float, float, float]]:
    out: Dict[str, Tuple[float, float, float]] = {}
    rows = scores.fam_scores[scores.fam_scores["row_index"].eq(row_index)]
    for _, row in rows.iterrows():
        fam = str(row["family"])
        gain = float(row["p_family_gain"])
        harm = float(row["p_family_harm"])
        out[fam] = (gain, harm, gain - harm)
    return out


def utility_for_family(fam_scores: Mapping[str, Tuple[float, float, float]], family: str, beta_harm: float) -> Tuple[float, float, float]:
    if family == "local_boundary":
        gain, harm, _ = fam_scores.get("residual_distribution", (0.0, 1.0, -1.0))
        utility = 0.8 * gain - beta_harm * harm
        return gain, harm, utility
    gain, harm, _ = fam_scores.get(family, (0.0, 1.0, -1.0))
    return gain, harm, gain - beta_harm * harm


def selected_delta_norm(pred: np.ndarray, batch: ForecastBatch) -> np.ndarray:
    return np.linalg.norm((pred - batch.lrbn_pred).reshape(len(batch.meta), -1), axis=1)


def apply_rpcm(batch: ForecastBatch, prepared: PreparedGeometry, policy: GeometryPolicy) -> Tuple[np.ndarray, pd.DataFrame]:
    pred = batch.lrbn_pred.copy()
    decisions: List[Dict[str, Any]] = []
    for i in range(len(batch.meta)):
        selected: List[Tuple[str, float, float, float]] = []
        if prepared.scores.leave_score[i] >= policy.tau_leave:
            fam_scores = family_score_map(prepared.scores, i)
            for fam in FAMILIES:
                if fam not in prepared.family_pred:
                    continue
                gain, harm, utility = utility_for_family(fam_scores, fam, policy.beta_harm)
                if utility >= policy.tau_utility:
                    selected.append((fam, gain, harm, utility))
        selected = sorted(selected, key=lambda x: x[3], reverse=True)[: policy.k_max]
        if selected:
            utilities = np.asarray([x[3] for x in selected], dtype=float)
            weights = np.exp(utilities - np.max(utilities))
            weights = weights / (weights.sum() + 1e-12)
            family_mass = 1.0 - policy.base_floor
            expected_harm = float(np.sum(weights * np.asarray([x[2] for x in selected], dtype=float)))
            shrink = max(0.0, min(1.0, 1.0 - expected_harm))
            acc = batch.lrbn_pred[i].copy()
            for w, (fam, _, _, _) in zip(weights, selected):
                delta = clip_delta(prepared.family_pred[fam] - batch.lrbn_pred, batch, policy.delta_cap)[i]
                acc += family_mass * shrink * float(w) * delta
            pred[i] = acc
            selected_name = ",".join(x[0] for x in selected)
        else:
            expected_harm = 0.0
            selected_name = "none"
        decisions.append(
            {
                "row_index": i,
                "selected": selected_name != "none",
                "selected_action": selected_name,
                "expected_harm": expected_harm,
                "accept_score": float(max([x[3] for x in selected], default=0.0)),
            }
        )
    return pred, pd.DataFrame(decisions)


def apply_tsge(batch: ForecastBatch, prepared: PreparedGeometry, policy: GeometryPolicy) -> Tuple[np.ndarray, pd.DataFrame]:
    smooth = prepared.family_pred.get("smoothing_teacher", batch.lrbn_pred)
    local = prepared.family_pred.get("local_boundary", prepared.family_pred.get("residual_distribution", batch.lrbn_pred))
    threshold = prepared.temporal_thresholds.get(float(policy.temporal_q))
    if threshold is None:
        threshold = float(np.nanquantile(prepared.temporal_score.reshape(-1), policy.temporal_q))
    mask = (prepared.temporal_score >= threshold).astype(float)
    smooth_delta = clip_delta(smooth - batch.lrbn_pred, batch, policy.delta_cap)
    local_delta = clip_delta(local - batch.lrbn_pred, batch, policy.delta_cap)
    pred = batch.lrbn_pred + (1.0 - mask) * policy.lam * smooth_delta + mask * policy.local_lam * local_delta
    norm = selected_delta_norm(pred, batch)
    decisions = pd.DataFrame(
        {
            "row_index": np.arange(len(batch.meta)),
            "selected": norm > 1e-12,
            "selected_action": "time_step_hybrid",
            "expected_harm": np.nanmean(mask, axis=(1, 2)),
            "accept_score": 1.0 - np.nanmean(mask, axis=(1, 2)),
        }
    )
    return pred, decisions


def _family_choice(prepared: PreparedGeometry, policy: GeometryPolicy, i: int) -> Tuple[str, float, float, float]:
    if prepared.scores.leave_score[i] < policy.tau_leave:
        return "none", 0.0, 0.0, 0.0
    fam_scores = family_score_map(prepared.scores, i)
    best = ("none", 0.0, 0.0, 0.0)
    for fam in FAMILIES:
        if fam not in prepared.family_pred:
            continue
        gain, harm, utility = utility_for_family(fam_scores, fam, policy.beta_harm)
        if utility > best[3]:
            best = (fam, gain, harm, utility)
    if best[3] < policy.tau_utility:
        return "none", 0.0, 0.0, best[3]
    return best


def fit_bias_centers(calib: ForecastBatch, prepared: PreparedGeometry, policy: GeometryPolicy) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    selected: Dict[str, List[int]] = {fam: [] for fam in FAMILIES}
    for i in range(len(calib.meta)):
        fam, _, harm, _ = _family_choice(prepared, policy, i)
        if fam != "none" and harm <= policy.risk_tau:
            selected[fam].append(i)
    bias: Dict[str, np.ndarray] = {}
    expected_harm: Dict[str, float] = {}
    base_mse = mse_per_sample(calib.lrbn_pred, calib.y_true)
    for fam, idxs in selected.items():
        if not idxs or fam not in prepared.family_pred:
            bias[fam] = np.zeros_like(calib.lrbn_pred[0])
            expected_harm[fam] = 1.0
            continue
        idx = np.asarray(idxs, dtype=int)
        raw_bias = np.nanmean(prepared.family_pred[fam][idx] - calib.y_true[idx], axis=0)
        bias[fam] = np.where(np.isfinite(raw_bias), raw_bias, 0.0)
        fam_mse = mse_per_sample(prepared.family_pred[fam][idx] - bias[fam], calib.y_true[idx])
        expected_harm[fam] = float(np.mean(fam_mse > base_mse[idx] + 1e-12))
    return bias, expected_harm


def apply_sccfe(
    batch: ForecastBatch,
    prepared: PreparedGeometry,
    policy: GeometryPolicy,
    bias: Dict[str, np.ndarray],
    expected_harm: Dict[str, float],
) -> Tuple[np.ndarray, pd.DataFrame]:
    pred = batch.lrbn_pred.copy()
    decisions: List[Dict[str, Any]] = []
    for i in range(len(batch.meta)):
        fam, _, harm, utility = _family_choice(prepared, policy, i)
        selected = False
        if fam != "none" and fam in prepared.family_pred and harm <= policy.risk_tau:
            b = bias.get(fam, np.zeros_like(batch.lrbn_pred[i]))
            if float(np.nanmean(np.abs(b))) > policy.deadzone:
                adjusted = prepared.family_pred[fam][i] - b
                delta = clip_delta(adjusted[None, ...] - batch.lrbn_pred[i : i + 1], batch.subset(np.arange(len(batch.meta)) == i), policy.delta_cap)[0]
                pred[i] = batch.lrbn_pred[i] + policy.lam * delta
                selected = True
        decisions.append(
            {
                "row_index": i,
                "selected": selected,
                "selected_action": fam if selected else "none",
                "expected_harm": float(expected_harm.get(fam, 1.0 if fam != "none" else 0.0)),
                "accept_score": float(utility),
            }
        )
    return pred, pd.DataFrame(decisions)


def apply_ralre(batch: ForecastBatch, prepared: PreparedGeometry, policy: GeometryPolicy) -> Tuple[np.ndarray, pd.DataFrame]:
    memory = prepared.family_pred.get("retrieval_memory", batch.lrbn_pred)
    agreement = np.exp(-np.maximum(prepared.retrieval_disagreement, 0.0))
    delta = clip_delta(memory - batch.lrbn_pred, batch, policy.delta_cap)
    selected = agreement >= policy.agreement_tau
    pred = batch.lrbn_pred.copy()
    pred[selected] = batch.lrbn_pred[selected] + policy.lam * agreement[selected, None, None] * delta[selected]
    decisions = pd.DataFrame(
        {
            "row_index": np.arange(len(batch.meta)),
            "selected": selected,
            "selected_action": np.where(selected, "retrieval_local_prior", "none"),
            "expected_harm": 1.0 - agreement,
            "accept_score": agreement,
        }
    )
    return pred, decisions


def apply_ccc(batch: ForecastBatch, prepared: PreparedGeometry, policy: GeometryPolicy) -> Tuple[np.ndarray, pd.DataFrame]:
    pred = batch.lrbn_pred.copy()
    decisions: List[Dict[str, Any]] = []
    for i in range(len(batch.meta)):
        fam, gain, harm, utility = _family_choice(prepared, policy, i)
        accept_score = gain - policy.beta_harm * harm
        selected = fam != "none" and accept_score >= policy.margin
        if selected:
            challenger = prepared.family_pred[fam][i]
            delta = clip_delta(challenger[None, ...] - batch.lrbn_pred[i : i + 1], batch.subset(np.arange(len(batch.meta)) == i), policy.delta_cap)[0]
            pred[i] = batch.lrbn_pred[i] + policy.lam * delta
        decisions.append(
            {
                "row_index": i,
                "selected": selected,
                "selected_action": fam if selected else "none",
                "expected_harm": harm if fam != "none" else 0.0,
                "accept_score": float(accept_score),
            }
        )
    return pred, pd.DataFrame(decisions)


def policy_grid(variant: str) -> Iterable[GeometryPolicy]:
    if variant == "Residual-Prior Convex Mixer":
        for k in [1, 2]:
            for beta in [1.5, 2.5, 4.0]:
                for floor in [0.65, 0.80, 0.90]:
                    for cap in [0.15, 0.25, 0.35]:
                        for tau in [0.0, 0.03]:
                            yield GeometryPolicy(variant, k_max=k, beta_harm=beta, base_floor=floor, delta_cap=cap, tau_utility=tau)
    elif variant == "Time-Step Gated Hybrid Editor":
        for q in [0.65, 0.75, 0.85]:
            for lam in [0.30, 0.45, 0.60]:
                for local_lam in [0.0, 0.20, 0.35]:
                    for cap in [0.20, 0.35]:
                        yield GeometryPolicy(variant, temporal_q=q, lam=lam, local_lam=local_lam, delta_cap=cap)
    elif variant == "Selection-Conditional Conformal Family Editor":
        for beta in [2.0, 4.0]:
            for tau in [0.0, 0.03, 0.06]:
                for risk_tau in [0.05, 0.10, 0.15]:
                    for lam in [0.30, 0.50]:
                        yield GeometryPolicy(variant, beta_harm=beta, tau_utility=tau, risk_tau=risk_tau, lam=lam, delta_cap=0.25)
    elif variant == "Retrieval-Augmented Local Residual Editor":
        for agree in [0.20, 0.35, 0.50, 0.65]:
            for lam in [0.30, 0.50, 0.70]:
                for cap in [0.20, 0.35]:
                    yield GeometryPolicy(variant, agreement_tau=agree, lam=lam, delta_cap=cap)
    elif variant == "Conservative Challenger Comparator":
        for beta in [2.0, 4.0, 6.0]:
            for margin in [0.00, 0.05, 0.10]:
                for lam in [0.25, 0.40, 0.60]:
                    for cap in [0.20, 0.35]:
                        yield GeometryPolicy(variant, beta_harm=beta, margin=margin, lam=lam, delta_cap=cap)
    else:
        raise ValueError(f"unknown variant {variant}")


def apply_variant(
    batch: ForecastBatch,
    prepared: PreparedGeometry,
    policy: GeometryPolicy,
    calib_bias: Optional[Dict[str, np.ndarray]] = None,
    calib_expected_harm: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    if policy.variant == "Residual-Prior Convex Mixer":
        return apply_rpcm(batch, prepared, policy)
    if policy.variant == "Time-Step Gated Hybrid Editor":
        return apply_tsge(batch, prepared, policy)
    if policy.variant == "Selection-Conditional Conformal Family Editor":
        return apply_sccfe(batch, prepared, policy, calib_bias or {}, calib_expected_harm or {})
    if policy.variant == "Retrieval-Augmented Local Residual Editor":
        return apply_ralre(batch, prepared, policy)
    if policy.variant == "Conservative Challenger Comparator":
        return apply_ccc(batch, prepared, policy)
    raise ValueError(f"unknown variant {policy.variant}")


def evaluate_variant(
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
        selected_mask = decisions["selected"].to_numpy(bool)
        row["accept_precision"] = float(np.mean(method[selected_mask] < base[selected_mask])) if selected_mask.any() else 0.0
        row["mean_expected_harm"] = float(decisions["expected_harm"].mean())
        observed = float(np.mean(method[selected_mask] > base[selected_mask] + 1e-12)) if selected_mask.any() else 0.0
        row["observed_selected_harm"] = observed
        row["expected_observed_harm_gap_pp"] = float(100.0 * abs(row["mean_expected_harm"] - observed))
        row["mean_accept_score"] = float(decisions["accept_score"].mean())
    return row


def calibration_score(row: Mapping[str, Any], variant: str) -> float:
    score = float(row["mse_delta_pct_vs_lrbn"])
    if variant == "Residual-Prior Convex Mixer":
        harm_cap, max_cap, oracle_min = 0.06, 0.15, 0.08
    elif variant == "Time-Step Gated Hybrid Editor":
        harm_cap, max_cap, oracle_min = 0.15, 0.15, 0.04
    elif variant == "Selection-Conditional Conformal Family Editor":
        harm_cap, max_cap, oracle_min = 0.10, 0.18, 0.03
    elif variant == "Retrieval-Augmented Local Residual Editor":
        harm_cap, max_cap, oracle_min = 0.15, 0.18, 0.03
    else:
        harm_cap, max_cap, oracle_min = 0.12, 0.18, 0.10
    score += 220.0 * max(0.0, float(row["harm_rate"]) - harm_cap)
    score += 180.0 * max(0.0, float(row["max_config_harm"]) - max_cap)
    score += 30.0 * max(0.0, oracle_min - float(row.get("oracle_gain_fraction", 0.0)))
    if variant == "Selection-Conditional Conformal Family Editor":
        score += 2.0 * max(0.0, float(row.get("expected_observed_harm_gap_pp", 999.0)) - 5.0)
    return score


def calibrate_variant(
    variant: str,
    calib: ForecastBatch,
    prepared: PreparedGeometry,
    oracle_mse: np.ndarray,
    seed: int,
) -> Tuple[GeometryPolicy, pd.DataFrame, Dict[str, np.ndarray], Dict[str, float]]:
    rows: List[Dict[str, Any]] = []
    best_policy: Optional[GeometryPolicy] = None
    best_score = float("inf")
    best_bias: Dict[str, np.ndarray] = {}
    best_expected: Dict[str, float] = {}
    for policy in policy_grid(variant):
        bias: Dict[str, np.ndarray] = {}
        expected: Dict[str, float] = {}
        if variant == "Selection-Conditional Conformal Family Editor":
            bias, expected = fit_bias_centers(calib, prepared, policy)
        pred, decisions = apply_variant(calib, prepared, policy, bias, expected)
        row = evaluate_variant(variant, pred, calib, decisions, oracle_mse, n_bootstrap=0, seed=seed)
        row.update(asdict(policy))
        row["calibration_score"] = calibration_score(row, variant)
        rows.append(row)
        if float(row["calibration_score"]) < best_score:
            best_score = float(row["calibration_score"])
            best_policy = policy
            best_bias = bias
            best_expected = expected
    assert best_policy is not None
    return best_policy, pd.DataFrame(rows), best_bias, best_expected


def selection_distribution(decisions_by_variant: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for variant, df in decisions_by_variant.items():
        vc = df["selected_action"].value_counts(normalize=True).rename_axis("selected_action").reset_index(name="share")
        vc.insert(0, "variant", variant)
        frames.append(vc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _slice_value(slice_df: pd.DataFrame, variant: str, name: str) -> float:
    row = slice_df[slice_df["variant"].eq(variant) & slice_df["slice"].eq(name)]
    if row.empty:
        return float("nan")
    return float(row["mse_delta_pct_vs_lrbn"].iloc[0])


def _known_delta(per_config: pd.DataFrame, variant: str) -> float:
    row = per_config[
        per_config["variant"].eq(variant)
        & per_config["dataset"].eq("ETTm1")
        & per_config["backbone"].eq("DLinear")
        & per_config["horizon"].eq(192)
    ]
    return float(row["mse_delta_pct_vs_lrbn"].iloc[0]) if not row.empty else float("nan")


def gate_table(overall: pd.DataFrame, slice_df: pd.DataFrame, per_config: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in overall[overall["variant"].isin(STAGE13_VARIANTS)].iterrows():
        variant = str(row["variant"])
        q4 = _slice_value(slice_df, variant, "q4_boundary")
        non_boundary = _slice_value(slice_df, variant, "non_boundary")
        low_gap = _slice_value(slice_df, variant, "low_gap_high_repair")
        known = _known_delta(per_config, variant)
        base = {
            "variant": variant,
            "mse_delta_pct_vs_lrbn": float(row["mse_delta_pct_vs_lrbn"]),
            "harm_rate": float(row["harm_rate"]),
            "max_config_harm": float(row["max_config_harm"]),
            "oracle_gain_fraction": float(row.get("oracle_gain_fraction", np.nan)),
            "coverage": float(row.get("coverage", np.nan)),
            "accept_precision": float(row.get("accept_precision", np.nan)),
            "selected_nonharm_rate": float(1.0 - row.get("observed_selected_harm", np.nan)),
            "expected_observed_harm_gap_pp": float(row.get("expected_observed_harm_gap_pp", np.nan)),
            "q4_boundary_delta_pct": q4,
            "non_boundary_delta_pct": non_boundary,
            "low_gap_high_repair_delta_pct": low_gap,
            "known_harmed_config_delta_pct": known,
            "bootstrap_high_delta_raw": float(row.get("ci95_high_delta_raw", np.nan)),
        }
        if variant == "Residual-Prior Convex Mixer":
            passed = (
                base["mse_delta_pct_vs_lrbn"] <= -1.8
                and base["harm_rate"] <= 0.06
                and base["max_config_harm"] <= 0.15
                and base["oracle_gain_fraction"] >= 0.08
                and base["bootstrap_high_delta_raw"] < 0
            )
        elif variant == "Time-Step Gated Hybrid Editor":
            passed = (
                base["q4_boundary_delta_pct"] < 0
                and base["non_boundary_delta_pct"] <= -2.0
                and base["max_config_harm"] <= 0.15
                and base["bootstrap_high_delta_raw"] < 0
            )
        elif variant == "Selection-Conditional Conformal Family Editor":
            passed = (
                base["expected_observed_harm_gap_pp"] <= 5.0
                and 0.88 <= base["selected_nonharm_rate"] <= 0.92
                and base["mse_delta_pct_vs_lrbn"] < 0
                and base["bootstrap_high_delta_raw"] < 0
            )
        elif variant == "Retrieval-Augmented Local Residual Editor":
            passed = (
                base["non_boundary_delta_pct"] < 0
                and base["low_gap_high_repair_delta_pct"] < 0
                and base["q4_boundary_delta_pct"] <= 0.1
                and base["oracle_gain_fraction"] >= 0.03
                and base["bootstrap_high_delta_raw"] < 0
            )
        else:
            passed = (
                base["accept_precision"] >= 0.60
                and base["oracle_gain_fraction"] >= 0.10
                and base["bootstrap_high_delta_raw"] < 0
            )
        base["compact_gate_pass"] = bool(passed)
        rows.append(base)
    return pd.DataFrame(rows)


def verify_outputs(output_dir: Path) -> pd.DataFrame:
    required = [
        "stage13_config.json",
        "stage13_candidate_metadata.csv",
        "stage13_topk_metrics.csv",
        "stage13_calibration_grid.csv",
        "stage13_overall.csv",
        "stage13_per_config.csv",
        "stage13_slice_metrics.csv",
        "stage13_selection_distribution.csv",
        "stage13_policies.json",
        "stage13_gate_table.csv",
        "stage13_verdict.json",
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
        "accept_precision",
        "expected_observed_harm_gap_pp",
        "ci95_high_delta_raw",
    ]
    show_cols = [c for c in cols if c in overall.columns]
    return "\n".join(
        [
            "# Stage 13 CGA Decision-Geometry Validation",
            "",
            f"Status: `{verdict['status']}`.",
            "",
            "## Overall",
            "",
            df_to_md(overall[show_cols], max_rows=32),
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
    print("[stage13-cga-geometry] preparing assets", flush=True)
    assets = prepare_assets(metrics_csv, stage5_dir, stage3_dir, seed)
    print(f"[stage13-cga-geometry] assets ready in {time.time() - start:.1f}s", flush=True)
    pools = build_cga_pools(assets)
    write_json(
        output_dir / "stage13_config.json",
        {
            "stage": "stage13_cga_decision_geometry",
            "source_plan": "deep-research-report (3).md",
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
    candidate_metadata(pools.test_candidates, "test").to_csv(output_dir / "stage13_candidate_metadata.csv", index=False)
    print("[stage13-cga-geometry] fitting CGA scoring models", flush=True)
    models, _, _, _ = fit_cga_models(
        assets.val_train,
        assets.val_calib,
        pools.train_candidates,
        pools.calib_candidates,
        assets.schema,
        seed=seed,
    )
    topk = topk_metrics(assets.test, deployable_candidates(pools.test_candidates), assets.schema, models, k=2)
    pd.DataFrame([topk]).to_csv(output_dir / "stage13_topk_metrics.csv", index=False)
    oracle_pred, oracle_mse, _ = oracle_best(deployable_candidates(pools.test_candidates), assets.test)
    calib_oracle_mse = oracle_best(deployable_candidates(pools.calib_candidates), assets.val_calib)[1]

    calib_thresholds = temporal_threshold_map(temporal_boundary_score(assets.val_calib), [0.5, 0.65, 0.75, 0.85, 0.9])
    calib_prepared = prepare_geometry(assets.val_calib, pools.calib_candidates, assets.schema, models, temporal_thresholds=calib_thresholds)
    test_scores = prepare_score_bundle(assets.test, pools.test_candidates, assets.schema, models)
    test_prepared = prepare_geometry(assets.test, pools.test_candidates, assets.schema, models, score_cache=test_scores, temporal_thresholds=calib_thresholds)

    policies: Dict[str, GeometryPolicy] = {}
    bias_by: Dict[str, Dict[str, np.ndarray]] = {}
    expected_by: Dict[str, Dict[str, float]] = {}
    grid_frames: List[pd.DataFrame] = []
    preds: Dict[str, np.ndarray] = {
        "LRBN": assets.test.lrbn_pred,
        "SRA-BP-balanced": candidate_dict(assets.old_test_candidates).get("sra_balanced", assets.old_test_candidates[0]).pred,
        "oracle_stage13_cga_full": oracle_pred,
    }
    decisions_by: Dict[str, pd.DataFrame] = {}
    print("[stage13-cga-geometry] calibrating five candidates", flush=True)
    for variant in STAGE13_VARIANTS:
        print(f"[stage13-cga-geometry] calibrating {variant}", flush=True)
        policy, grid, bias, expected = calibrate_variant(variant, assets.val_calib, calib_prepared, calib_oracle_mse, seed)
        grid["target_variant"] = variant
        grid_frames.append(grid)
        policies[variant] = policy
        bias_by[variant] = bias
        expected_by[variant] = expected
        pred, decisions = apply_variant(assets.test, test_prepared, policy, bias, expected)
        preds[variant] = pred
        decisions_by[variant] = decisions

    print("[stage13-cga-geometry] evaluating", flush=True)
    overall_rows: List[Dict[str, Any]] = []
    for variant, pred in preds.items():
        decisions = decisions_by.get(variant)
        overall_rows.append(evaluate_variant(variant, pred, assets.test, decisions, oracle_mse if variant != "LRBN" else None, n_bootstrap, seed))
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(output_dir / "stage13_overall.csv", index=False)
    pd.concat(grid_frames, ignore_index=True).to_csv(output_dir / "stage13_calibration_grid.csv", index=False)
    write_json(output_dir / "stage13_policies.json", {variant: asdict(policy) for variant, policy in policies.items()})

    per_config = pd.concat(
        [per_config_rows(variant, pred, assets.test, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    per_config.to_csv(output_dir / "stage13_per_config.csv", index=False)
    masks = stage8_slice_masks(assets.test, assets.schema, stage8_slice_thresholds(assets.val, assets.schema))
    slice_df = pd.concat(
        [slice_rows(variant, pred, assets.test, masks, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    slice_df.to_csv(output_dir / "stage13_slice_metrics.csv", index=False)
    selection_distribution(decisions_by).to_csv(output_dir / "stage13_selection_distribution.csv", index=False)
    gates = gate_table(overall, slice_df, per_config)
    gates.to_csv(output_dir / "stage13_gate_table.csv", index=False)

    deployable = overall[overall["variant"].isin(STAGE13_VARIANTS)].copy()
    best = deployable.sort_values(["mse_delta_pct_vs_lrbn", "max_config_harm", "harm_rate"], ascending=[True, True, True]).iloc[0]
    passed = gates[gates["compact_gate_pass"]]
    verdict = {
        "stage": "stage13_cga_decision_geometry",
        "status": "compact_pass_ready_for_mini_extension" if not passed.empty else "compact_failed_stop_before_mini_extension",
        "compact_pass": bool(not passed.empty),
        "passed_variants": passed["variant"].astype(str).tolist(),
        "best_variant": str(best["variant"]),
        "best_mse": float(best["mse"]),
        "best_mae": float(best["mae"]),
        "best_mse_delta_pct_vs_lrbn": float(best["mse_delta_pct_vs_lrbn"]),
        "best_harm_rate": float(best["harm_rate"]),
        "best_max_config_harm": float(best["max_config_harm"]),
        "best_oracle_gain_fraction": float(best["oracle_gain_fraction"]),
        "family_top2_hit": float(topk.get("family_top2_hit", np.nan)),
        "candidate_top2_hit": float(topk.get("candidate_top2_hit", np.nan)),
        "test_threshold_leakage": False,
        "stop_reason": None,
    }
    if passed.empty:
        verdict["stop_reason"] = "no decision-geometry candidate passed compact gates"
    write_json(output_dir / "stage13_verdict.json", verdict)
    (output_dir / "summary.md").write_text(build_summary(output_dir, verdict, overall, gates), encoding="utf-8")
    completeness = verify_outputs(output_dir)
    completeness.to_csv(output_dir / "stage13_output_completeness.csv", index=False)
    print(f"[stage13-cga-geometry] done in {time.time() - start:.1f}s", flush=True)
    return {
        "verdict": verdict,
        "overall": overall,
        "per_config": per_config,
        "slice": slice_df,
        "gates": gates,
        "completeness": completeness,
    }
