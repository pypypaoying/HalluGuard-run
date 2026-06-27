#!/usr/bin/env python
"""Stage 7 Safe-TAE validation utilities.

Safe-TAE keeps HalluGuard-LRBN as the default trajectory, then admits
candidate experts only when validation-fitted pairwise heads predict both
non-trivial gain and low harm. All trainable choices are validation-only; the
test split is used only after the policy is fixed.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from halluguard_lrbn_bp import (
    EPS,
    ForecastBatch,
    lrbn_optional_bp,
    mae_per_sample,
    mse_per_sample,
)
from halluguard_sra_bp import SRABPParams, apply_sra_bp
from halluguard_stage4_bp_harm_control import apply_candidate
from halluguard_stage6_mechanism import (
    amplitude_candidate,
    bootstrap_ci,
    feature_frame,
    feature_schema,
    fit_ridge_residual_models,
    horizons,
    level_bias_candidate,
    metric_summary,
    phase_shift_candidate,
    predict_harm_risk,
    predict_ridge_residual,
    safe_pct,
    select_abstention_threshold,
    select_mrc_shrink_cap,
    slice_masks,
    slice_thresholds,
    train_harm_model,
    volatility_shrink_candidate,
    apply_mrc_shrink_cap,
)


@dataclass
class ExpertCandidate:
    name: str
    tier: str
    family: str
    pred: np.ndarray
    deployable: bool = True


@dataclass
class PolicyParams:
    variant: str
    tau_leave: float
    tau_gain: float
    tau_harm: float
    risk_beta: float
    lambda_safe: float
    lambda_balanced: float
    lambda_aggressive: float
    cos_min: Optional[float]
    hard_replacement: bool = False
    allow_aggressive: bool = False

    def lambda_for(self, tier: str) -> float:
        if self.hard_replacement:
            return 1.0
        if tier == "safe":
            return self.lambda_safe
        if tier == "balanced":
            return self.lambda_balanced
        if tier == "aggressive":
            return self.lambda_aggressive
        return 0.0


class ConstantProbability:
    def __init__(self, p: float):
        self.p = float(np.clip(p, 0.0, 1.0))

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        return np.column_stack([np.full(n, 1.0 - self.p), np.full(n, self.p)])


@dataclass
class FittedHeads:
    leave_head: Any
    gain_heads: Dict[str, Any]
    harm_heads: Dict[str, Any]
    sample_columns: List[str]
    expert_columns: Dict[str, List[str]]
    metrics: pd.DataFrame


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


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def split_batch(batch: ForecastBatch) -> Tuple[ForecastBatch, ForecastBatch]:
    split = batch.meta["split"].astype(str).to_numpy()
    return batch.subset(split == "val"), batch.subset(split == "test")


def stratified_inner_split(meta: pd.DataFrame, train_frac: float = 0.70, seed: int = 2026) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train = np.zeros(len(meta), dtype=bool)
    group_cols = ["dataset", "backbone", "horizon", "seed"]
    for _, group in meta.reset_index().groupby(group_cols, observed=True):
        idx = group["index"].to_numpy(int)
        rng.shuffle(idx)
        n_train = int(round(len(idx) * train_frac))
        n_train = min(max(1, n_train), max(1, len(idx) - 1))
        train[idx[:n_train]] = True
    return train, ~train


def subset_candidates(candidates: Sequence[ExpertCandidate], mask: np.ndarray) -> List[ExpertCandidate]:
    return [ExpertCandidate(c.name, c.tier, c.family, c.pred[mask], c.deployable) for c in candidates]


def array_cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    aa = a.reshape(a.shape[0], -1)
    bb = b.reshape(b.shape[0], -1)
    return np.sum(aa * bb, axis=1) / (np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1) + eps)


def robust_nan_to_num(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def fit_binary_head(x: pd.DataFrame, y: np.ndarray) -> Any:
    labels = np.asarray(y, dtype=int)
    if len(np.unique(labels)) < 2:
        return ConstantProbability(float(labels.mean()) if len(labels) else 0.0)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, random_state=2026),
    )
    model.fit(x.to_numpy(float), labels)
    return model


def predict_head(head: Any, x: pd.DataFrame) -> np.ndarray:
    return np.asarray(head.predict_proba(x.to_numpy(float))[:, 1], dtype=float)


def binary_metrics(name: str, y: np.ndarray, p: np.ndarray, split: str, expert: str = "") -> Dict[str, Any]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    row: Dict[str, Any] = {
        "split": split,
        "head": name,
        "expert": expert,
        "n": int(len(y)),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "mean_probability": float(p.mean()) if len(p) else 0.0,
    }
    if len(np.unique(y)) >= 2:
        row["roc_auc"] = float(roc_auc_score(y, p))
        row["pr_auc"] = float(average_precision_score(y, p))
        row["brier"] = float(brier_score_loss(y, p))
    else:
        row["roc_auc"] = float("nan")
        row["pr_auc"] = float("nan")
        row["brier"] = float("nan")
    return row


def load_stage3_params(stage3_dir: Optional[Path]) -> Optional[Dict[str, Any]]:
    if stage3_dir is None:
        return None
    path = stage3_dir / "selected_lrbn_bp_params.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("HalluGuard-LRBN-BP-gated")


def select_basic_sra_params(stage5_dir: Path) -> Dict[str, Any]:
    grid = pd.read_csv(stage5_dir / "stage5_calibration_grid.csv")
    rows = grid[(grid["split"].eq("val")) & (grid["method"].eq("LRBN-SRA-BP-basic"))].copy()
    if rows.empty:
        return {
            "method_family": "basic",
            "anchor_mode": "last",
            "tail_len": 16,
            "tau_g": 2.0,
            "tau_r": 0.8,
            "tau_j": None,
            "alpha": 0.75,
            "K": "H_div_4",
            "continuous": False,
            "kg": 4.0,
            "kr": 4.0,
            "kj": 4.0,
        }
    feasible = rows[rows["harm_rate"] <= 0.15].copy()
    chosen = (feasible if not feasible.empty else rows).sort_values(["mse_delta_pct_vs_lrbn", "harm_rate"]).iloc[0]
    out = {
        "method_family": "basic",
        "anchor_mode": str(chosen.get("anchor_mode", "last")),
        "tail_len": int(chosen.get("tail_len", 16)),
        "tau_g": float(chosen.get("tau_g", 2.0)),
        "tau_r": float(chosen.get("tau_r", 0.8)),
        "tau_j": None if pd.isna(chosen.get("tau_j", np.nan)) else float(chosen.get("tau_j")),
        "alpha": float(chosen.get("alpha", 0.75)),
        "K": chosen.get("K", "H_div_4"),
        "continuous": str(chosen.get("continuous", "False")).lower() == "true",
        "kg": float(chosen.get("kg", 4.0)),
        "kr": float(chosen.get("kr", 4.0)),
        "kj": float(chosen.get("kj", 4.0)),
    }
    return out


def make_sra_pred(batch: ForecastBatch, params: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    pred, aux = apply_sra_bp(batch.context, batch.raw_pred, batch.lrbn_pred, horizons(batch), dict(params))
    return pred, np.asarray(aux["strength"], dtype=float)


def make_stage3_gated(batch: ForecastBatch, params: Optional[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    if not params:
        return batch.lrbn_pred.copy(), np.zeros(len(batch.meta), dtype=float)
    pred, selected, _ = lrbn_optional_bp(
        batch.context,
        batch.lrbn_pred,
        alpha=float(params.get("alpha", 0.5)),
        tau=float(params.get("tau", np.inf)),
        tail=int(params.get("tail", 24)),
        decay=str(params.get("decay", "linear")),
        horizons=horizons(batch),
    )
    return pred, selected.astype(float)


def build_mrc_artifacts(val: ForecastBatch, test: ForecastBatch, schema: Dict[str, List[Any]]) -> Dict[str, Any]:
    x_val = feature_frame(val, schema)
    x_test = feature_frame(test, schema)
    ridge = fit_ridge_residual_models(val, x_val)
    raw_test_delta = predict_ridge_residual(ridge, test, x_test)
    shrink_cap_params, shrink_cap_grid, val_delta = select_mrc_shrink_cap(val, ridge.cv_pred)
    test_delta = apply_mrc_shrink_cap(
        test,
        raw_test_delta,
        shrink=float(shrink_cap_params["shrink"]),
        cap_mult=float(shrink_cap_params["cap_mult"]),
    )
    val_point = val.lrbn_pred + val_delta
    harm_label = mse_per_sample(val_point, val.y_true) > mse_per_sample(val.lrbn_pred, val.y_true) + 1e-12
    harm_model = train_harm_model(x_val, harm_label)
    val_risk = predict_harm_risk(harm_model, x_val)
    tau, abstain_curve = select_abstention_threshold(val, val_delta, val_risk)
    test_risk = predict_harm_risk(harm_model, x_test)
    val_selected = val_risk <= tau
    test_selected = test_risk <= tau
    return {
        "val_delta": val_delta,
        "test_delta": test_delta,
        "val_abstain_pred": val.lrbn_pred + val_delta * val_selected.reshape(-1, 1, 1),
        "test_abstain_pred": test.lrbn_pred + test_delta * test_selected.reshape(-1, 1, 1),
        "val_selected": val_selected,
        "test_selected": test_selected,
        "shrink_cap_params": shrink_cap_params,
        "shrink_cap_grid": shrink_cap_grid,
        "abstain_curve": abstain_curve,
        "risk_threshold": tau,
    }


def build_candidate_pool(
    batch: ForecastBatch,
    safe_params: Mapping[str, Any],
    balanced_params: Mapping[str, Any],
    basic_params: Mapping[str, Any],
    stage3_params: Optional[Mapping[str, Any]],
    mrc_pred: np.ndarray,
    mrc_abstain_pred: np.ndarray,
) -> List[ExpertCandidate]:
    sra_safe, _ = make_sra_pred(batch, safe_params)
    sra_balanced, _ = make_sra_pred(batch, balanced_params)
    sra_basic, _ = make_sra_pred(batch, basic_params)
    stage3, _ = make_stage3_gated(batch, stage3_params)
    bp_always = apply_candidate(
        batch,
        {"method": "LRBN-BP-always", "alpha": 0.5, "anchor_mode": "last", "bridge_mode": "linear"},
    ).pred
    level_light = level_bias_candidate(batch, alpha=0.25)
    level = level_bias_candidate(batch, alpha=0.50)
    amp_shrink = amplitude_candidate(batch, scale=0.85)
    amp_expand = amplitude_candidate(batch, scale=1.10)
    phase_short = phase_shift_candidate(batch, shift=1, blend=0.35)
    vol_shrink = volatility_shrink_candidate(batch, alpha=0.35)
    safe_stack = np.stack([sra_safe, stage3, mrc_abstain_pred, vol_shrink, amp_shrink], axis=0)
    ensemble_median = np.nanmedian(safe_stack, axis=0)
    return [
        ExpertCandidate("keep_lrbn", "default", "default", batch.lrbn_pred.copy(), True),
        ExpertCandidate("sra_safe", "safe", "boundary", sra_safe, True),
        ExpertCandidate("sra_balanced", "balanced", "boundary", sra_balanced, True),
        ExpertCandidate("sra_basic_ablation", "balanced", "boundary", sra_basic, True),
        ExpertCandidate("stage3_gated_bp", "safe", "boundary", stage3, True),
        ExpertCandidate("mrc_ridge_abstain", "safe", "residual", mrc_abstain_pred, True),
        ExpertCandidate("mrc_ridge_residual_blend", "balanced", "residual", batch.lrbn_pred + mrc_pred, True),
        ExpertCandidate("volatility_shrink", "safe", "volatility", vol_shrink, True),
        ExpertCandidate("level_bias_bounded", "balanced", "level", level, True),
        ExpertCandidate("level_bias_light", "safe", "level", level_light, True),
        ExpertCandidate("amplitude_scale_bounded", "balanced", "amplitude", amp_shrink, True),
        ExpertCandidate("amplitude_expand", "aggressive", "amplitude", amp_expand, False),
        ExpertCandidate("phase_shift_short", "aggressive", "phase", phase_short, False),
        ExpertCandidate("ensemble_median", "safe", "ensemble", ensemble_median, True),
        ExpertCandidate("bp_always", "aggressive", "boundary", bp_always, False),
        ExpertCandidate("raw", "aggressive", "raw", batch.raw_pred.copy(), False),
    ]


def candidate_dict(candidates: Sequence[ExpertCandidate]) -> Dict[str, ExpertCandidate]:
    return {c.name: c for c in candidates}


def candidate_metric_rows(candidates: Sequence[ExpertCandidate], batch: ForecastBatch, split: str) -> List[Dict[str, Any]]:
    rows = []
    for c in candidates:
        row = metric_summary(c.name, c.pred, batch)
        row.update({"split": split, "tier": c.tier, "family": c.family, "deployable": bool(c.deployable)})
        rows.append(row)
    return rows


def sample_candidate_table(candidates: Sequence[ExpertCandidate], batch: ForecastBatch, split: str) -> pd.DataFrame:
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    base_mae = mae_per_sample(batch.lrbn_pred, batch.y_true)
    rows: List[Dict[str, Any]] = []
    meta = batch.meta.reset_index(drop=True)
    for c in candidates:
        cmse = mse_per_sample(c.pred, batch.y_true)
        cmae = mae_per_sample(c.pred, batch.y_true)
        corr_norm = np.linalg.norm((c.pred - batch.lrbn_pred).reshape(len(meta), -1), axis=1)
        for i in range(len(meta)):
            rows.append(
                {
                    "split": split,
                    "row": i,
                    "sample_id": meta.loc[i, "sample_id"],
                    "dataset": meta.loc[i, "dataset"],
                    "backbone": meta.loc[i, "backbone"],
                    "horizon": int(meta.loc[i, "horizon"]),
                    "seed": int(meta.loc[i, "seed"]),
                    "candidate": c.name,
                    "tier": c.tier,
                    "family": c.family,
                    "deployable": bool(c.deployable),
                    "mse": float(cmse[i]),
                    "mae": float(cmae[i]),
                    "mse_delta_vs_lrbn": float(cmse[i] - base_mse[i]),
                    "mae_delta_vs_lrbn": float(cmae[i] - base_mae[i]),
                    "gain_label": bool(cmse[i] < base_mse[i] - 1e-4),
                    "harm_label": bool(cmse[i] > base_mse[i] + 1e-4),
                    "corr_norm": float(corr_norm[i]),
                }
            )
    return pd.DataFrame(rows)


def expert_features(batch: ForecastBatch, schema: Dict[str, List[Any]], candidates: Sequence[ExpertCandidate], mrc_prior: np.ndarray) -> Dict[str, pd.DataFrame]:
    base = feature_frame(batch, schema)
    out: Dict[str, pd.DataFrame] = {}
    base_norm = np.linalg.norm(batch.lrbn_pred.reshape(len(batch.meta), -1), axis=1) + EPS
    for c in candidates:
        delta = c.pred - batch.lrbn_pred
        dflat = delta.reshape(len(batch.meta), -1)
        corr_norm = np.linalg.norm(dflat, axis=1)
        early = delta[:, : max(1, delta.shape[1] // 4), :]
        mid = delta[:, delta.shape[1] // 4 : 3 * delta.shape[1] // 4, :]
        late = delta[:, 3 * delta.shape[1] // 4 :, :]
        extra = pd.DataFrame(
            {
                "corr_norm": corr_norm,
                "corr_norm_ratio": corr_norm / base_norm,
                "corr_mean": np.nanmean(delta, axis=(1, 2)),
                "corr_std": np.nanstd(delta, axis=(1, 2)),
                "early_energy": np.nanmean(early ** 2, axis=(1, 2)),
                "mid_energy": np.nanmean(mid ** 2, axis=(1, 2)) if mid.size else np.zeros(len(batch.meta)),
                "late_energy": np.nanmean(late ** 2, axis=(1, 2)),
                "cos_mrc_prior": array_cosine(delta, mrc_prior),
                "tier_safe": float(c.tier == "safe"),
                "tier_balanced": float(c.tier == "balanced"),
                "tier_aggressive": float(c.tier == "aggressive"),
                "family_boundary": float(c.family == "boundary"),
                "family_residual": float(c.family == "residual"),
                "family_volatility": float(c.family == "volatility"),
                "family_level": float(c.family == "level"),
                "family_amplitude": float(c.family == "amplitude"),
                "family_phase": float(c.family == "phase"),
                "family_ensemble": float(c.family == "ensemble"),
            }
        )
        out[c.name] = robust_nan_to_num(pd.concat([base.reset_index(drop=True), extra], axis=1))
    return out


def align_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return robust_nan_to_num(df.reindex(columns=columns, fill_value=0.0))


def fit_heads(
    train_batch: ForecastBatch,
    calib_batch: ForecastBatch,
    train_candidates: Sequence[ExpertCandidate],
    calib_candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    train_mrc_prior: np.ndarray,
    calib_mrc_prior: np.ndarray,
    eps_gain: float = 1e-4,
    eps_harm: float = 1e-4,
) -> FittedHeads:
    train_base_mse = mse_per_sample(train_batch.lrbn_pred, train_batch.y_true)
    train_sample_x = feature_frame(train_batch, schema)
    calib_sample_x = feature_frame(calib_batch, schema)
    sample_columns = list(train_sample_x.columns)
    deploy_train = [c for c in train_candidates if c.name != "keep_lrbn" and c.deployable]
    deploy_calib = [c for c in calib_candidates if c.name != "keep_lrbn" and c.deployable]
    any_gain = np.zeros(len(train_batch.meta), dtype=bool)
    for c in deploy_train:
        any_gain |= mse_per_sample(c.pred, train_batch.y_true) < train_base_mse - eps_gain
    calib_base_mse = mse_per_sample(calib_batch.lrbn_pred, calib_batch.y_true)
    calib_any_gain = np.zeros(len(calib_batch.meta), dtype=bool)
    for c in deploy_calib:
        calib_any_gain |= mse_per_sample(c.pred, calib_batch.y_true) < calib_base_mse - eps_gain
    leave_head = fit_binary_head(train_sample_x, any_gain)

    train_expert_x = expert_features(train_batch, schema, train_candidates, train_mrc_prior)
    calib_expert_x = expert_features(calib_batch, schema, calib_candidates, calib_mrc_prior)
    rows = [
        binary_metrics("leave", any_gain, predict_head(leave_head, train_sample_x), split="inner_train"),
        binary_metrics("leave", calib_any_gain, predict_head(leave_head, calib_sample_x), split="inner_calib"),
    ]
    gain_heads: Dict[str, Any] = {}
    harm_heads: Dict[str, Any] = {}
    expert_columns: Dict[str, List[str]] = {}
    train_by = candidate_dict(train_candidates)
    calib_by = candidate_dict(calib_candidates)
    for name, c in train_by.items():
        if name == "keep_lrbn" or not c.deployable:
            continue
        delta = mse_per_sample(c.pred, train_batch.y_true) - train_base_mse
        gain_label = delta < -eps_gain
        harm_label = delta > eps_harm
        x_train = train_expert_x[name]
        cols = list(x_train.columns)
        expert_columns[name] = cols
        gain_heads[name] = fit_binary_head(x_train, gain_label)
        harm_heads[name] = fit_binary_head(x_train, harm_label)
        rows.append(binary_metrics("gain", gain_label, predict_head(gain_heads[name], x_train), "inner_train", name))
        rows.append(binary_metrics("harm", harm_label, predict_head(harm_heads[name], x_train), "inner_train", name))
        if name in calib_by:
            calib_delta = mse_per_sample(calib_by[name].pred, calib_batch.y_true) - mse_per_sample(calib_batch.lrbn_pred, calib_batch.y_true)
            x_calib = align_frame(calib_expert_x[name], cols)
            rows.append(binary_metrics("gain", calib_delta < -eps_gain, predict_head(gain_heads[name], x_calib), "inner_calib", name))
            rows.append(binary_metrics("harm", calib_delta > eps_harm, predict_head(harm_heads[name], x_calib), "inner_calib", name))
    return FittedHeads(
        leave_head=leave_head,
        gain_heads=gain_heads,
        harm_heads=harm_heads,
        sample_columns=sample_columns,
        expert_columns=expert_columns,
        metrics=pd.DataFrame(rows),
    )


def compute_probabilities(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    mrc_prior: np.ndarray,
    heads: FittedHeads,
) -> Dict[str, Any]:
    sample_x = align_frame(feature_frame(batch, schema), heads.sample_columns)
    ex_x = expert_features(batch, schema, candidates, mrc_prior)
    by_name = candidate_dict(candidates)
    probs = {
        "p_leave": predict_head(heads.leave_head, sample_x),
        "p_gain": {},
        "p_harm": {},
        "cos": {},
    }
    for name in heads.gain_heads:
        if name not in by_name:
            continue
        x = align_frame(ex_x[name], heads.expert_columns[name])
        probs["p_gain"][name] = predict_head(heads.gain_heads[name], x)
        probs["p_harm"][name] = predict_head(heads.harm_heads[name], x)
        probs["cos"][name] = np.asarray(x["cos_mrc_prior"], dtype=float) if "cos_mrc_prior" in x.columns else np.zeros(len(batch.meta))
    return probs


def policy_from_row(row: Mapping[str, Any]) -> PolicyParams:
    return PolicyParams(
        variant=str(row["variant"]),
        tau_leave=float(row["tau_leave"]),
        tau_gain=float(row["tau_gain"]),
        tau_harm=float(row["tau_harm"]),
        risk_beta=float(row["risk_beta"]),
        lambda_safe=float(row["lambda_safe"]),
        lambda_balanced=float(row["lambda_balanced"]),
        lambda_aggressive=float(row.get("lambda_aggressive", 0.0)),
        cos_min=None if pd.isna(row.get("cos_min", np.nan)) else float(row.get("cos_min")),
        hard_replacement=bool(row.get("hard_replacement", False)),
        allow_aggressive=bool(row.get("allow_aggressive", False)),
    )


def apply_policy(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    probs: Mapping[str, Any],
    policy: PolicyParams,
) -> Tuple[np.ndarray, pd.DataFrame]:
    n = len(batch.meta)
    out = batch.lrbn_pred.copy()
    selected = np.array(["keep_lrbn"] * n, dtype=object)
    selected_tier = np.array(["default"] * n, dtype=object)
    selected_lambda = np.zeros(n, dtype=float)
    selected_score = np.full(n, -np.inf, dtype=float)
    leave = np.asarray(probs["p_leave"], dtype=float) >= policy.tau_leave
    for c in candidates:
        if c.name == "keep_lrbn":
            continue
        if not c.deployable and not policy.allow_aggressive:
            continue
        if c.name not in probs["p_gain"]:
            continue
        pg = np.asarray(probs["p_gain"][c.name], dtype=float)
        ph = np.asarray(probs["p_harm"][c.name], dtype=float)
        cos = np.asarray(probs["cos"].get(c.name, np.zeros(n)), dtype=float)
        admissible = leave & (pg >= policy.tau_gain) & (ph <= policy.tau_harm)
        if policy.cos_min is not None:
            admissible &= cos >= policy.cos_min
        score = pg - policy.risk_beta * ph + 0.05 * cos
        score[~admissible] = -np.inf
        take = score > selected_score
        if take.any():
            lam = float(policy.lambda_for(c.tier))
            out[take] = batch.lrbn_pred[take] + lam * (c.pred[take] - batch.lrbn_pred[take])
            selected[take] = c.name
            selected_tier[take] = c.tier
            selected_lambda[take] = lam
            selected_score[take] = score[take]
    decisions = batch.meta[["dataset", "backbone", "horizon", "seed", "sample_id"]].reset_index(drop=True).copy()
    decisions["variant"] = policy.variant
    decisions["selected_expert"] = selected
    decisions["selected_tier"] = selected_tier
    decisions["selected_lambda"] = selected_lambda
    decisions["selected_score"] = np.where(np.isfinite(selected_score), selected_score, 0.0)
    decisions["p_leave"] = np.asarray(probs["p_leave"], dtype=float)
    return out, decisions


def per_config_rows(variant: str, pred: np.ndarray, batch: ForecastBatch) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    method_mse = mse_per_sample(pred, batch.y_true)
    base_mae = mae_per_sample(batch.lrbn_pred, batch.y_true)
    method_mae = mae_per_sample(pred, batch.y_true)
    meta = batch.meta.assign(row_index=np.arange(len(batch.meta)))
    for keys, group in meta.groupby(["dataset", "backbone", "horizon", "seed"], observed=True):
        idx = group["row_index"].to_numpy(int)
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
                "lrbn_mse": float(np.mean(base_mse[idx])),
                "lrbn_mae": float(np.mean(base_mae[idx])),
                "mse_delta_pct_vs_lrbn": safe_pct(float(np.mean(method_mse[idx])), float(np.mean(base_mse[idx]))),
                "mae_delta_pct_vs_lrbn": safe_pct(float(np.mean(method_mae[idx])), float(np.mean(base_mae[idx]))),
                "harm_rate": float(np.mean(method_mse[idx] > base_mse[idx] + 1e-12)),
                "win_rate": float(np.mean(method_mse[idx] < base_mse[idx])),
            }
        )
    return rows


def summary_row(
    variant: str,
    pred: np.ndarray,
    batch: ForecastBatch,
    selected: Optional[np.ndarray] = None,
    oracle_mse: Optional[np.ndarray] = None,
    ci_bootstrap: int = 2000,
    seed: int = 2026,
) -> Dict[str, Any]:
    row = metric_summary(variant, pred, batch, selected=selected)
    method_mse = mse_per_sample(pred, batch.y_true)
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    delta = method_mse - base_mse
    if ci_bootstrap > 0:
        ci = bootstrap_ci(delta, n_boot=ci_bootstrap, seed=seed)
    else:
        ci = {
            "mean": float(np.mean(delta)),
            "ci95_low": float(np.mean(delta)),
            "ci95_high": float(np.mean(delta)),
            "p_lt_zero": float(np.mean(delta < 0.0)),
        }
    pcs = pd.DataFrame(per_config_rows(variant, pred, batch))
    row["ci95_low_mse_delta"] = ci["ci95_low"]
    row["ci95_high_mse_delta"] = ci["ci95_high"]
    row["p_bootstrap_delta_lt_zero"] = ci["p_lt_zero"]
    row["max_per_config_harm_rate"] = float(pcs["harm_rate"].max()) if not pcs.empty else float("nan")
    row["worst_config_mse_delta_pct_vs_lrbn"] = float(pcs["mse_delta_pct_vs_lrbn"].max()) if not pcs.empty else float("nan")
    row["improved_configs"] = int((pcs["mse_delta_pct_vs_lrbn"] < 0.0).sum()) if not pcs.empty else 0
    row["total_configs"] = int(len(pcs))
    row["test_threshold_leakage"] = False
    if oracle_mse is not None:
        denom = float(np.mean(base_mse - oracle_mse))
        row["oracle_gain_fraction"] = float(np.mean(base_mse - method_mse) / (denom + EPS))
    else:
        row["oracle_gain_fraction"] = float("nan")
    return row


def oracle_best(candidates: Sequence[ExpertCandidate], batch: ForecastBatch) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = [c.name for c in candidates]
    stack = np.stack([mse_per_sample(c.pred, batch.y_true) for c in candidates], axis=1)
    best = np.argmin(stack, axis=1)
    pred = batch.lrbn_pred.copy()
    for j, c in enumerate(candidates):
        mask = best == j
        if mask.any():
            pred[mask] = c.pred[mask]
    return pred, stack[np.arange(len(batch.meta)), best], np.asarray([names[j] for j in best], dtype=object)


def policy_grid(variant: str) -> Iterable[PolicyParams]:
    tau_leave = [0.3, 0.4, 0.5, 0.6, 0.7]
    tau_gain = [0.5, 0.6, 0.7, 0.8]
    tau_harm = [0.05, 0.10, 0.15, 0.20]
    risk_beta = [1.0, 2.0, 4.0]
    lambdas = [(0.25, 0.25), (0.50, 0.25), (0.50, 0.50), (0.75, 0.50), (1.00, 0.75)]
    if variant == "SafeTAE-pairwise-hard":
        for tl, tg, th, rb in itertools.product(tau_leave, tau_gain, tau_harm, risk_beta):
            yield PolicyParams(variant, tl, tg, th, rb, 1.0, 1.0, 0.0, None, hard_replacement=True)
        return
    if variant == "SafeTAE-pairwise-blend":
        for tl, tg, th, rb in itertools.product(tau_leave, tau_gain, tau_harm, risk_beta):
            yield PolicyParams(variant, tl, tg, th, rb, 0.50, 0.50, 0.0, None)
        return
    cos_values: List[Optional[float]] = [None] if variant == "SafeTAE-tiered-blend" else [None, -0.1, 0.0, 0.1, 0.2]
    for tl, tg, th, rb, (ls, lb), cm in itertools.product(tau_leave, tau_gain, tau_harm, risk_beta, lambdas, cos_values):
        yield PolicyParams(variant, tl, tg, th, rb, ls, lb, 0.05, cm)


def calibrate_variant(
    variant: str,
    calib_batch: ForecastBatch,
    calib_candidates: Sequence[ExpertCandidate],
    calib_probs: Mapping[str, Any],
    reference_rows: Mapping[str, Dict[str, Any]],
    objective: str,
) -> Tuple[PolicyParams, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    oracle_pred, oracle_mse, _ = oracle_best(calib_candidates, calib_batch)
    _ = oracle_pred
    for policy in policy_grid(variant):
        pred, decisions = apply_policy(calib_batch, calib_candidates, calib_probs, policy)
        row = summary_row(
            variant,
            pred,
            calib_batch,
            selected=(decisions["selected_expert"].to_numpy(str) != "keep_lrbn"),
            oracle_mse=oracle_mse,
            ci_bootstrap=0,
        )
        row.update(asdict(policy))
        row["cos_min"] = np.nan if policy.cos_min is None else policy.cos_min
        sra_safe = reference_rows.get("sra_safe", {})
        sra_bal = reference_rows.get("sra_balanced", {})
        if objective == "safe":
            feasible = (
                row["harm_rate"] <= 0.05
                and row["max_per_config_harm_rate"] <= 0.10
                and row.get("selected_rate", row.get("coverage", 0.0)) >= float(sra_safe.get("selected_rate", sra_safe.get("coverage", 0.0))) - 1e-9
            )
            score = float(row["mse_delta_pct_vs_lrbn"]) + 100.0 * max(0.0, float(row["harm_rate"]) - 0.05)
        elif objective == "balanced":
            max_harm = max(0.12, float(sra_bal.get("harm_rate", 0.0)) + 0.03)
            feasible = row["harm_rate"] <= max_harm and row["max_per_config_harm_rate"] <= 0.20
            score = float(row["mse_delta_pct_vs_lrbn"]) + 20.0 * max(0.0, float(row["harm_rate"]) - max_harm) - 0.1 * float(row["oracle_gain_fraction"])
        else:
            feasible = True
            score = float(row["mse_delta_pct_vs_lrbn"]) + 10.0 * max(0.0, float(row["harm_rate"]) - 0.15)
        row["calibration_feasible"] = bool(feasible)
        row["calibration_score"] = float(score)
        rows.append(row)
        ranked_score = score if feasible else score + 1000.0
        if ranked_score < best_score:
            best_score = ranked_score
            best_row = row
    if best_row is None:
        best_row = rows[0]
    return policy_from_row(best_row), pd.DataFrame(rows)


def slice_rows(variant: str, pred: np.ndarray, batch: ForecastBatch, thresholds: Dict[str, float], schema: Dict[str, List[Any]]) -> pd.DataFrame:
    masks = slice_masks(batch, thresholds, schema)
    base_mse = mse_per_sample(batch.lrbn_pred, batch.y_true)
    method_mse = mse_per_sample(pred, batch.y_true)
    rows: List[Dict[str, Any]] = []
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "variant": variant,
                "slice": name,
                "n": int(mask.sum()),
                "mse": float(method_mse[mask].mean()),
                "lrbn_mse": float(base_mse[mask].mean()),
                "mse_delta_pct_vs_lrbn": safe_pct(float(method_mse[mask].mean()), float(base_mse[mask].mean())),
                "harm_rate": float(np.mean(method_mse[mask] > base_mse[mask] + 1e-12)),
                "win_rate": float(np.mean(method_mse[mask] < base_mse[mask])),
            }
        )
    return pd.DataFrame(rows)


def expert_distribution(variant: str, decisions: pd.DataFrame) -> pd.DataFrame:
    counts = decisions["selected_expert"].value_counts().rename_axis("selected_expert").reset_index(name="count")
    counts["variant"] = variant
    counts["rate"] = counts["count"] / max(1, len(decisions))
    tier = decisions.groupby("selected_expert", observed=True)["selected_tier"].first().to_dict()
    counts["selected_tier"] = counts["selected_expert"].map(tier)
    return counts


def pairwise_acceptance_diagnostics(
    variant: str,
    decisions: pd.DataFrame,
    candidates: Sequence[ExpertCandidate],
    batch: ForecastBatch,
) -> pd.DataFrame:
    base = mse_per_sample(batch.lrbn_pred, batch.y_true)
    by_name = candidate_dict(candidates)
    rows = []
    for name, group in decisions.groupby("selected_expert", observed=True):
        mask = group.index.to_numpy(int)
        if name == "keep_lrbn" or name not in by_name:
            delta = np.zeros(len(mask), dtype=float)
        else:
            delta = mse_per_sample(by_name[str(name)].pred, batch.y_true)[mask] - base[mask]
        rows.append(
            {
                "variant": variant,
                "selected_expert": name,
                "n": int(len(mask)),
                "mean_selected_delta_vs_lrbn": float(delta.mean()) if len(delta) else 0.0,
                "selected_harm_rate": float(np.mean(delta > 1e-12)) if len(delta) else 0.0,
                "selected_win_rate": float(np.mean(delta < 0.0)) if len(delta) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def write_parquet_safe(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_parquet(path, index=False)
    except Exception:
        df.to_csv(path.with_suffix(".csv"), index=False)
        path.write_text("Parquet engine unavailable; see adjacent CSV fallback.\n", encoding="utf-8")


def df_to_md(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_empty_"
    try:
        return df.head(max_rows).to_markdown(index=False)
    except ImportError:
        return "```\n" + df.head(max_rows).to_string(index=False) + "\n```"


def run_stage7_safe_tae(
    metrics_csv: Path,
    stage5_dir: Path,
    stage6_dir: Path,
    output_dir: Path,
    stage3_dir: Optional[Path] = None,
    n_bootstrap: int = 2000,
    seed: int = 2026,
) -> Dict[str, Any]:
    from halluguard_lrbn_bp import load_forecast_batch_from_metrics

    output_dir.mkdir(parents=True, exist_ok=True)
    batch = load_forecast_batch_from_metrics(metrics_csv)
    val, test = split_batch(batch)
    schema = feature_schema(val)
    thresholds = slice_thresholds(val)
    train_mask, calib_mask = stratified_inner_split(val.meta, seed=seed)
    val_train = val.subset(train_mask)
    val_calib = val.subset(calib_mask)

    safe_params = json.loads((stage5_dir / "stage5_selected_safe_params.json").read_text(encoding="utf-8"))
    balanced_params = json.loads((stage5_dir / "stage5_selected_balanced_params.json").read_text(encoding="utf-8"))
    basic_params = select_basic_sra_params(stage5_dir)
    stage3_params = load_stage3_params(stage3_dir)
    mrc = build_mrc_artifacts(val, test, schema)

    val_candidates = build_candidate_pool(
        val,
        safe_params,
        balanced_params,
        basic_params,
        stage3_params,
        mrc["val_delta"],
        mrc["val_abstain_pred"],
    )
    test_candidates = build_candidate_pool(
        test,
        safe_params,
        balanced_params,
        basic_params,
        stage3_params,
        mrc["test_delta"],
        mrc["test_abstain_pred"],
    )
    train_candidates = subset_candidates(val_candidates, train_mask)
    calib_candidates = subset_candidates(val_candidates, calib_mask)
    train_mrc = mrc["val_delta"][train_mask]
    calib_mrc = mrc["val_delta"][calib_mask]
    heads = fit_heads(val_train, val_calib, train_candidates, calib_candidates, schema, train_mrc, calib_mrc)
    calib_probs = compute_probabilities(val_calib, calib_candidates, schema, calib_mrc, heads)
    test_probs = compute_probabilities(test, test_candidates, schema, mrc["test_delta"], heads)

    candidate_metrics = pd.DataFrame(
        candidate_metric_rows(val_candidates, val, "val") + candidate_metric_rows(test_candidates, test, "test")
    )
    candidate_table = pd.concat(
        [sample_candidate_table(val_candidates, val, "val"), sample_candidate_table(test_candidates, test, "test")],
        ignore_index=True,
    )
    pairwise_labels = candidate_table[candidate_table["candidate"].ne("keep_lrbn")].copy()
    pairwise_labels["inner_split"] = "test_or_full_val"
    pairwise_labels.loc[pairwise_labels["split"].eq("val") & pairwise_labels["row"].isin(np.where(train_mask)[0]), "inner_split"] = "inner_train"
    pairwise_labels.loc[pairwise_labels["split"].eq("val") & pairwise_labels["row"].isin(np.where(calib_mask)[0]), "inner_split"] = "inner_calib"

    reference_rows = {
        "sra_safe": metric_summary("sra_safe", candidate_dict(calib_candidates)["sra_safe"].pred, val_calib),
        "sra_balanced": metric_summary("sra_balanced", candidate_dict(calib_candidates)["sra_balanced"].pred, val_calib),
    }
    variant_objectives = {
        "SafeTAE-pairwise-hard": "generic",
        "SafeTAE-pairwise-blend": "generic",
        "SafeTAE-tiered-blend": "generic",
        "SafeTAE-mrc-consistency": "generic",
        "SafeTAE-safe": "safe",
        "SafeTAE-balanced": "balanced",
    }
    selected: Dict[str, PolicyParams] = {}
    grid_frames = []
    for variant, objective in variant_objectives.items():
        policy, grid = calibrate_variant(variant, val_calib, calib_candidates, calib_probs, reference_rows, objective)
        selected[variant] = policy
        grid["selected_policy"] = False
        best_mask = (
            (grid["tau_leave"] == policy.tau_leave)
            & (grid["tau_gain"] == policy.tau_gain)
            & (grid["tau_harm"] == policy.tau_harm)
            & (grid["risk_beta"] == policy.risk_beta)
            & (grid["lambda_safe"] == policy.lambda_safe)
            & (grid["lambda_balanced"] == policy.lambda_balanced)
            & (grid["lambda_aggressive"] == policy.lambda_aggressive)
        )
        if policy.cos_min is None:
            best_mask &= grid["cos_min"].isna()
        else:
            best_mask &= grid["cos_min"].eq(policy.cos_min)
        grid.loc[best_mask, "selected_policy"] = True
        grid_frames.append(grid)
    calibration_grid = pd.concat(grid_frames, ignore_index=True)

    oracle_pred, oracle_mse, oracle_names = oracle_best(test_candidates, test)
    overall_rows: List[Dict[str, Any]] = [
        summary_row("LRBN", test.lrbn_pred, test, oracle_mse=oracle_mse, ci_bootstrap=n_bootstrap, seed=seed),
        summary_row("TAE-oracle-best", oracle_pred, test, oracle_mse=oracle_mse, ci_bootstrap=n_bootstrap, seed=seed),
    ]
    decision_frames = []
    per_config_frames = []
    slice_frames = []
    dist_frames = []
    diag_frames = []
    base_variants = {
        "sra_safe": candidate_dict(test_candidates)["sra_safe"].pred,
        "sra_balanced": candidate_dict(test_candidates)["sra_balanced"].pred,
        "sra_basic_ablation": candidate_dict(test_candidates)["sra_basic_ablation"].pred,
        "mrc_ridge_abstain": candidate_dict(test_candidates)["mrc_ridge_abstain"].pred,
        "mrc_ridge_residual_blend": candidate_dict(test_candidates)["mrc_ridge_residual_blend"].pred,
    }
    for name, pred in base_variants.items():
        overall_rows.append(summary_row(name, pred, test, oracle_mse=oracle_mse, ci_bootstrap=n_bootstrap, seed=seed))
        per_config_frames.append(pd.DataFrame(per_config_rows(name, pred, test)))
        slice_frames.append(slice_rows(name, pred, test, thresholds, schema))

    for variant, policy in selected.items():
        pred, decisions = apply_policy(test, test_candidates, test_probs, policy)
        selected_mask = decisions["selected_expert"].to_numpy(str) != "keep_lrbn"
        overall_rows.append(
            summary_row(variant, pred, test, selected=selected_mask, oracle_mse=oracle_mse, ci_bootstrap=n_bootstrap, seed=seed)
        )
        decisions["mse_delta_vs_lrbn"] = mse_per_sample(pred, test.y_true) - mse_per_sample(test.lrbn_pred, test.y_true)
        decision_frames.append(decisions)
        per_config_frames.append(pd.DataFrame(per_config_rows(variant, pred, test)))
        slice_frames.append(slice_rows(variant, pred, test, thresholds, schema))
        dist_frames.append(expert_distribution(variant, decisions))
        diag_frames.append(pairwise_acceptance_diagnostics(variant, decisions, test_candidates, test))

    stage6_decision = stage6_dir / "tae" / "decision_eval.csv"
    if stage6_decision.exists():
        stage6 = pd.read_csv(stage6_decision)
        for source, target in [("TAE-router", "TAE-router-stage6"), ("TAE-ranker", "TAE-ranker-stage6")]:
            rows = stage6[stage6["method"].eq(source)]
            if not rows.empty:
                row = rows.iloc[0].to_dict()
                row["method"] = target
                row["variant"] = target
                row["reference_only"] = True
                row["test_threshold_leakage"] = False
                overall_rows.append(row)

    overall = pd.DataFrame(overall_rows)
    if "method" in overall.columns and "variant" not in overall.columns:
        overall["variant"] = overall["method"]
    elif "variant" not in overall.columns:
        overall["variant"] = overall.get("method", "")
    elif "method" in overall.columns:
        overall["variant"] = overall["variant"].fillna(overall["method"])
    per_config = pd.concat(per_config_frames, ignore_index=True) if per_config_frames else pd.DataFrame()
    slices = pd.concat(slice_frames, ignore_index=True) if slice_frames else pd.DataFrame()
    distribution = pd.concat(dist_frames, ignore_index=True) if dist_frames else pd.DataFrame()
    pairwise_diag = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    decisions_all = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()

    selected_params = {name: asdict(policy) for name, policy in selected.items()}
    selected_params["selection_source"] = "validation_inner_calib_only"
    selected_params["test_threshold_leakage"] = False
    selected_params["mrc_validation_params"] = {
        "risk_threshold": mrc["risk_threshold"],
        "shrink_cap_params": mrc["shrink_cap_params"],
    }

    bootstrap_map = {}
    for _, row in overall.iterrows():
        name = str(row.get("variant", row.get("method", "")))
        bootstrap_map[name] = {
            "ci95_low_mse_delta": row.get("ci95_low_mse_delta"),
            "ci95_high_mse_delta": row.get("ci95_high_mse_delta"),
            "p_bootstrap_delta_lt_zero": row.get("p_bootstrap_delta_lt_zero"),
        }

    def row_for(name: str) -> pd.Series:
        rows = overall[overall["variant"].eq(name)]
        if rows.empty and "method" in overall.columns:
            rows = overall[overall["method"].eq(name)]
        return rows.iloc[0]

    safe = row_for("SafeTAE-safe")
    balanced = row_for("SafeTAE-balanced")
    sra_safe = row_for("sra_safe")
    sra_bal = row_for("sra_balanced")
    hard = row_for("SafeTAE-pairwise-hard")
    blend = row_for("SafeTAE-pairwise-blend")
    mrc_cons = row_for("SafeTAE-mrc-consistency")
    tiered = row_for("SafeTAE-tiered-blend")
    oracle = row_for("TAE-oracle-best")

    safe_pass = bool(
        float(safe["mse_delta_pct_vs_lrbn"]) <= -1.5
        and float(safe["harm_rate"]) <= 0.05
        and float(safe["max_per_config_harm_rate"]) <= 0.10
        and float(safe.get("selected_rate", safe.get("coverage", 0.0))) >= float(sra_safe.get("selected_rate", sra_safe.get("coverage", 0.0))) - 1e-9
        and (float(safe["mse"]) <= float(sra_safe["mse"]) or float(safe["harm_rate"]) <= float(sra_safe["harm_rate"]))
        and float(safe["ci95_high_mse_delta"]) < 0.0
    )
    max_bal_harm = max(0.12, float(sra_bal["harm_rate"]) + 0.03)
    balanced_pass = bool(
        float(balanced["mse_delta_pct_vs_lrbn"]) <= -3.0
        and float(balanced["harm_rate"]) <= max_bal_harm
        and float(balanced["max_per_config_harm_rate"]) <= 0.20
        and float(balanced["oracle_gain_fraction"]) > 0.20
        and (float(balanced["mse"]) <= float(sra_bal["mse"]) or float(balanced["harm_rate"]) < float(sra_bal["harm_rate"]))
        and float(balanced["ci95_high_mse_delta"]) < 0.0
    )
    verdict = {
        "stage": "stage7_safe_tae",
        "test_threshold_leakage": False,
        "safe_pass": safe_pass,
        "balanced_pass": balanced_pass,
        "promote_to_tablea": bool(safe_pass or balanced_pass),
        "h1_pairwise_safer_than_stage6_top1": bool(float(safe["harm_rate"]) < 0.15 and float(balanced["harm_rate"]) < 0.20),
        "h2_no_change_gate_protects_lrbn": bool(float(safe["mse_delta_pct_vs_lrbn"]) < 0.0 and float(safe["harm_rate"]) < float(hard["harm_rate"])),
        "h3_blend_safer_than_hard": bool(float(blend["harm_rate"]) <= float(hard["harm_rate"]) and float(blend["worst_config_mse_delta_pct_vs_lrbn"]) <= float(hard["worst_config_mse_delta_pct_vs_lrbn"])),
        "h4_mrc_consistency_helped": bool(float(mrc_cons["harm_rate"]) <= float(tiered["harm_rate"]) and float(mrc_cons["mse_delta_pct_vs_lrbn"]) <= float(tiered["mse_delta_pct_vs_lrbn"]) + 0.10),
        "h5_beats_sra_frontier": bool(
            (float(safe["mse"]) <= float(sra_safe["mse"]) and float(safe["harm_rate"]) <= float(sra_safe["harm_rate"]) + 1e-9)
            or (float(balanced["mse"]) <= float(sra_bal["mse"]) and float(balanced["harm_rate"]) <= float(sra_bal["harm_rate"]) + 0.03)
        ),
        "oracle_delta_pct_vs_lrbn": float(oracle["mse_delta_pct_vs_lrbn"]),
        "safe_delta_pct_vs_lrbn": float(safe["mse_delta_pct_vs_lrbn"]),
        "balanced_delta_pct_vs_lrbn": float(balanced["mse_delta_pct_vs_lrbn"]),
        "sra_safe_delta_pct_vs_lrbn": float(sra_safe["mse_delta_pct_vs_lrbn"]),
        "sra_balanced_delta_pct_vs_lrbn": float(sra_bal["mse_delta_pct_vs_lrbn"]),
    }

    config = {
        "metrics_csv": str(metrics_csv),
        "stage5_dir": str(stage5_dir),
        "stage6_dir": str(stage6_dir),
        "stage3_dir": str(stage3_dir) if stage3_dir else None,
        "output_dir": str(output_dir),
        "scope": "stage7_safe_tae_compact_validation",
        "datasets": sorted(test.meta["dataset"].astype(str).unique().tolist()),
        "backbones": sorted(test.meta["backbone"].astype(str).unique().tolist()),
        "horizons": sorted([int(x) for x in test.meta["horizon"].unique().tolist()]),
        "seeds": sorted([int(x) for x in test.meta["seed"].unique().tolist()]),
        "n_val_samples": int(len(val.meta)),
        "n_inner_train_samples": int(len(val_train.meta)),
        "n_inner_calib_samples": int(len(val_calib.meta)),
        "n_test_samples": int(len(test.meta)),
        "n_test_configs": int(test.meta.groupby(["dataset", "backbone", "horizon", "seed"], observed=True).ngroups),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
        "safe_params": safe_params,
        "balanced_params": balanced_params,
        "basic_params": basic_params,
        "stage3_params": stage3_params,
        "feature_schema": schema,
        "slice_thresholds_validation_only": thresholds,
        "test_threshold_leakage": False,
    }

    write_json(output_dir / "safe_tae_config.json", config)
    candidate_metrics.to_csv(output_dir / "candidate_expert_metrics.csv", index=False)
    write_parquet_safe(candidate_table, output_dir / "candidate_expert_table.parquet")
    write_parquet_safe(pairwise_labels, output_dir / "pairwise_labels.parquet")
    heads.metrics.to_csv(output_dir / "pairwise_head_metrics.csv", index=False)
    calibration_grid.to_csv(output_dir / "threshold_calibration_grid.csv", index=False)
    write_json(output_dir / "selected_safe_tae_params.json", selected_params)
    overall.to_csv(output_dir / "safe_tae_overall.csv", index=False)
    per_config.to_csv(output_dir / "safe_tae_per_config.csv", index=False)
    slices.to_csv(output_dir / "safe_tae_slices.csv", index=False)
    distribution.to_csv(output_dir / "safe_tae_expert_distribution.csv", index=False)
    pairwise_diag.to_csv(output_dir / "safe_tae_pairwise_diagnostics.csv", index=False)
    write_json(output_dir / "safe_tae_bootstrap_ci.json", bootstrap_map)
    failure = decisions_all.sort_values("mse_delta_vs_lrbn", ascending=False).head(80) if not decisions_all.empty else pd.DataFrame()
    failure.to_csv(output_dir / "safe_tae_failure_cases.csv", index=False)
    write_json(output_dir / "stage7_verdict.json", verdict)

    summary = [
        "# Stage 7 Safe-TAE Summary",
        "",
        "## Setup",
        "",
        f"- Validation samples: `{len(val.meta)}`",
        f"- Inner train/calib: `{len(val_train.meta)}` / `{len(val_calib.meta)}`",
        f"- Test samples: `{len(test.meta)}`",
        f"- Test configs: `{config['n_test_configs']}`",
        "- Test threshold leakage: `False`",
        "",
        "## Verdict",
        "",
        f"- Safe pass: `{verdict['safe_pass']}`",
        f"- Balanced pass: `{verdict['balanced_pass']}`",
        f"- Promote to TableA: `{verdict['promote_to_tablea']}`",
        f"- H1 pairwise safer than top-1: `{verdict['h1_pairwise_safer_than_stage6_top1']}`",
        f"- H2 no-change gate protects LRBN: `{verdict['h2_no_change_gate_protects_lrbn']}`",
        f"- H3 residual blend safer than hard: `{verdict['h3_blend_safer_than_hard']}`",
        f"- H4 MRC consistency helped: `{verdict['h4_mrc_consistency_helped']}`",
        f"- H5 beats SRA frontier: `{verdict['h5_beats_sra_frontier']}`",
        "",
        "## Overall",
        "",
        df_to_md(overall[["variant", "mse", "mae", "mse_delta_pct_vs_lrbn", "harm_rate", "max_per_config_harm_rate", "oracle_gain_fraction", "ci95_high_mse_delta"]].sort_values("mse")),
        "",
        "## Selected Params",
        "",
        "```json",
        json.dumps(selected_params, indent=2, ensure_ascii=False, default=json_default),
        "```",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    return {
        "config": config,
        "overall": overall,
        "per_config": per_config,
        "slices": slices,
        "verdict": verdict,
    }
