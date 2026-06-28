#!/usr/bin/env python
"""Stage 12 compact validation for low-harm CGA arbitration.

The goal is not to add new candidate families. It tests whether Stage 10 CGA
can become deployable through sparse family admission, residual-family simplex
mixing, no-harm gating, and boundary-aware lambda shrink.
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

from halluguard_lrbn_bp import ForecastBatch, mse_per_sample
from halluguard_stage6_mechanism import feature_frame, safe_pct
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
    CGAModels,
    ScoreBundle,
    build_cga_pools,
    candidate_metadata,
    df_to_md,
    fit_cga_models,
    json_default,
    prepare_score_bundle,
    topk_metrics,
)


ACTIVE_FAMILIES = ["smoothing_teacher", "residual_distribution", "retrieval_memory"]
RESIDUAL_SIMPLEX_CANDIDATES = {
    "residual_q25",
    "residual_config_median",
    "residual_q75",
    "residual_slice_quantile_median",
}


@dataclass(frozen=True)
class LowHarmPolicy:
    variant: str
    family_rep: str
    k_max: int
    tau_leave: float
    tau_family_gain: float
    tau_family_harm: float
    beta_harm: float
    safe_floor: float
    temperature: float
    risk_gate: bool
    tau_expected_harm: float
    dynamic_lambda: bool
    boundary_veto: bool
    boundary_veto_mult: float
    boundary_gap_quantile: float
    boundary_repair_quantile: float


@dataclass
class LowHarmPrepared:
    scores: ScoreBundle
    family_pred: Dict[str, np.ndarray]
    family_candidate_count: Dict[str, int]
    boundary_mask: np.ndarray


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    z = (x - np.max(x)) / max(float(temperature), 1e-6)
    w = np.exp(z)
    return w / (np.sum(w) + 1e-12)


def policy_grid(variant: str) -> Iterable[LowHarmPolicy]:
    if variant == "Sparse-Family-CGA":
        reps = ["median"]
        risk_options = [(False, 1.0, False)]
        veto_options = [(False, 1.0)]
        safe_floors = [0.65, 0.80]
        tau_harms = [0.04, 0.06, 0.10]
        betas = [2.0, 4.0]
    elif variant == "Sparse-Residual-Simplex-CGA":
        reps = ["residual_simplex"]
        risk_options = [(False, 1.0, False)]
        veto_options = [(False, 1.0)]
        safe_floors = [0.65, 0.80]
        tau_harms = [0.04, 0.06, 0.10]
        betas = [2.0, 4.0]
    elif variant == "NoHarm-Selective-CGA":
        reps = ["residual_simplex"]
        risk_options = [(True, tau, False) for tau in [0.04, 0.06, 0.08]]
        veto_options = [(False, 1.0)]
        safe_floors = [0.80, 0.90]
        tau_harms = [0.03, 0.05, 0.08]
        betas = [3.0, 5.0]
    elif variant == "LambdaVeto-CGA":
        reps = ["residual_simplex"]
        risk_options = [(True, tau, True) for tau in [0.06, 0.08, 0.10]]
        veto_options = [(True, 0.0), (True, 0.25)]
        safe_floors = [0.70, 0.80, 0.90]
        tau_harms = [0.05, 0.08, 0.10]
        betas = [2.0, 4.0]
    else:
        raise ValueError(f"unknown variant: {variant}")

    for rep in reps:
        for k_max in [1, 2]:
            for tau_leave in [0.55, 0.65]:
                for tau_gain in [0.50, 0.60]:
                    for tau_harm in tau_harms:
                        for beta in betas:
                            for safe_floor in safe_floors:
                                for risk_gate, tau_expected, dynamic_lambda in risk_options:
                                    for veto, veto_mult in veto_options:
                                        yield LowHarmPolicy(
                                            variant=variant,
                                            family_rep=rep,
                                            k_max=int(k_max),
                                            tau_leave=float(tau_leave),
                                            tau_family_gain=float(tau_gain),
                                            tau_family_harm=float(tau_harm),
                                            beta_harm=float(beta),
                                            safe_floor=float(safe_floor),
                                            temperature=1.0,
                                            risk_gate=bool(risk_gate),
                                            tau_expected_harm=float(tau_expected),
                                            dynamic_lambda=bool(dynamic_lambda),
                                            boundary_veto=bool(veto),
                                            boundary_veto_mult=float(veto_mult),
                                            boundary_gap_quantile=0.75,
                                            boundary_repair_quantile=0.50,
                                        )


def boundary_veto_mask(batch: ForecastBatch, calib_batch: ForecastBatch, schema: Dict[str, List[Any]], policy: LowHarmPolicy) -> np.ndarray:
    calib_feat = feature_frame(calib_batch, schema)
    feat = feature_frame(batch, schema)
    gap_tau = float(calib_feat["boundary_gap_lrbn"].quantile(policy.boundary_gap_quantile))
    repair_tau = float(calib_feat["repair_ratio"].quantile(policy.boundary_repair_quantile))
    return (feat["boundary_gap_lrbn"].to_numpy(float) >= gap_tau) & (feat["repair_ratio"].to_numpy(float) <= repair_tau)


def build_family_predictions(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    scores: ScoreBundle,
    policy: LowHarmPolicy,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    family_pred: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    deployable = [c for c in candidates if c.deployable and c.name != "keep_lrbn"]
    for fam in ACTIVE_FAMILIES:
        fam_cands = [c for c in deployable if c.family == fam]
        counts[fam] = len(fam_cands)
        if not fam_cands:
            continue
        if fam == "residual_distribution" and policy.family_rep == "residual_simplex":
            simplex_cands = [c for c in fam_cands if c.name in RESIDUAL_SIMPLEX_CANDIDATES]
            if not simplex_cands:
                simplex_cands = fam_cands
            by_name = {c.name: c for c in simplex_cands}
            pred = np.zeros_like(batch.lrbn_pred)
            for i in range(len(batch.meta)):
                rows = scores.cand_scores[
                    (scores.cand_scores["row_index"].eq(i))
                    & (scores.cand_scores["candidate"].isin(list(by_name.keys())))
                ].copy()
                if rows.empty:
                    pred[i] = np.nanmedian(np.stack([c.pred[i] for c in simplex_cands], axis=0), axis=0)
                    continue
                rows["utility"] = rows["p_candidate_gain"] - policy.beta_harm * rows["p_candidate_harm"]
                util = rows["utility"].to_numpy(float)
                weights = _softmax(util, policy.temperature)
                acc = np.zeros_like(batch.lrbn_pred[i])
                for weight, name in zip(weights, rows["candidate"].astype(str).tolist()):
                    acc += float(weight) * by_name[name].pred[i]
                pred[i] = acc
            family_pred[fam] = pred
        else:
            family_pred[fam] = np.nanmedian(np.stack([c.pred for c in fam_cands], axis=0), axis=0)
    return family_pred, counts


def prepare_low_harm(
    batch: ForecastBatch,
    calib_batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    models: CGAModels,
    policy: LowHarmPolicy,
    score_cache: Optional[ScoreBundle] = None,
) -> LowHarmPrepared:
    scores = score_cache or prepare_score_bundle(batch, candidates, schema, models)
    family_pred, counts = build_family_predictions(batch, candidates, scores, policy)
    return LowHarmPrepared(
        scores=scores,
        family_pred=family_pred,
        family_candidate_count=counts,
        boundary_mask=boundary_veto_mask(batch, calib_batch, schema, policy),
    )


def apply_low_harm_policy(batch: ForecastBatch, prepared: LowHarmPrepared, policy: LowHarmPolicy) -> Tuple[np.ndarray, pd.DataFrame]:
    pred = batch.lrbn_pred.copy()
    fam_scores = prepared.scores.fam_scores
    rows_out: List[Dict[str, Any]] = []
    for i in range(len(batch.meta)):
        selected_families: List[str] = []
        expected_harm = 0.0
        lambda_scale = 1.0
        reason = "keep_lrbn"
        family_weights: Dict[str, float] = {}
        if prepared.scores.leave_score[i] >= policy.tau_leave:
            fam_i = fam_scores[fam_scores["row_index"].eq(i)].copy()
            raw_rows: List[Dict[str, Any]] = []
            for fam in ACTIVE_FAMILIES:
                row = fam_i[fam_i["family"].eq(fam)]
                if row.empty or fam not in prepared.family_pred:
                    continue
                gain = float(row["p_family_gain"].iloc[0])
                harm = float(row["p_family_harm"].iloc[0])
                if gain < policy.tau_family_gain or harm > policy.tau_family_harm:
                    continue
                utility = max(0.0, gain - policy.beta_harm * harm)
                if policy.boundary_veto and prepared.boundary_mask[i] and fam == "smoothing_teacher":
                    utility *= policy.boundary_veto_mult
                if utility > 0.0:
                    raw_rows.append({"family": fam, "utility": utility, "harm": harm, "gain": gain})
            if raw_rows:
                raw_rows = sorted(raw_rows, key=lambda r: r["utility"], reverse=True)[: policy.k_max]
                total_utility = sum(float(r["utility"]) for r in raw_rows)
                if total_utility > 0.0:
                    unit_weights = {str(r["family"]): float(r["utility"]) / (total_utility + 1e-12) for r in raw_rows}
                    expected_harm = float(sum(unit_weights[str(r["family"])] * float(r["harm"]) for r in raw_rows))
                    if policy.risk_gate and expected_harm > policy.tau_expected_harm:
                        if policy.dynamic_lambda:
                            lambda_scale = max(0.0, min(1.0, policy.tau_expected_harm / (expected_harm + 1e-12)))
                            reason = "risk_shrink"
                        else:
                            lambda_scale = 0.0
                            reason = "risk_abstain"
                    else:
                        reason = "selected"
                    if lambda_scale > 0.0:
                        family_mass = (1.0 - policy.safe_floor) * lambda_scale
                        acc = batch.lrbn_pred[i].copy()
                        for fam, unit_w in unit_weights.items():
                            w = family_mass * float(unit_w)
                            family_weights[fam] = w
                            acc += w * (prepared.family_pred[fam][i] - batch.lrbn_pred[i])
                            selected_families.append(fam)
                        pred[i] = acc
        row_out = {
            "row_index": i,
            "selected": bool(selected_families),
            "selected_families": ",".join(selected_families) if selected_families else "none",
            "reason": reason,
            "leave_score": float(prepared.scores.leave_score[i]),
            "expected_harm": float(expected_harm),
            "lambda_scale": float(lambda_scale),
            "boundary_veto_active": bool(policy.boundary_veto and prepared.boundary_mask[i]),
        }
        for fam in ACTIVE_FAMILIES:
            row_out[f"weight_{fam}"] = float(family_weights.get(fam, 0.0))
        rows_out.append(row_out)
    return pred, pd.DataFrame(rows_out)


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
        row["mean_expected_harm"] = float(decisions["expected_harm"].mean())
        row["mean_lambda_scale"] = float(decisions["lambda_scale"].mean())
        row["boundary_veto_rate"] = float(decisions["boundary_veto_active"].mean())
        for fam in ACTIVE_FAMILIES:
            row[f"mean_weight_{fam}"] = float(decisions[f"weight_{fam}"].mean())
    return row


def calibration_caps(variant: str) -> Tuple[float, float, float, float]:
    if variant == "NoHarm-Selective-CGA":
        return 0.05, 0.12, 0.08, 0.85
    if variant == "LambdaVeto-CGA":
        return 0.05, 0.12, 0.08, 0.85
    return 0.10, 0.18, 0.08, 0.90


def calibrate_policy(
    variant: str,
    calib: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    models: CGAModels,
    oracle_mse: np.ndarray,
    seed: int,
) -> Tuple[LowHarmPolicy, pd.DataFrame]:
    harm_cap, max_config_cap, oracle_min, coverage_cap = calibration_caps(variant)
    rows: List[Dict[str, Any]] = []
    best_policy: Optional[LowHarmPolicy] = None
    best_ranked = float("inf")
    policies = list(policy_grid(variant))
    scores = prepare_score_bundle(calib, candidates, schema, models)
    prepared_cache: Dict[Tuple[Any, ...], LowHarmPrepared] = {}
    for policy in policies:
        cache_key = (
            policy.family_rep,
            policy.beta_harm,
            policy.temperature,
            policy.boundary_veto,
            policy.boundary_veto_mult,
            policy.boundary_gap_quantile,
            policy.boundary_repair_quantile,
        )
        if cache_key not in prepared_cache:
            prepared_cache[cache_key] = prepare_low_harm(calib, calib, candidates, schema, models, policy, score_cache=scores)
        pred, decisions = apply_low_harm_policy(calib, prepared_cache[cache_key], policy)
        row = evaluate_variant(variant, pred, calib, decisions, oracle_mse, n_bootstrap=0, seed=seed)
        row.update(asdict(policy))
        feasible = (
            row["mse_delta_pct_vs_lrbn"] < 0.0
            and row["harm_rate"] <= harm_cap
            and row["max_config_harm"] <= max_config_cap
            and row["coverage"] <= coverage_cap
            and row["oracle_gain_fraction"] >= oracle_min
        )
        row["calibration_feasible"] = bool(feasible)
        score = float(row["mse_delta_pct_vs_lrbn"])
        score += 220.0 * max(0.0, float(row["harm_rate"]) - harm_cap)
        score += 160.0 * max(0.0, float(row["max_config_harm"]) - max_config_cap)
        score += 35.0 * max(0.0, float(row["coverage"]) - coverage_cap)
        score += 25.0 * max(0.0, oracle_min - float(row["oracle_gain_fraction"]))
        row["calibration_score"] = float(score)
        rows.append(row)
        ranked = score if feasible else score + 1000.0
        if ranked < best_ranked:
            best_ranked = ranked
            best_policy = policy
    assert best_policy is not None
    return best_policy, pd.DataFrame(rows)


def selection_distribution(decisions_by_variant: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for variant, df in decisions_by_variant.items():
        vc = df["selected_families"].value_counts(normalize=True).rename_axis("selected_families").reset_index(name="share")
        vc.insert(0, "variant", variant)
        frames.append(vc)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def known_config_delta(per_config: pd.DataFrame, variant: str) -> float:
    row = per_config[
        per_config["variant"].eq(variant)
        & per_config["dataset"].eq("ETTm1")
        & per_config["backbone"].eq("DLinear")
        & per_config["horizon"].eq(192)
    ]
    if row.empty:
        return float("nan")
    return float(row["mse_delta_pct_vs_lrbn"].iloc[0])


def boundary_slice_delta(slice_df: pd.DataFrame, variant: str) -> float:
    sub = slice_df[slice_df["variant"].eq(variant)].copy()
    if sub.empty:
        return float("nan")
    boundary = sub[sub["slice"].astype(str).str.contains("boundary|gap", case=False, regex=True)]
    if boundary.empty:
        boundary = sub
    # Worst boundary-like mean delta is the relevant safety signal.
    return float(boundary["mse_delta_pct_vs_lrbn"].max())


def verify_outputs(output_dir: Path) -> pd.DataFrame:
    required = [
        "stage12_config.json",
        "stage12_candidate_metadata.csv",
        "stage12_topk_metrics.csv",
        "stage12_calibration_grid.csv",
        "stage12_overall.csv",
        "stage12_per_config.csv",
        "stage12_slice_metrics.csv",
        "stage12_selection_distribution.csv",
        "stage12_policies.json",
        "stage12_verdict.json",
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


def build_summary(output_dir: Path, verdict: Mapping[str, Any], overall: pd.DataFrame) -> str:
    cols = [
        "variant",
        "mse",
        "mae",
        "mse_delta_pct_vs_lrbn",
        "harm_rate",
        "max_config_harm",
        "coverage",
        "oracle_gain_fraction",
        "ci95_high_delta_raw",
        "test_threshold_leakage",
    ]
    show_cols = [c for c in cols if c in overall.columns]
    return "\n".join(
        [
            "# Stage 12 CGA Low-Harm Priority Validation",
            "",
            f"Status: `{verdict['status']}`.",
            "",
            "## Overall",
            "",
            df_to_md(overall[show_cols], max_rows=32),
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
    print("[stage12-cga-low-harm] preparing assets", flush=True)
    assets = prepare_assets(metrics_csv, stage5_dir, stage3_dir, seed)
    print(f"[stage12-cga-low-harm] assets ready in {time.time() - start:.1f}s", flush=True)
    pools = build_cga_pools(assets)
    write_json(
        output_dir / "stage12_config.json",
        {
            "stage": "stage12_cga_low_harm",
            "source_plan": "deep-research-report (1).md",
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
    candidate_metadata(pools.test_candidates, "test").to_csv(output_dir / "stage12_candidate_metadata.csv", index=False)
    print("[stage12-cga-low-harm] fitting CGA scoring models", flush=True)
    models, selector_metrics, _, _ = fit_cga_models(
        assets.val_train,
        assets.val_calib,
        pools.train_candidates,
        pools.calib_candidates,
        assets.schema,
        seed=seed,
    )
    topk = topk_metrics(assets.test, deployable_candidates(pools.test_candidates), assets.schema, models, k=2)
    pd.DataFrame([topk]).to_csv(output_dir / "stage12_topk_metrics.csv", index=False)
    oracle_pred, oracle_mse, _ = oracle_best(deployable_candidates(pools.test_candidates), assets.test)
    calib_oracle_mse = oracle_best(deployable_candidates(pools.calib_candidates), assets.val_calib)[1]

    variants = [
        "Sparse-Family-CGA",
        "Sparse-Residual-Simplex-CGA",
        "NoHarm-Selective-CGA",
        "LambdaVeto-CGA",
    ]
    policies: Dict[str, LowHarmPolicy] = {}
    grid_frames: List[pd.DataFrame] = []
    preds: Dict[str, np.ndarray] = {
        "LRBN": assets.test.lrbn_pred,
        "SRA-BP-balanced": candidate_dict(assets.old_test_candidates).get("sra_balanced", assets.old_test_candidates[0]).pred,
        "oracle_stage12_cga_full": oracle_pred,
    }
    decisions_by: Dict[str, pd.DataFrame] = {}
    test_scores = prepare_score_bundle(assets.test, pools.test_candidates, assets.schema, models)
    test_prepared_cache: Dict[Tuple[Any, ...], LowHarmPrepared] = {}
    print("[stage12-cga-low-harm] calibrating priority variants", flush=True)
    for variant in variants:
        print(f"[stage12-cga-low-harm] calibrating {variant}", flush=True)
        policy, grid = calibrate_policy(
            variant,
            assets.val_calib,
            pools.calib_candidates,
            assets.schema,
            models,
            calib_oracle_mse,
            seed=seed,
        )
        grid["target_variant"] = variant
        grid_frames.append(grid)
        policies[variant] = policy
        key = (
            policy.family_rep,
            policy.beta_harm,
            policy.temperature,
            policy.boundary_veto,
            policy.boundary_veto_mult,
            policy.boundary_gap_quantile,
            policy.boundary_repair_quantile,
        )
        if key not in test_prepared_cache:
            test_prepared_cache[key] = prepare_low_harm(
                assets.test,
                assets.val_calib,
                pools.test_candidates,
                assets.schema,
                models,
                policy,
                score_cache=test_scores,
            )
        pred, decisions = apply_low_harm_policy(assets.test, test_prepared_cache[key], policy)
        preds[variant] = pred
        decisions_by[variant] = decisions

    print("[stage12-cga-low-harm] evaluating", flush=True)
    overall_rows: List[Dict[str, Any]] = []
    for variant, pred in preds.items():
        decisions = decisions_by.get(variant)
        overall_rows.append(evaluate_variant(variant, pred, assets.test, decisions, oracle_mse if variant != "LRBN" else None, n_bootstrap, seed))
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(output_dir / "stage12_overall.csv", index=False)
    pd.concat(grid_frames, ignore_index=True).to_csv(output_dir / "stage12_calibration_grid.csv", index=False)
    write_json(output_dir / "stage12_policies.json", {variant: asdict(policy) for variant, policy in policies.items()})

    per_config = pd.concat(
        [per_config_rows(variant, pred, assets.test, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    per_config.to_csv(output_dir / "stage12_per_config.csv", index=False)
    masks = stage8_slice_masks(assets.test, assets.schema, stage8_slice_thresholds(assets.val, assets.schema))
    slice_df = pd.concat(
        [slice_rows(variant, pred, assets.test, masks, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    slice_df.to_csv(output_dir / "stage12_slice_metrics.csv", index=False)
    selection_distribution(decisions_by).to_csv(output_dir / "stage12_selection_distribution.csv", index=False)

    deployable = overall[overall["variant"].isin(variants)].copy()
    best = deployable.sort_values(["mse_delta_pct_vs_lrbn", "max_config_harm", "harm_rate"], ascending=[True, True, True]).iloc[0]
    final = overall[overall["variant"].eq("LambdaVeto-CGA")].iloc[0]
    known_delta = known_config_delta(per_config, "LambdaVeto-CGA")
    boundary_delta = boundary_slice_delta(slice_df, "LambdaVeto-CGA")
    residual = overall[overall["variant"].eq("Sparse-Residual-Simplex-CGA")].iloc[0]
    sparse = overall[overall["variant"].eq("Sparse-Family-CGA")].iloc[0]
    residual_extra_pp = float(residual["mse_delta_pct_vs_lrbn"] - sparse["mse_delta_pct_vs_lrbn"])
    sparse_pass = bool(
        sparse["mse_delta_pct_vs_lrbn"] < 0.0
        and sparse.get("ci95_high_delta_raw", 1.0) < 0.0
        and sparse["oracle_gain_fraction"] >= 0.08
        and sparse["max_config_harm"] <= 0.18
        and topk.get("family_top2_hit", 0.0) >= 0.65
    )
    residual_pass = bool(residual_extra_pp <= -0.30 and residual["harm_rate"] <= sparse["harm_rate"] + 0.02)
    noharm = overall[overall["variant"].eq("NoHarm-Selective-CGA")].iloc[0]
    noharm_pass = bool(noharm["mse_delta_pct_vs_lrbn"] < 0.0 and noharm["harm_rate"] <= 0.05 and noharm["max_config_harm"] <= 0.12)
    lambda_pass = bool(
        final["mse_delta_pct_vs_lrbn"] < 0.0
        and final.get("ci95_high_delta_raw", 1.0) < 0.0
        and final["max_config_harm"] <= 0.12
        and known_delta <= 0.0
        and boundary_delta <= 0.0
    )
    compact_pass = bool(sparse_pass and lambda_pass)
    verdict = {
        "stage": "stage12_cga_low_harm",
        "status": "compact_pass_ready_for_mini_extension" if compact_pass else "compact_failed_stop_before_mini_extension",
        "compact_pass": compact_pass,
        "best_variant": str(best["variant"]),
        "best_mse": float(best["mse"]),
        "best_mae": float(best["mae"]),
        "best_mse_delta_pct_vs_lrbn": float(best["mse_delta_pct_vs_lrbn"]),
        "best_harm_rate": float(best["harm_rate"]),
        "best_max_config_harm": float(best["max_config_harm"]),
        "best_oracle_gain_fraction": float(best["oracle_gain_fraction"]),
        "lambda_veto_mse_delta_pct_vs_lrbn": float(final["mse_delta_pct_vs_lrbn"]),
        "lambda_veto_harm_rate": float(final["harm_rate"]),
        "lambda_veto_max_config_harm": float(final["max_config_harm"]),
        "lambda_veto_oracle_gain_fraction": float(final["oracle_gain_fraction"]),
        "known_harmed_config_delta_pct": known_delta,
        "boundary_like_worst_slice_delta_pct": boundary_delta,
        "residual_simplex_extra_delta_pp_vs_sparse": residual_extra_pp,
        "family_top2_hit": float(topk.get("family_top2_hit", np.nan)),
        "candidate_top2_hit": float(topk.get("candidate_top2_hit", np.nan)),
        "gates": {
            "sparse_family_pass": sparse_pass,
            "residual_simplex_pass": residual_pass,
            "noharm_selective_pass": noharm_pass,
            "lambda_veto_pass": lambda_pass,
            "oracle_gain_fraction_min": 0.08,
            "max_config_harm_max": 0.18,
            "final_max_config_harm_target": 0.12,
        },
        "stop_reason": None,
        "test_threshold_leakage": False,
    }
    if not compact_pass:
        failed = [k for k, v in verdict["gates"].items() if k.endswith("_pass") and not v]
        verdict["stop_reason"] = "compact gate failed: " + ", ".join(failed)
    write_json(output_dir / "stage12_verdict.json", verdict)
    (output_dir / "summary.md").write_text(build_summary(output_dir, verdict, overall), encoding="utf-8")
    completeness = verify_outputs(output_dir)
    completeness.to_csv(output_dir / "stage12_output_completeness.csv", index=False)
    print(f"[stage12-cga-low-harm] done in {time.time() - start:.1f}s", flush=True)
    return {"verdict": verdict, "overall": overall, "per_config": per_config, "slice": slice_df, "completeness": completeness}

