#!/usr/bin/env python
"""Stage 11 CGA-HM compact validation.

CGA-HM replaces Stage 10 exact candidate hard selection with family-level soft
mixtures and validation-only harm-aware admission.
"""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from halluguard_lrbn_bp import ForecastBatch, mae_per_sample, mse_per_sample
from halluguard_stage6_mechanism import feature_frame, feature_schema, safe_pct
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
from halluguard_stage10_cga import (
    NEW_FAMILIES,
    CGAModels,
    ScoreBundle,
    build_cga_pools,
    candidate_metadata,
    candidate_sample_table,
    df_to_md,
    family_scores,
    fit_cga_models,
    json_default,
    prepare_score_bundle,
    scored_candidates,
    selector_test_metrics,
)
from halluguard_stage7_safe_tae import ExpertCandidate, candidate_dict, write_json


ACTIVE_FAMILIES = ["smoothing_teacher", "residual_distribution", "retrieval_memory"]


@dataclass
class HMPolicy:
    variant: str
    family_rep: str
    tau_leave: float
    tau_family_gain: float
    tau_family_harm: float
    safe_floor: float
    beta_harm: float
    temperature: float
    family_mass_cap: float
    boundary_veto: bool
    boundary_veto_mult: float
    boundary_gap_quantile: float
    boundary_repair_quantile: float


@dataclass
class HMPrepared:
    scores: ScoreBundle
    family_pred: Dict[str, np.ndarray]
    family_candidate_count: Dict[str, int]
    boundary_mask: np.ndarray


def bootstrap_ci(values: np.ndarray, n_boot: int, seed: int) -> Dict[str, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if len(values) == 0 or n_boot <= 0:
        return {"ci95_low": float("nan"), "ci95_high": float("nan"), "p_lt_zero": float("nan")}
    means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, len(values), size=len(values))
        means[b] = float(np.mean(values[idx]))
    return {
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "p_lt_zero": float(np.mean(means < 0.0)),
    }


def policy_grid(variant: str) -> Iterable[HMPolicy]:
    family_rep_options = ["median"]
    if "veto" in variant:
        veto_options = [(True, 0.0), (True, 0.25)]
    else:
        veto_options = [(False, 1.0)]
    for family_rep in family_rep_options:
        for tau_leave in [0.55]:
            for tau_gain in [0.50, 0.60]:
                for tau_harm in [0.06, 0.10]:
                    for safe_floor in [0.80, 0.90]:
                        for beta_harm in [2.0]:
                            for temp in [1.0]:
                                for cap in [0.25, 0.40]:
                                    for veto, mult in veto_options:
                                        yield HMPolicy(
                                            variant=variant,
                                            family_rep=family_rep,
                                            tau_leave=float(tau_leave),
                                            tau_family_gain=float(tau_gain),
                                            tau_family_harm=float(tau_harm),
                                            safe_floor=float(safe_floor),
                                            beta_harm=float(beta_harm),
                                            temperature=float(temp),
                                            family_mass_cap=float(cap),
                                            boundary_veto=bool(veto),
                                            boundary_veto_mult=float(mult),
                                            boundary_gap_quantile=0.75,
                                            boundary_repair_quantile=0.50,
                                        )


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    z = (x - np.max(x)) / max(float(temperature), 1e-6)
    w = np.exp(z)
    return w / (np.sum(w) + 1e-12)


def boundary_veto_mask(batch: ForecastBatch, calib_batch: ForecastBatch, schema: Dict[str, List[Any]], policy: HMPolicy) -> np.ndarray:
    calib_feat = feature_frame(calib_batch, schema)
    feat = feature_frame(batch, schema)
    gap_tau = float(calib_feat["boundary_gap_lrbn"].quantile(policy.boundary_gap_quantile))
    repair_tau = float(calib_feat["repair_ratio"].quantile(policy.boundary_repair_quantile))
    return (feat["boundary_gap_lrbn"].to_numpy(float) >= gap_tau) & (feat["repair_ratio"].to_numpy(float) <= repair_tau)


def build_family_predictions(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    scores: ScoreBundle,
    policy: HMPolicy,
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    family_pred: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    for fam in ACTIVE_FAMILIES:
        fam_cands = [c for c in candidates if c.deployable and c.family == fam]
        counts[fam] = len(fam_cands)
        if not fam_cands:
            continue
        if policy.family_rep == "median":
            family_pred[fam] = np.nanmedian(np.stack([c.pred for c in fam_cands], axis=0), axis=0)
            continue
        pred = np.zeros_like(batch.lrbn_pred)
        for i in range(len(batch.meta)):
            rows = scores.cand_scores[(scores.cand_scores["row_index"].eq(i)) & (scores.cand_scores["family"].eq(fam))]
            rows = rows[rows["candidate"].isin([c.name for c in fam_cands])]
            if rows.empty:
                pred[i] = np.nanmedian(np.stack([c.pred[i] for c in fam_cands], axis=0), axis=0)
                continue
            rows = rows.copy()
            rows["utility"] = rows["p_candidate_gain"] - policy.beta_harm * rows["p_candidate_harm"]
            weights = _softmax(rows["utility"].to_numpy(float), policy.temperature)
            acc = np.zeros_like(batch.lrbn_pred[i])
            by_name = {c.name: c for c in fam_cands}
            for weight, name in zip(weights, rows["candidate"].astype(str).tolist()):
                acc += float(weight) * by_name[name].pred[i]
            pred[i] = acc
        family_pred[fam] = pred
    return family_pred, counts


def prepare_hm(
    batch: ForecastBatch,
    calib_batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    models: CGAModels,
    policy: HMPolicy,
) -> HMPrepared:
    scores = prepare_score_bundle(batch, candidates, schema, models)
    family_pred, counts = build_family_predictions(batch, candidates, scores, policy)
    return HMPrepared(
        scores=scores,
        family_pred=family_pred,
        family_candidate_count=counts,
        boundary_mask=boundary_veto_mask(batch, calib_batch, schema, policy),
    )


def apply_hm_policy(batch: ForecastBatch, prepared: HMPrepared, policy: HMPolicy) -> Tuple[np.ndarray, pd.DataFrame]:
    pred = batch.lrbn_pred.copy()
    fam_scores = prepared.scores.fam_scores
    rows_out: List[Dict[str, Any]] = []
    for i in range(len(batch.meta)):
        selected_families: List[str] = []
        weights: Dict[str, float] = {"safe_base": 1.0}
        reason = "keep_lrbn"
        if prepared.scores.leave_score[i] >= policy.tau_leave:
            fam_i = fam_scores[fam_scores["row_index"].eq(i)].copy()
            raw: Dict[str, float] = {}
            for fam in ACTIVE_FAMILIES:
                row = fam_i[fam_i["family"].eq(fam)]
                if row.empty or fam not in prepared.family_pred:
                    continue
                gain = float(row["p_family_gain"].iloc[0])
                harm = float(row["p_family_harm"].iloc[0])
                if gain < policy.tau_family_gain or harm > policy.tau_family_harm:
                    continue
                score = max(0.0, gain - policy.beta_harm * harm)
                if policy.boundary_veto and prepared.boundary_mask[i] and fam == "smoothing_teacher":
                    score *= policy.boundary_veto_mult
                if score > 0.0:
                    raw[fam] = score
            if raw:
                total = sum(raw.values())
                family_mass = 1.0 - policy.safe_floor
                safe_weight = policy.safe_floor
                acc = safe_weight * batch.lrbn_pred[i]
                weights = {"safe_base": safe_weight}
                for fam, score in raw.items():
                    w = family_mass * score / (total + 1e-12)
                    w = min(w, policy.family_mass_cap)
                    weights[fam] = w
                    acc += w * prepared.family_pred[fam][i]
                    selected_families.append(fam)
                used = sum(weights.values())
                if used < 1.0:
                    acc += (1.0 - used) * batch.lrbn_pred[i]
                    weights["safe_base"] = weights.get("safe_base", 0.0) + (1.0 - used)
                elif used > 1.0:
                    acc = acc / used
                    weights = {k: v / used for k, v in weights.items()}
                pred[i] = acc
                reason = "family_mixture"
        row_out = {
            "row_index": i,
            "selected": bool(selected_families),
            "selected_families": ",".join(selected_families) if selected_families else "none",
            "reason": reason,
            "safe_weight": float(weights.get("safe_base", 1.0)),
            "boundary_veto_active": bool(policy.boundary_veto and prepared.boundary_mask[i]),
        }
        for fam in ACTIVE_FAMILIES:
            row_out[f"weight_{fam}"] = float(weights.get(fam, 0.0))
        rows_out.append(row_out)
    return pred, pd.DataFrame(rows_out)


def evaluate_policy(
    variant: str,
    pred: np.ndarray,
    batch: ForecastBatch,
    decisions: pd.DataFrame,
    oracle_mse: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Dict[str, Any]:
    row = metric_row(
        variant,
        pred,
        batch,
        selected=decisions["selected"].to_numpy(bool),
        oracle_mse=oracle_mse,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    row["variant"] = variant
    row["mean_safe_weight"] = float(decisions["safe_weight"].mean())
    for fam in ACTIVE_FAMILIES:
        row[f"mean_weight_{fam}"] = float(decisions[f"weight_{fam}"].mean())
    row["boundary_veto_rate"] = float(decisions["boundary_veto_active"].mean())
    return row


def calibrate_hm_policy(
    variant: str,
    calib: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    schema: Dict[str, List[Any]],
    models: CGAModels,
    oracle_mse: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> Tuple[HMPolicy, pd.DataFrame]:
    safe = "safe" in variant.lower()
    harm_budget = 0.03 if safe else 0.08
    max_harm_budget = 0.10 if safe else 0.18
    rows: List[Dict[str, Any]] = []
    best_policy: Optional[HMPolicy] = None
    best_score = float("inf")
    policies = list(policy_grid(variant))
    scores = prepare_score_bundle(calib, candidates, schema, models)
    boundary_cache: Dict[Tuple[float, float], np.ndarray] = {}
    family_cache: Dict[Tuple[str, float, float], Tuple[Dict[str, np.ndarray], Dict[str, int]]] = {}
    for policy in policies:
        boundary_key = (policy.boundary_gap_quantile, policy.boundary_repair_quantile)
        if boundary_key not in boundary_cache:
            boundary_cache[boundary_key] = boundary_veto_mask(calib, calib, schema, policy)
        family_key = (policy.family_rep, policy.beta_harm, policy.temperature)
        if family_key not in family_cache:
            family_cache[family_key] = build_family_predictions(calib, candidates, scores, policy)
        family_pred, family_counts = family_cache[family_key]
        prepared = HMPrepared(
            scores=scores,
            family_pred=family_pred,
            family_candidate_count=family_counts,
            boundary_mask=boundary_cache[boundary_key],
        )
        pred, decisions = apply_hm_policy(calib, prepared, policy)
        row = evaluate_policy(variant, pred, calib, decisions, oracle_mse, n_bootstrap=0, seed=seed)
        row.update(asdict(policy))
        feasible = row["harm_rate"] <= harm_budget and row["max_config_harm"] <= max_harm_budget
        row["calibration_feasible"] = bool(feasible)
        score = float(row["mse_delta_pct_vs_lrbn"])
        score += 200.0 * max(0.0, float(row["harm_rate"]) - harm_budget)
        score += 120.0 * max(0.0, float(row["max_config_harm"]) - max_harm_budget)
        score += 20.0 * max(0.0, 0.08 - float(row["oracle_gain_fraction"]))
        row["calibration_score"] = float(score)
        rows.append(row)
        ranked = score if feasible else score + 1000.0
        if ranked < best_score:
            best_score = ranked
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


def verify_outputs(output_dir: Path) -> pd.DataFrame:
    required = [
        "stage11_config.json",
        "stage11_candidate_metadata.csv",
        "stage11_calibration_grid.csv",
        "stage11_overall.csv",
        "stage11_per_config.csv",
        "stage11_slice_metrics.csv",
        "stage11_selection_distribution.csv",
        "stage11_selector_metrics.csv",
        "stage11_verdict.json",
        "stage11_bootstrap_ci.json",
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
        "mean_safe_weight",
        "test_threshold_leakage",
    ]
    show_cols = [c for c in cols if c in overall.columns]
    return "\n".join(
        [
            "# Stage 11 CGA-HM Mechanism Validation",
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
    start_time = time.time()
    print("[stage11-cga-hm] preparing assets", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets = prepare_assets(metrics_csv, stage5_dir, stage3_dir, seed)
    print(f"[stage11-cga-hm] assets ready in {time.time() - start_time:.1f}s", flush=True)
    print("[stage11-cga-hm] building CGA pools", flush=True)
    pools = build_cga_pools(assets)
    write_json(
        output_dir / "stage11_config.json",
        {
            "stage": "stage11_cga_hm",
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
    candidate_metadata(pools.test_candidates, "test").to_csv(output_dir / "stage11_candidate_metadata.csv", index=False)
    print("[stage11-cga-hm] fitting CGA scoring models", flush=True)
    models, selector_metrics, _, _ = fit_cga_models(
        assets.val_train,
        assets.val_calib,
        pools.train_candidates,
        pools.calib_candidates,
        assets.schema,
        seed=seed,
    )
    test_selector_metrics, topk_df, _ = selector_test_metrics(
        selector_metrics,
        assets.test,
        pools.test_candidates,
        assets.schema,
        models,
    )
    selector_out = pd.concat([test_selector_metrics, topk_df.assign(selector="topk", split="test", level="topk")], ignore_index=True, sort=False)
    selector_out.to_csv(output_dir / "stage11_selector_metrics.csv", index=False)
    print("[stage11-cga-hm] calibrating HM variants", flush=True)

    oracle_pred, oracle_mse, _ = oracle_best(deployable_candidates(pools.test_candidates), assets.test)
    calib_oracle_mse = oracle_best(deployable_candidates(pools.calib_candidates), assets.val_calib)[1]

    variants = ["CGA-HM-safe", "CGA-HM-balanced", "CGA-HM-veto-safe", "CGA-HM-veto-balanced"]
    grid_frames: List[pd.DataFrame] = []
    policies: Dict[str, HMPolicy] = {}
    preds: Dict[str, np.ndarray] = {
        "LRBN": assets.test.lrbn_pred,
        "SRA-BP-balanced": candidate_dict(assets.old_test_candidates).get("sra_balanced", assets.old_test_candidates[0]).pred,
        "oracle_stage11_cga_full": oracle_pred,
    }
    decisions_by: Dict[str, pd.DataFrame] = {}
    overall_rows: List[Dict[str, Any]] = []
    for variant in variants:
        print(f"[stage11-cga-hm] calibrating {variant}", flush=True)
        policy, grid = calibrate_hm_policy(
            variant,
            assets.val_calib,
            pools.calib_candidates,
            assets.schema,
            models,
            calib_oracle_mse,
            n_bootstrap=0,
            seed=seed,
        )
        grid["target_variant"] = variant
        grid_frames.append(grid)
        policies[variant] = policy
        prepared = prepare_hm(assets.test, assets.val_calib, pools.test_candidates, assets.schema, models, policy)
        pred, decisions = apply_hm_policy(assets.test, prepared, policy)
        preds[variant] = pred
        decisions_by[variant] = decisions

    print("[stage11-cga-hm] evaluating final variants", flush=True)
    for variant, pred in preds.items():
        selected = decisions_by[variant]["selected"].to_numpy(bool) if variant in decisions_by else None
        row = metric_row(
            variant,
            pred,
            assets.test,
            selected=selected,
            oracle_mse=oracle_mse if variant.startswith("CGA-HM") or variant.startswith("oracle") else None,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        row["variant"] = variant
        if variant in decisions_by:
            decisions = decisions_by[variant]
            row["mean_safe_weight"] = float(decisions["safe_weight"].mean())
            for fam in ACTIVE_FAMILIES:
                row[f"mean_weight_{fam}"] = float(decisions[f"weight_{fam}"].mean())
            row["boundary_veto_rate"] = float(decisions["boundary_veto_active"].mean())
            row.update({f"policy_{k}": v for k, v in asdict(policies[variant]).items()})
        overall_rows.append(row)
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(output_dir / "stage11_overall.csv", index=False)
    pd.concat(grid_frames, ignore_index=True).to_csv(output_dir / "stage11_calibration_grid.csv", index=False)

    masks = stage8_slice_masks(assets.test, assets.schema, stage8_slice_thresholds(assets.val, assets.schema))
    per_config = pd.concat(
        [per_config_rows(variant, pred, assets.test, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    per_config.to_csv(output_dir / "stage11_per_config.csv", index=False)
    slice_df = pd.concat(
        [slice_rows(variant, pred, assets.test, masks, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    slice_df.to_csv(output_dir / "stage11_slice_metrics.csv", index=False)
    selection_distribution(decisions_by).to_csv(output_dir / "stage11_selection_distribution.csv", index=False)
    write_json(
        output_dir / "stage11_bootstrap_ci.json",
        {
            str(row["variant"]): {
                "ci95_low_delta_raw": float(row.get("ci95_low_delta_raw", np.nan)),
                "ci95_high_delta_raw": float(row.get("ci95_high_delta_raw", np.nan)),
                "p_bootstrap_delta_lt_zero": float(row.get("p_bootstrap_delta_lt_zero", np.nan)),
            }
            for _, row in overall.iterrows()
        },
    )

    deployable = overall[overall["variant"].str.startswith("CGA-HM")].copy()
    deployable["stage1_gate_score"] = deployable["mse_delta_pct_vs_lrbn"].astype(float)
    feasible = deployable[
        (deployable["oracle_gain_fraction"] >= 0.08)
        & (deployable["max_config_harm"] <= 0.18)
        & (deployable["mse_delta_pct_vs_lrbn"] < 0.0)
        & (~deployable["test_threshold_leakage"].astype(bool))
    ].copy()
    best = (feasible if not feasible.empty else deployable).sort_values(
        ["stage1_gate_score", "max_config_harm", "harm_rate"], ascending=[True, True, True]
    ).iloc[0]
    stage1_pass = bool(not feasible.empty)
    status = "stage1_pass_ready_for_mini_extension" if stage1_pass else "stage1_failed_stop_before_mini_extension"
    verdict = {
        "stage": "stage11_cga_hm",
        "status": status,
        "stage1_pass": stage1_pass,
        "best_variant": str(best["variant"]),
        "best_mse": float(best["mse"]),
        "best_mae": float(best["mae"]),
        "best_mse_delta_pct_vs_lrbn": float(best["mse_delta_pct_vs_lrbn"]),
        "best_harm_rate": float(best["harm_rate"]),
        "best_max_config_harm": float(best["max_config_harm"]),
        "best_oracle_gain_fraction": float(best["oracle_gain_fraction"]),
        "oracle_full_mse": float(overall.loc[overall["variant"].eq("oracle_stage11_cga_full"), "mse"].iloc[0]),
        "stage1_gate": {
            "oracle_gain_fraction_min": 0.08,
            "max_config_harm_max": 0.18,
            "mse_delta_negative": True,
        },
        "stop_reason": ""
        if stage1_pass
        else (
            f"best deployable policy failed gate: oracle_gain_fraction={float(best['oracle_gain_fraction']):.6f}, "
            f"max_config_harm={float(best['max_config_harm']):.6f}, "
            f"mse_delta_pct_vs_lrbn={float(best['mse_delta_pct_vs_lrbn']):.6f}"
        ),
        "policies": {variant: asdict(policy) for variant, policy in policies.items()},
        "test_threshold_leakage": False,
    }
    write_json(output_dir / "stage11_verdict.json", verdict)
    (output_dir / "summary.md").write_text(build_summary(output_dir, verdict, overall), encoding="utf-8")
    completeness = verify_outputs(output_dir)
    completeness.to_csv(output_dir / "stage11_output_completeness.csv", index=False)
    print(f"[stage11-cga-hm] done in {time.time() - start_time:.1f}s", flush=True)
    return {"verdict": verdict, "overall": overall, "completeness": completeness}
