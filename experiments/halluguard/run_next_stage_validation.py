#!/usr/bin/env python
"""Second-stage validation experiments for HalluGuard innovation directions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


THIS_DIR = Path(__file__).resolve().parent
BASE_PATH = THIS_DIR / "run_research_direction_validation.py"
SPEC = importlib.util.spec_from_file_location("stage1_validation", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot import {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

EPS = 1e-8
FEATURE_COLUMNS = list(base.FEATURE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-features", type=Path, default=Path("experiments/halluguard/results/research_direction_validation/sample_features.csv"))
    parser.add_argument("--action-alignment", type=Path, default=Path("experiments/halluguard/results/research_direction_validation/action_alignment.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halluguard/results/research_direction_validation_stage2"))
    parser.add_argument("--gradient-samples", type=int, default=48)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_df = pd.read_csv(args.sample_features)
    action_df = pd.read_csv(args.action_alignment)

    bp = run_boundary_projection(sample_df, action_df)
    write_csvs(args.output_dir, bp)

    critic = run_critic_assisted_bp(sample_df, bp["bp_sample_table"], args.gradient_samples)
    write_csvs(args.output_dir, critic)

    basis = run_residual_basis_sign_bucket(sample_df, bp["bp_sample_table"])
    write_csvs(args.output_dir, basis)

    multiscale = run_multiscale_retest(sample_df)
    write_csvs(args.output_dir, multiscale)

    regime = run_regime_validation(sample_df, action_df, bp["bp_sample_table"], multiscale["multiscale_sample_table"])
    write_csvs(args.output_dir, regime)

    verdicts = stage2_verdicts(bp, critic, basis, multiscale, regime)
    pd.DataFrame(verdicts).to_csv(args.output_dir / "stage2_direction_verdicts.csv", index=False)
    write_summary(args.output_dir, sample_df, bp, critic, basis, multiscale, regime, verdicts)
    print(json.dumps({"output_dir": str(args.output_dir), "samples": len(sample_df), "verdicts": len(verdicts)}))


def run_boundary_projection(sample_df: pd.DataFrame, action_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    alphas = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
    decays = ("linear", "exp", "constant")
    anchors = ("last", "extrapolate")
    records = []
    for anchor in anchors:
        for decay in decays:
            for alpha in alphas:
                records.append({"scope": "global", "anchor": anchor, "decay": decay, "alpha": alpha, **bp_metrics(sample_df, alpha, decay, anchor, "val")})
    val_table = pd.DataFrame(records)
    best = val_table.sort_values(["mean_mse", "harm_rate"]).iloc[0]
    global_test = bp_metrics(sample_df, float(best["alpha"]), str(best["decay"]), str(best["anchor"]), "test")
    rows = [
        {
            "variant": "HalluGuard-BP-global",
            "calibration": "validation_global",
            "alpha": float(best["alpha"]),
            "decay": str(best["decay"]),
            "anchor": str(best["anchor"]),
            **global_test,
        }
    ]

    domain_rows = []
    sample_rows = []
    for config_id, g in sample_df.groupby("config_id"):
        local_records = []
        for anchor in anchors:
            for decay in decays:
                for alpha in alphas:
                    local_records.append({"config_id": config_id, "anchor": anchor, "decay": decay, "alpha": alpha, **bp_metrics(g, alpha, decay, anchor, "val")})
        local_val = pd.DataFrame(local_records)
        local_best = local_val.sort_values(["mean_mse", "harm_rate"]).iloc[0]
        local_test = bp_metrics(g, float(local_best["alpha"]), str(local_best["decay"]), str(local_best["anchor"]), "test")
        domain_rows.append(
            {
                "config_id": config_id,
                "variant": "HalluGuard-BP-domain",
                "calibration": "validation_per_config",
                "alpha": float(local_best["alpha"]),
                "decay": str(local_best["decay"]),
                "anchor": str(local_best["anchor"]),
                **local_test,
            }
        )
        sample_rows.extend(bp_sample_predictions(g, float(local_best["alpha"]), str(local_best["decay"]), str(local_best["anchor"]), "HalluGuard-BP-domain"))
    domain_df = pd.DataFrame(domain_rows)
    rows.append(aggregate_config_rows(domain_df, "HalluGuard-BP-domain", "validation_per_config"))

    global_samples = bp_sample_predictions(sample_df, float(best["alpha"]), str(best["decay"]), str(best["anchor"]), "HalluGuard-BP-global")
    bp_sample_table = pd.DataFrame(global_samples + sample_rows)
    extended = pd.DataFrame(rows)

    baseline = summarize_existing_actions(action_df)
    extended = pd.concat([extended, baseline], ignore_index=True, sort=False)
    alignment = bp_sample_table.groupby(["variant", "split"]).apply(alignment_agg).reset_index()
    quant = boundary_gap_quantiles(sample_df, bp_sample_table, "HalluGuard-BP-global")
    ablation = pd.concat([val_table, domain_df], ignore_index=True, sort=False)
    return {
        "bp_extended_table": extended,
        "bp_alignment_analysis": alignment,
        "bp_boundary_gap_quantile": quant,
        "bp_ablation_alpha_global_vs_domain": ablation,
        "bp_sample_table": bp_sample_table,
    }


def run_critic_assisted_bp(sample_df: pd.DataFrame, bp_samples: pd.DataFrame, gradient_samples: int) -> Dict[str, pd.DataFrame]:
    critic = fit_plausibility_critic(sample_df)
    global_bp = bp_samples[bp_samples["variant"].eq("HalluGuard-BP-global")].copy()
    score_rows = []
    for _, r in global_bp.iterrows():
        srow = sample_df[(sample_df["config_id"].eq(r["config_id"])) & (sample_df["sample_key"].eq(r["sample_key"]))].iloc[0]
        ctx = arrj(srow["context_json"])
        raw = arrj(srow["raw_prediction_json"])
        bp_pred = arrj(r["prediction_json"])
        h = int(srow["horizon"])
        s_raw = critic_score(critic, ctx, raw, h)
        s_bp = critic_score(critic, ctx, bp_pred, h)
        score_rows.append(
            {
                "config_id": r["config_id"],
                "sample_key": r["sample_key"],
                "split": r["split"],
                "score_raw": s_raw,
                "score_bp": s_bp,
                "score_delta": s_bp - s_raw,
                "bp_mse_delta": r["mse_delta"],
                "bp_gain": -r["mse_delta"],
                "bp_harm": r["harm"],
                "boundary_mismatch": srow["boundary_mismatch"],
            }
        )
    score_df = pd.DataFrame(score_rows)
    score_delta = score_summary(score_df)
    selector_curve, selector_compare = selector_experiment(sample_df, global_bp, score_df)
    grad = finite_difference_critic_gradient(sample_df, critic, gradient_samples)
    return {
        "critic_score_delta_vs_gain": score_delta,
        "critic_selected_risk_coverage": selector_curve,
        "critic_vs_boundarygap_selector": selector_compare,
        "critic_gradient_alignment": grad,
    }


def run_residual_basis_sign_bucket(sample_df: pd.DataFrame, bp_samples: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    sign_rows = []
    correction_rows = []
    for config_id, g in sample_df.groupby("config_id"):
        val = g[g["split"].eq("val")]
        test = g[g["split"].eq("test")]
        if len(val) < 16 or len(test) < 16:
            continue
        e_val = residual_matrix(val)
        e_test = residual_matrix(test)
        k = min(5, e_val.shape[0], e_val.shape[1])
        pca = PCA(n_components=k, random_state=2026).fit(e_val)
        w_val = pca.transform(e_val)
        w_test = pca.transform(e_test)
        x_val = val[FEATURE_COLUMNS].to_numpy(float)
        x_test = test[FEATURE_COLUMNS].to_numpy(float)
        scaler = StandardScaler().fit(x_val)
        predicted_weights = np.zeros_like(w_test)
        for j in range(k):
            y_val = (w_val[:, j] >= 0).astype(int)
            y_test = (w_test[:, j] >= 0).astype(int)
            majority = int(np.round(np.mean(y_val)))
            if len(set(y_val)) < 2:
                pred = np.repeat(majority, len(y_test))
                auc = float("nan")
            else:
                clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=2026 + j)
                clf.fit(scaler.transform(x_val), y_val)
                prob = clf.predict_proba(scaler.transform(x_test))[:, 1]
                pred = (prob >= 0.5).astype(int)
                auc = safe_auc(y_test, prob)
            sign = np.where(pred == 1, 1.0, -1.0)
            median_mag = float(np.median(np.abs(w_val[:, j])))
            predicted_weights[:, j] = sign * median_mag
            sign_rows.append(
                {
                    "config_id": config_id,
                    "component": j + 1,
                    "sign_accuracy": float(accuracy_score(y_test, pred)),
                    "majority_accuracy": float(accuracy_score(y_test, np.repeat(majority, len(y_test)))),
                    "auc": auc,
                    "median_abs_weight": median_mag,
                }
            )
        recon = predicted_weights @ pca.components_
        raw_preds = np.vstack([arrj(v) for v in test["raw_prediction_json"]])
        targets = np.vstack([arrj(v) for v in test["target_json"]])
        best_gamma, best_val = calibrate_basis_gamma(val, pca, w_val)
        corrected = raw_preds + best_gamma * recon
        raw_mse = np.mean((raw_preds - targets) ** 2, axis=1)
        mse = np.mean((corrected - targets) ** 2, axis=1)
        bp_cfg = bp_samples[(bp_samples["variant"].eq("HalluGuard-BP-global")) & (bp_samples["config_id"].eq(config_id)) & (bp_samples["split"].eq("test"))]
        correction_rows.append(
            {
                "config_id": config_id,
                "gamma": best_gamma,
                "val_mean_mse": best_val,
                "test_mean_mse_delta": float(np.mean(mse - raw_mse)),
                "test_mean_mse_delta_pct": float(100.0 * np.mean(mse - raw_mse) / (np.mean(raw_mse) + EPS)),
                "test_harm_rate": float(np.mean(mse > raw_mse + 1e-9)),
                "bp_global_mean_mse_delta": float(np.mean(bp_cfg["mse_delta"])) if len(bp_cfg) else float("nan"),
            }
        )
    return {
        "residual_basis_sign_bucket": pd.DataFrame(sign_rows),
        "residual_basis_lite_correction": pd.DataFrame(correction_rows),
    }


def run_multiscale_retest(sample_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    rows = []
    sample_rows = []
    for _, r in sample_df.iterrows():
        ctx = arrj(r["context_json"])
        raw = arrj(r["raw_prediction_json"])
        target = arrj(r["target_json"])
        residual = target - raw
        low_res, mid_res, high_res = scale_components(residual)
        scores = support_scores(ctx, raw)
        rows.append(
            {
                "config_id": r["config_id"],
                "sample_key": r["sample_key"],
                "split": r["split"],
                "low_residual_energy": float(np.mean(low_res**2)),
                "mid_residual_energy": float(np.mean(mid_res**2)),
                "high_residual_energy": float(np.mean(high_res**2)),
                **scores,
            }
        )
    score_df = pd.DataFrame(rows)
    summary_rows = []
    for split, g in score_df.groupby("split"):
        for score_name, energy_name in (
            ("energy_support", "high_residual_energy"),
            ("phase_support", "mid_residual_energy"),
            ("local_band_boundary", "high_residual_energy"),
        ):
            summary_rows.append(
                {
                    "split": split,
                    "score": score_name,
                    "target_energy": energy_name,
                    "spearman": base.spearman(g[score_name], g[energy_name]),
                }
            )
    val_records = []
    for score_name in ("energy_support", "phase_support", "local_band_boundary"):
        for q in (0.5, 0.65, 0.75, 0.85):
            for strength in (0.15, 0.3, 0.5, 0.75):
                val_records.append({"score": score_name, "q": q, "strength": strength, **adaptive_scale_metrics(sample_df, score_df, score_name, q, strength, "val")})
    val_df = pd.DataFrame(val_records)
    best = val_df.sort_values(["mean_mse", "harm_rate"]).iloc[0]
    test_metrics, test_samples = adaptive_scale_metrics(sample_df, score_df, str(best["score"]), float(best["q"]), float(best["strength"]), "test", return_samples=True)
    summary_rows.append({"split": "test", "score": "adaptive_scale_shrink", "q": float(best["q"]), "strength": float(best["strength"]), **test_metrics})
    sample_rows.extend(test_samples)
    return {
        "multiscale_support_retest": pd.DataFrame(summary_rows),
        "multiscale_adaptive_ablation": val_df,
        "multiscale_sample_table": pd.DataFrame(sample_rows),
    }


def run_regime_validation(sample_df: pd.DataFrame, action_df: pd.DataFrame, bp_samples: pd.DataFrame, multiscale_samples: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    labels = mechanism_labels(sample_df, action_df, bp_samples, multiscale_samples)
    label_df = pd.DataFrame(labels)
    merged = sample_df.merge(label_df, on=["config_id", "sample_key", "split"], how="inner")
    train = merged[merged["split"].eq("val")]
    test = merged[merged["split"].eq("test")]
    rows = []
    pred_rows = []
    if train["regime"].nunique() >= 2:
        scaler = StandardScaler().fit(train[FEATURE_COLUMNS].to_numpy(float))
        clf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=8, random_state=2026)
        clf.fit(scaler.transform(train[FEATURE_COLUMNS].to_numpy(float)), train["regime"])
        pred = clf.predict(scaler.transform(test[FEATURE_COLUMNS].to_numpy(float)))
        majority = train["regime"].value_counts().idxmax()
        rows.append(
            {
                "split": "test",
                "validation": "global",
                "accuracy": float(accuracy_score(test["regime"], pred)),
                "macro_f1": float(f1_score(test["regime"], pred, average="macro")),
                "majority_accuracy": float(accuracy_score(test["regime"], np.repeat(majority, len(test)))),
                "top_action_purity": top_action_purity(test),
                "cross_domain_consistency": cross_domain_regime_consistency(test),
            }
        )
        pred_rows.extend(selector_from_regime_predictions(test, pred, bp_samples, multiscale_samples, action_df))
    for leave_col in ("dataset", "backbone"):
        for heldout in sorted(merged[leave_col].unique()):
            tr = merged[(merged["split"].eq("val")) & (~merged[leave_col].eq(heldout))]
            te = merged[(merged["split"].eq("test")) & (merged[leave_col].eq(heldout))]
            if len(tr) < 16 or len(te) < 16 or tr["regime"].nunique() < 2:
                continue
            scaler = StandardScaler().fit(tr[FEATURE_COLUMNS].to_numpy(float))
            clf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=8, random_state=2030)
            clf.fit(scaler.transform(tr[FEATURE_COLUMNS].to_numpy(float)), tr["regime"])
            pred = clf.predict(scaler.transform(te[FEATURE_COLUMNS].to_numpy(float)))
            maj = tr["regime"].value_counts().idxmax()
            rows.append(
                {
                    "split": "test",
                    "validation": f"leave_one_{leave_col}",
                    "heldout": heldout,
                    "accuracy": float(accuracy_score(te["regime"], pred)),
                    "macro_f1": float(f1_score(te["regime"], pred, average="macro")),
                    "majority_accuracy": float(accuracy_score(te["regime"], np.repeat(maj, len(te)))),
                    "top_action_purity": top_action_purity(te),
                    "cross_domain_consistency": cross_domain_regime_consistency(te),
                }
            )
    selector_df = pd.DataFrame(pred_rows)
    selector_summary = pd.DataFrame([summarize_selector(selector_df, "regime_assisted_selector")]) if len(selector_df) else pd.DataFrame()
    return {
        "regime_mechanism_stability": pd.DataFrame(rows),
        "regime_assisted_selector": selector_summary,
        "regime_labels": label_df,
    }


def bp_metrics(sample_df: pd.DataFrame, alpha: float, decay: str, anchor: str, split: str) -> Dict[str, float]:
    selected = sample_df[sample_df["split"].eq(split)]
    out = [bp_one(r, alpha, decay, anchor) for _, r in selected.iterrows()]
    return summarize_prediction_rows(out)


def bp_sample_predictions(sample_df: pd.DataFrame, alpha: float, decay: str, anchor: str, variant: str) -> List[dict]:
    rows = []
    for _, r in sample_df.iterrows():
        pred_row = bp_one(r, alpha, decay, anchor)
        rows.append({"variant": variant, "alpha": alpha, "decay": decay, "anchor": anchor, **pred_row})
    return rows


def bp_one(r: pd.Series, alpha: float, decay: str, anchor: str) -> dict:
    ctx = arrj(r["context_json"])
    raw = arrj(r["raw_prediction_json"])
    target = arrj(r["target_json"])
    corrected = boundary_projection(ctx, raw, alpha, decay, anchor)
    return prediction_row(r, corrected)


def boundary_projection(ctx: np.ndarray, forecast: np.ndarray, alpha: float, decay: str, anchor: str) -> np.ndarray:
    h = len(forecast)
    if anchor == "last":
        desired_first = ctx[-1]
    elif anchor == "extrapolate":
        desired_first = ctx[-1] + (ctx[-1] - ctx[-2] if len(ctx) >= 2 else 0.0)
    else:
        raise ValueError(anchor)
    gap = desired_first - forecast[0]
    t = np.arange(h, dtype=float)
    if decay == "linear":
        w = np.linspace(1.0, 0.0, h)
    elif decay == "exp":
        w = np.exp(-4.0 * t / max(1.0, h - 1))
    elif decay == "constant":
        w = np.ones(h)
    else:
        raise ValueError(decay)
    return forecast + alpha * w * gap


def prediction_row(r: pd.Series, corrected: np.ndarray) -> dict:
    raw = arrj(r["raw_prediction_json"])
    target = arrj(r["target_json"])
    raw_mse = base.mse(raw, target)
    corrected_mse = base.mse(corrected, target)
    delta = corrected - raw
    residual = target - raw
    alignment = 2.0 * float(np.dot(delta, residual)) / (float(np.dot(delta, delta)) + EPS)
    cosine = float(np.dot(delta, residual)) / ((float(np.linalg.norm(delta)) * float(np.linalg.norm(residual))) + EPS)
    return {
        "config_id": r["config_id"],
        "sample_key": r["sample_key"],
        "dataset": r["dataset"],
        "backbone": r["backbone"],
        "horizon": int(r["horizon"]),
        "seed": int(r["seed"]),
        "split": r["split"],
        "raw_mse": raw_mse,
        "mse": corrected_mse,
        "mse_delta": corrected_mse - raw_mse,
        "mse_delta_pct": 100.0 * (corrected_mse - raw_mse) / (raw_mse + EPS),
        "harm": int(corrected_mse > raw_mse + 1e-9),
        "alignment_ratio": alignment,
        "cosine_alignment": cosine,
        "prediction_json": json.dumps(base.round_list(corrected)),
    }


def summarize_prediction_rows(rows: Sequence[dict]) -> Dict[str, float]:
    if not rows:
        return {"rows": 0}
    df = pd.DataFrame(rows)
    return {
        "rows": len(df),
        "mean_mse": mean(df["mse"]),
        "raw_mean_mse": mean(df["raw_mse"]),
        "mean_mse_delta": mean(df["mse_delta"]),
        "mean_mse_delta_pct": 100.0 * mean(df["mse_delta"]) / (mean(df["raw_mse"]) + EPS),
        "harm_rate": mean(df["harm"]),
        "win_rate_vs_raw": mean((df["mse_delta"] < 0).astype(float)),
        "alignment_A_gt_1_rate": mean((df["alignment_ratio"] > 1.0).astype(float)),
        "mean_cosine": mean(df["cosine_alignment"]),
    }


def summarize_existing_actions(action_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action, g in action_df[action_df["split"].eq("test")].groupby("action"):
        rows.append(
            {
                "variant": action,
                "calibration": "existing_action",
                "rows": len(g),
                "mean_mse": mean(g["mse"]),
                "mean_mse_delta": mean(g["mse_delta"]),
                "mean_mse_delta_pct": mean(g["mse_delta_pct"]),
                "harm_rate": mean(g["harm"]),
                "win_rate_vs_raw": mean((g["mse_delta"] < 0).astype(float)),
                "alignment_A_gt_1_rate": mean((g["alignment_ratio"] > 1.0).astype(float)),
                "mean_cosine": mean(g["cosine_alignment"]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_config_rows(df: pd.DataFrame, variant: str, calibration: str) -> dict:
    return {
        "variant": variant,
        "calibration": calibration,
        "rows": int(df["rows"].sum()),
        "mean_mse": mean(df["mean_mse"]),
        "raw_mean_mse": mean(df["raw_mean_mse"]),
        "mean_mse_delta": mean(df["mean_mse_delta"]),
        "mean_mse_delta_pct": mean(df["mean_mse_delta_pct"]),
        "harm_rate": mean(df["harm_rate"]),
        "win_rate_vs_raw": mean(df["win_rate_vs_raw"]),
        "alignment_A_gt_1_rate": mean(df["alignment_A_gt_1_rate"]),
        "mean_cosine": mean(df["mean_cosine"]),
        "alpha": "per_config",
        "decay": "per_config",
        "anchor": "per_config",
    }


def alignment_agg(g: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "rows": len(g),
            "mean_mse_delta": mean(g["mse_delta"]),
            "harm_rate": mean(g["harm"]),
            "alignment_A_gt_1_rate": mean((g["alignment_ratio"] > 1.0).astype(float)),
            "mean_cosine": mean(g["cosine_alignment"]),
        }
    )


def boundary_gap_quantiles(sample_df: pd.DataFrame, bp_samples: pd.DataFrame, variant: str) -> pd.DataFrame:
    bp = bp_samples[(bp_samples["variant"].eq(variant)) & (bp_samples["split"].eq("test"))]
    merged = sample_df[["config_id", "sample_key", "split", "boundary_mismatch"]].merge(bp, on=["config_id", "sample_key", "split"])
    merged["boundary_gap_bin"] = pd.qcut(merged["boundary_mismatch"], 4, labels=["q1_low", "q2", "q3", "q4_high"], duplicates="drop")
    rows = []
    for b, g in merged.groupby("boundary_gap_bin"):
        rows.append({"boundary_gap_bin": str(b), **summarize_prediction_rows(g.to_dict("records"))})
    return pd.DataFrame(rows)


def fit_plausibility_critic(sample_df: pd.DataFrame) -> dict:
    rows = []
    for _, r in sample_df.iterrows():
        ctx = arrj(r["context_json"])
        raw = arrj(r["raw_prediction_json"])
        target = arrj(r["target_json"])
        h = int(r["horizon"])
        rows.append({"split": r["split"], "label": 1, **base.candidate_features(ctx, target, h)})
        rows.append({"split": r["split"], "label": 0, **base.candidate_features(ctx, raw, h)})
        rows.append({"split": r["split"], "label": 0, **base.candidate_features(ctx, base.corrupt_boundary(target), h)})
        rows.append({"split": r["split"], "label": 0, **base.candidate_features(ctx, base.corrupt_highfreq(target), h)})
    df = pd.DataFrame(rows)
    train = df[df["split"].eq("val")]
    scaler = StandardScaler().fit(train[FEATURE_COLUMNS].to_numpy(float))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=2026)
    clf.fit(scaler.transform(train[FEATURE_COLUMNS].to_numpy(float)), train["label"].to_numpy())
    return {"scaler": scaler, "clf": clf}


def critic_score(critic: dict, ctx: np.ndarray, candidate: np.ndarray, horizon: int) -> float:
    feat = pd.DataFrame([base.candidate_features(ctx, candidate, horizon)])[FEATURE_COLUMNS].to_numpy(float)
    return float(critic["clf"].predict_proba(critic["scaler"].transform(feat))[0, 1])


def score_summary(score_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, g in score_df.groupby("split"):
        rows.append(
            {
                "split": split,
                "rows": len(g),
                "spearman_score_delta_vs_bp_gain": base.spearman(g["score_delta"], g["bp_gain"]),
                "spearman_score_delta_vs_bp_harm": base.spearman(g["score_delta"], g["bp_harm"]),
                "mean_score_delta": mean(g["score_delta"]),
                "mean_bp_gain": mean(g["bp_gain"]),
                "bp_harm_rate": mean(g["bp_harm"]),
            }
        )
    return pd.DataFrame(rows)


def selector_experiment(sample_df: pd.DataFrame, bp_samples: pd.DataFrame, score_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_scores = score_df[score_df["split"].eq("test")].copy()
    bp_test = bp_samples[(bp_samples["variant"].eq("HalluGuard-BP-global")) & (bp_samples["split"].eq("test"))]
    base_lookup = {(r.config_id, r.sample_key): r for r in bp_test.itertuples()}
    sample_lookup = {(r.config_id, r.sample_key): r for r in sample_df[sample_df["split"].eq("test")].itertuples()}
    curves = []
    compare = []
    rng = np.random.default_rng(2026)
    for coverage in (0.25, 0.5, 0.75, 1.0):
        n = max(1, int(round(len(test_scores) * coverage)))
        selectors = {
            "critic_score_delta": test_scores.sort_values("score_delta", ascending=False).head(n),
            "boundary_gap": test_scores.assign(boundary_gap=[getattr(sample_lookup[(r.config_id, r.sample_key)], "boundary_mismatch") for r in test_scores.itertuples()]).sort_values("boundary_gap", ascending=False).head(n),
            "random": test_scores.iloc[rng.permutation(len(test_scores))[:n]],
        }
        for selector, selected in selectors.items():
            selected_keys = set(zip(selected["config_id"], selected["sample_key"]))
            rows = []
            for r in test_scores.itertuples():
                key = (r.config_id, r.sample_key)
                if key in selected_keys:
                    rows.append(base_lookup[key]._asdict())
                else:
                    sr = sample_lookup[key]
                    rows.append({"mse": sr.raw_mse, "raw_mse": sr.raw_mse, "mse_delta": 0.0, "harm": 0})
            metric = summarize_prediction_rows(rows)
            curves.append({"coverage": coverage, "selector": selector, **metric})
    curve_df = pd.DataFrame(curves)
    for coverage, g in curve_df.groupby("coverage"):
        crit = g[g["selector"].eq("critic_score_delta")].iloc[0]
        rand = g[g["selector"].eq("random")].iloc[0]
        bnd = g[g["selector"].eq("boundary_gap")].iloc[0]
        compare.append(
            {
                "coverage": coverage,
                "critic_delta": crit["mean_mse_delta"],
                "boundary_delta": bnd["mean_mse_delta"],
                "random_delta": rand["mean_mse_delta"],
                "critic_harm": crit["harm_rate"],
                "boundary_harm": bnd["harm_rate"],
                "random_harm": rand["harm_rate"],
                "critic_minus_random_delta": crit["mean_mse_delta"] - rand["mean_mse_delta"],
                "critic_minus_boundary_delta": crit["mean_mse_delta"] - bnd["mean_mse_delta"],
            }
        )
    return curve_df, pd.DataFrame(compare)


def finite_difference_critic_gradient(sample_df: pd.DataFrame, critic: dict, max_samples: int) -> pd.DataFrame:
    test = sample_df[sample_df["split"].eq("test")].head(max_samples)
    rows = []
    for _, r in test.iterrows():
        ctx = arrj(r["context_json"])
        raw = arrj(r["raw_prediction_json"])
        target = arrj(r["target_json"])
        h = int(r["horizon"])
        scale = np.std(ctx) + 1e-6
        step = 1e-3 * scale
        grad = np.zeros_like(raw)
        for i in range(len(raw)):
            plus = raw.copy()
            minus = raw.copy()
            plus[i] += step
            minus[i] -= step
            grad[i] = (critic_score(critic, ctx, plus, h) - critic_score(critic, ctx, minus, h)) / (2 * step)
        norm = np.linalg.norm(grad)
        if norm <= EPS:
            delta = grad
        else:
            delta = grad / norm * (0.05 * scale * math.sqrt(len(raw)))
        residual = target - raw
        rows.append(
            {
                "config_id": r["config_id"],
                "sample_key": r["sample_key"],
                "cosine_grad_residual": float(np.dot(delta, residual) / ((np.linalg.norm(delta) * np.linalg.norm(residual)) + EPS)),
                "alignment_ratio": float(2 * np.dot(delta, residual) / (np.dot(delta, delta) + EPS)),
                "A_gt_1": int((2 * np.dot(delta, residual) / (np.dot(delta, delta) + EPS)) > 1.0),
                "raw_mse": base.mse(raw, target),
                "gradient_step_mse_delta": base.mse(raw + delta, target) - base.mse(raw, target),
            }
        )
    if rows:
        rows.append(
            {
                "config_id": "SUMMARY",
                "sample_key": "",
                "cosine_grad_residual": mean([r["cosine_grad_residual"] for r in rows]),
                "alignment_ratio": mean([r["alignment_ratio"] for r in rows]),
                "A_gt_1": mean([r["A_gt_1"] for r in rows]),
                "gradient_step_mse_delta": mean([r["gradient_step_mse_delta"] for r in rows]),
            }
        )
    return pd.DataFrame(rows)


def residual_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.vstack([arrj(r["target_json"]) - arrj(r["raw_prediction_json"]) for _, r in df.iterrows()])


def calibrate_basis_gamma(val: pd.DataFrame, pca: PCA, w_val: np.ndarray) -> Tuple[float, float]:
    x_val = val[FEATURE_COLUMNS].to_numpy(float)
    scaler = StandardScaler().fit(x_val)
    pred_w = np.zeros_like(w_val)
    for j in range(w_val.shape[1]):
        y = (w_val[:, j] >= 0).astype(int)
        median_mag = np.median(np.abs(w_val[:, j]))
        if len(set(y)) < 2:
            pred = np.repeat(int(np.round(np.mean(y))), len(y))
        else:
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=2040 + j)
            clf.fit(scaler.transform(x_val), y)
            pred = (clf.predict_proba(scaler.transform(x_val))[:, 1] >= 0.5).astype(int)
        pred_w[:, j] = np.where(pred == 1, 1.0, -1.0) * median_mag
    recon = pred_w @ pca.components_
    raw = np.vstack([arrj(v) for v in val["raw_prediction_json"]])
    target = np.vstack([arrj(v) for v in val["target_json"]])
    best_gamma, best_mse = 0.0, float("inf")
    for gamma in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
        mse = float(np.mean((raw + gamma * recon - target) ** 2))
        if mse < best_mse:
            best_gamma, best_mse = gamma, mse
    return best_gamma, best_mse


def scale_components(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = base.moving_average(x, 15)
    mid_smooth = base.moving_average(x, 5)
    mid = mid_smooth - low
    high = x - mid_smooth
    return low, mid, high


def support_scores(ctx: np.ndarray, raw: np.ndarray) -> Dict[str, float]:
    ctail = ctx[-len(raw) :] if len(ctx) >= len(raw) else np.pad(ctx, (len(raw) - len(ctx), 0), mode="edge")
    _, _, c_high = scale_components(ctail)
    _, raw_mid, raw_high = scale_components(raw)
    energy_support = max(0.0, np.std(raw_high) - np.std(c_high)) / (np.std(c_high) + EPS)
    phase_support = dominant_phase_gap(ctail, raw)
    local_band_boundary = abs(raw_high[0] - c_high[-1]) / (np.std(c_high) + EPS)
    return {"energy_support": float(energy_support), "phase_support": float(phase_support), "local_band_boundary": float(local_band_boundary), "mid_band_energy": float(np.std(raw_mid))}


def dominant_phase_gap(ctx_tail: np.ndarray, raw: np.ndarray) -> float:
    n = len(raw)
    c = ctx_tail[-n:] if len(ctx_tail) >= n else np.pad(ctx_tail, (n - len(ctx_tail), 0), mode="edge")
    cf = np.fft.rfft(c - np.mean(c))
    rf = np.fft.rfft(raw - np.mean(raw))
    if len(rf) <= 2:
        return 0.0
    idx = int(np.argmax(np.abs(rf[1:])) + 1)
    return float(abs(np.angle(rf[idx]) - np.angle(cf[min(idx, len(cf) - 1)])) / np.pi)


def adaptive_scale_metrics(sample_df: pd.DataFrame, score_df: pd.DataFrame, score_name: str, q: float, strength: float, split: str, return_samples: bool = False):
    scores = score_df[score_df["split"].eq("val")][score_name]
    threshold = float(np.quantile(scores, q))
    merged = sample_df.merge(score_df[["config_id", "sample_key", "split", score_name]], on=["config_id", "sample_key", "split"], how="inner")
    selected = merged[merged["split"].eq(split)]
    rows = []
    for _, r in selected.iterrows():
        ctx = arrj(r["context_json"])
        raw = arrj(r["raw_prediction_json"])
        pred = base.multiscale_shrink(ctx, raw, strength) if float(r[score_name]) >= threshold else raw
        rows.append({"variant": "adaptive_scale_shrink", "score": score_name, "threshold": threshold, **prediction_row(r, pred)})
    metrics = summarize_prediction_rows(rows)
    return (metrics, rows) if return_samples else metrics


def mechanism_labels(sample_df: pd.DataFrame, action_df: pd.DataFrame, bp_samples: pd.DataFrame, multiscale_samples: pd.DataFrame) -> List[dict]:
    lrbn = action_df[action_df["action"].eq("HalluGuard-LRBN")][["config_id", "sample_key", "split", "mse_delta"]].rename(columns={"mse_delta": "lrbn_delta"})
    ema = action_df[action_df["action"].eq("ema_smoothing")][["config_id", "sample_key", "split", "mse_delta"]].rename(columns={"mse_delta": "ema_delta"})
    bp = bp_samples[bp_samples["variant"].eq("HalluGuard-BP-global")][["config_id", "sample_key", "split", "mse_delta"]].rename(columns={"mse_delta": "bp_delta"})
    ms = multiscale_samples[["config_id", "sample_key", "split", "mse_delta"]].rename(columns={"mse_delta": "scale_delta"}) if len(multiscale_samples) else pd.DataFrame(columns=["config_id", "sample_key", "split", "scale_delta"])
    merged = sample_df.merge(lrbn, on=["config_id", "sample_key", "split"], how="left").merge(ema, on=["config_id", "sample_key", "split"], how="left").merge(bp, on=["config_id", "sample_key", "split"], how="left").merge(ms, on=["config_id", "sample_key", "split"], how="left")
    low_raw = merged["raw_mse"].quantile(0.25)
    high_raw = merged["raw_mse"].quantile(0.75)
    high_boundary = merged["boundary_mismatch"].quantile(0.75)
    rows = []
    for _, r in merged.iterrows():
        deltas = {"bp": r.get("bp_delta", 0.0), "scale": r.get("scale_delta", 0.0), "ema": r.get("ema_delta", 0.0), "lrbn": r.get("lrbn_delta", 0.0)}
        best_action = min(deltas, key=lambda k: deltas[k] if not pd.isna(deltas[k]) else 0.0)
        best_delta = deltas[best_action]
        if r["raw_mse"] <= low_raw and min(deltas.values()) > -0.01:
            regime = "R0_stable_raw"
        elif r["boundary_mismatch"] >= high_boundary and r.get("bp_delta", 0.0) < -0.01:
            regime = "R1_boundary_mismatch"
        elif r.get("scale_delta", 0.0) < min(r.get("bp_delta", 0.0), r.get("ema_delta", 0.0), 0.0):
            regime = "R2_hf_or_scale_unsupported"
        elif r["raw_mse"] <= low_raw:
            regime = "R3_residual_low_energy"
        elif r["raw_mse"] >= high_raw and best_delta >= -0.01:
            regime = "R4_large_residual_unaligned"
        else:
            regime = "R5_general_correctable"
        rows.append({"config_id": r["config_id"], "sample_key": r["sample_key"], "split": r["split"], "regime": regime, "oracle_best_action": best_action, "oracle_best_delta": best_delta})
    return rows


def top_action_purity(df: pd.DataFrame) -> float:
    vals = []
    for _, g in df.groupby("regime"):
        vals.append(g["oracle_best_action"].value_counts(normalize=True).iloc[0])
    return mean(vals)


def cross_domain_regime_consistency(df: pd.DataFrame) -> float:
    vals = []
    for _, g in df.groupby("regime"):
        winners = []
        for _, dg in g.groupby(["dataset", "backbone"]):
            winners.append(dg["oracle_best_action"].value_counts().idxmax())
        if winners:
            vals.append(max(Counter(winners).values()) / len(winners))
    return mean(vals)


def selector_from_regime_predictions(test: pd.DataFrame, pred_regimes: Sequence[str], bp_samples: pd.DataFrame, multiscale_samples: pd.DataFrame, action_df: pd.DataFrame) -> List[dict]:
    bp_lookup = {(r.config_id, r.sample_key): r for r in bp_samples[(bp_samples["variant"].eq("HalluGuard-BP-global")) & (bp_samples["split"].eq("test"))].itertuples()}
    ms_lookup = {(r.config_id, r.sample_key): r for r in multiscale_samples.itertuples()} if len(multiscale_samples) else {}
    lrbn = action_df[(action_df["action"].eq("HalluGuard-LRBN")) & (action_df["split"].eq("test"))]
    lrbn_lookup = {(r.config_id, r.sample_key): r for r in lrbn.itertuples()}
    rows = []
    for row, regime in zip(test.itertuples(), pred_regimes):
        key = (row.config_id, row.sample_key)
        if regime == "R1_boundary_mismatch" and key in bp_lookup:
            src = bp_lookup[key]
            rows.append({"mse": src.mse, "raw_mse": src.raw_mse, "mse_delta": src.mse_delta, "harm": src.harm, "alignment_ratio": src.alignment_ratio, "cosine_alignment": src.cosine_alignment})
        elif regime == "R2_hf_or_scale_unsupported" and key in ms_lookup:
            src = ms_lookup[key]
            rows.append({"mse": src.mse, "raw_mse": src.raw_mse, "mse_delta": src.mse_delta, "harm": src.harm, "alignment_ratio": src.alignment_ratio, "cosine_alignment": src.cosine_alignment})
        elif regime == "R5_general_correctable" and key in lrbn_lookup:
            src = lrbn_lookup[key]
            rows.append({"mse": src.mse, "raw_mse": row.raw_mse, "mse_delta": src.mse_delta, "harm": src.harm, "alignment_ratio": src.alignment_ratio, "cosine_alignment": src.cosine_alignment})
        else:
            rows.append({"mse": row.raw_mse, "raw_mse": row.raw_mse, "mse_delta": 0.0, "harm": 0, "alignment_ratio": 0.0, "cosine_alignment": 0.0})
    return rows


def summarize_selector(df: pd.DataFrame, name: str) -> dict:
    if not len(df):
        return {"selector": name, "rows": 0}
    return {"selector": name, **summarize_prediction_rows(df.to_dict("records"))}


def stage2_verdicts(bp, critic, basis, multiscale, regime) -> List[dict]:
    verdicts = []
    bp_table = bp["bp_extended_table"]
    bp_global = bp_table[bp_table["variant"].eq("HalluGuard-BP-global")].iloc[0]
    bp_domain = bp_table[bp_table["variant"].eq("HalluGuard-BP-domain")].iloc[0]
    verdicts.append(
        verdict(
            "Experiment_A_HalluGuard_BP",
            "promising" if bp_global["mean_mse_delta"] < 0 and bp_global["alignment_A_gt_1_rate"] > 0.55 else "weak",
            f"BP-global delta {bp_global['mean_mse_delta']:.6g}, harm {bp_global['harm_rate']:.3f}, A>1 {bp_global['alignment_A_gt_1_rate']:.3f}; BP-domain delta {bp_domain['mean_mse_delta']:.6g}.",
        )
    )
    score = critic["critic_score_delta_vs_gain"]
    test_score = score[score["split"].eq("test")].iloc[0]
    selector = critic["critic_vs_boundarygap_selector"]
    cov50 = selector[selector["coverage"].eq(0.5)].iloc[0]
    verdicts.append(
        verdict(
            "Experiment_B_Critic_Assisted_BP",
            "weak" if test_score["spearman_score_delta_vs_bp_gain"] <= 0.2 or cov50["critic_minus_random_delta"] >= 0 else "promising",
            f"score-delta/gain Spearman {test_score['spearman_score_delta_vs_bp_gain']:.3f}; at 50% coverage critic-random delta gap {cov50['critic_minus_random_delta']:.6g}.",
        )
    )
    sign = basis["residual_basis_sign_bucket"]
    lite = basis["residual_basis_lite_correction"]
    sign_gain = mean(sign["sign_accuracy"] - sign["majority_accuracy"]) if len(sign) else float("nan")
    lite_delta = mean(lite["test_mean_mse_delta"]) if len(lite) else float("nan")
    verdicts.append(
        verdict(
            "Experiment_C_Residual_Basis_Lite",
            "promising" if sign_gain > 0.05 and lite_delta < 0 else "weak",
            f"mean sign accuracy gain vs majority {sign_gain:.3f}; basis-lite test delta {lite_delta:.6g}.",
        )
    )
    ms = multiscale["multiscale_support_retest"]
    support = ms[(ms["split"].eq("test")) & (ms["score"].isin(["energy_support", "phase_support", "local_band_boundary"]))]
    adapt = ms[ms["score"].eq("adaptive_scale_shrink")].iloc[0]
    best_spear = float(support["spearman"].abs().max()) if len(support) else 0.0
    verdicts.append(
        verdict(
            "Experiment_D_Multiscale_Support",
            "promising" if best_spear > 0.2 and adapt["mean_mse_delta"] < 0 else "weak",
            f"best support/residual |Spearman| {best_spear:.3f}; adaptive shrink delta {adapt['mean_mse_delta']:.6g}.",
        )
    )
    reg = regime["regime_mechanism_stability"]
    global_row = reg[reg["validation"].eq("global")].iloc[0] if len(reg[reg["validation"].eq("global")]) else None
    reg_selector = regime["regime_assisted_selector"]
    selector_delta = float(reg_selector.iloc[0]["mean_mse_delta"]) if len(reg_selector) else float("nan")
    verdicts.append(
        verdict(
            "Experiment_E_Regime_Invariant",
            "promising" if global_row is not None and global_row["top_action_purity"] > 0.65 and global_row["cross_domain_consistency"] > 0.75 else "weak",
            f"global accuracy {global_row['accuracy']:.3f}, purity {global_row['top_action_purity']:.3f}, consistency {global_row['cross_domain_consistency']:.3f}; regime selector delta {selector_delta:.6g}.",
        )
    )
    return verdicts


def write_csvs(output_dir: Path, tables: Dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        if name.endswith("_sample_table") or name == "regime_labels":
            # Large/intermediate tables are useful locally but not part of the headline report.
            df.to_csv(output_dir / f"{name}.csv", index=False)
        else:
            df.to_csv(output_dir / f"{name}.csv", index=False)


def write_summary(output_dir: Path, sample_df: pd.DataFrame, bp, critic, basis, multiscale, regime, verdicts: List[dict]) -> None:
    lines = [
        "# HalluGuard Next-Stage Validation Report",
        "",
        f"- Samples: {len(sample_df)}",
        f"- Configs: {sample_df['config_id'].nunique()}",
        "- Contract: validation selects alphas/thresholds/models; test evaluates.",
        "- Scope: second-stage compact validation, not final TableA.",
        "",
        "## Verdicts",
        "",
    ]
    for v in verdicts:
        lines.append(f"- `{v['direction']}`: **{v['status']}** — {v['reason']}")
    lines.extend(["", "## Experiment A: Boundary Projection", ""])
    for _, r in bp["bp_extended_table"].iterrows():
        lines.append(f"- `{r['variant']}`: delta {float(r['mean_mse_delta']):.6g}, harm {float(r['harm_rate']):.3f}, A>1 {float(r['alignment_A_gt_1_rate']):.3f}")
    lines.extend(["", "## Experiment B: Critic Selector", ""])
    for _, r in critic["critic_score_delta_vs_gain"].iterrows():
        lines.append(f"- split `{r['split']}`: score_delta/gain Spearman {float(r['spearman_score_delta_vs_bp_gain']):.3f}, score_delta/harm Spearman {float(r['spearman_score_delta_vs_bp_harm']):.3f}")
    lines.extend(["", "## Experiment C/D/E Outputs", ""])
    for filename in [
        "residual_basis_sign_bucket.csv",
        "residual_basis_lite_correction.csv",
        "multiscale_support_retest.csv",
        "regime_mechanism_stability.csv",
        "stage2_direction_verdicts.csv",
    ]:
        lines.append(f"- `{filename}`")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def arrj(text: str) -> np.ndarray:
    return np.asarray(json.loads(text), dtype=float)


def mean(values: Iterable) -> float:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def safe_auc(y, score) -> float:
    try:
        if len(set(map(int, y))) < 2:
            return float("nan")
        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


def verdict(direction: str, status: str, reason: str) -> dict:
    return {"direction": direction, "status": status, "reason": reason}


if __name__ == "__main__":
    main()
