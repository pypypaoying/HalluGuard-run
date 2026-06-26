#!/usr/bin/env python
"""Validate HalluGuard research directions on sample-level forecast tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

EPS = 1e-8
ACTIONS = (
    "HalluGuard-LRBN",
    "matched_sparse_smoothing",
    "naive_smoothing",
    "ema_smoothing",
    "median_smoothing",
)
FEATURE_COLUMNS = (
    "boundary_mismatch",
    "slope_mismatch",
    "curvature_mismatch",
    "spectral_distance",
    "highfreq_excess",
    "var_ratio",
    "diffstd_ratio",
    "context_volatility",
    "raw_roughness",
    "pred_context_mean_gap",
    "pred_context_last_gap",
    "horizon_norm",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, default=Path("experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halluguard/results/research_direction_validation"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.metrics_csv)
    sample_rows, action_rows = build_tables(metrics)
    sample_df = pd.DataFrame(sample_rows)
    action_df = pd.DataFrame(action_rows)
    sample_df.to_csv(args.output_dir / "sample_features.csv", index=False)
    action_df.to_csv(args.output_dir / "action_alignment.csv", index=False)

    alignment_summary = summarize_alignment(action_df)
    alignment_summary.to_csv(args.output_dir / "alignment_summary.csv", index=False)

    oracle_sep = oracle_action_separability(sample_df, action_df)
    pd.DataFrame([oracle_sep]).to_csv(args.output_dir / "oracle_action_separability.csv", index=False)

    risk_curve = risk_coverage(sample_df, action_df, action="HalluGuard-LRBN")
    pd.DataFrame(risk_curve).to_csv(args.output_dir / "risk_coverage.csv", index=False)

    basis_summary = residual_basis_summary(sample_df)
    pd.DataFrame(basis_summary).to_csv(args.output_dir / "basis_summary.csv", index=False)

    projection_summary = projection_experiment(sample_df)
    pd.DataFrame(projection_summary).to_csv(args.output_dir / "projection_summary.csv", index=False)

    multiscale_summary = multiscale_experiment(sample_df)
    pd.DataFrame(multiscale_summary).to_csv(args.output_dir / "multiscale_summary.csv", index=False)

    critic_summary = critic_separability(sample_df)
    pd.DataFrame([critic_summary]).to_csv(args.output_dir / "critic_summary.csv", index=False)

    regime_summary = regime_stability(sample_df, action_df)
    pd.DataFrame(regime_summary).to_csv(args.output_dir / "regime_summary.csv", index=False)

    verdicts = direction_verdicts(
        alignment_summary,
        oracle_sep,
        risk_curve,
        basis_summary,
        projection_summary,
        multiscale_summary,
        critic_summary,
        regime_summary,
    )
    pd.DataFrame(verdicts).to_csv(args.output_dir / "direction_verdicts.csv", index=False)
    write_summary(args.output_dir, sample_df, action_df, alignment_summary, oracle_sep, risk_curve, basis_summary, projection_summary, multiscale_summary, critic_summary, regime_summary, verdicts)
    print(json.dumps({"output_dir": str(args.output_dir), "samples": len(sample_df), "action_rows": len(action_df), "directions": len(verdicts)}))


def build_tables(metrics: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    completed = metrics[metrics["status"].eq("completed")].copy()
    grouped: Dict[Tuple[str, str, int, int], Dict[str, str]] = defaultdict(dict)
    for _, row in completed.iterrows():
        key = (row["dataset"], row["backbone"], int(row["horizon"]), int(row["seed"]))
        grouped[key][row["method"]] = row["prediction_path"]

    sample_rows: List[dict] = []
    action_rows: List[dict] = []
    for key, paths in sorted(grouped.items()):
        if "raw_no_correction" not in paths:
            continue
        raw_samples = {sample_key(s): s for s in read_jsonl(Path(paths["raw_no_correction"]))}
        action_samples = {
            action: {sample_key(s): s for s in read_jsonl(Path(path))}
            for action, path in paths.items()
            if action in ACTIONS and Path(path).exists()
        }
        for skey, raw in raw_samples.items():
            context = arr(raw["context"])
            pred = arr(raw["prediction"])
            target = arr(raw["target"])
            residual = target - pred
            features = feature_dict(context, pred, int(key[2]))
            raw_mse = mse(pred, target)
            raw_mae = mae(pred, target)
            row = {
                "config_id": config_id(*key),
                "sample_key": skey,
                "dataset": key[0],
                "backbone": key[1],
                "horizon": key[2],
                "seed": key[3],
                "split": raw["split"],
                "raw_mse": raw_mse,
                "raw_mae": raw_mae,
                "residual_norm": float(np.linalg.norm(residual)),
                "context_json": json.dumps(round_list(context)),
                "raw_prediction_json": json.dumps(round_list(pred)),
                "target_json": json.dumps(round_list(target)),
                **features,
            }
            sample_rows.append(row)
            for action, samples in action_samples.items():
                if skey not in samples:
                    continue
                apred = arr(samples[skey]["prediction"])
                delta = apred - pred
                amse = mse(apred, target)
                amae = mae(apred, target)
                delta_mse = amse - raw_mse
                alignment = 2.0 * float(np.dot(delta, residual)) / (float(np.dot(delta, delta)) + EPS)
                cosine = float(np.dot(delta, residual)) / ((float(np.linalg.norm(delta)) * float(np.linalg.norm(residual))) + EPS)
                action_rows.append(
                    {
                        "config_id": row["config_id"],
                        "sample_key": skey,
                        "dataset": key[0],
                        "backbone": key[1],
                        "horizon": key[2],
                        "seed": key[3],
                        "split": raw["split"],
                        "action": action,
                        "mse": amse,
                        "mae": amae,
                        "mse_delta": delta_mse,
                        "mse_delta_pct": 100.0 * delta_mse / (raw_mse + EPS),
                        "harm": int(delta_mse > 1e-9),
                        "alignment_ratio": alignment,
                        "cosine_alignment": cosine,
                        "delta_norm": float(np.linalg.norm(delta)),
                        "correction_energy_ratio": float(np.linalg.norm(delta) / (np.linalg.norm(pred) + EPS)),
                    }
                )
    if not sample_rows or not action_rows:
        raise RuntimeError("No usable forecast samples/action rows found.")
    return sample_rows, action_rows


def summarize_alignment(action_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action, g in action_df.groupby("action"):
        rows.append(
            {
                "action": action,
                "rows": len(g),
                "test_rows": int(g["split"].eq("test").sum()),
                "mean_mse_delta_test": mean(g[g["split"].eq("test")]["mse_delta"]),
                "mean_mse_delta_pct_test": mean(g[g["split"].eq("test")]["mse_delta_pct"]),
                "harm_rate_test": mean(g[g["split"].eq("test")]["harm"]),
                "alignment_A_gt_1_rate_test": mean((g[g["split"].eq("test")]["alignment_ratio"] > 1.0).astype(float)),
                "mean_alignment_ratio_test": mean(g[g["split"].eq("test")]["alignment_ratio"]),
                "mean_cosine_test": mean(g[g["split"].eq("test")]["cosine_alignment"]),
                "mean_correction_energy_ratio_test": mean(g[g["split"].eq("test")]["correction_energy_ratio"]),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_mse_delta_test")


def oracle_action_separability(sample_df: pd.DataFrame, action_df: pd.DataFrame) -> dict:
    best = action_df.loc[action_df.groupby(["config_id", "sample_key"])["mse"].idxmin()][["config_id", "sample_key", "action"]]
    data = sample_df.merge(best, on=["config_id", "sample_key"], how="inner")
    train = data[data["split"].eq("val")]
    test = data[data["split"].eq("test")]
    if train["action"].nunique() < 2 or test.empty:
        return {"status": "inconclusive", "reason": "not enough action label diversity"}
    x_train, x_test = train[list(FEATURE_COLUMNS)].to_numpy(float), test[list(FEATURE_COLUMNS)].to_numpy(float)
    y_train, y_test = train["action"].to_numpy(), test["action"].to_numpy()
    scaler = StandardScaler().fit(x_train)
    clf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=8, random_state=2026)
    clf.fit(scaler.transform(x_train), y_train)
    pred = clf.predict(scaler.transform(x_test))
    majority = Counter(y_train).most_common(1)[0][0]
    rng = np.random.default_rng(2026)
    shuffled = y_train.copy()
    rng.shuffle(shuffled)
    shuf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=8, random_state=2027)
    shuf.fit(scaler.transform(x_train), shuffled)
    shuf_pred = shuf.predict(scaler.transform(x_test))
    return {
        "status": "completed",
        "train_rows": len(train),
        "test_rows": len(test),
        "n_actions": int(train["action"].nunique()),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "majority_accuracy": float(accuracy_score(y_test, np.repeat(majority, len(y_test)))),
        "shuffled_feature_accuracy": float(accuracy_score(y_test, shuf_pred)),
        "oracle_best_action_entropy_test": entropy(test["action"]),
        "top_predicted_action_rate": float(pd.Series(pred).value_counts(normalize=True).max()),
    }


def risk_coverage(sample_df: pd.DataFrame, action_df: pd.DataFrame, action: str) -> List[dict]:
    action_data = action_df[action_df["action"].eq(action)][["config_id", "sample_key", "harm", "mse_delta"]]
    data = sample_df.merge(action_data, on=["config_id", "sample_key"], how="inner")
    train = data[data["split"].eq("val")]
    test = data[data["split"].eq("test")]
    if train["harm"].nunique() < 2:
        return [{"action": action, "status": "inconclusive", "reason": "validation harm labels have one class"}]
    scaler = StandardScaler().fit(train[list(FEATURE_COLUMNS)].to_numpy(float))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=2026)
    clf.fit(scaler.transform(train[list(FEATURE_COLUMNS)].to_numpy(float)), train["harm"].to_numpy())
    risk = clf.predict_proba(scaler.transform(test[list(FEATURE_COLUMNS)].to_numpy(float)))[:, 1]
    labels = test["harm"].to_numpy()
    auc = safe_auc(labels, risk)
    ap = safe_ap(labels, risk)
    rows = []
    for coverage in (0.25, 0.4, 0.5, 0.6, 0.75, 1.0):
        n = max(1, int(round(len(test) * coverage)))
        idx = np.argsort(risk)[:n]
        selected = test.iloc[idx]
        rows.append(
            {
                "action": action,
                "status": "completed",
                "coverage": coverage,
                "selected_rows": len(selected),
                "risk_auc": auc,
                "risk_auprc": ap,
                "mean_mse_delta_selected": mean(selected["mse_delta"]),
                "harm_rate_selected": mean(selected["harm"]),
                "full_mean_mse_delta": mean(test["mse_delta"]),
                "full_harm_rate": mean(test["harm"]),
            }
        )
    return rows


def residual_basis_summary(sample_df: pd.DataFrame) -> List[dict]:
    rows = []
    for (dataset, backbone, horizon), g in sample_df.groupby(["dataset", "backbone", "horizon"]):
        train = g[g["split"].eq("val")]
        test = g[g["split"].eq("test")]
        if len(train) < 8 or len(test) < 8:
            continue
        e_train = np.vstack([arr_json(r["target_json"]) - arr_json(r["raw_prediction_json"]) for _, r in train.iterrows()])
        e_test = np.vstack([arr_json(r["target_json"]) - arr_json(r["raw_prediction_json"]) for _, r in test.iterrows()])
        kmax = min(20, e_train.shape[0], e_train.shape[1])
        pca = PCA(n_components=kmax, random_state=2026).fit(e_train)
        evr = np.cumsum(pca.explained_variance_ratio_)
        recon = pca.inverse_transform(pca.transform(e_test))
        test_evr = 1.0 - (np.sum((e_test - recon) ** 2) / (np.sum((e_test - e_test.mean(axis=0)) ** 2) + EPS))
        x_train = train[list(FEATURE_COLUMNS)].to_numpy(float)
        x_test = test[list(FEATURE_COLUMNS)].to_numpy(float)
        scaler = StandardScaler().fit(x_train)
        weights_train = pca.transform(e_train)[:, : min(5, kmax)]
        weights_test = pca.transform(e_test)[:, : min(5, kmax)]
        ridge = Ridge(alpha=1.0).fit(scaler.transform(x_train), weights_train)
        pred_w = ridge.predict(scaler.transform(x_test))
        r2 = 1.0 - np.sum((weights_test - pred_w) ** 2) / (np.sum((weights_test - weights_test.mean(axis=0)) ** 2) + EPS)
        sign_acc = float(np.mean(np.sign(pred_w) == np.sign(weights_test)))
        dct_evr = dct_lowfreq_evr(e_test, min(20, e_test.shape[1]))
        rows.append(
            {
                "dataset": dataset,
                "backbone": backbone,
                "horizon": int(horizon),
                "val_rows": len(train),
                "test_rows": len(test),
                "pca_top5_val_evr": float(evr[min(4, len(evr) - 1)]),
                "pca_top10_val_evr": float(evr[min(9, len(evr) - 1)]),
                "pca_top20_val_evr": float(evr[-1]),
                "pca_top20_test_recon_evr": float(test_evr),
                "dct_top20_test_evr": float(dct_evr),
                "basis_weight_r2_test": float(r2),
                "basis_weight_sign_accuracy_test": sign_acc,
            }
        )
    rows.append(aggregate_numeric(rows, {"dataset": "ALL", "backbone": "ALL", "horizon": "ALL"}))
    return rows


def projection_experiment(sample_df: pd.DataFrame) -> List[dict]:
    variants = ("boundary_projection", "slope_projection", "curvature_projection", "dynamic_combo_projection")
    rows = []
    for variant in variants:
        strengths = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0]
        val_scores = []
        for strength in strengths:
            val_scores.append((strength, projection_metrics(sample_df, variant, strength, "val")))
        best_strength, _ = min(val_scores, key=lambda x: x[1]["mean_mse"])
        test = projection_metrics(sample_df, variant, best_strength, "test")
        rows.append({"variant": variant, "best_strength_val": best_strength, **test})
    return rows


def projection_metrics(sample_df: pd.DataFrame, variant: str, strength: float, split: str) -> dict:
    rows = sample_df[sample_df["split"].eq(split)]
    deltas = []
    harms = []
    aligns = []
    mses = []
    raw_mses = []
    for _, r in rows.iterrows():
        ctx = arr_json(r["context_json"])
        pred = arr_json(r["raw_prediction_json"])
        target = arr_json(r["target_json"])
        corrected = apply_projection(ctx, pred, variant, strength)
        raw_mse = mse(pred, target)
        amse = mse(corrected, target)
        delta = corrected - pred
        residual = target - pred
        mses.append(amse)
        raw_mses.append(raw_mse)
        deltas.append(amse - raw_mse)
        harms.append(int(amse > raw_mse + 1e-9))
        aligns.append(2.0 * float(np.dot(delta, residual)) / (float(np.dot(delta, delta)) + EPS))
    return {
        "split": split,
        "rows": len(rows),
        "mean_mse": mean(mses),
        "raw_mean_mse": mean(raw_mses),
        "mean_mse_delta": mean(deltas),
        "mean_mse_delta_pct": 100.0 * mean(deltas) / (mean(raw_mses) + EPS),
        "harm_rate": mean(harms),
        "alignment_A_gt_1_rate": mean([a > 1.0 for a in aligns]),
        "mean_alignment_ratio": mean(aligns),
    }


def multiscale_experiment(sample_df: pd.DataFrame) -> List[dict]:
    corr_rows = []
    for split, g in sample_df.groupby("split"):
        hf_residual_energy = []
        hf_excess = []
        low_residual_energy = []
        var_ratio = []
        concentration = []
        for _, r in g.iterrows():
            ctx = arr_json(r["context_json"])
            pred = arr_json(r["raw_prediction_json"])
            target = arr_json(r["target_json"])
            residual = target - pred
            low = moving_average(residual, 15)
            high = residual - moving_average(residual, 7)
            energies = np.array([np.sum(low**2), np.sum((moving_average(residual, 7) - low) ** 2), np.sum(high**2)], dtype=float)
            hf_residual_energy.append(float(energies[-1] / (np.sum(energies) + EPS)))
            low_residual_energy.append(float(energies[0] / (np.sum(energies) + EPS)))
            hf_excess.append(float(r["highfreq_excess"]))
            var_ratio.append(float(r["var_ratio"]))
            concentration.append(gini(energies))
        corr_rows.append(
            {
                "variant": "scale_residual_energy_analysis",
                "split": split,
                "rows": len(g),
                "mean_energy_gini": mean(concentration),
                "spearman_highfreq_excess_vs_hf_residual": spearman(hf_excess, hf_residual_energy),
                "spearman_var_ratio_vs_low_residual": spearman(var_ratio, low_residual_energy),
            }
        )
    strengths = [0.0, 0.15, 0.3, 0.5, 0.75]
    val_scores = [(s, multiscale_metrics(sample_df, s, "val")) for s in strengths]
    best_strength, _ = min(val_scores, key=lambda x: x[1]["mean_mse"])
    corr_rows.append({"variant": "multiscale_unsupported_hf_shrink", "best_strength_val": best_strength, **multiscale_metrics(sample_df, best_strength, "test")})
    return corr_rows


def multiscale_metrics(sample_df: pd.DataFrame, strength: float, split: str) -> dict:
    rows = sample_df[sample_df["split"].eq(split)]
    mses, raw_mses, deltas, harms = [], [], [], []
    for _, r in rows.iterrows():
        ctx = arr_json(r["context_json"])
        pred = arr_json(r["raw_prediction_json"])
        target = arr_json(r["target_json"])
        corrected = multiscale_shrink(ctx, pred, strength)
        raw_mse = mse(pred, target)
        amse = mse(corrected, target)
        mses.append(amse)
        raw_mses.append(raw_mse)
        deltas.append(amse - raw_mse)
        harms.append(int(amse > raw_mse + 1e-9))
    return {
        "split": split,
        "rows": len(rows),
        "mean_mse": mean(mses),
        "raw_mean_mse": mean(raw_mses),
        "mean_mse_delta": mean(deltas),
        "mean_mse_delta_pct": 100.0 * mean(deltas) / (mean(raw_mses) + EPS),
        "harm_rate": mean(harms),
    }


def critic_separability(sample_df: pd.DataFrame) -> dict:
    rows = []
    for _, r in sample_df.iterrows():
        ctx = arr_json(r["context_json"])
        raw = arr_json(r["raw_prediction_json"])
        target = arr_json(r["target_json"])
        rows.append({"split": r["split"], "label": 1, **candidate_features(ctx, target, int(r["horizon"]))})
        rows.append({"split": r["split"], "label": 0, "negative_type": "raw_forecast", **candidate_features(ctx, raw, int(r["horizon"]))})
        rows.append({"split": r["split"], "label": 0, "negative_type": "boundary_corrupt", **candidate_features(ctx, corrupt_boundary(target), int(r["horizon"]))})
        rows.append({"split": r["split"], "label": 0, "negative_type": "hf_corrupt", **candidate_features(ctx, corrupt_highfreq(target), int(r["horizon"]))})
    df = pd.DataFrame(rows).fillna(0.0)
    feat_cols = list(FEATURE_COLUMNS)
    train = df[df["split"].eq("val")]
    test = df[df["split"].eq("test")]
    scaler = StandardScaler().fit(train[feat_cols].to_numpy(float))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=2026)
    clf.fit(scaler.transform(train[feat_cols].to_numpy(float)), train["label"].to_numpy())
    score = clf.predict_proba(scaler.transform(test[feat_cols].to_numpy(float)))[:, 1]
    return {
        "status": "completed",
        "train_rows": len(train),
        "test_rows": len(test),
        "auc_true_vs_raw_corrupted": safe_auc(test["label"].to_numpy(), score),
        "auprc_true_vs_raw_corrupted": safe_ap(test["label"].to_numpy(), score),
        "gradient_alignment_status": "not_evaluated_feature_critic_only",
        "note": "A strong AUC supports critic separability; gradient refinement still needs differentiable energy validation.",
    }


def regime_stability(sample_df: pd.DataFrame, action_df: pd.DataFrame) -> List[dict]:
    best = action_df.loc[action_df.groupby(["config_id", "sample_key"])["mse"].idxmin()][["config_id", "sample_key", "action", "mse_delta"]]
    data = sample_df.merge(best, on=["config_id", "sample_key"], how="inner")
    x = data[list(FEATURE_COLUMNS)].to_numpy(float)
    x = StandardScaler().fit_transform(x)
    n_clusters = min(6, max(2, len(data) // 64))
    labels = KMeans(n_clusters=n_clusters, random_state=2026, n_init=20).fit_predict(x)
    data = data.copy()
    data["regime"] = labels
    rows = []
    for regime, g in data.groupby("regime"):
        action_counts = g["action"].value_counts(normalize=True)
        domain_best = []
        for _, dg in g.groupby(["dataset", "backbone"]):
            domain_best.append(dg["action"].value_counts().idxmax())
        rows.append(
            {
                "regime": int(regime),
                "rows": len(g),
                "top_action": action_counts.index[0],
                "top_action_rate": float(action_counts.iloc[0]),
                "action_entropy": entropy(g["action"]),
                "mean_best_action_mse_delta": mean(g["mse_delta"]),
                "cross_domain_top_action_consistency": max(Counter(domain_best).values()) / max(1, len(domain_best)),
                "lrbtn_harm_rate": lrbn_harm_rate(g, action_df),
            }
        )
    rows.append(aggregate_numeric(rows, {"regime": "ALL", "top_action": "", "action_entropy": mean([r["action_entropy"] for r in rows])}))
    return rows


def direction_verdicts(alignment_summary, oracle_sep, risk_curve, basis_summary, projection_summary, multiscale_summary, critic_summary, regime_summary) -> List[dict]:
    verdicts = []
    best_align = alignment_summary.iloc[0].to_dict()
    verdicts.append(
        verdict(
            "E1_residual_alignment",
            "promising" if best_align["alignment_A_gt_1_rate_test"] > 0.5 and best_align["mean_mse_delta_test"] < 0 else "weak",
            f"Best action {best_align['action']} test delta {best_align['mean_mse_delta_test']:.6g}, A>1 rate {best_align['alignment_A_gt_1_rate_test']:.3f}.",
        )
    )
    verdicts.append(
        verdict(
            "E2_oracle_action_separability",
            "promising" if oracle_sep.get("test_accuracy", 0) > max(oracle_sep.get("majority_accuracy", 0), oracle_sep.get("shuffled_feature_accuracy", 0)) + 0.05 else "weak",
            f"Accuracy {oracle_sep.get('test_accuracy')}, majority {oracle_sep.get('majority_accuracy')}, shuffled {oracle_sep.get('shuffled_feature_accuracy')}.",
        )
    )
    risk_df = pd.DataFrame(risk_curve)
    low_cov = risk_df[risk_df.get("coverage", pd.Series(dtype=float)).eq(0.5)] if "coverage" in risk_df else pd.DataFrame()
    if not low_cov.empty:
        r = low_cov.iloc[0]
        status = "promising" if r["risk_auc"] > 0.6 and r["harm_rate_selected"] < r["full_harm_rate"] else "weak"
        reason = f"Risk AUC {r['risk_auc']:.3f}; 50% coverage harm {r['harm_rate_selected']:.3f} vs full {r['full_harm_rate']:.3f}."
    else:
        status, reason = "inconclusive", "Risk model could not be trained."
    verdicts.append(verdict("E3_no_harm_selective_correction", status, reason))
    basis_all = pd.DataFrame(basis_summary)
    all_row = basis_all[basis_all["dataset"].eq("ALL")].iloc[0]
    verdicts.append(
        verdict(
            "E4_residual_basis_decomposition",
            "promising" if all_row["pca_top10_val_evr"] > 0.5 else "weak",
            f"Mean top10 PCA EVR {all_row['pca_top10_val_evr']:.3f}, test recon EVR {all_row['pca_top20_test_recon_evr']:.3f}, weight R2 {all_row['basis_weight_r2_test']:.3f}.",
        )
    )
    proj = pd.DataFrame(projection_summary).sort_values("mean_mse_delta")
    p = proj.iloc[0]
    verdicts.append(
        verdict(
            "E5_dynamic_consistency_projection",
            "promising" if p["mean_mse_delta"] < 0 else "weak",
            f"Best projection {p['variant']} delta {p['mean_mse_delta']:.6g} ({p['mean_mse_delta_pct']:.3f}%).",
        )
    )
    ms = pd.DataFrame(multiscale_summary)
    edit = ms[ms["variant"].eq("multiscale_unsupported_hf_shrink")].iloc[0]
    corr = ms[ms["variant"].eq("scale_residual_energy_analysis")]
    corr_test = corr[corr["split"].eq("test")].iloc[0]
    verdicts.append(
        verdict(
            "E6_multiscale_amplitude_phase_support",
            "promising" if edit["mean_mse_delta"] < 0 or abs(corr_test["spearman_highfreq_excess_vs_hf_residual"]) > 0.2 else "weak",
            f"Multiscale edit delta {edit['mean_mse_delta']:.6g}; hf mismatch/residual Spearman {corr_test['spearman_highfreq_excess_vs_hf_residual']:.3f}.",
        )
    )
    verdicts.append(
        verdict(
            "E7_energy_critic_separability",
            "promising" if critic_summary["auc_true_vs_raw_corrupted"] > 0.7 else "weak",
            f"Critic AUC {critic_summary['auc_true_vs_raw_corrupted']:.3f}; gradient alignment not yet evaluated.",
        )
    )
    verdicts.append(verdict("E8_tsfm_disagreement", "blocked", "No local Chronos/TimesFM/Moirai forecast files were available; do not fake TSFM disagreement."))
    reg = pd.DataFrame(regime_summary)
    reg_all = reg[reg["regime"].eq("ALL")].iloc[0]
    verdicts.append(
        verdict(
            "E9_regime_invariant_correction",
            "promising" if reg_all["top_action_rate"] > 0.5 and reg_all["cross_domain_top_action_consistency"] > 0.5 else "weak",
            f"Mean top-action rate {reg_all['top_action_rate']:.3f}, cross-domain consistency {reg_all['cross_domain_top_action_consistency']:.3f}.",
        )
    )
    return verdicts


def write_summary(output_dir: Path, sample_df, action_df, alignment_summary, oracle_sep, risk_curve, basis_summary, projection_summary, multiscale_summary, critic_summary, regime_summary, verdicts) -> None:
    lines = [
        "# HalluGuard Research Direction Validation",
        "",
        f"- Samples: {len(sample_df)}",
        f"- Action rows: {len(action_df)}",
        f"- Configs: {sample_df['config_id'].nunique()}",
        "- Split contract: validation trains/calibrates diagnostics; test evaluates.",
        "",
        "## Direction Verdicts",
        "",
    ]
    for v in verdicts:
        lines.append(f"- `{v['direction']}`: **{v['status']}** — {v['reason']}")
    lines.extend(["", "## Alignment Summary", ""])
    for _, r in alignment_summary.iterrows():
        lines.append(f"- `{r['action']}`: test mean delta {r['mean_mse_delta_test']:.6g}, harm {r['harm_rate_test']:.3f}, A>1 {r['alignment_A_gt_1_rate_test']:.3f}")
    lines.extend(["", "## Risk / Coverage", ""])
    for r in risk_curve:
        if r.get("status") == "completed":
            lines.append(f"- coverage {r['coverage']}: selected delta {r['mean_mse_delta_selected']:.6g}, harm {r['harm_rate_selected']:.3f}, risk AUC {r['risk_auc']:.3f}")
        else:
            lines.append(f"- {r}")
    lines.extend(["", "## Key Output Files", ""])
    for name in [
        "sample_features.csv",
        "action_alignment.csv",
        "alignment_summary.csv",
        "oracle_action_separability.csv",
        "risk_coverage.csv",
        "basis_summary.csv",
        "projection_summary.csv",
        "multiscale_summary.csv",
        "critic_summary.csv",
        "regime_summary.csv",
        "direction_verdicts.csv",
    ]:
        lines.append(f"- `{name}`")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sample_key(sample: dict) -> str:
    return f"{sample.get('split')}::{sample.get('sample_id')}"


def config_id(dataset: str, backbone: str, horizon: int, seed: int) -> str:
    return f"{dataset}_{backbone}_{horizon}_seed{seed}"


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def arr_json(x: str) -> np.ndarray:
    return np.asarray(json.loads(x), dtype=float)


def round_list(x: np.ndarray) -> List[float]:
    return [round(float(v), 6) for v in np.asarray(x, dtype=float).tolist()]


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def feature_dict(context: np.ndarray, pred: np.ndarray, horizon: int) -> Dict[str, float]:
    return candidate_features(context, pred, horizon)


def candidate_features(context: np.ndarray, candidate: np.ndarray, horizon: int) -> Dict[str, float]:
    ctx = np.asarray(context, dtype=float)
    z = np.asarray(candidate, dtype=float)
    ctail = ctx[-min(len(ctx), len(z)) :]
    ctx_diff = np.diff(ctx)
    z_diff = np.diff(z)
    ctx_last_diff = ctx_diff[-1] if len(ctx_diff) else 0.0
    z_first_diff = z_diff[0] if len(z_diff) else 0.0
    ctx_curv = ctx[-1] - 2 * ctx[-2] + ctx[-3] if len(ctx) >= 3 else 0.0
    z_curv = z[2] - 2 * z[1] + z[0] if len(z) >= 3 else 0.0
    scale = float(np.std(ctx)) + EPS
    ctx_hf = highfreq_energy(ctail)
    z_hf = highfreq_energy(z)
    ctx_fft = spectral_profile(ctail, len(z))
    z_fft = spectral_profile(z, len(z))
    return {
        "boundary_mismatch": abs(float(z[0] - (ctx[-1] + ctx_last_diff))) / scale,
        "slope_mismatch": abs(float(z_first_diff - ctx_last_diff)) / scale,
        "curvature_mismatch": abs(float(z_curv - ctx_curv)) / scale,
        "spectral_distance": float(np.linalg.norm(z_fft - ctx_fft)),
        "highfreq_excess": max(0.0, z_hf - ctx_hf) / (ctx_hf + EPS),
        "var_ratio": float(np.std(z) / (np.std(ctail) + EPS)),
        "diffstd_ratio": float((np.std(z_diff) + EPS) / (np.std(np.diff(ctail)) + EPS)) if len(z_diff) and len(ctail) > 1 else 1.0,
        "context_volatility": float(np.std(np.diff(ctx)) / scale) if len(ctx) > 1 else 0.0,
        "raw_roughness": float(np.mean(np.abs(np.diff(z, n=2))) / scale) if len(z) >= 3 else 0.0,
        "pred_context_mean_gap": abs(float(np.mean(z) - np.mean(ctail))) / scale,
        "pred_context_last_gap": abs(float(z[0] - ctx[-1])) / scale,
        "horizon_norm": float(horizon / 720.0),
    }


def spectral_profile(x: np.ndarray, length: int) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    if len(y) != length:
        y = y[-min(len(y), length) :]
        if len(y) < length:
            y = np.pad(y, (length - len(y), 0), mode="edge")
    mag = np.abs(np.fft.rfft(y - np.mean(y)))
    total = float(np.sum(mag)) + EPS
    return mag / total


def highfreq_energy(x: np.ndarray) -> float:
    y = np.asarray(x, dtype=float)
    if len(y) < 4:
        return 0.0
    mag = np.abs(np.fft.rfft(y - np.mean(y))) ** 2
    cut = max(1, int(0.6 * len(mag)))
    return float(np.sum(mag[cut:]) / (np.sum(mag) + EPS))


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    y = np.asarray(x, dtype=float)
    if window <= 1 or len(y) < 3:
        return y.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def apply_projection(ctx: np.ndarray, pred: np.ndarray, variant: str, strength: float) -> np.ndarray:
    h = len(pred)
    t = np.arange(h, dtype=float)
    decay = max(6.0, h / 8.0)
    w = np.exp(-t / decay)
    ctx_diff = np.diff(ctx)
    last_diff = ctx_diff[-1] if len(ctx_diff) else 0.0
    boundary_jump = (ctx[-1] + last_diff) - pred[0]
    pred_diff = pred[1] - pred[0] if h > 1 else 0.0
    slope_mismatch = last_diff - pred_diff
    ctx_curv = ctx[-1] - 2 * ctx[-2] + ctx[-3] if len(ctx) >= 3 else 0.0
    pred_curv = pred[2] - 2 * pred[1] + pred[0] if h >= 3 else 0.0
    curv_mismatch = ctx_curv - pred_curv
    delta = np.zeros_like(pred)
    if variant in ("boundary_projection", "dynamic_combo_projection"):
        delta += boundary_jump * w
    if variant in ("slope_projection", "dynamic_combo_projection"):
        delta += slope_mismatch * (t + 1.0) * w / max(1.0, decay)
    if variant in ("curvature_projection", "dynamic_combo_projection"):
        delta += 0.5 * curv_mismatch * ((t + 1.0) / max(1.0, decay)) ** 2 * w
    return pred + strength * delta


def multiscale_shrink(ctx: np.ndarray, pred: np.ndarray, strength: float) -> np.ndarray:
    p_smooth = moving_average(pred, 7)
    p_high = pred - p_smooth
    ctail = ctx[-len(pred) :] if len(ctx) >= len(pred) else np.pad(ctx, (len(pred) - len(ctx), 0), mode="edge")
    c_high = ctail - moving_average(ctail, 7)
    support = np.std(c_high) + EPS
    excess = max(0.0, np.std(p_high) - support) / support
    gate = min(1.0, excess)
    return p_smooth + (1.0 - strength * gate) * p_high


def corrupt_boundary(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    out[: max(1, len(out) // 8)] += 0.75 * (np.std(y) + EPS)
    return out


def corrupt_highfreq(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    t = np.arange(len(out))
    out += 0.35 * (np.std(y) + EPS) * np.sin(2 * np.pi * t / 3.0)
    return out


def dct_lowfreq_evr(x: np.ndarray, k: int) -> float:
    coeff = np.fft.rfft(x - x.mean(axis=1, keepdims=True), axis=1)
    power = np.abs(coeff) ** 2
    return float(np.sum(power[:, : min(k, power.shape[1])]) / (np.sum(power) + EPS))


def gini(x: np.ndarray) -> float:
    y = np.sort(np.asarray(x, dtype=float))
    if np.sum(y) <= EPS:
        return 0.0
    n = len(y)
    return float((2 * np.sum((np.arange(n) + 1) * y) / (n * np.sum(y))) - (n + 1) / n)


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    a = pd.Series(a).rank().to_numpy(float)
    b = pd.Series(b).rank().to_numpy(float)
    if np.std(a) <= EPS or np.std(b) <= EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def lrbn_harm_rate(g: pd.DataFrame, action_df: pd.DataFrame) -> float:
    keys = set(zip(g["config_id"], g["sample_key"]))
    lrbn = action_df[action_df["action"].eq("HalluGuard-LRBN")]
    selected = lrbn[[key in keys for key in zip(lrbn["config_id"], lrbn["sample_key"])]]
    return mean(selected["harm"]) if len(selected) else 0.0


def aggregate_numeric(rows: List[dict], prefix: Dict[str, object]) -> dict:
    if not rows:
        return prefix
    out = dict(prefix)
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float, np.floating))})
    for k in keys:
        if k in prefix:
            continue
        out[k] = mean([r[k] for r in rows if k in r and isinstance(r[k], (int, float, np.floating))])
    return out


def safe_auc(y, score) -> float:
    try:
        if len(set(map(int, y))) < 2:
            return float("nan")
        return float(roc_auc_score(y, score))
    except Exception:
        return float("nan")


def safe_ap(y, score) -> float:
    try:
        return float(average_precision_score(y, score))
    except Exception:
        return float("nan")


def entropy(values: Iterable[str]) -> float:
    counts = np.asarray(list(Counter(values).values()), dtype=float)
    if len(counts) == 0:
        return 0.0
    p = counts / np.sum(counts)
    return float(-np.sum(p * np.log(p + EPS)))


def mean(values: Iterable) -> float:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def verdict(direction: str, status: str, reason: str) -> dict:
    return {"direction": direction, "status": status, "reason": reason}


if __name__ == "__main__":
    main()
