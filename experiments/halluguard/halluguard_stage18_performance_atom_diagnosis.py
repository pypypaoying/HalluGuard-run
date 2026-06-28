#!/usr/bin/env python
"""Stage18 performance atom extraction mechanism diagnosis.

Stage18 does not train another selector.  It uses the old TAE / Stage10 CGA
candidate pools as a performance microscope and asks whether oracle-selected
corrections after SRA-BP-balanced can be compressed into stable, interpretable
atoms.  Distillation/prototype checks are diagnostic only and use validation
inner-train / inner-calib for all fitted thresholds.
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
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from halluguard_lrbn_bp import ForecastBatch, mae_per_sample, mse_per_sample
from halluguard_stage6_mechanism import bootstrap_ci, feature_frame, horizons, safe_pct
from halluguard_stage7_safe_tae import ExpertCandidate, candidate_dict, write_json
from halluguard_stage8_safe_tae_pareto import stage8_slice_masks, stage8_slice_thresholds
from halluguard_stage9_architecture_validation import deployable_candidates, prepare_assets
from halluguard_stage10_cga import NEW_FAMILIES, build_cga_pools, df_to_md, json_default
from halluguard_stage15_endogenous_editors import scale_matrix


MAIN_PARENT = "SRA-BP-balanced"
PARENT_NAMES = ("LRBN", "SRA-BP-safe", "SRA-BP-balanced")
ATOM_EXCLUDE_FAMILIES = {"boundary", "default", "raw", "phase"}
FAMILY_GROUPS = {
    "residual_distribution": {"residual_distribution"},
    "smoothing_teacher": {"smoothing_teacher"},
    "retrieval_memory": {"retrieval_memory"},
    "volatility_amplitude_level": {"volatility", "amplitude", "level"},
    "old_residual": {"residual"},
    "ensemble": {"ensemble"},
}


@dataclass(frozen=True)
class Stage18Config:
    seed: int = 2026
    bootstrap: int = 2000
    output_dir: str = "experiments/halluguard/results/stage18_performance_atom_diagnosis"
    n_atoms: int = 5
    pca_components: int = 8
    prototype_threshold_quantiles: Tuple[float, ...] = (0.50, 0.70, 0.80, 0.90, 0.95)
    prototype_shrinks: Tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    max_harm_for_proto: float = 0.05
    max_config_harm_for_proto: float = 0.10


@dataclass
class SplitPools:
    batch: ForecastBatch
    old: List[ExpertCandidate]
    cga: List[ExpertCandidate]


@dataclass
class OracleResult:
    pred: np.ndarray
    mse: np.ndarray
    candidate: np.ndarray
    family: np.ndarray
    delta_vs_parent: np.ndarray


def finite_arr(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def parent_predictions(batch: ForecastBatch, old_candidates: Sequence[ExpertCandidate]) -> Dict[str, np.ndarray]:
    by = candidate_dict(old_candidates)
    return {
        "LRBN": np.asarray(batch.lrbn_pred, dtype=float),
        "SRA-BP-safe": np.asarray(by.get("sra_safe", ExpertCandidate("fallback", "safe", "boundary", batch.lrbn_pred, True)).pred, dtype=float),
        "SRA-BP-balanced": np.asarray(
            by.get("sra_balanced", ExpertCandidate("fallback", "balanced", "boundary", batch.lrbn_pred, True)).pred,
            dtype=float,
        ),
    }


def parent_candidate(parent_name: str, pred: np.ndarray) -> ExpertCandidate:
    return ExpertCandidate(f"parent::{parent_name}", "parent", "parent", np.asarray(pred, dtype=float), True)


def unique_candidates(candidates: Sequence[ExpertCandidate]) -> List[ExpertCandidate]:
    out: Dict[str, ExpertCandidate] = {}
    for c in candidates:
        if c.deployable:
            out[c.name] = c
    return list(out.values())


def candidate_pools(old: Sequence[ExpertCandidate], cga: Sequence[ExpertCandidate]) -> Dict[str, List[ExpertCandidate]]:
    old_deploy = unique_candidates(deployable_candidates(old))
    cga_deploy = unique_candidates(deployable_candidates(cga))
    new_fam = [c for c in cga_deploy if c.family in NEW_FAMILIES and c.name != "keep_lrbn"]
    atom = [
        c
        for c in cga_deploy
        if c.name != "keep_lrbn" and c.deployable and c.family not in ATOM_EXCLUDE_FAMILIES
    ]
    return {
        "tae_old_pool": old_deploy,
        "cga_new_family_pool": new_fam,
        "union_full_pool": cga_deploy,
        "sra_complement_atom_pool": atom,
    }


def oracle_against_parent(
    batch: ForecastBatch,
    candidates: Sequence[ExpertCandidate],
    parent_name: str,
    parent_pred: np.ndarray,
) -> OracleResult:
    cand = [parent_candidate(parent_name, parent_pred)] + [c for c in candidates if c.deployable]
    names = np.asarray([c.name for c in cand], dtype=object)
    fams = np.asarray([c.family for c in cand], dtype=object)
    losses = np.stack([mse_per_sample(c.pred, batch.y_true) for c in cand], axis=1)
    best_idx = np.argmin(losses, axis=1)
    pred = np.asarray(parent_pred, dtype=float).copy()
    for j, c in enumerate(cand):
        mask = best_idx == j
        if mask.any():
            pred[mask] = c.pred[mask]
    parent_mse = mse_per_sample(parent_pred, batch.y_true)
    return OracleResult(
        pred=pred,
        mse=losses[np.arange(len(batch.meta)), best_idx],
        candidate=names[best_idx],
        family=fams[best_idx],
        delta_vs_parent=losses[np.arange(len(batch.meta)), best_idx] - parent_mse,
    )


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
    name: str,
    pred: np.ndarray,
    batch: ForecastBatch,
    parent_name: str,
    parent_pred: np.ndarray,
    n_bootstrap: int,
    seed: int,
    selected: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    method_mse = mse_per_sample(pred, batch.y_true)
    parent_mse = mse_per_sample(parent_pred, batch.y_true)
    method_mae = mae_per_sample(pred, batch.y_true)
    parent_mae = mae_per_sample(parent_pred, batch.y_true)
    delta = method_mse - parent_mse
    max_harm, improved, total = config_harm(delta, batch)
    ci = bootstrap_ci(delta, n_boot=n_bootstrap, seed=seed) if n_bootstrap > 0 else {}
    row: Dict[str, Any] = {
        "variant": name,
        "parent": parent_name,
        "n": int(len(batch.meta)),
        "mse": float(np.mean(method_mse)),
        "mae": float(np.mean(method_mae)),
        "parent_mse": float(np.mean(parent_mse)),
        "parent_mae": float(np.mean(parent_mae)),
        "mse_delta_vs_parent": float(np.mean(delta)),
        "mse_delta_pct_vs_parent": safe_pct(float(np.mean(method_mse)), float(np.mean(parent_mse))),
        "mae_delta_pct_vs_parent": safe_pct(float(np.mean(method_mae)), float(np.mean(parent_mae))),
        "harm_rate_vs_parent": float(np.mean(delta > 1e-12)),
        "win_rate_vs_parent": float(np.mean(delta < 0.0)),
        "max_config_harm": max_harm,
        "improved_configs": improved,
        "total_configs": total,
        "test_threshold_leakage": False,
    }
    if selected is not None:
        selected = np.asarray(selected, dtype=bool)
        row["coverage"] = float(np.mean(selected))
        row["selected_count"] = int(np.sum(selected))
        row["selected_harm_rate"] = float(np.mean((delta > 1e-12)[selected])) if selected.any() else 0.0
    else:
        row["coverage"] = 1.0
        row["selected_count"] = int(len(batch.meta))
        row["selected_harm_rate"] = row["harm_rate_vs_parent"]
    if ci:
        row["ci95_low_delta_raw"] = ci["ci95_low"]
        row["ci95_high_delta_raw"] = ci["ci95_high"]
        row["p_bootstrap_delta_lt_zero"] = ci["p_lt_zero"]
    return row


def parent_oracle_table(
    split_pool: SplitPools,
    cfg: Stage18Config,
    split_name: str,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], OracleResult]]:
    parents = parent_predictions(split_pool.batch, split_pool.old)
    pools = candidate_pools(split_pool.old, split_pool.cga)
    rows: List[Dict[str, Any]] = []
    results: Dict[Tuple[str, str], OracleResult] = {}
    # Denominator: full union oracle gain over LRBN.
    lrbn_oracle = oracle_against_parent(split_pool.batch, pools["union_full_pool"], "LRBN", parents["LRBN"])
    lrbn_base = mse_per_sample(parents["LRBN"], split_pool.batch.y_true)
    lrbn_oracle_gain = float(np.mean(lrbn_base - lrbn_oracle.mse))
    for parent_name in PARENT_NAMES:
        parent_pred = parents[parent_name]
        parent_mse = mse_per_sample(parent_pred, split_pool.batch.y_true)
        for pool_name, cands in pools.items():
            oracle = oracle_against_parent(split_pool.batch, cands, parent_name, parent_pred)
            results[(parent_name, pool_name)] = oracle
            row = metric_vs_parent(
                f"oracle::{pool_name}",
                oracle.pred,
                split_pool.batch,
                parent_name,
                parent_pred,
                cfg.bootstrap if split_name == "test" else 0,
                cfg.seed,
            )
            row.update(
                {
                    "split": split_name,
                    "oracle_pool": pool_name,
                    "parent_method": parent_name,
                    "oracle_mse_over_parent": float(np.mean(oracle.mse)),
                    "incremental_oracle_delta_pct": safe_pct(float(np.mean(oracle.mse)), float(np.mean(parent_mse))),
                    "oracle_gain_fraction_vs_LRBN_oracle": float(np.mean(parent_mse - oracle.mse) / (lrbn_oracle_gain + 1e-8)),
                    "non_parent_selection_rate": float(np.mean(~pd.Series(oracle.family).eq("parent"))),
                }
            )
            fam_share = pd.Series(oracle.family).value_counts(normalize=True)
            for fam, share in fam_share.items():
                row[f"oracle_family_share::{fam}"] = float(share)
            rows.append(row)
    return pd.DataFrame(rows), results


def candidate_family_group(c: ExpertCandidate) -> str:
    for group, fams in FAMILY_GROUPS.items():
        if c.family in fams:
            return group
    return c.family


def family_tables(
    split_pool: SplitPools,
    cfg: Stage18Config,
) -> Tuple[pd.DataFrame, pd.DataFrame, OracleResult, List[ExpertCandidate]]:
    parents = parent_predictions(split_pool.batch, split_pool.old)
    parent_pred = parents[MAIN_PARENT]
    atom_pool = candidate_pools(split_pool.old, split_pool.cga)["sra_complement_atom_pool"]
    full = oracle_against_parent(split_pool.batch, atom_pool, MAIN_PARENT, parent_pred)
    parent_mse = mse_per_sample(parent_pred, split_pool.batch.y_true)
    full_mean = float(np.mean(full.mse))
    loo_rows: List[Dict[str, Any]] = []
    only_rows: List[Dict[str, Any]] = []
    all_groups = sorted({candidate_family_group(c) for c in atom_pool})
    full_share = pd.Series(full.family).value_counts(normalize=True)
    for group in all_groups:
        fams = FAMILY_GROUPS.get(group, {group})
        without = [c for c in atom_pool if c.family not in fams]
        only = [c for c in atom_pool if c.family in fams]
        loo = oracle_against_parent(split_pool.batch, without, MAIN_PARENT, parent_pred)
        only_oracle = oracle_against_parent(split_pool.batch, only, MAIN_PARENT, parent_pred)
        degradation = safe_pct(float(np.mean(loo.mse)), full_mean)
        only_gain = safe_pct(float(np.mean(only_oracle.mse)), float(np.mean(parent_mse)))
        # Best single full-family candidate as a naive deployment diagnostic.
        best_single: Optional[Dict[str, Any]] = None
        for c in only:
            row = metric_vs_parent(c.name, c.pred, split_pool.batch, MAIN_PARENT, parent_pred, 0, cfg.seed)
            if best_single is None or row["mse"] < best_single["mse"]:
                best_single = row
        share = float(full_share[[f for f in full_share.index if f in fams]].sum()) if not full_share.empty else 0.0
        loo_rows.append(
            {
                "family_group": group,
                "removed_families": ",".join(sorted(fams)),
                "full_oracle_mse": full_mean,
                "leave_one_out_mse": float(np.mean(loo.mse)),
                "leave_one_out_degradation_pct_vs_full": degradation,
                "leave_one_out_degradation_pct_vs_parent": safe_pct(float(np.mean(loo.mse)), float(np.mean(parent_mse))),
                "family_oracle_share": share,
                "candidate_count": int(len(only)),
                "test_threshold_leakage": False,
            }
        )
        only_row = metric_vs_parent(
            f"only::{group}",
            only_oracle.pred,
            split_pool.batch,
            MAIN_PARENT,
            parent_pred,
            cfg.bootstrap,
            cfg.seed,
        )
        only_row.update(
            {
                "family_group": group,
                "families": ",".join(sorted(fams)),
                "only_family_oracle_gain_pct_vs_parent": only_gain,
                "only_family_selection_rate": float(np.mean(~pd.Series(only_oracle.family).eq("parent"))),
                "best_single_candidate": best_single["variant"] if best_single else "",
                "best_single_delta_pct_vs_parent": float(best_single["mse_delta_pct_vs_parent"]) if best_single else np.nan,
                "best_single_harm_vs_parent": float(best_single["harm_rate_vs_parent"]) if best_single else np.nan,
                "best_single_max_config_harm": float(best_single["max_config_harm"]) if best_single else np.nan,
            }
        )
        only_rows.append(only_row)
    return pd.DataFrame(loo_rows), pd.DataFrame(only_rows), full, atom_pool


def slice_labels(batch: ForecastBatch, masks: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {name: np.asarray(mask, dtype=bool) for name, mask in masks.items()}
    out["overall"] = np.ones(len(batch.meta), dtype=bool)
    return out


def segment_delta(parent_pred: np.ndarray, cand_pred: np.ndarray, y: np.ndarray, h: int, segment: str) -> float:
    if h <= 0:
        return 0.0
    if segment == "early":
        s, e = 0, max(1, h // 3)
    elif segment == "mid":
        s, e = max(1, h // 3), max(2, 2 * h // 3)
    else:
        s, e = max(2, 2 * h // 3), h
    pm = np.nanmean((parent_pred[s:e] - y[s:e]) ** 2)
    cm = np.nanmean((cand_pred[s:e] - y[s:e]) ** 2)
    return float(cm - pm)


def vector_for_sample(batch: ForecastBatch, parent_pred: np.ndarray, cand_pred: np.ndarray, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    h = int(batch.meta.loc[idx, "horizon"])
    scale = float(scale_matrix(batch)[idx, 0, 0] + 1e-8)
    delta = np.zeros_like(batch.lrbn_pred[idx], dtype=float)
    residual = np.zeros_like(batch.lrbn_pred[idx], dtype=float)
    delta[:h, :] = (cand_pred[idx, :h, :] - parent_pred[idx, :h, :]) / scale
    residual[:h, :] = (batch.y_true[idx, :h, :] - parent_pred[idx, :h, :]) / scale
    return finite_arr(delta.reshape(-1)), finite_arr(residual.reshape(-1))


def residual_alignment(vec: np.ndarray, residual: np.ndarray) -> Tuple[float, float]:
    denom = float(np.sum(vec * vec)) + 1e-8
    a = 2.0 * float(np.sum(vec * residual)) / denom
    cos = float(np.sum(vec * residual) / ((np.linalg.norm(vec) * np.linalg.norm(residual)) + 1e-8))
    return a, cos


def oracle_vector_frame(
    split_name: str,
    split_pool: SplitPools,
    oracle: OracleResult,
    atom_pool: Sequence[ExpertCandidate],
    masks: Optional[Mapping[str, np.ndarray]] = None,
) -> pd.DataFrame:
    parent_pred = parent_predictions(split_pool.batch, split_pool.old)[MAIN_PARENT]
    by_name = {c.name: c for c in atom_pool}
    rows: List[Dict[str, Any]] = []
    selected_mask = ~pd.Series(oracle.family).eq("parent").to_numpy(bool)
    parent_mse = mse_per_sample(parent_pred, split_pool.batch.y_true)
    labels = slice_labels(split_pool.batch, masks or {}) if masks is not None else {"overall": np.ones(len(split_pool.batch.meta), dtype=bool)}
    for i in np.where(selected_mask)[0]:
        cname = str(oracle.candidate[i])
        cand = by_name.get(cname)
        if cand is None:
            continue
        vec, residual = vector_for_sample(split_pool.batch, parent_pred, cand.pred, int(i))
        a, cos = residual_alignment(vec, residual)
        h = int(split_pool.batch.meta.loc[i, "horizon"])
        delta = float(oracle.delta_vs_parent[i])
        scale = float(scale_matrix(split_pool.batch)[i, 0, 0] + 1e-8)
        row: Dict[str, Any] = {
            "split": split_name,
            "row_index": int(i),
            "dataset": split_pool.batch.meta.loc[i, "dataset"],
            "backbone": split_pool.batch.meta.loc[i, "backbone"],
            "horizon": h,
            "seed": int(split_pool.batch.meta.loc[i, "seed"]),
            "sample_id": split_pool.batch.meta.loc[i, "sample_id"],
            "sample_key": split_pool.batch.meta.loc[i, "sample_key"],
            "candidate": cname,
            "family": cand.family,
            "family_group": candidate_family_group(cand),
            "parent_mse": float(parent_mse[i]),
            "candidate_mse": float(oracle.mse[i]),
            "delta_mse_vs_parent": delta,
            "delta_pct_vs_parent_sample": safe_pct(float(oracle.mse[i]), float(parent_mse[i])),
            "A": a,
            "cosine_with_parent_residual": cos,
            "corr_norm": float(np.linalg.norm(vec)),
            "scale": scale,
            "early_delta_mse": segment_delta(parent_pred[i], cand.pred[i], split_pool.batch.y_true[i], h, "early"),
            "mid_delta_mse": segment_delta(parent_pred[i], cand.pred[i], split_pool.batch.y_true[i], h, "mid"),
            "late_delta_mse": segment_delta(parent_pred[i], cand.pred[i], split_pool.batch.y_true[i], h, "late"),
            "vector": vec.astype(np.float32).tolist(),
        }
        for name, mask in labels.items():
            row[f"slice::{name}"] = bool(mask[i])
        rows.append(row)
    return pd.DataFrame(rows)


def matrix_from_vectors(df: pd.DataFrame) -> np.ndarray:
    if df.empty:
        return np.zeros((0, 1), dtype=float)
    return finite_arr(np.vstack(df["vector"].map(lambda x: np.asarray(x, dtype=float)).to_list()))


def fit_atoms(vectors: pd.DataFrame, cfg: Stage18Config) -> Tuple[PCA, KMeans, pd.DataFrame, pd.DataFrame]:
    train = vectors[vectors["split"].eq("inner_train")].reset_index(drop=True)
    if train.empty:
        train = vectors.reset_index(drop=True)
    x_train = matrix_from_vectors(train)
    n_components = min(cfg.pca_components, x_train.shape[0], x_train.shape[1])
    pca = PCA(n_components=max(1, n_components), random_state=cfg.seed)
    z_train = pca.fit_transform(x_train)
    n_atoms = min(cfg.n_atoms, max(1, len(train)))
    kmeans = KMeans(n_clusters=n_atoms, random_state=cfg.seed, n_init=20)
    labels = kmeans.fit_predict(z_train[:, : min(n_atoms, z_train.shape[1])])
    pca_rows = []
    cum = 0.0
    for i, evr in enumerate(pca.explained_variance_ratio_, start=1):
        cum += float(evr)
        pca_rows.append({"component": i, "explained_variance_ratio": float(evr), "cumulative_evr": cum})
    train_shares = pd.Series(labels).value_counts(normalize=True).sort_index()
    cluster_rows = []
    for split in sorted(vectors["split"].unique()):
        split_df = vectors[vectors["split"].eq(split)].reset_index(drop=True)
        x = matrix_from_vectors(split_df)
        z = pca.transform(x)
        lab = kmeans.predict(z[:, : min(n_atoms, z.shape[1])])
        split_shares = pd.Series(lab).value_counts(normalize=True).sort_index()
        for atom in range(n_atoms):
            mask = lab == atom
            sub = split_df.loc[mask]
            family_top = sub["family_group"].value_counts(normalize=True).head(1)
            js = float(jensenshannon(
                np.asarray([train_shares.get(i, 0.0) for i in range(n_atoms)]) + 1e-12,
                np.asarray([split_shares.get(i, 0.0) for i in range(n_atoms)]) + 1e-12,
            ))
            cluster_rows.append(
                {
                    "split": split,
                    "atom_id": int(atom),
                    "n": int(mask.sum()),
                    "coverage_within_oracle_selected": float(mask.mean()) if len(mask) else 0.0,
                    "top_family_group": str(family_top.index[0]) if not family_top.empty else "",
                    "top_family_share": float(family_top.iloc[0]) if not family_top.empty else 0.0,
                    "mean_A": float(sub["A"].mean()) if not sub.empty else np.nan,
                    "A_gt1_rate": float((sub["A"] > 1.0).mean()) if not sub.empty else np.nan,
                    "mean_delta_mse_vs_parent": float(sub["delta_mse_vs_parent"].mean()) if not sub.empty else np.nan,
                    "cluster_share_js_vs_train": js,
                }
            )
        vectors.loc[vectors["split"].eq(split), "atom_id"] = lab
    return pca, kmeans, pd.DataFrame(pca_rows), pd.DataFrame(cluster_rows)


def atom_alignment_report(vectors: pd.DataFrame) -> pd.DataFrame:
    test = vectors[vectors["split"].eq("test")].copy()
    rows = []
    for atom, sub in test.groupby("atom_id", observed=True):
        d = sub["delta_mse_vs_parent"].to_numpy(float)
        wins = d < 0.0
        losses = d > 1e-12
        rows.append(
            {
                "atom_id": int(atom),
                "n": int(len(sub)),
                "coverage_within_test_oracle_selected": float(len(sub) / max(1, len(test))),
                "mean_delta_mse_vs_parent": float(np.mean(d)),
                "harm_rate_if_oracle_supported": float(np.mean(losses)),
                "A_gt1_rate": float(np.mean(sub["A"].to_numpy(float) > 1.0)),
                "mean_A": float(sub["A"].mean()),
                "mean_cosine_with_parent_residual": float(sub["cosine_with_parent_residual"].mean()),
                "mean_win_size": float((-d[wins]).mean()) if wins.any() else 0.0,
                "mean_loss_size": float(d[losses].mean()) if losses.any() else 0.0,
                "early_delta_mse": float(sub["early_delta_mse"].mean()),
                "mid_delta_mse": float(sub["mid_delta_mse"].mean()),
                "late_delta_mse": float(sub["late_delta_mse"].mean()),
                "top_family_group": str(sub["family_group"].value_counts().idxmax()) if not sub.empty else "",
            }
        )
    return pd.DataFrame(rows)


def atom_slice_profile(vectors: pd.DataFrame, test_batch: ForecastBatch, masks: Mapping[str, np.ndarray]) -> pd.DataFrame:
    test = vectors[vectors["split"].eq("test")].copy()
    rows = []
    for atom, sub_atom in test.groupby("atom_id", observed=True):
        for name, mask in slice_labels(test_batch, masks).items():
            idx_set = set(np.where(mask)[0].tolist())
            sub = sub_atom[sub_atom["row_index"].isin(idx_set)]
            if sub.empty:
                continue
            d = sub["delta_mse_vs_parent"].to_numpy(float)
            parent_mse = sub["parent_mse"].to_numpy(float)
            rows.append(
                {
                    "atom_id": int(atom),
                    "slice": name,
                    "n": int(len(sub)),
                    "slice_coverage": float(len(sub) / max(1, int(mask.sum()))),
                    "mse_delta_pct_vs_sra_balanced": safe_pct(float(np.mean(parent_mse + d)), float(np.mean(parent_mse))),
                    "mean_delta_mse_vs_sra_balanced": float(np.mean(d)),
                    "harm_rate": float(np.mean(d > 1e-12)),
                    "A_gt1_rate": float(np.mean(sub["A"].to_numpy(float) > 1.0)),
                    "mean_cosine": float(sub["cosine_with_parent_residual"].mean()),
                    "top_family_group": str(sub["family_group"].value_counts().idxmax()),
                }
            )
    return pd.DataFrame(rows)


def feature_matrices(split_pools: Mapping[str, SplitPools], schema: Mapping[str, List[Any]]) -> Dict[str, pd.DataFrame]:
    return {split: feature_frame(pool.batch, dict(schema)).reset_index(drop=True) for split, pool in split_pools.items()}


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, score))


def labels_for_atom(vectors: pd.DataFrame, split_pool: SplitPools, atom_id: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(split_pool.batch.meta)
    active = np.zeros(n, dtype=int)
    coeff = np.zeros(n, dtype=float)
    sub = vectors[(vectors["split"].eq(split_pool.batch.meta["split"].iloc[0] if "split" in split_pool.batch.meta else ""))]
    # Caller passes split-filtered vectors in practice; this fallback is not used.
    return active, coeff


def atom_targets(vectors_split: pd.DataFrame, n: int, atom_id: int, centers: Mapping[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    active = np.zeros(n, dtype=int)
    coeff = np.zeros(n, dtype=float)
    center = centers[int(atom_id)]
    denom = float(np.sum(center * center)) + 1e-8
    for _, row in vectors_split[vectors_split["atom_id"].eq(atom_id)].iterrows():
        i = int(row["row_index"])
        vec = np.asarray(row["vector"], dtype=float)
        active[i] = 1
        coeff[i] = float(np.sum(vec * center) / denom)
    return active, coeff


def predict_proba_model(model: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def apply_atom_center(batch: ForecastBatch, parent_pred: np.ndarray, center: np.ndarray, selected: np.ndarray, shrink: float) -> np.ndarray:
    out = np.asarray(parent_pred, dtype=float).copy()
    hmax = out.shape[1]
    channels = out.shape[2]
    center_seq = center.reshape(hmax, channels)
    scale = scale_matrix(batch)
    for i, h in enumerate(horizons(batch)):
        if not selected[i]:
            continue
        out[i, : int(h), :] = parent_pred[i, : int(h), :] + float(shrink) * center_seq[: int(h), :] * scale[i]
    return out


def distillability_and_prototypes(
    vectors: pd.DataFrame,
    split_pools: Mapping[str, SplitPools],
    schema: Mapping[str, List[Any]],
    cfg: Stage18Config,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feats = feature_matrices(split_pools, schema)
    atom_ids = sorted([int(x) for x in vectors["atom_id"].dropna().unique()])
    train_vectors = vectors[vectors["split"].eq("inner_train")]
    centers: Dict[int, np.ndarray] = {}
    for atom in atom_ids:
        sub = train_vectors[train_vectors["atom_id"].eq(atom)]
        if sub.empty:
            sub = vectors[vectors["atom_id"].eq(atom)]
        centers[atom] = matrix_from_vectors(sub).mean(axis=0)
    distill_rows: List[Dict[str, Any]] = []
    proto_rows: List[Dict[str, Any]] = []
    train_pool = split_pools["inner_train"]
    calib_pool = split_pools["inner_calib"]
    test_pool = split_pools["test"]
    train_parent = parent_predictions(train_pool.batch, train_pool.old)[MAIN_PARENT]
    calib_parent = parent_predictions(calib_pool.batch, calib_pool.old)[MAIN_PARENT]
    test_parent = parent_predictions(test_pool.batch, test_pool.old)[MAIN_PARENT]
    for atom in atom_ids:
        y_train, coeff_train = atom_targets(vectors[vectors["split"].eq("inner_train")], len(train_pool.batch.meta), atom, centers)
        y_calib, coeff_calib = atom_targets(vectors[vectors["split"].eq("inner_calib")], len(calib_pool.batch.meta), atom, centers)
        y_test, coeff_test = atom_targets(vectors[vectors["split"].eq("test")], len(test_pool.batch.meta), atom, centers)
        if len(np.unique(y_train)) < 2:
            distill_rows.append({"atom_id": atom, "status": "skipped_single_class_train"})
            continue
        x_train = feats["inner_train"]
        x_calib = feats["inner_calib"].reindex(columns=x_train.columns, fill_value=0.0)
        x_test = feats["test"].reindex(columns=x_train.columns, fill_value=0.0)
        models = {
            "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg.seed)),
            "random_forest": RandomForestClassifier(
                n_estimators=240,
                max_depth=7,
                min_samples_leaf=10,
                class_weight="balanced_subsample",
                random_state=cfg.seed + atom,
                n_jobs=1,
            ),
        }
        model_scores: Dict[str, Dict[str, float]] = {}
        fitted: Dict[str, Any] = {}
        for name, model in models.items():
            model.fit(x_train, y_train)
            fitted[name] = model
            p_calib = predict_proba_model(model, x_calib)
            p_test = predict_proba_model(model, x_test)
            model_scores[name] = {
                "calib_auroc": safe_auc(y_calib, p_calib),
                "test_auroc": safe_auc(y_test, p_test),
                "test_pr_auc": safe_ap(y_test, p_test),
            }
        choose_name = max(model_scores, key=lambda k: np.nan_to_num(model_scores[k]["calib_auroc"], nan=-1.0))
        chosen = fitted[choose_name]
        p_calib = predict_proba_model(chosen, x_calib)
        p_test = predict_proba_model(chosen, x_test)
        sign_train = (coeff_train > 0).astype(int)
        sign_test = (coeff_test > 0).astype(int)
        sign_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg.seed + 700 + atom))
        sign_acc = float("nan")
        if len(np.unique(sign_train)) >= 2:
            sign_model.fit(x_train, sign_train)
            sign_acc = float(np.mean(sign_model.predict(x_test) == sign_test))
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        reg.fit(x_train, coeff_train)
        pred_coeff = np.asarray(reg.predict(x_test), dtype=float)
        r2 = float(r2_score(coeff_test, pred_coeff)) if np.nanstd(coeff_test) > 1e-12 else float("nan")
        distill_rows.append(
            {
                "atom_id": atom,
                "status": "completed",
                "chosen_model": choose_name,
                "activation_train_rate": float(np.mean(y_train)),
                "activation_calib_rate": float(np.mean(y_calib)),
                "activation_test_rate": float(np.mean(y_test)),
                "activation_auroc_calib": model_scores[choose_name]["calib_auroc"],
                "activation_auroc_test": model_scores[choose_name]["test_auroc"],
                "activation_pr_auc_test": model_scores[choose_name]["test_pr_auc"],
                "coefficient_sign_accuracy_test": sign_acc,
                "coefficient_r2_test": r2,
                "test_threshold_leakage": False,
            }
        )
        # Diagnostic prototype: validation-only threshold/shrink over atom center.
        best_score = float("inf")
        best_policy: Dict[str, Any] = {}
        for q in cfg.prototype_threshold_quantiles:
            tau = float(np.nanquantile(p_calib, q))
            selected_calib = p_calib >= tau
            for shrink in cfg.prototype_shrinks:
                pred_calib = apply_atom_center(calib_pool.batch, calib_parent, centers[atom], selected_calib, shrink)
                row = metric_vs_parent(f"atom_{atom}_prototype", pred_calib, calib_pool.batch, MAIN_PARENT, calib_parent, 0, cfg.seed, selected_calib)
                score = float(row["mse_delta_pct_vs_parent"])
                score += 150.0 * max(0.0, float(row["harm_rate_vs_parent"]) - cfg.max_harm_for_proto)
                score += 100.0 * max(0.0, float(row["max_config_harm"]) - cfg.max_config_harm_for_proto)
                score += 5.0 * max(0.0, 0.02 - float(row["coverage"]))
                if score < best_score:
                    best_score = score
                    best_policy = {"tau": tau, "threshold_quantile": q, "shrink": shrink, "calib_score": score}
        selected_test = p_test >= float(best_policy["tau"])
        pred_test = apply_atom_center(test_pool.batch, test_parent, centers[atom], selected_test, float(best_policy["shrink"]))
        row = metric_vs_parent(
            f"atom_{atom}_prototype",
            pred_test,
            test_pool.batch,
            MAIN_PARENT,
            test_parent,
            cfg.bootstrap,
            cfg.seed + atom,
            selected_test,
        )
        row.update({"atom_id": atom, **best_policy, "status": "diagnostic_only", "test_threshold_leakage": False})
        proto_rows.append(row)
    return pd.DataFrame(distill_rows), pd.DataFrame(proto_rows)


def family_slice_oracle(
    split_pool: SplitPools,
    family_only: pd.DataFrame,
    atom_pool: Sequence[ExpertCandidate],
    masks: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    parent_pred = parent_predictions(split_pool.batch, split_pool.old)[MAIN_PARENT]
    rows = []
    for group in sorted({candidate_family_group(c) for c in atom_pool}):
        fams = FAMILY_GROUPS.get(group, {group})
        cands = [c for c in atom_pool if c.family in fams]
        oracle = oracle_against_parent(split_pool.batch, cands, MAIN_PARENT, parent_pred)
        for name, mask in slice_labels(split_pool.batch, masks).items():
            if not mask.any():
                continue
            parent_mse = mse_per_sample(parent_pred, split_pool.batch.y_true)[mask]
            oracle_mse = oracle.mse[mask]
            rows.append(
                {
                    "family_group": group,
                    "slice": name,
                    "n": int(mask.sum()),
                    "only_family_oracle_delta_pct_vs_sra_balanced": safe_pct(float(np.mean(oracle_mse)), float(np.mean(parent_mse))),
                    "selection_rate": float(np.mean(~pd.Series(oracle.family[mask]).eq("parent"))),
                    "test_threshold_leakage": False,
                }
            )
    return pd.DataFrame(rows)


def build_summary(output_dir: Path, verdict: Mapping[str, Any], parent_oracle: pd.DataFrame, loo: pd.DataFrame, pca: pd.DataFrame, align: pd.DataFrame, distill: pd.DataFrame, proto: pd.DataFrame) -> str:
    show_oracle = parent_oracle[
        parent_oracle["split"].eq("test") & parent_oracle["parent_method"].eq(MAIN_PARENT)
    ][
        [
            "oracle_pool",
            "incremental_oracle_delta_pct",
            "oracle_gain_fraction_vs_LRBN_oracle",
            "non_parent_selection_rate",
            "ci95_high_delta_raw",
        ]
    ]
    return "\n".join(
        [
            "# Stage18 Performance Atom Extraction Diagnosis",
            "",
            f"Status: `{verdict['status']}`.",
            "",
            "## Oracle over SRA-BP-balanced",
            "",
            df_to_md(show_oracle, max_rows=16),
            "",
            "## Family Leave-One-Out",
            "",
            df_to_md(loo.sort_values("leave_one_out_degradation_pct_vs_full", ascending=False), max_rows=16),
            "",
            "## PCA",
            "",
            df_to_md(pca.head(8), max_rows=8),
            "",
            "## Atom Alignment",
            "",
            df_to_md(align, max_rows=16),
            "",
            "## Distillability",
            "",
            df_to_md(distill, max_rows=16),
            "",
            "## Prototype Diagnostics",
            "",
            df_to_md(proto.sort_values("mse_delta_pct_vs_parent") if not proto.empty else proto, max_rows=16),
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
        "stage18_config.json",
        "parent_oracle_table.csv",
        "family_leave_one_out.csv",
        "only_family_oracle.csv",
        "oracle_selected_candidates.csv",
        "correction_vectors.parquet",
        "atom_pca_report.csv",
        "atom_cluster_report.csv",
        "atom_alignment_report.csv",
        "atom_slice_profile.csv",
        "sra_complementarity_matrix.csv",
        "atom_distillability_report.csv",
        "prototype_atom_metrics.csv",
        "bootstrap_ci.json",
        "stage18_verdict.json",
        "summary.md",
    ]
    rows = []
    for name in required:
        p = output_dir / name
        rows.append({"artifact": name, "exists": bool(p.exists()), "bytes": int(p.stat().st_size) if p.exists() else 0})
    return pd.DataFrame(rows)


def build_all_artifacts(
    metrics_csv: Path,
    stage5_dir: Path,
    output_dir: Path,
    stage3_dir: Optional[Path] = None,
    cfg: Optional[Stage18Config] = None,
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    start = time.time()
    cfg = cfg or Stage18Config(bootstrap=n_bootstrap)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[stage18-atoms] preparing assets", flush=True)
    assets = prepare_assets(metrics_csv, stage5_dir, stage3_dir, cfg.seed)
    pools = build_cga_pools(assets)
    split_pools: Dict[str, SplitPools] = {
        "inner_train": SplitPools(assets.val_train, assets.old_train_candidates, pools.train_candidates),
        "inner_calib": SplitPools(assets.val_calib, assets.old_calib_candidates, pools.calib_candidates),
        "test": SplitPools(assets.test, assets.old_test_candidates, pools.test_candidates),
    }
    write_json(
        output_dir / "stage18_config.json",
        {
            "stage": "stage18_performance_atom_diagnosis",
            "source_plan": "halluguard_stage18_performance_atom_diagnosis_validation_doc.md",
            "metrics_csv": metrics_csv,
            "stage5_dir": stage5_dir,
            "stage3_dir": stage3_dir,
            "config": asdict(cfg),
            "compact_protocol": "ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192 x seed 2026",
            "main_parent": MAIN_PARENT,
            "calibration": "distillation/prototype thresholds fit on validation inner-train and inner-calib only",
            "test_threshold_leakage": False,
        },
    )
    print("[stage18-atoms] running parent oracle diagnostics", flush=True)
    parent_rows: List[pd.DataFrame] = []
    oracle_results_by_split: Dict[str, Dict[Tuple[str, str], OracleResult]] = {}
    for split, sp in split_pools.items():
        df, results = parent_oracle_table(sp, cfg, split)
        parent_rows.append(df)
        oracle_results_by_split[split] = results
    parent_oracle = pd.concat(parent_rows, ignore_index=True)
    parent_oracle.to_csv(output_dir / "parent_oracle_table.csv", index=False)

    print("[stage18-atoms] running family leave-one-out diagnostics", flush=True)
    family_loo, only_family, test_full_oracle, test_atom_pool = family_tables(split_pools["test"], cfg)
    family_loo.to_csv(output_dir / "family_leave_one_out.csv", index=False)
    only_family.to_csv(output_dir / "only_family_oracle.csv", index=False)

    masks = stage8_slice_masks(assets.test, assets.schema, stage8_slice_thresholds(assets.val, assets.schema))
    family_slice = family_slice_oracle(split_pools["test"], only_family, test_atom_pool, masks)

    print("[stage18-atoms] extracting oracle correction vectors", flush=True)
    vector_frames: List[pd.DataFrame] = []
    oracle_selected_rows: List[pd.DataFrame] = []
    atom_pool_by_split: Dict[str, List[ExpertCandidate]] = {}
    for split, sp in split_pools.items():
        atom_pool = candidate_pools(sp.old, sp.cga)["sra_complement_atom_pool"]
        atom_pool_by_split[split] = atom_pool
        oracle = oracle_against_parent(sp.batch, atom_pool, MAIN_PARENT, parent_predictions(sp.batch, sp.old)[MAIN_PARENT])
        split_masks = masks if split == "test" else None
        vf = oracle_vector_frame(split, sp, oracle, atom_pool, split_masks)
        vector_frames.append(vf)
        meta = sp.batch.meta.reset_index(drop=True).copy()
        meta["split_eval"] = split
        meta["selected_candidate"] = oracle.candidate
        meta["selected_family"] = oracle.family
        meta["delta_mse_vs_sra_balanced"] = oracle.delta_vs_parent
        meta["selected_non_parent"] = ~pd.Series(oracle.family).eq("parent").to_numpy(bool)
        oracle_selected_rows.append(meta)
    vectors = pd.concat(vector_frames, ignore_index=True)
    selected = pd.concat(oracle_selected_rows, ignore_index=True)

    print("[stage18-atoms] fitting atom PCA/KMeans", flush=True)
    pca, kmeans, pca_report, cluster_report = fit_atoms(vectors, cfg)
    vectors.to_parquet(output_dir / "correction_vectors.parquet", index=False)
    selected.to_csv(output_dir / "oracle_selected_candidates.csv", index=False)
    pca_report.to_csv(output_dir / "atom_pca_report.csv", index=False)
    cluster_report.to_csv(output_dir / "atom_cluster_report.csv", index=False)

    print("[stage18-atoms] profiling atoms and slices", flush=True)
    alignment = atom_alignment_report(vectors)
    slice_profile = atom_slice_profile(vectors, assets.test, masks)
    complement = pd.concat([slice_profile.assign(profile_type="atom"), family_slice.assign(profile_type="family")], ignore_index=True, sort=False)
    alignment.to_csv(output_dir / "atom_alignment_report.csv", index=False)
    slice_profile.to_csv(output_dir / "atom_slice_profile.csv", index=False)
    complement.to_csv(output_dir / "sra_complementarity_matrix.csv", index=False)

    print("[stage18-atoms] running distillability/prototype diagnostics", flush=True)
    distill, proto = distillability_and_prototypes(vectors, split_pools, assets.schema, cfg)
    distill.to_csv(output_dir / "atom_distillability_report.csv", index=False)
    proto.to_csv(output_dir / "prototype_atom_metrics.csv", index=False)
    boot = {
        str(row["oracle_pool"]): {
            "parent": str(row["parent_method"]),
            "ci95_low_delta_raw": float(row.get("ci95_low_delta_raw", np.nan)),
            "ci95_high_delta_raw": float(row.get("ci95_high_delta_raw", np.nan)),
            "p_bootstrap_delta_lt_zero": float(row.get("p_bootstrap_delta_lt_zero", np.nan)),
        }
        for _, row in parent_oracle[parent_oracle["split"].eq("test") & parent_oracle["parent_method"].eq(MAIN_PARENT)].iterrows()
    }
    for _, row in proto.iterrows():
        boot[f"prototype_atom_{int(row['atom_id'])}"] = {
            "parent": MAIN_PARENT,
            "ci95_low_delta_raw": float(row.get("ci95_low_delta_raw", np.nan)),
            "ci95_high_delta_raw": float(row.get("ci95_high_delta_raw", np.nan)),
            "p_bootstrap_delta_lt_zero": float(row.get("p_bootstrap_delta_lt_zero", np.nan)),
        }
    write_json(output_dir / "bootstrap_ci.json", boot)

    # Decision logic.
    test_oracle = parent_oracle[parent_oracle["split"].eq("test") & parent_oracle["parent_method"].eq(MAIN_PARENT)]
    union_row = test_oracle[test_oracle["oracle_pool"].eq("sra_complement_atom_pool")]
    union_gain = float(union_row["incremental_oracle_delta_pct"].iloc[0]) if not union_row.empty else float("nan")
    full_union_row = test_oracle[test_oracle["oracle_pool"].eq("union_full_pool")]
    full_union_gain = float(full_union_row["incremental_oracle_delta_pct"].iloc[0]) if not full_union_row.empty else float("nan")
    max_loo = float(family_loo["leave_one_out_degradation_pct_vs_full"].max()) if not family_loo.empty else 0.0
    top5_evr = float(pca_report[pca_report["component"].le(5)]["explained_variance_ratio"].sum()) if not pca_report.empty else 0.0
    max_atom_a = float(alignment["A_gt1_rate"].max()) if not alignment.empty else 0.0
    best_proto = proto.sort_values("mse_delta_pct_vs_parent").head(1).to_dict("records")
    best_proto_row = best_proto[0] if best_proto else {}
    prototype_pass = bool(
        best_proto_row
        and float(best_proto_row["mse_delta_pct_vs_parent"]) <= -0.8
        and float(best_proto_row["harm_rate_vs_parent"]) <= cfg.max_harm_for_proto
        and float(best_proto_row["max_config_harm"]) <= cfg.max_config_harm_for_proto
        and float(best_proto_row.get("ci95_high_delta_raw", 1.0)) < 0.0
    )
    atom_route_pass = bool(full_union_gain <= -5.0 or union_gain <= -5.0 or max_loo >= 2.0 or top5_evr >= 0.60 or max_atom_a >= 0.60)
    if prototype_pass:
        status = "prototype_atom_pass_ready_for_mini_extension"
        recommendation = "Promote the best distilled atom prototype to mini-extension as SRA-BP complement."
    elif atom_route_pass:
        status = "atom_route_mechanism_pass_distillation_not_ready"
        recommendation = "Keep atom extraction as a promising mechanism diagnosis, but do not deploy until distillation/prototype safety improves."
    else:
        status = "oracle_not_distillable_stop_atom_route"
        recommendation = "Stop atom route and fall back to SRA-BP/LRBN main line."
    verdict = {
        "stage": "stage18_performance_atom_diagnosis",
        "status": status,
        "atom_route_pass": atom_route_pass,
        "prototype_pass": prototype_pass,
        "union_full_oracle_gain_pct_vs_sra_balanced": full_union_gain,
        "sra_complement_atom_pool_oracle_gain_pct_vs_sra_balanced": union_gain,
        "max_leave_one_out_degradation_pct_vs_full": max_loo,
        "top5_atom_explained_variance_ratio": top5_evr,
        "max_atom_A_gt1_rate": max_atom_a,
        "best_prototype": best_proto_row,
        "recommendation": recommendation,
        "test_threshold_leakage": False,
        "runtime_seconds": float(time.time() - start),
    }
    write_json(output_dir / "stage18_verdict.json", verdict)
    (output_dir / "summary.md").write_text(build_summary(output_dir, verdict, parent_oracle, family_loo, pca_report, alignment, distill, proto), encoding="utf-8")
    completeness = verify_outputs(output_dir)
    completeness.to_csv(output_dir / "stage18_output_completeness.csv", index=False)
    print(f"[stage18-atoms] done in {time.time() - start:.1f}s", flush=True)
    return {
        "verdict": verdict,
        "parent_oracle": parent_oracle,
        "family_leave_one_out": family_loo,
        "only_family_oracle": only_family,
        "vectors": vectors,
        "pca": pca_report,
        "clusters": cluster_report,
        "alignment": alignment,
        "slice_profile": slice_profile,
        "distillability": distill,
        "prototype": proto,
        "completeness": completeness,
    }
