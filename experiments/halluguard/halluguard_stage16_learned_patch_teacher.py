#!/usr/bin/env python
"""Stage 16 learned patch representation and teacher manifold compact validation."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from halluguard_lrbn_bp import ForecastBatch, mae_per_sample, mse_per_sample
from halluguard_stage6_mechanism import horizons, safe_pct
from halluguard_stage7_safe_tae import write_json
from halluguard_stage8_safe_tae_pareto import stage8_slice_masks, stage8_slice_thresholds
from halluguard_stage9_architecture_validation import (
    deployable_candidates,
    metric_row,
    oracle_best,
    per_config_rows,
    prepare_assets,
    slice_rows,
)
from halluguard_stage10_cga import build_cga_pools, df_to_md, json_default
from halluguard_stage15_endogenous_editors import nontriviality_metrics, scale_matrix


STAGE16_VARIANTS = [
    "H2L Learned Patch Residual Editor",
    "H6 Denoising Teacher Manifold Projector",
    "H2H6 Learned Patch Teacher Hybrid",
    "H6-Safe Sparse Teacher Projector",
    "H2H6-Safe Sparse Learned Teacher Hybrid",
]


@dataclass(frozen=True)
class Stage16Config:
    patch_len: int = 16
    stride: int = 8
    latent_dim: int = 24
    hidden_dim: int = 96
    epochs: int = 120
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.05
    seed: int = 2026
    device: str = "auto"


@dataclass(frozen=True)
class PatchPolicy:
    variant: str
    shrink: float = 0.50
    residual_cap: float = 0.25
    gate_quantile: float = 0.50
    score_threshold: float = 0.0
    mix_residual: float = 0.50


@dataclass
class PatchPack:
    features: np.ndarray
    residual_target: np.ndarray
    true_patch: np.ndarray
    pred_patch: np.ndarray
    row_index: np.ndarray
    start: np.ndarray
    horizon: np.ndarray
    score: np.ndarray
    scale: np.ndarray


@dataclass
class PatchPrediction:
    residual_delta: np.ndarray
    teacher_delta: np.ndarray
    row_index: np.ndarray
    start: np.ndarray
    score_residual: np.ndarray
    score_teacher: np.ndarray


class ResidualPatchMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenoisingPatchTeacher(nn.Module):
    def __init__(self, patch_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, patch_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def patch_starts(horizon: int, patch_len: int, stride: int) -> List[int]:
    h = int(horizon)
    p = int(patch_len)
    s = max(1, int(stride))
    if h <= p:
        return [0]
    starts = list(range(0, h - p + 1, s))
    if starts[-1] != h - p:
        starts.append(h - p)
    return starts


def temporal_boundary_score(batch: ForecastBatch) -> np.ndarray:
    p = np.asarray(batch.lrbn_pred, dtype=float)
    c = np.asarray(batch.context, dtype=float)
    score = np.zeros_like(p)
    score[:, 0, :] = np.abs(p[:, 0, :] - c[:, -1, :])
    if p.shape[1] > 1:
        score[:, 1:, :] += np.abs(np.diff(p, axis=1))
    if p.shape[1] > 2:
        score[:, 2:, :] += np.abs(np.diff(p, n=2, axis=1))
    return score / (scale_matrix(batch) + 1e-8)


def patch_feature_pack(batch: ForecastBatch, cfg: Stage16Config) -> PatchPack:
    c = int(batch.lrbn_pred.shape[2])
    ctx_tail = batch.context[:, -cfg.patch_len :, :]
    if ctx_tail.shape[1] < cfg.patch_len:
        pad = cfg.patch_len - ctx_tail.shape[1]
        ctx_tail = np.pad(ctx_tail, ((0, 0), (pad, 0), (0, 0)), mode="edge")
    scale = scale_matrix(batch).reshape(-1)
    boundary = temporal_boundary_score(batch)
    hvec = horizons(batch)
    features: List[np.ndarray] = []
    residual_target: List[np.ndarray] = []
    true_patch: List[np.ndarray] = []
    pred_patch: List[np.ndarray] = []
    row_index: List[int] = []
    starts: List[int] = []
    hs: List[int] = []
    scores: List[float] = []
    scales: List[float] = []
    residual = batch.y_true - batch.lrbn_pred
    for i, h in enumerate(hvec):
        h = int(h)
        for start in patch_starts(h, cfg.patch_len, cfg.stride):
            end = min(h, start + cfg.patch_len)
            if end - start != cfg.patch_len:
                continue
            s = float(scale[i] + 1e-8)
            pred = batch.lrbn_pred[i, start:end, :] / s
            ctx = ctx_tail[i] / s
            diff = pred - ctx
            bscore = float(np.nanmean(boundary[i, start:end, :]))
            pos = float(start / max(1, h - cfg.patch_len))
            hnorm = float(h / 720.0)
            stats = np.array(
                [
                    pos,
                    hnorm,
                    float(np.nanmean(pred)),
                    float(np.nanstd(pred)),
                    float(np.nanmean(ctx)),
                    float(np.nanstd(ctx)),
                    float(np.nanmean(diff)),
                    float(np.nanstd(diff)),
                    bscore,
                ],
                dtype=float,
            )
            feat = np.concatenate([pred.reshape(-1), ctx.reshape(-1), diff.reshape(-1), stats])
            features.append(feat)
            residual_target.append((residual[i, start:end, :] / s).reshape(-1))
            true_patch.append((batch.y_true[i, start:end, :] / s).reshape(-1))
            pred_patch.append(pred.reshape(-1))
            row_index.append(i)
            starts.append(start)
            hs.append(h)
            scores.append(float(bscore + 0.25 * np.sqrt(np.nanmean(diff**2))))
            scales.append(s)
    return PatchPack(
        features=np.nan_to_num(np.vstack(features), nan=0.0, posinf=0.0, neginf=0.0),
        residual_target=np.nan_to_num(np.vstack(residual_target), nan=0.0, posinf=0.0, neginf=0.0),
        true_patch=np.nan_to_num(np.vstack(true_patch), nan=0.0, posinf=0.0, neginf=0.0),
        pred_patch=np.nan_to_num(np.vstack(pred_patch), nan=0.0, posinf=0.0, neginf=0.0),
        row_index=np.asarray(row_index, dtype=int),
        start=np.asarray(starts, dtype=int),
        horizon=np.asarray(hs, dtype=int),
        score=np.asarray(scores, dtype=float),
        scale=np.asarray(scales, dtype=float),
    )


def standardize(train_x: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    train_z = np.nan_to_num((train_x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    other_z = [np.nan_to_num((x - mean) / std, nan=0.0, posinf=0.0, neginf=0.0) for x in others]
    return train_z, other_z, mean, std


def train_residual_mlp(train: PatchPack, cfg: Stage16Config, device: torch.device) -> Tuple[ResidualPatchMLP, Dict[str, Any]]:
    set_seed(cfg.seed)
    model = ResidualPatchMLP(
        input_dim=train.features.shape[1],
        output_dim=train.residual_target.shape[1],
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
    ).to(device)
    ds = TensorDataset(
        torch.tensor(train.features, dtype=torch.float32),
        torch.tensor(train.residual_target, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.25)
    history: List[Dict[str, float]] = []
    model.train()
    for epoch in range(cfg.epochs):
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "residual_loss": float(np.mean(losses))})
    return model.eval(), {"residual_train_loss": history[-1]["residual_loss"], "residual_epochs": cfg.epochs, "history": history}


def train_teacher(train: PatchPack, cfg: Stage16Config, device: torch.device) -> Tuple[DenoisingPatchTeacher, Dict[str, Any]]:
    set_seed(cfg.seed + 17)
    model = DenoisingPatchTeacher(
        patch_dim=train.true_patch.shape[1],
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        dropout=cfg.dropout,
    ).to(device)
    ds = TensorDataset(torch.tensor(train.true_patch, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=0.25)
    history: List[Dict[str, float]] = []
    model.train()
    for epoch in range(cfg.epochs):
        losses = []
        for (yb,) in loader:
            yb = yb.to(device)
            noise = 0.05 * torch.randn_like(yb)
            mask = (torch.rand_like(yb) > 0.10).float()
            xb = yb * mask + noise
            opt.zero_grad(set_to_none=True)
            recon = model(xb)
            loss = loss_fn(recon, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": float(epoch + 1), "teacher_loss": float(np.mean(losses))})
    return model.eval(), {"teacher_train_loss": history[-1]["teacher_loss"], "teacher_epochs": cfg.epochs, "history": history}


@torch.no_grad()
def predict_patches(
    pack: PatchPack,
    residual_model: ResidualPatchMLP,
    teacher: DenoisingPatchTeacher,
    device: torch.device,
    batch_size: int,
) -> PatchPrediction:
    residual_delta: List[np.ndarray] = []
    teacher_delta: List[np.ndarray] = []
    score_residual: List[np.ndarray] = []
    score_teacher: List[np.ndarray] = []
    residual_model.eval()
    teacher.eval()
    for start in range(0, len(pack.features), batch_size):
        end = min(len(pack.features), start + batch_size)
        xb = torch.tensor(pack.features[start:end], dtype=torch.float32, device=device)
        pb = torch.tensor(pack.pred_patch[start:end], dtype=torch.float32, device=device)
        rd = residual_model(xb).detach().cpu().numpy()
        recon = teacher(pb).detach().cpu().numpy()
        td = recon - pack.pred_patch[start:end]
        residual_delta.append(rd)
        teacher_delta.append(td)
        score_residual.append(np.sqrt(np.nanmean(rd**2, axis=1)))
        score_teacher.append(np.sqrt(np.nanmean(td**2, axis=1)))
    return PatchPrediction(
        residual_delta=np.nan_to_num(np.vstack(residual_delta), nan=0.0, posinf=0.0, neginf=0.0),
        teacher_delta=np.nan_to_num(np.vstack(teacher_delta), nan=0.0, posinf=0.0, neginf=0.0),
        row_index=pack.row_index,
        start=pack.start,
        score_residual=np.nan_to_num(np.concatenate(score_residual), nan=0.0, posinf=0.0, neginf=0.0),
        score_teacher=np.nan_to_num(np.concatenate(score_teacher), nan=0.0, posinf=0.0, neginf=0.0),
    )


def policy_grid(variant: str) -> Iterable[PatchPolicy]:
    if variant == "H2L Learned Patch Residual Editor":
        for shrink in [0.25, 0.50, 0.75]:
            for cap in [0.15, 0.25, 0.35]:
                for q in [0.00, 0.50, 0.75, 0.90]:
                    yield PatchPolicy(variant=variant, shrink=shrink, residual_cap=cap, gate_quantile=q, mix_residual=1.0)
    elif variant == "H6 Denoising Teacher Manifold Projector":
        for shrink in [0.10, 0.25, 0.50]:
            for cap in [0.10, 0.20, 0.35]:
                for q in [0.00, 0.50, 0.75, 0.90]:
                    yield PatchPolicy(variant=variant, shrink=shrink, residual_cap=cap, gate_quantile=q, mix_residual=0.0)
    elif variant == "H2H6 Learned Patch Teacher Hybrid":
        for mix in [0.25, 0.50, 0.75]:
            for shrink in [0.25, 0.50]:
                for cap in [0.15, 0.25, 0.35]:
                    for q in [0.00, 0.50, 0.75]:
                        yield PatchPolicy(variant=variant, shrink=shrink, residual_cap=cap, gate_quantile=q, mix_residual=mix)
    elif variant == "H6-Safe Sparse Teacher Projector":
        for shrink in [0.05, 0.10, 0.15, 0.25]:
            for cap in [0.05, 0.10, 0.15, 0.20]:
                for q in [0.75, 0.90, 0.95, 0.98]:
                    yield PatchPolicy(variant=variant, shrink=shrink, residual_cap=cap, gate_quantile=q, mix_residual=0.0)
    elif variant == "H2H6-Safe Sparse Learned Teacher Hybrid":
        for mix in [0.25, 0.50]:
            for shrink in [0.10, 0.15, 0.25]:
                for cap in [0.05, 0.10, 0.15, 0.20]:
                    for q in [0.75, 0.90, 0.95, 0.98]:
                        yield PatchPolicy(variant=variant, shrink=shrink, residual_cap=cap, gate_quantile=q, mix_residual=mix)
    else:
        raise ValueError(f"unknown variant {variant}")


def variant_patch_delta(variant: str, pred: PatchPrediction, policy: PatchPolicy) -> Tuple[np.ndarray, np.ndarray]:
    if variant == "H2L Learned Patch Residual Editor":
        return pred.residual_delta, pred.score_residual
    if variant == "H6 Denoising Teacher Manifold Projector":
        return pred.teacher_delta, pred.score_teacher
    if variant == "H2H6 Learned Patch Teacher Hybrid":
        mix = float(policy.mix_residual)
        delta = mix * pred.residual_delta + (1.0 - mix) * pred.teacher_delta
        score = mix * pred.score_residual + (1.0 - mix) * pred.score_teacher
        return delta, score
    if variant == "H6-Safe Sparse Teacher Projector":
        return pred.teacher_delta, pred.score_teacher
    if variant == "H2H6-Safe Sparse Learned Teacher Hybrid":
        mix = float(policy.mix_residual)
        delta = mix * pred.residual_delta + (1.0 - mix) * pred.teacher_delta
        score = mix * pred.score_residual + (1.0 - mix) * pred.score_teacher
        return delta, score
    raise ValueError(f"unknown variant {variant}")


def clip_delta(delta: np.ndarray, batch: ForecastBatch, cap: float) -> np.ndarray:
    return np.clip(delta, -float(cap) * scale_matrix(batch), float(cap) * scale_matrix(batch))


def apply_patch_prediction(
    variant: str,
    batch: ForecastBatch,
    patch_pred: PatchPrediction,
    cfg: Stage16Config,
    policy: PatchPolicy,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, pd.DataFrame, PatchPolicy]:
    patch_delta_norm, scores = variant_patch_delta(variant, patch_pred, policy)
    if threshold is None:
        threshold = float(np.nanquantile(scores, float(policy.gate_quantile))) if len(scores) else float("inf")
    policy = replace(policy, score_threshold=threshold)
    delta_sum = np.zeros_like(batch.lrbn_pred, dtype=float)
    weight_sum = np.zeros_like(batch.lrbn_pred, dtype=float)
    selected = np.zeros(len(batch.meta), dtype=bool)
    accept_scores = np.zeros(len(batch.meta), dtype=float)
    scale = scale_matrix(batch).reshape(-1)
    c = int(batch.lrbn_pred.shape[2])
    for j in range(len(scores)):
        if scores[j] < threshold:
            continue
        i = int(patch_pred.row_index[j])
        start = int(patch_pred.start[j])
        end = start + cfg.patch_len
        patch = patch_delta_norm[j].reshape(cfg.patch_len, c) * (scale[i] + 1e-8)
        delta_sum[i, start:end, :] += float(policy.shrink) * patch
        weight_sum[i, start:end, :] += 1.0
        selected[i] = True
        accept_scores[i] = max(accept_scores[i], float(scores[j]))
    delta = np.divide(delta_sum, np.maximum(weight_sum, 1.0))
    delta = clip_delta(delta, batch, policy.residual_cap)
    decisions = pd.DataFrame(
        {
            "row_index": np.arange(len(batch.meta)),
            "selected": selected,
            "selected_action": np.where(selected, variant, "keep_lrbn"),
            "accept_score": accept_scores,
        }
    )
    return batch.lrbn_pred + delta, decisions, policy


def calibration_score(row: Mapping[str, Any]) -> float:
    score = float(row["mse_delta_pct_vs_lrbn"])
    score += 180.0 * max(0.0, float(row["harm_rate"]) - 0.10)
    score += 150.0 * max(0.0, float(row["max_config_harm"]) - 0.18)
    score += 40.0 * max(0.0, float(row["lrbn_equiv_rate"]) - 0.80)
    score += 25.0 * max(0.0, 0.08 - float(row["active_patch_ratio"]))
    score += 30.0 * max(0.0, 0.08 - float(row.get("oracle_gain_fraction", 0.0)))
    return float(score)


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
    row.update(nontriviality_metrics(pred, batch))
    row["mean_accept_score"] = float(decisions["accept_score"].mean()) if decisions is not None else float("nan")
    return row


def calibrate_variant(
    variant: str,
    calib: ForecastBatch,
    calib_pred: PatchPrediction,
    cfg: Stage16Config,
    oracle_mse: np.ndarray,
    seed: int,
) -> Tuple[PatchPolicy, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best_policy: Optional[PatchPolicy] = None
    best_score = float("inf")
    for policy0 in policy_grid(variant):
        base_delta, scores = variant_patch_delta(variant, calib_pred, policy0)
        threshold = float(np.nanquantile(scores, float(policy0.gate_quantile))) if len(scores) else float("inf")
        pred, decisions, policy = apply_patch_prediction(variant, calib, calib_pred, cfg, policy0, threshold=threshold)
        row = evaluate_variant(variant, pred, calib, decisions, oracle_mse, n_bootstrap=0, seed=seed)
        row.update(asdict(policy))
        row["patch_delta_rms"] = float(np.sqrt(np.nanmean(base_delta**2)))
        row["calibration_score"] = calibration_score(row)
        rows.append(row)
        if float(row["calibration_score"]) < best_score:
            best_score = float(row["calibration_score"])
            best_policy = policy
    assert best_policy is not None
    return best_policy, pd.DataFrame(rows)


def known_harmed_delta(per_config: pd.DataFrame, variant: str) -> float:
    row = per_config[
        per_config["variant"].eq(variant)
        & per_config["dataset"].eq("ETTm1")
        & per_config["backbone"].eq("DLinear")
        & per_config["horizon"].eq(192)
    ]
    return float(row["mse_delta_pct_vs_lrbn"].iloc[0]) if not row.empty else float("nan")


def slice_value(slice_df: pd.DataFrame, variant: str, name: str) -> float:
    row = slice_df[slice_df["variant"].eq(variant) & slice_df["slice"].eq(name)]
    return float(row["mse_delta_pct_vs_lrbn"].iloc[0]) if not row.empty else float("nan")


def gate_table(overall: pd.DataFrame, per_config: pd.DataFrame, slice_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in overall[overall["variant"].isin(STAGE16_VARIANTS)].iterrows():
        variant = str(row["variant"])
        q4 = slice_value(slice_df, variant, "q4_boundary")
        known = known_harmed_delta(per_config, variant)
        base = {
            "variant": variant,
            "mse_delta_pct_vs_lrbn": float(row["mse_delta_pct_vs_lrbn"]),
            "harm_rate": float(row["harm_rate"]),
            "max_config_harm": float(row["max_config_harm"]),
            "oracle_gain_fraction": float(row.get("oracle_gain_fraction", np.nan)),
            "lrbn_equiv_rate": float(row["lrbn_equiv_rate"]),
            "active_patch_ratio": float(row["active_patch_ratio"]),
            "edit_energy_ratio": float(row["edit_energy_ratio"]),
            "q4_boundary_delta_pct": q4,
            "known_harmed_config_delta_pct": known,
            "bootstrap_high_delta_raw": float(row.get("ci95_high_delta_raw", np.nan)),
        }
        safe = (
            base["mse_delta_pct_vs_lrbn"] <= -1.8
            and base["harm_rate"] <= 0.02
            and base["max_config_harm"] <= 0.08
            and base["bootstrap_high_delta_raw"] < 0
            and base["lrbn_equiv_rate"] < 0.80
            and base["active_patch_ratio"] >= 0.08
            and base["q4_boundary_delta_pct"] <= 0.0
            and base["known_harmed_config_delta_pct"] <= 0.5
        )
        tradeoff = (
            base["mse_delta_pct_vs_lrbn"] <= -2.6
            and base["harm_rate"] <= 0.10
            and base["max_config_harm"] <= 0.18
            and base["bootstrap_high_delta_raw"] < 0
            and base["lrbn_equiv_rate"] < 0.70
            and base["active_patch_ratio"] >= 0.12
            and base["q4_boundary_delta_pct"] <= 0.0
            and base["known_harmed_config_delta_pct"] <= 0.5
        )
        mechanism = base["oracle_gain_fraction"] >= 0.08 and base["q4_boundary_delta_pct"] <= 0.0 and base["known_harmed_config_delta_pct"] <= 0.5
        base["safe_gate_pass"] = bool(safe)
        base["tradeoff_gate_pass"] = bool(tradeoff)
        base["mechanism_gate_pass"] = bool(mechanism)
        base["compact_gate_pass"] = bool(safe or tradeoff)
        rows.append(base)
    return pd.DataFrame(rows)


def add_reference_rows(overall: pd.DataFrame, stage7_dir: Path, stage14_dir: Path, stage15_dir: Path) -> pd.DataFrame:
    frames = [overall]
    stage7_overall = stage7_dir / "safe_tae_overall.csv"
    if stage7_overall.exists():
        s7 = pd.read_csv(stage7_overall)
        row = s7[s7["variant"].eq("SafeTAE-safe")].head(1).copy()
        if not row.empty:
            for col in overall.columns:
                if col not in row.columns:
                    row[col] = np.nan
            row = row[overall.columns]
            row["variant"] = "SafeTAE-safe (Stage7 table)"
            frames.append(row)
    stage14_overall = stage14_dir / "stage14_overall.csv"
    if stage14_overall.exists():
        s14 = pd.read_csv(stage14_overall)
        row = s14[s14["variant"].eq("FamilyMix Selector")].head(1).copy()
        if not row.empty:
            for col in overall.columns:
                if col not in row.columns:
                    row[col] = np.nan
            row = row[overall.columns]
            row["variant"] = "Stage14 FamilyMix Selector"
            frames.append(row)
    stage15_overall = stage15_dir / "stage15_overall.csv"
    if stage15_overall.exists():
        s15 = pd.read_csv(stage15_overall)
        refs = s15[s15["variant"].isin(["H1 Residual Atom Simplex Editor", "H2 Prototype Codebook Local Editor"])].copy()
        for _, ref in refs.iterrows():
            row = pd.DataFrame([ref])
            for col in overall.columns:
                if col not in row.columns:
                    row[col] = np.nan
            row = row[overall.columns]
            row["variant"] = str(ref["variant"]) + " (Stage15)"
            frames.append(row)
    return pd.concat(frames, ignore_index=True)


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
        "lrbn_equiv_rate",
        "active_patch_ratio",
        "ci95_high_delta_raw",
    ]
    show = [c for c in cols if c in overall.columns]
    return "\n".join(
        [
            "# Stage 16 Learned Patch / Teacher Projector",
            "",
            f"Status: `{verdict['status']}`.",
            "",
            "## Overall",
            "",
            df_to_md(overall[show], max_rows=40),
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


def verify_outputs(output_dir: Path) -> pd.DataFrame:
    required = [
        "stage16_config.json",
        "stage16_training_log.csv",
        "stage16_calibration_grid.csv",
        "stage16_policies.json",
        "stage16_overall.csv",
        "stage16_per_config.csv",
        "stage16_slice_metrics.csv",
        "stage16_gate_table.csv",
        "stage16_verdict.json",
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
    stage7_dir: Path,
    stage14_dir: Path,
    stage15_dir: Path,
    output_dir: Path,
    stage3_dir: Optional[Path] = None,
    cfg: Optional[Stage16Config] = None,
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    start = time.time()
    cfg = cfg or Stage16Config()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[stage16-learned] preparing assets", flush=True)
    assets = prepare_assets(metrics_csv, stage5_dir, stage3_dir, cfg.seed)
    pools = build_cga_pools(assets)
    write_json(
        output_dir / "stage16_config.json",
        {
            "stage": "stage16_learned_patch_teacher",
            "source_plan": "Stage15 follow-up from deep-research-report (5).md",
            "compact_protocol": "ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192 x seed 2026",
            "metrics_csv": metrics_csv,
            "stage5_dir": stage5_dir,
            "stage7_dir": stage7_dir,
            "stage14_dir": stage14_dir,
            "stage15_dir": stage15_dir,
            "stage3_dir": stage3_dir,
            "stage16_config": asdict(cfg),
            "resolved_device": str(device),
            "n_bootstrap": n_bootstrap,
            "test_threshold_leakage": False,
        },
    )

    print("[stage16-learned] extracting patch packs", flush=True)
    train_pack = patch_feature_pack(assets.val_train, cfg)
    calib_pack = patch_feature_pack(assets.val_calib, cfg)
    test_pack = patch_feature_pack(assets.test, cfg)
    train_z, [calib_z, test_z], mean, std = standardize(train_pack.features, calib_pack.features, test_pack.features)
    train_pack.features = train_z
    calib_pack.features = calib_z
    test_pack.features = test_z

    print("[stage16-learned] training residual patch MLP", flush=True)
    residual_model, residual_log = train_residual_mlp(train_pack, cfg, device)
    print("[stage16-learned] training denoising teacher", flush=True)
    teacher, teacher_log = train_teacher(train_pack, cfg, device)
    train_rows = []
    for row in residual_log["history"]:
        row = dict(row)
        row["model"] = "residual_patch_mlp"
        train_rows.append(row)
    for row in teacher_log["history"]:
        row = dict(row)
        row["model"] = "denoising_patch_teacher"
        train_rows.append(row)
    pd.DataFrame(train_rows).to_csv(output_dir / "stage16_training_log.csv", index=False)

    print("[stage16-learned] predicting patch deltas", flush=True)
    calib_patch_pred = predict_patches(calib_pack, residual_model, teacher, device, cfg.batch_size)
    test_patch_pred = predict_patches(test_pack, residual_model, teacher, device, cfg.batch_size)
    calib_oracle_mse = oracle_best(deployable_candidates(pools.calib_candidates), assets.val_calib)[1]
    oracle_pred, oracle_mse, _ = oracle_best(deployable_candidates(pools.test_candidates), assets.test)

    preds: Dict[str, np.ndarray] = {
        "LRBN": assets.test.lrbn_pred,
        "SRA-BP-safe": next((c.pred for c in assets.old_test_candidates if c.name == "sra_safe"), assets.test.lrbn_pred),
        "SRA-BP-balanced": next((c.pred for c in assets.old_test_candidates if c.name == "sra_balanced"), assets.test.lrbn_pred),
        "oracle_stage16_cga_full": oracle_pred,
    }
    decisions_by: Dict[str, pd.DataFrame] = {}
    policies: Dict[str, Any] = {}
    grid_frames: List[pd.DataFrame] = []

    print("[stage16-learned] calibrating learned patch/teacher variants", flush=True)
    for variant in STAGE16_VARIANTS:
        print(f"[stage16-learned] calibrating {variant}", flush=True)
        policy, grid = calibrate_variant(variant, assets.val_calib, calib_patch_pred, cfg, calib_oracle_mse, cfg.seed)
        grid["target_variant"] = variant
        grid_frames.append(grid)
        policies[variant] = asdict(policy)
        pred, decisions, _ = apply_patch_prediction(variant, assets.test, test_patch_pred, cfg, policy, threshold=policy.score_threshold)
        preds[variant] = pred
        decisions_by[variant] = decisions

    print("[stage16-learned] evaluating", flush=True)
    overall_rows = []
    for variant, pred in preds.items():
        decisions = decisions_by.get(variant)
        overall_rows.append(evaluate_variant(variant, pred, assets.test, decisions, oracle_mse if variant != "LRBN" else None, n_bootstrap, cfg.seed))
    overall = pd.DataFrame(overall_rows)
    overall_with_refs = add_reference_rows(overall, stage7_dir, stage14_dir, stage15_dir)
    overall_with_refs.to_csv(output_dir / "stage16_overall.csv", index=False)
    pd.concat(grid_frames, ignore_index=True).to_csv(output_dir / "stage16_calibration_grid.csv", index=False)
    write_json(output_dir / "stage16_policies.json", policies)

    per_config = pd.concat(
        [per_config_rows(variant, pred, assets.test, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    per_config.to_csv(output_dir / "stage16_per_config.csv", index=False)
    masks = stage8_slice_masks(assets.test, assets.schema, stage8_slice_thresholds(assets.val, assets.schema))
    slice_df = pd.concat(
        [slice_rows(variant, pred, assets.test, masks, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    slice_df.to_csv(output_dir / "stage16_slice_metrics.csv", index=False)
    gates = gate_table(overall, per_config, slice_df)
    gates.to_csv(output_dir / "stage16_gate_table.csv", index=False)
    deployable = overall[overall["variant"].isin(STAGE16_VARIANTS)].copy()
    best = deployable.sort_values(["mse_delta_pct_vs_lrbn", "max_config_harm", "harm_rate"], ascending=[True, True, True]).iloc[0]
    passed = gates[gates["compact_gate_pass"]]
    verdict = {
        "stage": "stage16_learned_patch_teacher",
        "status": "compact_pass_ready_for_mini_extension" if not passed.empty else "compact_failed_stop_before_mini_extension",
        "compact_pass": bool(not passed.empty),
        "passed_variants": passed["variant"].astype(str).tolist(),
        "best_variant": str(best["variant"]),
        "best_mse": float(best["mse"]),
        "best_mae": float(best["mae"]),
        "best_mse_delta_pct_vs_lrbn": float(best["mse_delta_pct_vs_lrbn"]),
        "best_harm_rate": float(best["harm_rate"]),
        "best_max_config_harm": float(best["max_config_harm"]),
        "best_oracle_gain_fraction": float(best.get("oracle_gain_fraction", np.nan)),
        "test_threshold_leakage": False,
        "stop_reason": None,
    }
    if passed.empty:
        verdict["stop_reason"] = "no learned patch/teacher variant passed compact safe/tradeoff gates"
    write_json(output_dir / "stage16_verdict.json", verdict)
    (output_dir / "summary.md").write_text(build_summary(output_dir, verdict, overall_with_refs, gates), encoding="utf-8")
    completeness = verify_outputs(output_dir)
    completeness.to_csv(output_dir / "stage16_output_completeness.csv", index=False)
    print(f"[stage16-learned] done in {time.time() - start:.1f}s", flush=True)
    return {
        "verdict": verdict,
        "overall": overall_with_refs,
        "per_config": per_config,
        "slice": slice_df,
        "gates": gates,
        "completeness": completeness,
    }
