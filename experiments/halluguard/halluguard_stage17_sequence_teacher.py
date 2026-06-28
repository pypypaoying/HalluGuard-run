#!/usr/bin/env python
"""Stage17 sequence-level teacher projection compact validation.

This stage is intentionally compact.  It reuses the existing LRBN compact
forecast assets, trains only on validation inner-train, calibrates only on
validation inner-calib, and evaluates on test.  The local compact assets do not
contain the original forecast-model training split, so the teacher training
source is recorded as a validation-inner-train proxy rather than silently
pretending to use unavailable data.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from halluguard_lrbn_bp import ForecastBatch, mae_per_sample, mse_per_sample
from halluguard_stage6_mechanism import bootstrap_ci, feature_frame, horizons, safe_pct
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


STAGE17_VARIANTS = [
    "SL-TMP Sequence Teacher Minimal-Norm Projection",
    "UTRE Uncertainty Teacher Residual Envelope",
    "SSP Structured Sequence Projector",
    "TRAP Teacher-Residual Agreement Projector",
    "IMDR Iterative Minimal-Norm Denoising Refiner",
]


@dataclass(frozen=True)
class Stage17Config:
    datasets: Tuple[str, ...] = ("ETTm1", "ETTh1")
    backbones: Tuple[str, ...] = ("DLinear", "PatchTST")
    horizons: Tuple[int, ...] = (96, 192)
    seed: int = 2026
    bootstrap: int = 2000
    output_dir: str = "experiments/halluguard/results/stage17_sequence_teacher"

    teacher_epochs: int = 20
    corrector_epochs: int = 24
    batch_size: int = 128
    lr_teacher: float = 8e-4
    lr_corrector: float = 8e-4
    weight_decay: float = 1e-4
    patch_len: int = 16
    d_model: int = 64
    depth: int = 2
    hidden_dim: int = 160
    dropout: float = 0.05
    max_delta_norm: float = 0.25
    mask_l1: float = 1e-3
    refine_steps: int = 3
    step_norm: float = 0.08
    device: str = "auto"


@dataclass(frozen=True)
class SequencePolicy:
    variant: str
    shrink: float = 0.50
    residual_cap: float = 0.20
    gate_quantile: float = 0.50
    score_threshold: float = 0.0
    width_coef: float = 1.0
    energy_only_accept: bool = False


@dataclass
class SequencePack:
    batch: ForecastBatch
    features: np.ndarray
    x_norm: np.ndarray
    y_base_norm: np.ndarray
    y_true_norm: np.ndarray
    valid_mask: np.ndarray
    scale: np.ndarray


@dataclass
class CandidateRaw:
    variant: str
    delta_norm: np.ndarray
    score: np.ndarray
    aux: Dict[str, np.ndarray]


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def finite_arr(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def valid_mask(batch: ForecastBatch) -> np.ndarray:
    mask = np.zeros_like(batch.y_true, dtype=float)
    hs = horizons(batch)
    for i, h in enumerate(hs):
        mask[i, : int(h), :] = 1.0
    mask *= np.isfinite(batch.y_true).astype(float)
    return mask


def standardize(train_x: np.ndarray, *others: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray]:
    mean = np.nanmean(train_x, axis=0)
    std = np.nanstd(train_x, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    train_z = finite_arr((train_x - mean) / std)
    other_z = [finite_arr((x - mean) / std) for x in others]
    return train_z, other_z, mean, std


def build_feature_matrix(batch: ForecastBatch, schema: Mapping[str, List[Any]]) -> np.ndarray:
    df = feature_frame(batch, dict(schema))
    numeric = df.select_dtypes(include=[np.number]).copy()
    for col in numeric.columns:
        numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
    return finite_arr(numeric.to_numpy(float))


def make_pack(batch: ForecastBatch, features: np.ndarray) -> SequencePack:
    scale = scale_matrix(batch)
    vm = valid_mask(batch)
    x_norm = finite_arr(batch.context / (scale + 1e-8))
    y_base_norm = finite_arr(batch.lrbn_pred / (scale + 1e-8)) * vm
    y_true_norm = finite_arr(batch.y_true / (scale + 1e-8)) * vm
    return SequencePack(
        batch=batch,
        features=finite_arr(features),
        x_norm=x_norm,
        y_base_norm=y_base_norm,
        y_true_norm=y_true_norm,
        valid_mask=vm,
        scale=scale,
    )


def pack_loader(pack: SequencePack, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(pack.features, dtype=torch.float32),
        torch.tensor(pack.x_norm, dtype=torch.float32),
        torch.tensor(pack.y_base_norm, dtype=torch.float32),
        torch.tensor(pack.y_true_norm, dtype=torch.float32),
        torch.tensor(pack.valid_mask, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + 1e-8)


def clip_by_norm(delta: torch.Tensor, max_norm: float, eps: float = 1e-8) -> torch.Tensor:
    flat = delta.flatten(1)
    norm = flat.norm(dim=1, keepdim=True).view(-1, 1, 1)
    scale = torch.clamp(max_norm / (norm + eps), max=1.0)
    return delta * scale


def lowpass_1d(y: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    if kernel_size <= 1:
        return y
    pad = kernel_size // 2
    x = y.transpose(1, 2)
    channels = x.size(1)
    weight = torch.ones(channels, 1, kernel_size, device=y.device, dtype=y.dtype) / float(kernel_size)
    out = F.conv1d(F.pad(x, (pad, pad), mode="replicate"), weight, groups=channels)
    return out.transpose(1, 2)


def sequence_consistency_loss(y_edit: torch.Tensor, x_hist: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dy = y_edit[:, 1:] - y_edit[:, :-1]
    ddy = dy[:, 1:] - dy[:, :-1]
    ddy_mask = mask[:, 2:]
    curvature = masked_mean(ddy.pow(2), ddy_mask) if ddy.shape[1] else y_edit.mean() * 0.0
    tail_d = x_hist[:, -1:] - x_hist[:, -2:-1]
    first_d = y_edit[:, :1] - x_hist[:, -1:]
    boundary = ((first_d - tail_d) ** 2).mean()
    return 0.5 * curvature + 0.5 * boundary


def highfreq_penalty(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    d1 = delta[:, 1:] - delta[:, :-1]
    d2 = d1[:, 1:] - d1[:, :-1]
    if d2.shape[1] == 0:
        return delta.mean() * 0.0
    return masked_mean(d2.pow(2), mask[:, 2:])


class PatchTransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, depth: int, patch_len: int, dropout: float):
        super().__init__()
        self.patch_len = int(patch_len)
        self.proj = nn.Linear(input_dim * self.patch_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length, channels = x.shape
        n_patch = max(1, length // self.patch_len)
        use_len = n_patch * self.patch_len
        if use_len > length:
            x = F.pad(x.transpose(1, 2), (0, use_len - length), mode="replicate").transpose(1, 2)
        x = x[:, :use_len, :]
        patches = x.reshape(bsz, n_patch, self.patch_len * channels)
        return self.encoder(self.proj(patches))


class PatchTransformerDecoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, depth: int, patch_len: int, dropout: float):
        super().__init__()
        self.patch_len = int(patch_len)
        self.input_proj = nn.Linear(input_dim * self.patch_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Linear(d_model, input_dim * self.patch_len)

    def forward(self, z_context: torch.Tensor, y_masked: torch.Tensor) -> torch.Tensor:
        bsz, horizon, channels = y_masked.shape
        n_patch = max(1, math.ceil(horizon / self.patch_len))
        use_len = n_patch * self.patch_len
        y = y_masked
        if use_len > horizon:
            y = F.pad(y.transpose(1, 2), (0, use_len - horizon)).transpose(1, 2)
        patches = y[:, :use_len, :].reshape(bsz, n_patch, self.patch_len * channels)
        z = self.input_proj(patches) + z_context.mean(dim=1, keepdim=True)
        out = self.out(self.decoder(z)).reshape(bsz, use_len, channels)
        return out[:, :horizon, :]


class SequenceSSLTeacher(nn.Module):
    def __init__(self, input_dim: int, d_model: int, depth: int, patch_len: int, dropout: float):
        super().__init__()
        self.context_encoder = PatchTransformerEncoder(input_dim, d_model, depth, patch_len, dropout)
        self.future_encoder = PatchTransformerEncoder(input_dim, d_model, depth, patch_len, dropout)
        self.decoder = PatchTransformerDecoder(input_dim, d_model, depth, patch_len, dropout)
        self.proj = nn.Linear(d_model, d_model)

    def encode_context(self, x_hist: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(x_hist)

    def encode_future(self, y: torch.Tensor) -> torch.Tensor:
        return self.future_encoder(y)

    def reconstruct_future(self, x_hist: torch.Tensor, y_masked: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode_context(x_hist), y_masked)

    def energy(self, x_hist: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        zc = self.encode_context(x_hist)
        zy = self.encode_future(y_pred)
        return ((self.proj(zc).mean(dim=1) - zy.mean(dim=1)) ** 2).mean(dim=-1)


def mask_future_patches(y: torch.Tensor, valid: torch.Tensor, p: float, patch_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, horizon, channels = y.shape
    n_patch = max(1, math.ceil(horizon / patch_len))
    patch_mask = (torch.rand(bsz, n_patch, 1, device=y.device) < p).float()
    mask = patch_mask.repeat_interleave(patch_len, dim=1)[:, :horizon, :]
    if channels > 1:
        mask = mask.repeat(1, 1, channels)
    mask = mask * valid
    return y * (1.0 - mask), mask


def contrastive_context_future_loss(z_context: torch.Tensor, z_future: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    c = F.normalize(z_context.mean(dim=1), dim=-1)
    f = F.normalize(z_future.mean(dim=1), dim=-1)
    logits = c @ f.t() / temperature
    labels = torch.arange(c.size(0), device=c.device)
    return F.cross_entropy(logits, labels)


def train_teacher(pack: SequencePack, cfg: Stage17Config, device: torch.device) -> Tuple[SequenceSSLTeacher, pd.DataFrame]:
    set_seed(cfg.seed + 1700)
    teacher = SequenceSSLTeacher(
        input_dim=int(pack.x_norm.shape[2]),
        d_model=cfg.d_model,
        depth=cfg.depth,
        patch_len=cfg.patch_len,
        dropout=cfg.dropout,
    ).to(device)
    loader = pack_loader(pack, cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(teacher.parameters(), lr=cfg.lr_teacher, weight_decay=cfg.weight_decay)
    logs: List[Dict[str, float]] = []
    teacher.train()
    for epoch in range(cfg.teacher_epochs):
        losses: List[float] = []
        rec_losses: List[float] = []
        con_losses: List[float] = []
        for _feat, x, _base, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            y_masked, rec_mask = mask_future_patches(y, mask, p=0.50, patch_len=cfg.patch_len)
            opt.zero_grad(set_to_none=True)
            recon = teacher.reconstruct_future(x, y_masked)
            loss_rec = masked_mean((recon - y).pow(2), rec_mask)
            loss_con = contrastive_context_future_loss(teacher.encode_context(x), teacher.encode_future(y))
            loss = loss_rec + 0.05 * loss_con
            loss.backward()
            nn.utils.clip_grad_norm_(teacher.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            rec_losses.append(float(loss_rec.detach().cpu()))
            con_losses.append(float(loss_con.detach().cpu()))
        logs.append(
            {
                "epoch": float(epoch + 1),
                "teacher_loss": float(np.mean(losses)),
                "masked_reconstruction_loss": float(np.mean(rec_losses)),
                "contrastive_loss": float(np.mean(con_losses)),
            }
        )
    return teacher.eval(), pd.DataFrame(logs)


class SequenceDeltaMLP(nn.Module):
    def __init__(self, feature_dim: int, horizon: int, channels: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.horizon = int(horizon)
        self.channels = int(channels)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.horizon * self.channels),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).view(-1, self.horizon, self.channels)


class ResidualQuantileEnvelope(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        horizon: int,
        channels: int,
        hidden_dim: int,
        dropout: float,
        taus: Tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90),
    ):
        super().__init__()
        self.horizon = int(horizon)
        self.channels = int(channels)
        self.taus = tuple(float(t) for t in taus)
        self.register_buffer("tau_tensor", torch.tensor(self.taus, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(feature_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.horizon * self.channels),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        outs = []
        for tau in self.tau_tensor.to(feat.device):
            tau_col = tau.expand(feat.size(0), 1)
            pred = self.net(torch.cat([feat, tau_col], dim=-1)).view(-1, self.horizon, self.channels)
            outs.append(pred)
        q = torch.stack(outs, dim=1)
        q_sorted, _ = torch.sort(q, dim=1)
        return q_sorted


class StructuredSequenceProjector(nn.Module):
    def __init__(self, feature_dim: int, horizon: int, channels: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.horizon = int(horizon)
        self.channels = int(channels)
        self.global_head = SequenceDeltaMLP(feature_dim, horizon, channels, hidden_dim, dropout)
        self.local_head = SequenceDeltaMLP(feature_dim, horizon, channels, hidden_dim, dropout)
        self.mask_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = feat.size(0)
        global_delta = lowpass_1d(self.global_head(feat), kernel_size=7)
        local_raw = self.local_head(feat)
        mask = torch.sigmoid(self.mask_head(feat)).view(bsz, self.horizon, 1)
        return global_delta, mask * local_raw, mask


def pinball_loss(q: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, taus: Sequence[float]) -> torch.Tensor:
    losses: List[torch.Tensor] = []
    for i, tau in enumerate(taus):
        err = target - q[:, i]
        loss = torch.maximum((tau - 1.0) * err, tau * err)
        losses.append(masked_mean(loss, mask))
    return torch.stack(losses).mean()


def train_sl_tmp(
    train: SequencePack,
    teacher: SequenceSSLTeacher,
    cfg: Stage17Config,
    device: torch.device,
) -> Tuple[SequenceDeltaMLP, pd.DataFrame]:
    set_seed(cfg.seed + 1710)
    model = SequenceDeltaMLP(train.features.shape[1], train.y_base_norm.shape[1], train.y_base_norm.shape[2], cfg.hidden_dim, cfg.dropout).to(device)
    loader = pack_loader(train, cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_corrector, weight_decay=cfg.weight_decay)
    logs: List[Dict[str, float]] = []
    teacher.eval()
    for epoch in range(cfg.corrector_epochs):
        losses: List[float] = []
        model.train()
        for feat, x, y_base, y_true, mask in loader:
            feat = feat.to(device)
            x = x.to(device)
            y_base = y_base.to(device)
            y_true = y_true.to(device)
            mask = mask.to(device)
            opt.zero_grad(set_to_none=True)
            delta = clip_by_norm(model(feat), cfg.max_delta_norm)
            y_edit = y_base + delta * mask
            mse_loss = masked_mean((y_edit - y_true).pow(2), mask)
            energy_loss = teacher.energy(x, y_edit).mean()
            norm_loss = masked_mean(delta.pow(2), mask)
            seq_loss = sequence_consistency_loss(y_edit, x, mask)
            loss = mse_loss + 0.03 * energy_loss + 0.03 * norm_loss + 0.03 * seq_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        logs.append({"epoch": float(epoch + 1), "model": "SL-TMP", "loss": float(np.mean(losses))})
    return model.eval(), pd.DataFrame(logs)


def train_utre(train: SequencePack, cfg: Stage17Config, device: torch.device) -> Tuple[ResidualQuantileEnvelope, pd.DataFrame]:
    set_seed(cfg.seed + 1720)
    model = ResidualQuantileEnvelope(train.features.shape[1], train.y_base_norm.shape[1], train.y_base_norm.shape[2], cfg.hidden_dim, cfg.dropout).to(device)
    loader = pack_loader(train, cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_corrector, weight_decay=cfg.weight_decay)
    logs: List[Dict[str, float]] = []
    for epoch in range(cfg.corrector_epochs):
        losses: List[float] = []
        model.train()
        for feat, _x, y_base, y_true, mask in loader:
            feat = feat.to(device)
            y_base = y_base.to(device)
            y_true = y_true.to(device)
            mask = mask.to(device)
            opt.zero_grad(set_to_none=True)
            residual = (y_true - y_base) * mask
            q = model(feat)
            loss = pinball_loss(q, residual, mask, model.taus)
            width_penalty = masked_mean((q[:, -1] - q[:, 0]).abs(), mask)
            loss = loss + 0.002 * width_penalty
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        logs.append({"epoch": float(epoch + 1), "model": "UTRE", "loss": float(np.mean(losses))})
    return model.eval(), pd.DataFrame(logs)


def train_ssp(
    train: SequencePack,
    teacher: SequenceSSLTeacher,
    cfg: Stage17Config,
    device: torch.device,
) -> Tuple[StructuredSequenceProjector, pd.DataFrame]:
    set_seed(cfg.seed + 1730)
    model = StructuredSequenceProjector(train.features.shape[1], train.y_base_norm.shape[1], train.y_base_norm.shape[2], cfg.hidden_dim, cfg.dropout).to(device)
    loader = pack_loader(train, cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_corrector, weight_decay=cfg.weight_decay)
    logs: List[Dict[str, float]] = []
    teacher.eval()
    for epoch in range(cfg.corrector_epochs):
        losses: List[float] = []
        model.train()
        for feat, x, y_base, y_true, mask in loader:
            feat = feat.to(device)
            x = x.to(device)
            y_base = y_base.to(device)
            y_true = y_true.to(device)
            mask = mask.to(device)
            opt.zero_grad(set_to_none=True)
            delta_g, delta_l, local_mask = model(feat)
            delta = clip_by_norm(delta_g + delta_l, cfg.max_delta_norm)
            y_edit = y_base + delta * mask
            mse_loss = masked_mean((y_edit - y_true).pow(2), mask)
            energy_loss = teacher.energy(x, y_edit).mean()
            norm_loss = masked_mean(delta.pow(2), mask)
            sparse_loss = local_mask.mean()
            hf_loss = highfreq_penalty(delta_g, mask)
            seq_loss = sequence_consistency_loss(y_edit, x, mask)
            loss = mse_loss + 0.03 * energy_loss + 0.02 * norm_loss + cfg.mask_l1 * sparse_loss + 0.02 * hf_loss + 0.02 * seq_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        logs.append({"epoch": float(epoch + 1), "model": "SSP", "loss": float(np.mean(losses))})
    return model.eval(), pd.DataFrame(logs)


@torch.no_grad()
def teacher_delta(pack: SequencePack, teacher: SequenceSSLTeacher, cfg: Stage17Config, device: torch.device) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    deltas: List[np.ndarray] = []
    e_before: List[np.ndarray] = []
    e_after: List[np.ndarray] = []
    teacher.eval()
    for _feat, x, y_base, _y_true, mask in pack_loader(pack, cfg.batch_size, shuffle=False):
        x = x.to(device)
        y_base = y_base.to(device)
        mask = mask.to(device)
        recon = teacher.reconstruct_future(x, y_base)
        delta = (recon - y_base) * mask
        y_edit = y_base + delta
        deltas.append(delta.cpu().numpy())
        e_before.append(teacher.energy(x, y_base).cpu().numpy())
        e_after.append(teacher.energy(x, y_edit).cpu().numpy())
    return finite_arr(np.concatenate(deltas, axis=0)), {
        "teacher_energy_before": finite_arr(np.concatenate(e_before)),
        "teacher_energy_after_raw": finite_arr(np.concatenate(e_after)),
    }


@torch.no_grad()
def predict_sl_tmp(pack: SequencePack, model: SequenceDeltaMLP, teacher: SequenceSSLTeacher, cfg: Stage17Config, device: torch.device) -> CandidateRaw:
    deltas: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    e_before: List[np.ndarray] = []
    e_after: List[np.ndarray] = []
    model.eval()
    teacher.eval()
    for feat, x, y_base, _y_true, mask in pack_loader(pack, cfg.batch_size, shuffle=False):
        feat = feat.to(device)
        x = x.to(device)
        y_base = y_base.to(device)
        mask = mask.to(device)
        delta = clip_by_norm(model(feat), cfg.max_delta_norm) * mask
        y_edit = y_base + delta
        eb = teacher.energy(x, y_base)
        ea = teacher.energy(x, y_edit)
        deltas.append(delta.cpu().numpy())
        score = delta.flatten(1).norm(dim=1).cpu().numpy() + np.maximum(0.0, (eb - ea).cpu().numpy())
        scores.append(score)
        e_before.append(eb.cpu().numpy())
        e_after.append(ea.cpu().numpy())
    return CandidateRaw(
        "SL-TMP Sequence Teacher Minimal-Norm Projection",
        finite_arr(np.concatenate(deltas, axis=0)),
        finite_arr(np.concatenate(scores)),
        {"teacher_energy_before": finite_arr(np.concatenate(e_before)), "teacher_energy_after_raw": finite_arr(np.concatenate(e_after))},
    )


@torch.no_grad()
def predict_utre(pack: SequencePack, model: ResidualQuantileEnvelope, cfg: Stage17Config, device: torch.device) -> CandidateRaw:
    deltas: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    widths: List[np.ndarray] = []
    model.eval()
    for feat, _x, _y_base, _y_true, mask in pack_loader(pack, cfg.batch_size, shuffle=False):
        feat = feat.to(device)
        mask = mask.to(device)
        q = model(feat)
        center = 0.5 * (q[:, 1] + q[:, 3]) * mask
        width = masked_sample_mean((q[:, -1] - q[:, 0]).abs(), mask)
        center_norm = center.flatten(1).norm(dim=1) / torch.sqrt(mask.flatten(1).sum(dim=1).clamp_min(1.0))
        score = center_norm / (width + 1e-6)
        deltas.append(center.cpu().numpy())
        scores.append(score.cpu().numpy())
        widths.append(width.cpu().numpy())
    return CandidateRaw(
        "UTRE Uncertainty Teacher Residual Envelope",
        finite_arr(np.concatenate(deltas, axis=0)),
        finite_arr(np.concatenate(scores)),
        {"uncertainty_width": finite_arr(np.concatenate(widths))},
    )


@torch.no_grad()
def predict_ssp(pack: SequencePack, model: StructuredSequenceProjector, teacher: SequenceSSLTeacher, cfg: Stage17Config, device: torch.device) -> CandidateRaw:
    deltas: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    e_before: List[np.ndarray] = []
    e_after: List[np.ndarray] = []
    model.eval()
    teacher.eval()
    for feat, x, y_base, _y_true, mask in pack_loader(pack, cfg.batch_size, shuffle=False):
        feat = feat.to(device)
        x = x.to(device)
        y_base = y_base.to(device)
        mask = mask.to(device)
        delta_g, delta_l, local_mask = model(feat)
        delta = clip_by_norm(delta_g + delta_l, cfg.max_delta_norm) * mask
        y_edit = y_base + delta
        eb = teacher.energy(x, y_base)
        ea = teacher.energy(x, y_edit)
        deltas.append(delta.cpu().numpy())
        score = delta.flatten(1).norm(dim=1).cpu().numpy() + 0.5 * local_mask.mean(dim=(1, 2)).cpu().numpy()
        scores.append(score)
        masks.append(local_mask.mean(dim=(1, 2)).cpu().numpy())
        e_before.append(eb.cpu().numpy())
        e_after.append(ea.cpu().numpy())
    return CandidateRaw(
        "SSP Structured Sequence Projector",
        finite_arr(np.concatenate(deltas, axis=0)),
        finite_arr(np.concatenate(scores)),
        {
            "local_mask_mean": finite_arr(np.concatenate(masks)),
            "teacher_energy_before": finite_arr(np.concatenate(e_before)),
            "teacher_energy_after_raw": finite_arr(np.concatenate(e_after)),
        },
    )


def masked_sample_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (x * mask).flatten(1).sum(dim=1) / (mask.flatten(1).sum(dim=1) + 1e-8)


def agreement_projection(delta_residual: np.ndarray, delta_teacher: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    r = delta_residual.reshape(delta_residual.shape[0], -1)
    t = delta_teacher.reshape(delta_teacher.shape[0], -1)
    coeff = np.sum(r * t, axis=1, keepdims=True) / (np.sum(t * t, axis=1, keepdims=True) + eps)
    coeff = np.clip(coeff, 0.0, 1.0)
    cos = np.sum(r * t, axis=1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(t, axis=1) + eps)
    return (coeff.reshape(-1, 1, 1) * delta_teacher), finite_arr(cos)


def make_trap(sl_raw: CandidateRaw, teacher_raw_delta: np.ndarray) -> CandidateRaw:
    delta, cosine = agreement_projection(sl_raw.delta_norm, teacher_raw_delta)
    score = np.maximum(0.0, cosine) * np.sqrt(np.nanmean(delta**2, axis=(1, 2)))
    return CandidateRaw(
        "TRAP Teacher-Residual Agreement Projector",
        finite_arr(delta),
        finite_arr(score),
        {"direction_cosine_RT": finite_arr(cosine)},
    )


def make_imdr(teacher_raw_delta: np.ndarray, teacher_aux: Mapping[str, np.ndarray], cfg: Stage17Config) -> CandidateRaw:
    energy_gain = np.asarray(teacher_aux.get("teacher_energy_before", 0.0)) - np.asarray(teacher_aux.get("teacher_energy_after_raw", 0.0))
    delta = np.zeros_like(teacher_raw_delta)
    active = energy_gain >= 0.0
    if active.any():
        step = np.clip(teacher_raw_delta, -cfg.step_norm, cfg.step_norm)
        delta[active] = min(cfg.refine_steps, 3) * step[active] / 3.0
    score = np.maximum(0.0, energy_gain) + np.sqrt(np.nanmean(delta**2, axis=(1, 2)))
    return CandidateRaw(
        "IMDR Iterative Minimal-Norm Denoising Refiner",
        finite_arr(delta),
        finite_arr(score),
        {"teacher_energy_gain_proxy": finite_arr(energy_gain)},
    )


def apply_sequence_policy(
    raw: CandidateRaw,
    pack: SequencePack,
    policy: SequencePolicy,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, pd.DataFrame, SequencePolicy]:
    score = finite_arr(raw.score)
    if threshold is None:
        threshold = float(np.nanquantile(score, float(policy.gate_quantile))) if len(score) else float("inf")
    selected = score >= threshold
    if policy.energy_only_accept and "teacher_energy_before" in raw.aux and "teacher_energy_after_raw" in raw.aux:
        selected &= np.asarray(raw.aux["teacher_energy_after_raw"]) <= np.asarray(raw.aux["teacher_energy_before"])
    policy = replace(policy, score_threshold=float(threshold))
    delta_norm = np.clip(float(policy.shrink) * raw.delta_norm, -float(policy.residual_cap), float(policy.residual_cap))
    delta_norm[~selected] = 0.0
    pred = pack.batch.lrbn_pred + delta_norm * pack.scale
    pred = np.where(pack.valid_mask > 0, pred, pack.batch.lrbn_pred)
    decisions = pd.DataFrame(
        {
            "row_index": np.arange(len(pack.batch.meta)),
            "selected": selected,
            "selected_action": np.where(selected, raw.variant, "keep_lrbn"),
            "accept_score": score,
        }
    )
    return pred, decisions, policy


def sequence_policy_grid(variant: str) -> Iterable[SequencePolicy]:
    if variant == "UTRE Uncertainty Teacher Residual Envelope":
        for shrink in [0.25, 0.50, 0.75]:
            for cap in [0.10, 0.20, 0.35]:
                for q in [0.00, 0.50, 0.75, 0.90]:
                    for width in [0.50, 1.00, 2.00]:
                        yield SequencePolicy(variant, shrink=shrink, residual_cap=cap, gate_quantile=q, width_coef=width)
    elif variant == "TRAP Teacher-Residual Agreement Projector":
        for shrink in [0.25, 0.50, 0.75, 1.00]:
            for cap in [0.05, 0.10, 0.20, 0.35]:
                for q in [0.00, 0.50, 0.75, 0.90]:
                    yield SequencePolicy(variant, shrink=shrink, residual_cap=cap, gate_quantile=q)
    elif variant == "IMDR Iterative Minimal-Norm Denoising Refiner":
        for shrink in [0.25, 0.50, 0.75, 1.00]:
            for cap in [0.05, 0.10, 0.20]:
                for q in [0.00, 0.50, 0.75, 0.90]:
                    yield SequencePolicy(variant, shrink=shrink, residual_cap=cap, gate_quantile=q, energy_only_accept=True)
    else:
        for shrink in [0.25, 0.50, 0.75]:
            for cap in [0.10, 0.20, 0.35]:
                for q in [0.00, 0.50, 0.75, 0.90]:
                    yield SequencePolicy(variant, shrink=shrink, residual_cap=cap, gate_quantile=q)


def calibration_score(row: Mapping[str, Any]) -> float:
    score = float(row["mse_delta_pct_vs_lrbn"])
    score += 180.0 * max(0.0, float(row["harm_rate"]) - 0.10)
    score += 150.0 * max(0.0, float(row["max_config_harm"]) - 0.18)
    score += 45.0 * max(0.0, float(row.get("lrbn_equiv_rate", 0.0)) - 0.80)
    score += 25.0 * max(0.0, 0.08 - float(row.get("active_patch_ratio", 0.0)))
    score += 20.0 * max(0.0, 0.08 - float(row.get("oracle_gain_fraction", 0.0)))
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
    raw: CandidateRaw,
    calib_pack: SequencePack,
    oracle_mse: np.ndarray,
    seed: int,
) -> Tuple[SequencePolicy, pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best: Optional[SequencePolicy] = None
    best_score = float("inf")
    for policy0 in sequence_policy_grid(raw.variant):
        pred, decisions, policy = apply_sequence_policy(raw, calib_pack, policy0)
        row = evaluate_variant(raw.variant, pred, calib_pack.batch, decisions, oracle_mse, n_bootstrap=0, seed=seed)
        row.update(asdict(policy))
        row["calibration_score"] = calibration_score(row)
        rows.append(row)
        if float(row["calibration_score"]) < best_score:
            best_score = float(row["calibration_score"])
            best = policy
    assert best is not None
    return best, pd.DataFrame(rows)


def teacher_energy_array(pack: SequencePack, teacher: SequenceSSLTeacher, pred: np.ndarray, cfg: Stage17Config, device: torch.device) -> np.ndarray:
    energies: List[np.ndarray] = []
    teacher.eval()
    scaled_pred = finite_arr(pred / (pack.scale + 1e-8)) * pack.valid_mask
    ds = TensorDataset(torch.tensor(pack.x_norm, dtype=torch.float32), torch.tensor(scaled_pred, dtype=torch.float32))
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            energies.append(teacher.energy(x.to(device), y.to(device)).cpu().numpy())
    return finite_arr(np.concatenate(energies))


def residual_alignment_A(pred: np.ndarray, batch: ForecastBatch, eps: float = 1e-8) -> np.ndarray:
    delta = pred - batch.lrbn_pred
    residual = batch.y_true - batch.lrbn_pred
    flat_d = finite_arr(delta).reshape(delta.shape[0], -1)
    flat_e = finite_arr(residual).reshape(residual.shape[0], -1)
    return 2.0 * np.sum(flat_d * flat_e, axis=1) / (np.sum(flat_d * flat_d, axis=1) + eps)


def mean_abs_second_diff(arr: np.ndarray) -> np.ndarray:
    d1 = np.diff(finite_arr(arr), axis=1)
    d2 = np.diff(d1, axis=1)
    if d2.shape[1] == 0:
        return np.zeros(arr.shape[0], dtype=float)
    return np.nanmean(np.abs(d2), axis=(1, 2))


def mechanism_row(
    variant: str,
    pred: np.ndarray,
    raw: CandidateRaw,
    pack: SequencePack,
    teacher: SequenceSSLTeacher,
    cfg: Stage17Config,
    device: torch.device,
) -> Dict[str, Any]:
    base_mse = mse_per_sample(pack.batch.lrbn_pred, pack.batch.y_true)
    method_mse = mse_per_sample(pred, pack.batch.y_true)
    mse_delta = method_mse - base_mse
    e_before = teacher_energy_array(pack, teacher, pack.batch.lrbn_pred, cfg, device)
    e_after = teacher_energy_array(pack, teacher, pred, cfg, device)
    energy_delta = e_after - e_before
    corr = spearmanr(energy_delta, mse_delta, nan_policy="omit")
    align = residual_alignment_A(pred, pack.batch)
    delta = pred - pack.batch.lrbn_pred
    scale = scale_matrix(pack.batch)
    edit_norm = np.sqrt(np.nanmean((delta / (scale + 1e-8)) ** 2, axis=(1, 2)))
    patch_active = edit_norm > 0.01
    direction_cos = raw.aux.get("direction_cosine_RT")
    width = raw.aux.get("uncertainty_width")
    return {
        "variant": variant,
        "teacher_energy_before_mean": float(np.mean(e_before)),
        "teacher_energy_after_mean": float(np.mean(e_after)),
        "teacher_energy_delta_mean": float(np.mean(energy_delta)),
        "latent_consistency_gain": float(-np.mean(energy_delta)),
        "teacher_energy_mse_delta_spearman": float(corr.correlation) if np.isfinite(corr.correlation) else 0.0,
        "residual_alignment_A_mean": float(np.nanmean(align)),
        "residual_alignment_A_gt1_rate": float(np.nanmean(align > 1.0)),
        "direction_cosine_RT_mean": float(np.nanmean(direction_cos)) if direction_cos is not None else float("nan"),
        "uncertainty_width_mean": float(np.nanmean(width)) if width is not None else float("nan"),
        "sequence_overlap_error": float(np.nanmean(mean_abs_second_diff(delta / (scale + 1e-8)))),
        "edit_energy_ratio": float(np.nanmean(delta**2) / (float(np.nanmean(pack.batch.lrbn_pred**2)) + 1e-12)),
        "lrbn_equiv_rate": float(np.nanmean(edit_norm <= 0.01)),
        "active_patch_ratio": float(np.nanmean(patch_active)),
    }


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


def gate_table(overall: pd.DataFrame, per_config: pd.DataFrame, slice_df: pd.DataFrame, mechanism: pd.DataFrame) -> pd.DataFrame:
    mech_by = {str(r["variant"]): r for _, r in mechanism.iterrows()}
    rows: List[Dict[str, Any]] = []
    for _, row in overall[overall["variant"].isin(STAGE17_VARIANTS)].iterrows():
        variant = str(row["variant"])
        mech = mech_by.get(variant, {})
        q4 = slice_value(slice_df, variant, "q4_boundary")
        non = slice_value(slice_df, variant, "non_boundary")
        known = known_harmed_delta(per_config, variant)
        base = {
            "variant": variant,
            "mse_delta_pct_vs_lrbn": float(row["mse_delta_pct_vs_lrbn"]),
            "harm_rate": float(row["harm_rate"]),
            "max_config_harm": float(row["max_config_harm"]),
            "oracle_gain_fraction": float(row.get("oracle_gain_fraction", np.nan)),
            "lrbn_equiv_rate": float(row.get("lrbn_equiv_rate", np.nan)),
            "active_patch_ratio": float(row.get("active_patch_ratio", np.nan)),
            "edit_energy_ratio": float(row.get("edit_energy_ratio", np.nan)),
            "q4_boundary_delta_pct": q4,
            "non_boundary_delta_pct": non,
            "known_harmed_config_delta_pct": known,
            "bootstrap_high_delta_raw": float(row.get("ci95_high_delta_raw", np.nan)),
            "teacher_energy_mse_delta_spearman": float(mech.get("teacher_energy_mse_delta_spearman", np.nan)),
            "residual_alignment_A_gt1_rate": float(mech.get("residual_alignment_A_gt1_rate", np.nan)),
        }
        safe = (
            base["mse_delta_pct_vs_lrbn"] <= -1.8
            and base["harm_rate"] <= 0.025
            and base["max_config_harm"] <= 0.08
            and base["bootstrap_high_delta_raw"] < 0
            and base["q4_boundary_delta_pct"] <= 0.0
            and base["non_boundary_delta_pct"] <= 0.0
            and base["known_harmed_config_delta_pct"] <= 0.5
            and base["lrbn_equiv_rate"] < 0.80
        )
        tradeoff = (
            base["mse_delta_pct_vs_lrbn"] <= -2.6
            and base["harm_rate"] <= 0.10
            and base["max_config_harm"] <= 0.18
            and base["bootstrap_high_delta_raw"] < 0
            and base["known_harmed_config_delta_pct"] <= 1.0
        )
        mechanism_pass = (
            base["teacher_energy_mse_delta_spearman"] >= 0.20
            and base["residual_alignment_A_gt1_rate"] >= 0.60
            and base["active_patch_ratio"] >= 0.08
            and base["edit_energy_ratio"] > 1e-8
        )
        base["safe_gate_pass"] = bool(safe)
        base["tradeoff_gate_pass"] = bool(tradeoff)
        base["mechanism_gate_pass"] = bool(mechanism_pass)
        base["compact_gate_pass"] = bool(safe or tradeoff)
        rows.append(base)
    return pd.DataFrame(rows)


def add_reference_rows(overall: pd.DataFrame, stage7_dir: Path, stage14_dir: Path, stage15_dir: Path, stage16_dir: Path) -> pd.DataFrame:
    frames = [overall]
    refs: List[Tuple[Path, List[str], str]] = [
        (stage7_dir / "safe_tae_overall.csv", ["SafeTAE-safe"], " (Stage7 table)"),
        (stage14_dir / "stage14_overall.csv", ["FamilyMix Selector"], " (Stage14)"),
        (stage15_dir / "stage15_overall.csv", ["H1 Residual Atom Simplex Editor", "H2 Prototype Codebook Local Editor"], " (Stage15)"),
        (
            stage16_dir / "stage16_overall.csv",
            [
                "H6-Safe Sparse Teacher Projector",
                "H2H6 Learned Patch Teacher Hybrid",
                "H6 Denoising Teacher Manifold Projector",
            ],
            " (Stage16)",
        ),
    ]
    for path, variants, suffix in refs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for _, ref in df[df["variant"].isin(variants)].iterrows():
            row = pd.DataFrame([ref])
            for col in overall.columns:
                if col not in row.columns:
                    row[col] = np.nan
            row = row[overall.columns]
            row["variant"] = str(ref["variant"]) + suffix
            frames.append(row)
    return pd.concat(frames, ignore_index=True)


def verify_outputs(output_dir: Path) -> pd.DataFrame:
    required = [
        "stage17_config.json",
        "stage17_training_log_teacher.csv",
        "stage17_training_log_correctors.csv",
        "stage17_calibration_grid.csv",
        "stage17_policies.json",
        "stage17_overall.csv",
        "stage17_per_config.csv",
        "stage17_slice_metrics.csv",
        "stage17_mechanism_metrics.csv",
        "stage17_alignment_metrics.csv",
        "stage17_uncertainty_metrics.csv",
        "stage17_bootstrap_ci.json",
        "stage17_gate_table.csv",
        "stage17_verdict.json",
        "summary.md",
    ]
    rows = []
    for name in required:
        p = output_dir / name
        rows.append({"artifact": name, "exists": bool(p.exists()), "bytes": int(p.stat().st_size) if p.exists() else 0})
    return pd.DataFrame(rows)


def build_summary(output_dir: Path, verdict: Mapping[str, Any], overall: pd.DataFrame, gates: pd.DataFrame, mechanism: pd.DataFrame) -> str:
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
    mech_cols = [
        "variant",
        "teacher_energy_delta_mean",
        "teacher_energy_mse_delta_spearman",
        "residual_alignment_A_gt1_rate",
        "uncertainty_width_mean",
        "edit_energy_ratio",
        "active_patch_ratio",
    ]
    return "\n".join(
        [
            "# Stage 17 Sequence Teacher Projection",
            "",
            f"Status: `{verdict['status']}`.",
            "",
            "## Overall",
            "",
            df_to_md(overall[show], max_rows=48),
            "",
            "## Gate Table",
            "",
            df_to_md(gates, max_rows=16),
            "",
            "## Mechanism Metrics",
            "",
            df_to_md(mechanism[[c for c in mech_cols if c in mechanism.columns]], max_rows=16),
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
    stage7_dir: Path,
    stage14_dir: Path,
    stage15_dir: Path,
    stage16_dir: Path,
    output_dir: Path,
    stage3_dir: Optional[Path] = None,
    cfg: Optional[Stage17Config] = None,
    n_bootstrap: int = 2000,
) -> Dict[str, Any]:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    start = time.time()
    cfg = cfg or Stage17Config(bootstrap=n_bootstrap)
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("[stage17-seq] preparing assets", flush=True)
    assets = prepare_assets(metrics_csv, stage5_dir, stage3_dir, cfg.seed)
    pools = build_cga_pools(assets)
    write_json(
        output_dir / "stage17_config.json",
        {
            "stage": "stage17_sequence_teacher_projection",
            "source_plan": "halluguard_stage17_sequence_teacher_projection_validation_doc.md",
            "compact_protocol": "ETTm1/ETTh1 x DLinear/PatchTST x horizons 96/192 x seed 2026",
            "teacher_training_source": "validation_inner_train_proxy_no_original_train_assets_available",
            "metrics_csv": metrics_csv,
            "stage5_dir": stage5_dir,
            "stage7_dir": stage7_dir,
            "stage14_dir": stage14_dir,
            "stage15_dir": stage15_dir,
            "stage16_dir": stage16_dir,
            "stage3_dir": stage3_dir,
            "stage17_config": asdict(cfg),
            "resolved_device": str(device),
            "n_bootstrap": n_bootstrap,
            "test_threshold_leakage": False,
        },
    )
    print("[stage17-seq] building sequence feature packs", flush=True)
    train_x = build_feature_matrix(assets.val_train, assets.schema)
    calib_x = build_feature_matrix(assets.val_calib, assets.schema)
    test_x = build_feature_matrix(assets.test, assets.schema)
    train_z, [calib_z, test_z], mean, std = standardize(train_x, calib_x, test_x)
    train_pack = make_pack(assets.val_train, train_z)
    calib_pack = make_pack(assets.val_calib, calib_z)
    test_pack = make_pack(assets.test, test_z)

    print("[stage17-seq] training sequence SSL teacher", flush=True)
    teacher, teacher_log = train_teacher(train_pack, cfg, device)
    teacher_log.to_csv(output_dir / "stage17_training_log_teacher.csv", index=False)

    corrector_logs: List[pd.DataFrame] = []
    print("[stage17-seq] training SL-TMP", flush=True)
    sl_model, log = train_sl_tmp(train_pack, teacher, cfg, device)
    corrector_logs.append(log)
    print("[stage17-seq] training UTRE", flush=True)
    utre_model, log = train_utre(train_pack, cfg, device)
    corrector_logs.append(log)
    print("[stage17-seq] training SSP", flush=True)
    ssp_model, log = train_ssp(train_pack, teacher, cfg, device)
    corrector_logs.append(log)
    pd.concat(corrector_logs, ignore_index=True).to_csv(output_dir / "stage17_training_log_correctors.csv", index=False)

    print("[stage17-seq] predicting raw deltas", flush=True)
    calib_teacher_delta, calib_teacher_aux = teacher_delta(calib_pack, teacher, cfg, device)
    test_teacher_delta, test_teacher_aux = teacher_delta(test_pack, teacher, cfg, device)
    calib_raws: Dict[str, CandidateRaw] = {}
    test_raws: Dict[str, CandidateRaw] = {}
    calib_raws["SL-TMP Sequence Teacher Minimal-Norm Projection"] = predict_sl_tmp(calib_pack, sl_model, teacher, cfg, device)
    test_raws["SL-TMP Sequence Teacher Minimal-Norm Projection"] = predict_sl_tmp(test_pack, sl_model, teacher, cfg, device)
    calib_raws["UTRE Uncertainty Teacher Residual Envelope"] = predict_utre(calib_pack, utre_model, cfg, device)
    test_raws["UTRE Uncertainty Teacher Residual Envelope"] = predict_utre(test_pack, utre_model, cfg, device)
    calib_raws["SSP Structured Sequence Projector"] = predict_ssp(calib_pack, ssp_model, teacher, cfg, device)
    test_raws["SSP Structured Sequence Projector"] = predict_ssp(test_pack, ssp_model, teacher, cfg, device)
    calib_raws["TRAP Teacher-Residual Agreement Projector"] = make_trap(calib_raws["SL-TMP Sequence Teacher Minimal-Norm Projection"], calib_teacher_delta)
    test_raws["TRAP Teacher-Residual Agreement Projector"] = make_trap(test_raws["SL-TMP Sequence Teacher Minimal-Norm Projection"], test_teacher_delta)
    calib_raws["IMDR Iterative Minimal-Norm Denoising Refiner"] = make_imdr(calib_teacher_delta, calib_teacher_aux, cfg)
    test_raws["IMDR Iterative Minimal-Norm Denoising Refiner"] = make_imdr(test_teacher_delta, test_teacher_aux, cfg)

    calib_oracle_mse = oracle_best(deployable_candidates(pools.calib_candidates), assets.val_calib)[1]
    oracle_pred, oracle_mse, _ = oracle_best(deployable_candidates(pools.test_candidates), assets.test)
    preds: Dict[str, np.ndarray] = {
        "LRBN": assets.test.lrbn_pred,
        "SRA-BP-safe": next((c.pred for c in assets.old_test_candidates if c.name == "sra_safe"), assets.test.lrbn_pred),
        "SRA-BP-balanced": next((c.pred for c in assets.old_test_candidates if c.name == "sra_balanced"), assets.test.lrbn_pred),
        "oracle_stage17_cga_full": oracle_pred,
    }
    decisions_by: Dict[str, pd.DataFrame] = {}
    policies: Dict[str, Any] = {}
    grid_frames: List[pd.DataFrame] = []

    print("[stage17-seq] calibrating sequence candidates", flush=True)
    for variant in STAGE17_VARIANTS:
        print(f"[stage17-seq] calibrating {variant}", flush=True)
        policy, grid = calibrate_variant(calib_raws[variant], calib_pack, calib_oracle_mse, cfg.seed)
        grid["target_variant"] = variant
        grid_frames.append(grid)
        policies[variant] = asdict(policy)
        pred, decisions, _ = apply_sequence_policy(test_raws[variant], test_pack, policy, threshold=policy.score_threshold)
        preds[variant] = pred
        decisions_by[variant] = decisions
    pd.concat(grid_frames, ignore_index=True).to_csv(output_dir / "stage17_calibration_grid.csv", index=False)
    write_json(output_dir / "stage17_policies.json", policies)

    print("[stage17-seq] evaluating", flush=True)
    overall_rows = []
    for variant, pred in preds.items():
        decisions = decisions_by.get(variant)
        overall_rows.append(evaluate_variant(variant, pred, assets.test, decisions, oracle_mse if variant != "LRBN" else None, n_bootstrap, cfg.seed))
    overall = pd.DataFrame(overall_rows)
    overall_with_refs = add_reference_rows(overall, stage7_dir, stage14_dir, stage15_dir, stage16_dir)
    overall_with_refs.to_csv(output_dir / "stage17_overall.csv", index=False)

    per_config = pd.concat(
        [per_config_rows(variant, pred, assets.test, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    per_config.to_csv(output_dir / "stage17_per_config.csv", index=False)
    masks = stage8_slice_masks(assets.test, assets.schema, stage8_slice_thresholds(assets.val, assets.schema))
    slice_df = pd.concat(
        [slice_rows(variant, pred, assets.test, masks, decisions_by.get(variant, pd.DataFrame()).get("selected", None)) for variant, pred in preds.items()],
        ignore_index=True,
    )
    slice_df.to_csv(output_dir / "stage17_slice_metrics.csv", index=False)

    mech_rows = []
    for variant in STAGE17_VARIANTS:
        mech_rows.append(mechanism_row(variant, preds[variant], test_raws[variant], test_pack, teacher, cfg, device))
    mechanism = pd.DataFrame(mech_rows)
    mechanism.to_csv(output_dir / "stage17_mechanism_metrics.csv", index=False)
    alignment_cols = [
        "variant",
        "residual_alignment_A_mean",
        "residual_alignment_A_gt1_rate",
        "direction_cosine_RT_mean",
        "teacher_energy_mse_delta_spearman",
    ]
    mechanism[[c for c in alignment_cols if c in mechanism.columns]].to_csv(output_dir / "stage17_alignment_metrics.csv", index=False)
    uncertainty_cols = ["variant", "uncertainty_width_mean", "lrbn_equiv_rate", "active_patch_ratio", "edit_energy_ratio"]
    mechanism[[c for c in uncertainty_cols if c in mechanism.columns]].to_csv(output_dir / "stage17_uncertainty_metrics.csv", index=False)

    boot = {}
    for _, row in overall[overall["variant"].isin(STAGE17_VARIANTS)].iterrows():
        boot[str(row["variant"])] = {
            "ci95_low_delta_raw": float(row.get("ci95_low_delta_raw", np.nan)),
            "ci95_high_delta_raw": float(row.get("ci95_high_delta_raw", np.nan)),
            "p_bootstrap_delta_lt_zero": float(row.get("p_bootstrap_delta_lt_zero", np.nan)),
        }
    write_json(output_dir / "stage17_bootstrap_ci.json", boot)

    gates = gate_table(overall, per_config, slice_df, mechanism)
    gates.to_csv(output_dir / "stage17_gate_table.csv", index=False)
    deployable = overall[overall["variant"].isin(STAGE17_VARIANTS)].copy()
    best = deployable.sort_values(["mse_delta_pct_vs_lrbn", "max_config_harm", "harm_rate"], ascending=[True, True, True]).iloc[0]
    passed = gates[gates["compact_gate_pass"]]
    mech_passed = gates[gates["mechanism_gate_pass"]]
    verdict = {
        "stage": "stage17_sequence_teacher_projection",
        "status": "compact_pass_ready_for_mini_extension" if not passed.empty else "compact_failed_stop_before_mini_extension",
        "compact_pass": bool(not passed.empty),
        "mechanism_pass_any": bool(not mech_passed.empty),
        "passed_variants": passed["variant"].astype(str).tolist(),
        "mechanism_passed_variants": mech_passed["variant"].astype(str).tolist(),
        "best_variant": str(best["variant"]),
        "best_mse": float(best["mse"]),
        "best_mae": float(best["mae"]),
        "best_mse_delta_pct_vs_lrbn": float(best["mse_delta_pct_vs_lrbn"]),
        "best_harm_rate": float(best["harm_rate"]),
        "best_max_config_harm": float(best["max_config_harm"]),
        "best_oracle_gain_fraction": float(best.get("oracle_gain_fraction", np.nan)),
        "teacher_training_source": "validation_inner_train_proxy_no_original_train_assets_available",
        "test_threshold_leakage": False,
        "stop_reason": None,
    }
    if passed.empty:
        verdict["stop_reason"] = "no Stage17 sequence teacher variant passed compact safe/tradeoff gates"
    write_json(output_dir / "stage17_verdict.json", verdict)
    (output_dir / "summary.md").write_text(build_summary(output_dir, verdict, overall_with_refs, gates, mechanism), encoding="utf-8")
    completeness = verify_outputs(output_dir)
    completeness.to_csv(output_dir / "stage17_output_completeness.csv", index=False)
    print(f"[stage17-seq] done in {time.time() - start:.1f}s", flush=True)
    return {
        "verdict": verdict,
        "overall": overall_with_refs,
        "per_config": per_config,
        "slice": slice_df,
        "mechanism": mechanism,
        "gates": gates,
        "completeness": completeness,
    }
