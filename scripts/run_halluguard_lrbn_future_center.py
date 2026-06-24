#!/usr/bin/env python
"""Run HalluGuard-LRBN future-center restoration ablations.

This runner tests the mechanism implied by the LRBN+NST decomposition: the
useful gain mostly came from level/center restoration. It reuses the existing
LRBN data, training, export, and metric contract, and only adds center-focused
normalization variants.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
LRBN_RUNNER_PATH = REPO_ROOT / "scripts" / "run_halluguard_lrbn.py"


def load_lrbn_runner():
    spec = importlib.util.spec_from_file_location("halluguard_lrbn_runner", LRBN_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LRBN runner from {LRBN_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


lrbn = load_lrbn_runner()


class FutureCenterSelector(nn.Module):
    """Select a reversible center from target-free context anchors."""

    def __init__(
        self,
        base: nn.Module,
        tail_len: int,
        init_gamma: float = 0.35,
        hidden: int = 8,
        drift: bool = False,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.base = base
        self.tail_len = int(tail_len)
        self.gamma_logit = nn.Parameter(lrbn.logit_tensor(init_gamma))
        self.selector = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, 4))
        self.drift = bool(drift)
        if self.drift:
            self.drift_head = nn.Sequential(nn.Linear(6, hidden), nn.GELU(), nn.Linear(hidden, 1))
            with torch.no_grad():
                self.drift_head[-1].weight.zero_()
                self.drift_head[-1].bias.zero_()
        self.eps = float(eps)
        with torch.no_grad():
            self.selector[-1].weight.zero_()
            # Start near the current LRBN parent: mostly boundary anchor with
            # some instance mean, little tail/trend.
            self.selector[-1].bias.copy_(torch.tensor([0.0, 1.3, -2.0, -2.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        anchors, scale, features = self._anchors_scale_features(x)
        weights = torch.softmax(self.selector(features), dim=-1).view(x.shape[0], 4, 1, 1)
        center = (weights * anchors).sum(dim=1)
        z = (x - center) / scale
        pred = self.base(z) * scale + center
        if self.drift:
            drift = self.drift_head(features).view(-1, 1, 1) * scale
            ramp = torch.linspace(0.0, 1.0, pred.shape[1], device=pred.device, dtype=pred.dtype).view(1, -1, 1)
            pred = pred + ramp * drift
        return pred

    def _anchors_scale_features(self, x: torch.Tensor):
        last, tail_median, robust_scale, instance = lrbn.context_stats(x, self.tail_len, self.eps)
        instance_mean, instance_std = instance
        tail = x[:, -max(4, min(self.tail_len, x.shape[1])) :, :]
        idx = torch.linspace(-1.0, 0.0, tail.shape[1], device=x.device, dtype=x.dtype).view(1, -1, 1)
        idx = idx - idx.mean(dim=1, keepdim=True)
        denom = torch.sum(idx * idx, dim=1, keepdim=True).clamp_min(self.eps)
        slope = torch.sum((tail - tail.mean(dim=1, keepdim=True)) * idx, dim=1, keepdim=True) / denom
        trend_anchor = last + slope
        boundary_anchor = 0.85 * last + 0.15 * tail_median
        anchors = torch.stack([instance_mean, boundary_anchor, tail_median, trend_anchor], dim=1)
        gamma = torch.sigmoid(self.gamma_logit)
        scale = gamma * robust_scale + (1.0 - gamma) * instance_std
        diff = x[:, 1:, :] - x[:, :-1, :]
        diff_std = torch.sqrt(torch.var(diff, dim=1, keepdim=True, unbiased=False) + self.eps)
        last_diff = torch.abs(x[:, -1:, :] - x[:, -2:-1, :])
        denom_feat = instance_std + self.eps
        features = torch.cat(
            [
                (torch.abs(last - instance_mean) / denom_feat).squeeze(1),
                (torch.abs(tail_median - instance_mean) / denom_feat).squeeze(1),
                (torch.abs(trend_anchor - boundary_anchor) / denom_feat).squeeze(1),
                (robust_scale / denom_feat).squeeze(1),
                (diff_std / denom_feat).squeeze(1),
                (last_diff / denom_feat).squeeze(1),
            ],
            dim=-1,
        ).detach()
        return anchors, scale, features


class FutureCenterStaticMix(nn.Module):
    """Global trainable anchor mixture without feature conditioning."""

    def __init__(self, base: nn.Module, tail_len: int, init_gamma: float = 0.35, drift: bool = False, eps: float = 1e-5):
        super().__init__()
        self.base = base
        self.tail_len = int(tail_len)
        self.gamma_logit = nn.Parameter(lrbn.logit_tensor(init_gamma))
        self.anchor_logits = nn.Parameter(torch.tensor([0.0, 1.3, -2.0, -2.0], dtype=torch.float32))
        self.drift = bool(drift)
        if self.drift:
            self.drift_logit = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last, tail_median, robust_scale, instance = lrbn.context_stats(x, self.tail_len, self.eps)
        instance_mean, instance_std = instance
        tail = x[:, -max(4, min(self.tail_len, x.shape[1])) :, :]
        idx = torch.linspace(-1.0, 0.0, tail.shape[1], device=x.device, dtype=x.dtype).view(1, -1, 1)
        idx = idx - idx.mean(dim=1, keepdim=True)
        denom = torch.sum(idx * idx, dim=1, keepdim=True).clamp_min(self.eps)
        slope = torch.sum((tail - tail.mean(dim=1, keepdim=True)) * idx, dim=1, keepdim=True) / denom
        trend_anchor = last + slope
        boundary_anchor = 0.85 * last + 0.15 * tail_median
        anchors = torch.stack([instance_mean, boundary_anchor, tail_median, trend_anchor], dim=1)
        weights = torch.softmax(self.anchor_logits, dim=0).view(1, 4, 1, 1)
        center = (weights * anchors).sum(dim=1)
        gamma = torch.sigmoid(self.gamma_logit)
        scale = gamma * robust_scale + (1.0 - gamma) * instance_std
        pred = self.base((x - center) / scale) * scale + center
        if self.drift:
            ramp = torch.linspace(0.0, 1.0, pred.shape[1], device=pred.device, dtype=pred.dtype).view(1, -1, 1)
            pred = pred + ramp * torch.tanh(self.drift_logit) * slope
        return pred


ORIGINAL_BUILD_VARIANT_MODEL = lrbn.build_variant_model


def build_variant_model(variant: str, backbone: str, seq_len: int, pred_len: int, tail_len: int, eps: float) -> nn.Module:
    if variant == "future_center_static":
        return FutureCenterStaticMix(lrbn.exporter.build_model(backbone, seq_len, pred_len), tail_len, eps=eps)
    if variant == "future_center_static_drift":
        return FutureCenterStaticMix(lrbn.exporter.build_model(backbone, seq_len, pred_len), tail_len, drift=True, eps=eps)
    if variant == "future_center_selector":
        return FutureCenterSelector(lrbn.exporter.build_model(backbone, seq_len, pred_len), tail_len, eps=eps)
    if variant == "future_center_selector_drift":
        return FutureCenterSelector(lrbn.exporter.build_model(backbone, seq_len, pred_len), tail_len, drift=True, eps=eps)
    return ORIGINAL_BUILD_VARIANT_MODEL(variant, backbone, seq_len, pred_len, tail_len, eps)


def main() -> None:
    lrbn.VARIANTS = tuple(
        dict.fromkeys(
            (
                *lrbn.VARIANTS,
                "future_center_static",
                "future_center_static_drift",
                "future_center_selector",
                "future_center_selector_drift",
            )
        )
    )
    lrbn.build_variant_model = build_variant_model
    lrbn.main()


if __name__ == "__main__":
    main()
