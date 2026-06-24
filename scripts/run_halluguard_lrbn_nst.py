#!/usr/bin/env python
"""Run HalluGuard-LRBN + NST complementarity ablations.

This script intentionally reuses the HalluGuard-LRBN runner's data, training,
export, and metric contract. It only adds NST-complementarity variants around
the claim-clean `unified_revin_rdn_hybrid` parent.
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


class NSTLightweight(nn.Module):
    """NST-style series stationarization boundary used in the core12 adapter."""

    def __init__(self, base: nn.Module, eps: float = 1e-5):
        super().__init__()
        self.base = base
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True).detach()
        centered = x - mean
        std = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        pred = self.base(centered / std)
        return pred * std[:, :1, :] + mean[:, :1, :]


class LRBNUnifiedNSTResidual(nn.Module):
    """Apply NST-style stationarization inside LRBN normalized coordinates."""

    def __init__(
        self,
        base: nn.Module,
        tail_len: int,
        init_beta: float = 0.7,
        init_gamma: float = 0.35,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.base = base
        self.tail_len = int(tail_len)
        self.beta_logit = nn.Parameter(lrbn.logit_tensor(init_beta))
        self.gamma_logit = nn.Parameter(lrbn.logit_tensor(init_gamma))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last, tail_median, robust_scale, instance = lrbn.context_stats(x, self.tail_len, self.eps)
        instance_mean, instance_std = instance
        beta = torch.sigmoid(self.beta_logit)
        gamma = torch.sigmoid(self.gamma_logit)
        boundary_anchor = 0.85 * last + 0.15 * tail_median
        center = beta * boundary_anchor + (1.0 - beta) * instance_mean
        scale = gamma * robust_scale + (1.0 - gamma) * instance_std

        z = (x - center) / scale
        residual_mean = z.mean(dim=1, keepdim=True).detach()
        residual_centered = z - residual_mean
        residual_std = torch.sqrt(torch.var(residual_centered, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        pred_z = self.base(residual_centered / residual_std)
        pred_lrbn = pred_z * residual_std[:, :1, :] + residual_mean[:, :1, :]
        return pred_lrbn * scale + center


class LRBNNSTOutputBlend(nn.Module):
    """Diagnostic only: train-split blend between LRBN and NST branches."""

    def __init__(self, lrbn_base: nn.Module, nst_base: nn.Module, tail_len: int, init_blend: float = 0.5, eps: float = 1e-5):
        super().__init__()
        self.lrbn_branch = lrbn.UnifiedRevINRDNHybrid(lrbn_base, tail_len, eps=eps)
        self.nst_branch = NSTLightweight(nst_base, eps=eps)
        self.blend_logit = nn.Parameter(lrbn.logit_tensor(init_blend))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        blend = torch.sigmoid(self.blend_logit)
        return blend * self.lrbn_branch(x) + (1.0 - blend) * self.nst_branch(x)


class LRBNNSTFeatureGate(nn.Module):
    """Train-split context gate that chooses between LRBN and NST branches."""

    def __init__(
        self,
        lrbn_base: nn.Module,
        nst_base: nn.Module,
        tail_len: int,
        hidden: int = 8,
        init_lrbn_weight: float = 0.75,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.lrbn_branch = lrbn.UnifiedRevINRDNHybrid(lrbn_base, tail_len, eps=eps)
        self.nst_branch = NSTLightweight(nst_base, eps=eps)
        self.tail_len = int(tail_len)
        self.eps = float(eps)
        self.gate = nn.Sequential(nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, 1))
        with torch.no_grad():
            self.gate[-1].weight.zero_()
            self.gate[-1].bias.fill_(lrbn.logit(init_lrbn_weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lrbn_pred = self.lrbn_branch(x)
        nst_pred = self.nst_branch(x)
        features = self._features(x)
        lrbn_weight = torch.sigmoid(self.gate(features)).view(-1, 1, 1)
        return lrbn_weight * lrbn_pred + (1.0 - lrbn_weight) * nst_pred

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        last, tail_median, robust_scale, instance = lrbn.context_stats(x, self.tail_len, self.eps)
        instance_mean, instance_std = instance
        diff = x[:, 1:, :] - x[:, :-1, :]
        diff_std = torch.sqrt(torch.var(diff, dim=1, keepdim=True, unbiased=False) + self.eps)
        last_diff = torch.abs(x[:, -1:, :] - x[:, -2:-1, :])
        denom = instance_std + self.eps
        return torch.cat(
            [
                (robust_scale / denom).squeeze(1),
                (torch.abs(last - instance_mean) / denom).squeeze(1),
                (torch.abs(last - tail_median) / denom).squeeze(1),
                (diff_std / denom).squeeze(1),
                (last_diff / denom).squeeze(1),
            ],
            dim=-1,
        ).detach()


ORIGINAL_BUILD_VARIANT_MODEL = lrbn.build_variant_model


def build_variant_model(variant: str, backbone: str, seq_len: int, pred_len: int, tail_len: int, eps: float) -> nn.Module:
    if variant == "nst_lightweight":
        return NSTLightweight(lrbn.exporter.build_model(backbone, seq_len, pred_len), eps=eps)
    if variant == "lrbn_unified_nst_residual":
        return LRBNUnifiedNSTResidual(lrbn.exporter.build_model(backbone, seq_len, pred_len), tail_len, eps=eps)
    if variant == "lrbn_nst_output_blend":
        return LRBNNSTOutputBlend(
            lrbn.exporter.build_model(backbone, seq_len, pred_len),
            lrbn.exporter.build_model(backbone, seq_len, pred_len),
            tail_len,
            eps=eps,
        )
    if variant == "lrbn_nst_feature_gate":
        return LRBNNSTFeatureGate(
            lrbn.exporter.build_model(backbone, seq_len, pred_len),
            lrbn.exporter.build_model(backbone, seq_len, pred_len),
            tail_len,
            eps=eps,
        )
    if variant == "lrbn_nst_conservative_gate":
        return LRBNNSTFeatureGate(
            lrbn.exporter.build_model(backbone, seq_len, pred_len),
            lrbn.exporter.build_model(backbone, seq_len, pred_len),
            tail_len,
            init_lrbn_weight=0.95,
            eps=eps,
        )
    return ORIGINAL_BUILD_VARIANT_MODEL(variant, backbone, seq_len, pred_len, tail_len, eps)


def main() -> None:
    lrbn.VARIANTS = tuple(dict.fromkeys((*lrbn.VARIANTS, "nst_lightweight", "lrbn_unified_nst_residual", "lrbn_nst_output_blend", "lrbn_nst_feature_gate", "lrbn_nst_conservative_gate")))
    lrbn.build_variant_model = build_variant_model
    lrbn.main()


if __name__ == "__main__":
    main()
