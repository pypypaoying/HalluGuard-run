#!/usr/bin/env python
"""Unified prediction exporter for the 12-method HalluGuard core table.

This runner deliberately uses one shared data/window/training/export contract.
It exports JSONL files for the same dataset/backbone/horizon matrix consumed by
the HalluGuard external evaluator and the core-table aggregator.

The official-method adapters are lightweight fair adapters around the local
DLinear/PatchTST exporter. They are not claimed to reproduce each official
repository's full leaderboard protocol.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = REPO_ROOT / "external" / "halluguard_real_pipeline" / "export_predictions.py"
METHODS = ("RevIN", "DishTS", "SAN", "NST", "TAFAS")
DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "ECL", "Traffic")
MODELS = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "TimeMixer")
HORIZONS = (96, 192, 336, 720)


def load_exporter():
    spec = importlib.util.spec_from_file_location("halluguard_real_exporter", EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load exporter from {EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exporter = load_exporter()


class RevINAdapter(nn.Module):
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


class DishTSAdapter(nn.Module):
    def __init__(self, base: nn.Module, seq_len: int, repo_root: Path):
        super().__init__()
        self.base = base
        module_path = repo_root / "external" / "plugin_baselines" / "Dish-TS" / "DishTS.py"
        spec = importlib.util.spec_from_file_location("dish_ts_official", module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load Dish-TS module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        args = SimpleNamespace(dish_init="standard", n_series=1, seq_len=int(seq_len))
        self.dish = module.DishTS(args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.dish(x, mode="forward")
        pred = self.base(z)
        return self.dish(pred, mode="inverse")


class SANLiteAdapter(nn.Module):
    """A compact SAN-style future-statistics adapter for the shared exporter."""

    def __init__(self, base: nn.Module, seq_len: int, pred_len: int, hidden: int = 64, eps: float = 1e-5):
        super().__init__()
        self.base = base
        self.pred_len = int(pred_len)
        self.eps = float(eps)
        self.mean_head = nn.Sequential(nn.Linear(seq_len, hidden), nn.GELU(), nn.Linear(hidden, pred_len))
        self.std_head = nn.Sequential(nn.Linear(seq_len, hidden), nn.GELU(), nn.Linear(hidden, pred_len), nn.Softplus())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.squeeze(-1)
        context_mean = x.mean(dim=1, keepdim=True).detach()
        centered = x - context_mean
        context_std = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        z = centered / context_std
        pred_z = self.base(z).squeeze(-1)
        future_mean = self.mean_head(flat).unsqueeze(-1)
        future_std = self.std_head(flat).unsqueeze(-1) + self.eps
        return pred_z.unsqueeze(-1) * future_std + future_mean


class NSTAdapter(RevINAdapter):
    """Series-stationarization adapter matching the NST normalization boundary."""


class TAFASLiteAdapter(nn.Module):
    """A bounded test-time calibration adapter with no test-target access."""

    def __init__(self, base: nn.Module, eps: float = 1e-5):
        super().__init__()
        self.base = base
        self.eps = float(eps)
        self.gate = nn.Parameter(torch.tensor(0.05))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pred = self.base(x)
        last = x[:, -1:, :]
        pred_start = pred[:, :1, :]
        boundary_adjust = last - pred_start
        ramp = torch.linspace(1.0, 0.0, pred.shape[1], device=pred.device, dtype=pred.dtype).view(1, -1, 1)
        return pred + torch.tanh(self.gate) * ramp * boundary_adjust


def build_method_model(method: str, backbone: str, seq_len: int, pred_len: int) -> nn.Module:
    base = exporter.build_model(backbone, seq_len, pred_len)
    if method == "RevIN":
        return RevINAdapter(base)
    if method == "DishTS":
        return DishTSAdapter(base, seq_len, REPO_ROOT)
    if method == "SAN":
        return SANLiteAdapter(base, seq_len, pred_len)
    if method == "NST":
        return NSTAdapter(base)
    if method == "TAFAS":
        return TAFASLiteAdapter(base)
    raise ValueError(f"Unknown method: {method}")


@dataclass(frozen=True)
class Job:
    dataset: str
    backbone: str
    horizon: int
    method: str

    @property
    def tag(self) -> str:
        return f"{self.dataset}_{self.backbone}_{self.horizon}_{self.method}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export unified 12-method core-table baseline predictions.")
    parser.add_argument("--datasets", default="ETTm1,ETTh1")
    parser.add_argument("--models", default="DLinear,PatchTST")
    parser.add_argument("--horizons", default="96,192,336,720")
    parser.add_argument("--methods", default="RevIN,DishTS,SAN,NST,TAFAS")
    parser.add_argument("--data-root", type=Path, default=Path("external/ETDataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("baseline_predictions/core_table"))
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-windows", type=int, default=4096)
    parser.add_argument("--max-eval-windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    datasets = parse_list(args.datasets, DATASETS, "dataset")
    models = parse_list(args.models, MODELS, "model")
    horizons = [int(v) for v in parse_list(args.horizons, [str(h) for h in HORIZONS], "horizon")]
    methods = parse_list(args.methods, METHODS, "method")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for job in [Job(d, m, h, method) for d in datasets for m in models for h in horizons for method in methods]:
        try:
            out_path = args.output_dir / f"{job.tag}.jsonl"
            manifest = run_job(job, out_path, args)
            records.append({**manifest, "status": "completed", "blocker_reason": ""})
            print(json.dumps({"status": "completed", "job": job.tag, "output": str(out_path)}), flush=True)
        except Exception as exc:
            record = {
                "dataset": job.dataset,
                "backbone": job.backbone,
                "horizon": job.horizon,
                "method": job.method,
                "model_label": f"{job.backbone}+{job.method}",
                "status": "blocked",
                "output": str(args.output_dir / f"{job.tag}.jsonl"),
                "blocker_reason": f"{type(exc).__name__}: {exc}",
                "adapter_mode": "lightweight_fair_adapter",
            }
            records.append(record)
            print(json.dumps({"status": "blocked", "job": job.tag, "reason": record["blocker_reason"]}), flush=True)
            if not args.continue_on_error:
                raise
    write_csv(records, args.output_dir / "manifest.csv")
    print(json.dumps({"output_dir": str(args.output_dir), "completed": sum(r["status"] == "completed" for r in records), "total": len(records)}))


def run_job(job: Job, out_path: Path, args: argparse.Namespace) -> dict:
    exporter.set_seed(args.seed + stable_offset(job.tag))
    device = exporter.choose_device(args.device)
    raw_series, scaler = exporter.load_series(job.dataset, args.data_root)
    scaled = scaler.transform(raw_series).astype(np.float32)
    train_starts = exporter.select_starts(exporter.split_starts(job.dataset, args.seq_len, job.horizon, "train"), args.max_train_windows)
    val_starts = exporter.select_starts(exporter.split_starts(job.dataset, args.seq_len, job.horizon, "val"), args.max_eval_windows)
    test_starts = exporter.select_starts(exporter.split_starts(job.dataset, args.seq_len, job.horizon, "test"), args.max_eval_windows)

    model = build_method_model(job.method, job.backbone, args.seq_len, job.horizon).to(device)
    exporter.train_model(model, scaled, train_starts, args.seq_len, job.horizon, args.epochs, args.batch_size, args.learning_rate, device)
    samples = []
    samples.extend(export_split(model, raw_series, scaled, scaler, val_starts, args, job, "val", device))
    samples.extend(export_split(model, raw_series, scaled, scaler, test_starts, args, job, "test", device))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": job.dataset,
        "backbone": job.backbone,
        "horizon": job.horizon,
        "method": job.method,
        "model_label": f"{job.backbone}+{job.method}",
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_train_windows": args.max_train_windows,
        "max_eval_windows": args.max_eval_windows,
        "device": str(device),
        "n_val": len(val_starts),
        "n_test": len(test_starts),
        "output": str(out_path),
        "adapter_mode": "lightweight_fair_adapter",
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def export_split(model: nn.Module, raw_series: np.ndarray, scaled: np.ndarray, scaler, starts: List[int], args, job: Job, split: str, device: torch.device) -> List[dict]:
    dataset = exporter.WindowDataset(scaled, starts, args.seq_len, job.horizon)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    samples = []
    model.eval()
    offset = 0
    with torch.no_grad():
        for x, _ in loader:
            pred = model(x.to(device)).detach().cpu().numpy()[:, :, 0]
            pred = scaler.inverse(pred)
            batch_size = pred.shape[0]
            for b in range(batch_size):
                start = starts[offset + b]
                context = raw_series[start : start + args.seq_len].astype(float)
                target = raw_series[start + args.seq_len : start + args.seq_len + job.horizon].astype(float)
                samples.append(
                    {
                        "sample_id": f"{split}_{offset + b:05d}",
                        "dataset": job.dataset,
                        "model": f"{job.backbone}+{job.method}",
                        "split": split,
                        "context": exporter.round_list(context),
                        "prediction": exporter.round_list(pred[b]),
                        "target": exporter.round_list(target),
                        "backbone": job.backbone,
                        "method": job.method,
                        "adapter_mode": "lightweight_fair_adapter",
                    }
                )
            offset += batch_size
    return samples


def parse_list(raw: str, allowed: Sequence[str], name: str) -> List[str]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise SystemExit(f"Unknown {name}(s): {', '.join(unknown)}. Allowed: {', '.join(map(str, allowed))}")
    return values


def stable_offset(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 100000


def write_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
