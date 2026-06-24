#!/usr/bin/env python
"""Run HalluGuard Reversible Dynamics Normalization (RDN).

HalluGuard-RDN is a RevIN-like input/output wrapper that replaces window
mean/std normalization with a context-only local dynamics baseline.  For each
sample, it fits a level/slope/scale policy from the context, trains the
backbone in residual space, and inverses the transform on the forecast:

    z_context = (context - dynamics_baseline_context) / scale
    z_future_hat = model(z_context)
    y_hat = dynamics_baseline_future + scale * z_future_hat

The target is transformed with the same context-only baseline during training,
so no future statistics or test targets are used to fit the reversible
normalizer.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = REPO_ROOT / "external" / "halluguard_real_pipeline" / "export_predictions.py"
DATASETS = ("ETTm1", "ETTh1")
MODELS = ("DLinear", "PatchTST")
HORIZONS = (96, 192, 336, 720)
VARIANTS = ("level_only", "level_scale", "level_slope", "level_slope_scale")


def load_exporter():
    spec = importlib.util.spec_from_file_location("halluguard_real_exporter", EXPORTER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load exporter from {EXPORTER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


exporter = load_exporter()


@dataclass(frozen=True)
class DynamicsTransform:
    baseline_context: np.ndarray
    baseline_future: np.ndarray
    scale: float
    slope: float
    level: float


@dataclass(frozen=True)
class Job:
    dataset: str
    backbone: str
    horizon: int
    variant: str

    @property
    def tag(self) -> str:
        return f"{self.dataset}_{self.backbone}_{self.horizon}_{self.variant}"

    @property
    def model_label(self) -> str:
        return f"{self.backbone}+HalluGuard-RDN-{self.variant}"


class RDNWindowDataset(Dataset):
    def __init__(
        self,
        scaled_series: np.ndarray,
        starts: Iterable[int],
        seq_len: int,
        pred_len: int,
        variant: str,
        tail_len: int,
        slope_shrink: float,
        scale_eps: float,
        max_scale: float,
    ):
        self.scaled_series = scaled_series.astype(np.float32)
        self.starts = list(starts)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.variant = variant
        self.tail_len = int(tail_len)
        self.slope_shrink = float(slope_shrink)
        self.scale_eps = float(scale_eps)
        self.max_scale = float(max_scale)

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        context = self.scaled_series[start : start + self.seq_len].astype(np.float32)
        target = self.scaled_series[start + self.seq_len : start + self.seq_len + self.pred_len].astype(np.float32)
        transform = fit_dynamics_transform(
            context=context,
            pred_len=self.pred_len,
            variant=self.variant,
            tail_len=self.tail_len,
            slope_shrink=self.slope_shrink,
            scale_eps=self.scale_eps,
            max_scale=self.max_scale,
        )
        x_z = (context - transform.baseline_context) / transform.scale
        y_z = (target - transform.baseline_future) / transform.scale
        return torch.from_numpy(x_z[:, None].astype(np.float32)), torch.from_numpy(y_z[:, None].astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HalluGuard Reversible Dynamics Normalization table.")
    parser.add_argument("--datasets", default="ETTm1,ETTh1")
    parser.add_argument("--models", default="DLinear,PatchTST")
    parser.add_argument("--horizons", default="96,192,336,720")
    parser.add_argument("--variants", default="level_slope_scale")
    parser.add_argument("--data-root", type=Path, default=Path("external/ETDataset"))
    parser.add_argument("--prediction-dir", type=Path, default=Path("baseline_predictions/halluguard_rdn"))
    parser.add_argument("--raw-prediction-dir", type=Path, default=Path("baseline_predictions/halluguard_rdn_raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halluguard/results/halluguard_rdn"))
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--tail-len", type=int, default=48)
    parser.add_argument("--slope-shrink", type=float, default=0.7)
    parser.add_argument("--scale-eps", type=float, default=1e-5)
    parser.add_argument("--max-scale", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-train-windows", type=int, default=4096)
    parser.add_argument("--max-eval-windows", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip-raw-baseline", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    datasets = parse_list(args.datasets, DATASETS, "dataset")
    models = parse_list(args.models, MODELS, "model")
    horizons = [int(v) for v in parse_list(args.horizons, [str(h) for h in HORIZONS], "horizon")]
    variants = parse_list(args.variants, VARIANTS, "variant")

    args.prediction_dir.mkdir(parents=True, exist_ok=True)
    args.raw_prediction_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    raw_metrics = {}
    if not args.skip_raw_baseline:
        for dataset in datasets:
            for backbone in models:
                for horizon in horizons:
                    raw_key = (dataset, backbone, horizon)
                    raw_path = args.raw_prediction_dir / f"{dataset}_{backbone}_{horizon}_raw_no_correction.jsonl"
                    try:
                        raw_manifest = run_raw_job(dataset, backbone, horizon, raw_path, args)
                        raw_metric = prediction_metric_row(raw_path)
                        raw_metrics[raw_key] = raw_metric
                        records.append(
                            {
                                **raw_manifest,
                                **raw_metric,
                                "variant": "raw_no_correction",
                                "method": "raw_no_correction",
                                "model_label": f"{backbone}+raw_no_correction",
                                "mse_delta_pct_vs_raw": 0.0,
                                "mae_delta_pct_vs_raw": 0.0,
                                "status": "completed",
                                "blocker_reason": "",
                            }
                        )
                        print(json.dumps({"status": "completed", "job": f"{dataset}_{backbone}_{horizon}_raw", "mse": raw_metric["mse"], "mae": raw_metric["mae"]}), flush=True)
                    except Exception as exc:
                        records.append(
                            {
                                "dataset": dataset,
                                "backbone": backbone,
                                "horizon": horizon,
                                "variant": "raw_no_correction",
                                "method": "raw_no_correction",
                                "model_label": f"{backbone}+raw_no_correction",
                                "status": "blocked",
                                "mse": "",
                                "mae": "",
                                "mse_delta_pct_vs_raw": "",
                                "mae_delta_pct_vs_raw": "",
                                "prediction_path": str(raw_path),
                                "blocker_reason": f"{type(exc).__name__}: {exc}",
                                "adapter_mode": "raw_backbone",
                                "test_threshold_leakage": False,
                            }
                        )
                        print(json.dumps({"status": "blocked", "job": f"{dataset}_{backbone}_{horizon}_raw", "reason": f"{type(exc).__name__}: {exc}"}), flush=True)
                        if not args.continue_on_error:
                            raise

    for job in [Job(d, m, h, v) for d in datasets for m in models for h in horizons for v in variants]:
        out_path = args.prediction_dir / f"{job.tag}.jsonl"
        try:
            manifest = run_job(job, out_path, args)
            metric = prediction_metric_row(out_path)
            raw_metric = raw_metrics.get((job.dataset, job.backbone, job.horizon))
            deltas = raw_deltas(metric, raw_metric)
            record = {**manifest, **metric, **deltas, "status": "completed", "blocker_reason": ""}
            records.append(record)
            print(json.dumps({"status": "completed", "job": job.tag, "mse": metric["mse"], "mae": metric["mae"]}), flush=True)
        except Exception as exc:
            record = {
                "dataset": job.dataset,
                "backbone": job.backbone,
                "horizon": job.horizon,
                "variant": job.variant,
                "method": "HalluGuard-RDN",
                "model_label": job.model_label,
                "status": "blocked",
                "mse": "",
                "mae": "",
                "mse_delta_pct_vs_raw": "",
                "mae_delta_pct_vs_raw": "",
                "prediction_path": str(out_path),
                "blocker_reason": f"{type(exc).__name__}: {exc}",
                "adapter_mode": "reversible_dynamics_normalization",
                "test_threshold_leakage": False,
            }
            records.append(record)
            print(json.dumps({"status": "blocked", "job": job.tag, "reason": record["blocker_reason"]}), flush=True)
            if not args.continue_on_error:
                raise

    write_csv(records, args.output_dir / "rdn_metrics.csv")
    summary = summarize(records)
    write_csv(summary, args.output_dir / "rdn_summary.csv")
    (args.output_dir / "rdn_metrics.json").write_text(json.dumps({"rows": records, "summary": summary}, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(summary_md(records, summary, args), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "completed": sum(r["status"] == "completed" for r in records), "total": len(records)}))


def run_job(job: Job, out_path: Path, args: argparse.Namespace) -> dict:
    set_seed(args.seed + stable_offset(job.tag))
    device = exporter.choose_device(args.device)
    raw_series, scaler = exporter.load_series(job.dataset, args.data_root)
    scaled = scaler.transform(raw_series).astype(np.float32)
    train_starts = exporter.select_starts(exporter.split_starts(job.dataset, args.seq_len, job.horizon, "train"), args.max_train_windows)
    val_starts = exporter.select_starts(exporter.split_starts(job.dataset, args.seq_len, job.horizon, "val"), args.max_eval_windows)
    test_starts = exporter.select_starts(exporter.split_starts(job.dataset, args.seq_len, job.horizon, "test"), args.max_eval_windows)

    model = exporter.build_model(job.backbone, args.seq_len, job.horizon).to(device)
    train_rdn_model(model, scaled, train_starts, args.seq_len, job.horizon, job.variant, args, device)

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
        "variant": job.variant,
        "method": "HalluGuard-RDN",
        "model_label": job.model_label,
        "seq_len": args.seq_len,
        "tail_len": args.tail_len,
        "slope_shrink": args.slope_shrink,
        "scale_eps": args.scale_eps,
        "max_scale": args.max_scale,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_train_windows": args.max_train_windows,
        "max_eval_windows": args.max_eval_windows,
        "device": str(device),
        "n_val": len(val_starts),
        "n_test": len(test_starts),
        "prediction_path": str(out_path),
        "adapter_mode": "reversible_dynamics_normalization",
        "test_threshold_leakage": False,
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_raw_job(dataset: str, backbone: str, horizon: int, out_path: Path, args: argparse.Namespace) -> dict:
    tag = f"{dataset}_{backbone}_{horizon}_raw_no_correction"
    exporter.set_seed(args.seed + stable_offset(tag))
    device = exporter.choose_device(args.device)
    raw_series, scaler = exporter.load_series(dataset, args.data_root)
    scaled = scaler.transform(raw_series).astype(np.float32)
    train_starts = exporter.select_starts(exporter.split_starts(dataset, args.seq_len, horizon, "train"), args.max_train_windows)
    val_starts = exporter.select_starts(exporter.split_starts(dataset, args.seq_len, horizon, "val"), args.max_eval_windows)
    test_starts = exporter.select_starts(exporter.split_starts(dataset, args.seq_len, horizon, "test"), args.max_eval_windows)

    model = exporter.build_model(backbone, args.seq_len, horizon).to(device)
    exporter.train_model(model, scaled, train_starts, args.seq_len, horizon, args.epochs, args.batch_size, args.learning_rate, device)
    raw_args = SimpleNamespace(seq_len=args.seq_len, horizon=horizon, batch_size=args.batch_size, dataset=dataset, model=f"{backbone}+raw_no_correction")
    samples = []
    samples.extend(exporter.export_split(model, raw_series, scaled, scaler, val_starts, raw_args, "val", device))
    samples.extend(exporter.export_split(model, raw_series, scaled, scaler, test_starts, raw_args, "test", device))
    for sample in samples:
        sample["backbone"] = backbone
        sample["method"] = "raw_no_correction"
        sample["variant"] = "raw_no_correction"
        sample["adapter_mode"] = "raw_backbone"
        sample["test_threshold_leakage"] = False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    manifest = {
        "dataset": dataset,
        "backbone": backbone,
        "horizon": horizon,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_train_windows": args.max_train_windows,
        "max_eval_windows": args.max_eval_windows,
        "device": str(device),
        "n_val": len(val_starts),
        "n_test": len(test_starts),
        "prediction_path": str(out_path),
        "adapter_mode": "raw_backbone",
        "test_threshold_leakage": False,
    }
    out_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def train_rdn_model(
    model: nn.Module,
    scaled: np.ndarray,
    starts: List[int],
    seq_len: int,
    pred_len: int,
    variant: str,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    dataset = RDNWindowDataset(
        scaled,
        starts,
        seq_len,
        pred_len,
        variant,
        args.tail_len,
        args.slope_shrink,
        args.scale_eps,
        args.max_scale,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(args.epochs):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()


def export_split(
    model: nn.Module,
    raw_series: np.ndarray,
    scaled: np.ndarray,
    scaler,
    starts: List[int],
    args: argparse.Namespace,
    job: Job,
    split: str,
    device: torch.device,
) -> List[dict]:
    dataset = RDNWindowDataset(
        scaled,
        starts,
        args.seq_len,
        job.horizon,
        job.variant,
        args.tail_len,
        args.slope_shrink,
        args.scale_eps,
        args.max_scale,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)
    samples = []
    model.eval()
    offset = 0
    with torch.no_grad():
        for x_z, _ in loader:
            pred_z = model(x_z.to(device)).detach().cpu().numpy()[:, :, 0]
            batch_size = pred_z.shape[0]
            for b in range(batch_size):
                start = starts[offset + b]
                context_scaled = scaled[start : start + args.seq_len].astype(np.float32)
                transform = fit_dynamics_transform(
                    context=context_scaled,
                    pred_len=job.horizon,
                    variant=job.variant,
                    tail_len=args.tail_len,
                    slope_shrink=args.slope_shrink,
                    scale_eps=args.scale_eps,
                    max_scale=args.max_scale,
                )
                pred_scaled = transform.baseline_future + transform.scale * pred_z[b]
                pred_raw = scaler.inverse(pred_scaled)
                context = raw_series[start : start + args.seq_len].astype(float)
                target = raw_series[start + args.seq_len : start + args.seq_len + job.horizon].astype(float)
                samples.append(
                    {
                        "sample_id": f"{split}_{offset + b:05d}",
                        "dataset": job.dataset,
                        "model": job.model_label,
                        "split": split,
                        "context": exporter.round_list(context),
                        "prediction": exporter.round_list(pred_raw),
                        "target": exporter.round_list(target),
                        "backbone": job.backbone,
                        "method": "HalluGuard-RDN",
                        "variant": job.variant,
                        "adapter_mode": "reversible_dynamics_normalization",
                        "normalizer_fit": "context_only",
                        "test_threshold_leakage": False,
                    }
                )
            offset += batch_size
    return samples


def fit_dynamics_transform(
    context: np.ndarray,
    pred_len: int,
    variant: str,
    tail_len: int,
    slope_shrink: float,
    scale_eps: float,
    max_scale: float,
) -> DynamicsTransform:
    context = np.asarray(context, dtype=np.float32)
    seq_len = len(context)
    tail_len = max(4, min(int(tail_len), seq_len))
    tail = context[-tail_len:]
    t_tail = np.arange(-tail_len + 1, 1, dtype=np.float32)

    use_slope = "slope" in variant
    use_scale = "scale" in variant
    if use_slope:
        t_centered = t_tail - float(t_tail.mean())
        denom = float(np.sum(t_centered**2)) + 1e-12
        raw_slope = float(np.sum(t_centered * (tail - float(tail.mean()))) / denom)
        slope = raw_slope * float(slope_shrink)
        level = float(tail.mean() - slope * float(t_tail.mean()))
    else:
        slope = 0.0
        level = float(context[-1])

    t_context = np.arange(-seq_len + 1, 1, dtype=np.float32)
    t_future = np.arange(1, pred_len + 1, dtype=np.float32)
    baseline_context = (level + slope * t_context).astype(np.float32)
    baseline_future = (level + slope * t_future).astype(np.float32)

    if use_scale:
        tail_baseline = level + slope * t_tail
        residual_tail = tail - tail_baseline.astype(np.float32)
        scale = float(np.std(residual_tail))
        fallback = float(np.std(tail - float(tail.mean())))
        if not math.isfinite(scale) or scale < scale_eps:
            scale = fallback
        if not math.isfinite(scale) or scale < scale_eps:
            scale = 1.0
        scale = float(np.clip(scale, scale_eps, max_scale))
    else:
        scale = 1.0
    return DynamicsTransform(baseline_context, baseline_future, scale, slope, level)


def prediction_metric_row(path: Path) -> dict:
    samples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    test = [s for s in samples if s.get("split") == "test"]
    if not test:
        raise ValueError(f"{path} has no test rows.")
    preds = np.asarray([s["prediction"] for s in test], dtype=float)
    targets = np.asarray([s["target"] for s in test], dtype=float)
    return {
        "mse": float(np.mean((preds - targets) ** 2)),
        "mae": float(np.mean(np.abs(preds - targets))),
        "n_test": len(test),
    }


def raw_deltas(metric: dict, raw_metric: dict | None) -> dict:
    if not raw_metric:
        return {"mse_delta_pct_vs_raw": "", "mae_delta_pct_vs_raw": ""}
    return {
        "mse_delta_pct_vs_raw": pct_delta(float(metric["mse"]), float(raw_metric["mse"])),
        "mae_delta_pct_vs_raw": pct_delta(float(metric["mae"]), float(raw_metric["mae"])),
    }


def pct_delta(value: float, baseline: float) -> float:
    if abs(baseline) < 1e-12:
        return 0.0
    return float(100.0 * (value - baseline) / baseline)


def summarize(rows: List[dict]) -> List[dict]:
    out = []
    keys = sorted({(str(r.get("variant", "")), str(r.get("backbone", ""))) for r in rows})
    for variant, backbone in keys:
        selected = [r for r in rows if r.get("variant") == variant and r.get("backbone") == backbone]
        completed = [r for r in selected if r.get("status") == "completed"]
        out.append(
            {
                "variant": variant,
                "backbone": backbone,
                "completed_rows": len(completed),
                "total_rows": len(selected),
                "mean_mse": mean(float(r["mse"]) for r in completed) if completed else "",
                "mean_mae": mean(float(r["mae"]) for r in completed) if completed else "",
                "mean_mse_delta_pct_vs_raw": safe_mean_delta(completed, "mse_delta_pct_vs_raw"),
                "mean_mae_delta_pct_vs_raw": safe_mean_delta(completed, "mae_delta_pct_vs_raw"),
                "blocked_rows": len(selected) - len(completed),
            }
        )
    for variant in sorted({str(r.get("variant", "")) for r in rows}):
        selected = [r for r in rows if r.get("variant") == variant]
        completed = [r for r in selected if r.get("status") == "completed"]
        out.append(
            {
                "variant": variant,
                "backbone": "ALL",
                "completed_rows": len(completed),
                "total_rows": len(selected),
                "mean_mse": mean(float(r["mse"]) for r in completed) if completed else "",
                "mean_mae": mean(float(r["mae"]) for r in completed) if completed else "",
                "mean_mse_delta_pct_vs_raw": safe_mean_delta(completed, "mse_delta_pct_vs_raw"),
                "mean_mae_delta_pct_vs_raw": safe_mean_delta(completed, "mae_delta_pct_vs_raw"),
                "blocked_rows": len(selected) - len(completed),
            }
        )
    return out


def summary_md(rows: List[dict], summary: List[dict], args: argparse.Namespace) -> str:
    completed = sum(r.get("status") == "completed" for r in rows)
    lines = [
        "# HalluGuard-RDN Summary",
        "",
        "HalluGuard-RDN is a reversible, context-only input/output dynamics normalizer.",
        "It is RevIN-like in placement but uses a local level/slope/scale dynamics baseline instead of future-aware statistics.",
        "",
        f"- Completed rows: {completed} / {len(rows)}",
        f"- Test threshold leakage: False",
        f"- seq_len: {args.seq_len}",
        f"- tail_len: {args.tail_len}",
        f"- slope_shrink: {args.slope_shrink}",
        "",
        "## Variant Summary",
        "",
    ]
    for row in summary:
        lines.append(
            f"- `{row['variant']}` / `{row['backbone']}`: completed {row['completed_rows']} / {row['total_rows']}, "
            f"mean MSE {row['mean_mse']}, mean MAE {row['mean_mae']}, "
            f"mean MSE delta vs raw {row['mean_mse_delta_pct_vs_raw']}, blocked {row['blocked_rows']}"
        )
    blocked = [r for r in rows if r.get("status") != "completed"]
    if blocked:
        lines.extend(["", "## Blocked Rows", ""])
        for row in blocked[:100]:
            lines.append(f"- {row.get('dataset')} {row.get('backbone')} {row.get('horizon')} `{row.get('variant')}`: {row.get('blocker_reason')}")
    return "\n".join(lines) + "\n"


def parse_list(raw: str, allowed: Sequence[str], name: str) -> List[str]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise SystemExit(f"Unknown {name}(s): {', '.join(unknown)}. Allowed: {', '.join(map(str, allowed))}")
    return values


def stable_offset(text: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(text)) % 100000


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def safe_mean_delta(rows: List[dict], key: str):
    values = []
    for row in rows:
        value = row.get(key, "")
        if value == "":
            continue
        values.append(float(value))
    return mean(values) if values else ""


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
