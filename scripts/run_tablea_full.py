#!/usr/bin/env python
"""Run the full HalluGuard-LRBN Table A matrix.

This is the no-shortcut server orchestrator for the clean-claim Table A.
It expands the complete requested dataset/backbone/horizon/seed/method matrix
and records every row as either completed or blocked with a reproducible reason.

The main method is frozen as:
    HalluGuard-LRBN unified_revin_rdn_hybrid

Table A intentionally excludes online/partially-observed-target protocols such as
TAFAS by default. Those can still be passed explicitly through --methods for an
appendix run, but they are not part of the clean offline Table A mean.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_RUNNER_PATH = REPO_ROOT / "scripts" / "run_lrbn_clean_claim_bigtable.py"
SRA_BP_PATH = REPO_ROOT / "experiments" / "halluguard" / "halluguard_sra_bp.py"
LRBN_SCRIPT = REPO_ROOT / "scripts" / "run_halluguard_lrbn.py"
CORE12_SCRIPT = REPO_ROOT / "scripts" / "run_core12_predictions.py"
FETCH_DATA_SCRIPT = REPO_ROOT / "scripts" / "fetch_core_datasets.py"
FETCH_PLUGIN_REPOS_SCRIPT = REPO_ROOT / "scripts" / "fetch_plugin_repos.sh"

DEFAULT_DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "Weather", "Exchange", "ECL", "Traffic")
DEFAULT_BACKBONES = ("DLinear", "PatchTST", "iTransformer", "TimesNet", "TimeMixer", "FreTS")
DEFAULT_HORIZONS = (96, 192, 336, 720)
DEFAULT_SEEDS = (2026, 2027, 2028)

TABLEA_METHODS = (
    "raw_no_correction",
    "HalluGuard-LRBN",
    "Safe-SRA",
    "Balanced-SRA",
    "RevIN",
    "DishTS",
    "SAN",
    "NST",
    "SoP-step-wise",
    "SoP-variable-wise",
    "SOLID-official-supported",
    "matched_sparse_smoothing",
    "naive_smoothing",
    "ema_smoothing",
    "median_smoothing",
)

ADAPTER_METHODS = (
    "RevIN",
    "DishTS",
    "SAN",
    "NST",
    "TAFAS",
    "SoP-step-wise",
    "SoP-variable-wise",
    "SOLID-official-supported",
)
SMOOTHING_METHODS = ("matched_sparse_smoothing", "naive_smoothing", "ema_smoothing", "median_smoothing")
SRA_METHODS = ("Safe-SRA", "Balanced-SRA", "LRBN-SRA-BP-safe", "LRBN-SRA-BP-balanced")
SRA_PARAM_FILES = {
    "Safe-SRA": "stage5_selected_safe_params.json",
    "LRBN-SRA-BP-safe": "stage5_selected_safe_params.json",
    "Balanced-SRA": "stage5_selected_balanced_params.json",
    "LRBN-SRA-BP-balanced": "stage5_selected_balanced_params.json",
}
SRA_FALLBACK_PARAMS = {
    "Safe-SRA": {
        "method_family": "short",
        "anchor_mode": "last",
        "tail_len": 16,
        "tau_g": 5.265299801054961,
        "tau_r": 0.8,
        "tau_j": None,
        "alpha": 0.75,
        "K": "H_div_4",
        "continuous": False,
        "kg": 4.0,
        "kr": 4.0,
        "kj": 4.0,
    },
    "Balanced-SRA": {
        "method_family": "support",
        "anchor_mode": "last",
        "tail_len": 16,
        "tau_g": 2.4260872328869336,
        "tau_r": 0.8,
        "tau_j": 0.3,
        "alpha": 0.75,
        "K": "H_div_4",
        "continuous": False,
        "kg": 4.0,
        "kr": 4.0,
        "kj": 4.0,
    },
}


def load_base_runner():
    spec = importlib.util.spec_from_file_location("halluguard_clean_claim_runner", BASE_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import base runner: {BASE_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = load_base_runner()


def load_sra_bp():
    spec = importlib.util.spec_from_file_location("halluguard_sra_bp_core", SRA_BP_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import SRA-BP core: {SRA_BP_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sra_bp = load_sra_bp()


@dataclass(frozen=True)
class Config:
    dataset: str
    backbone: str
    horizon: int
    seed: int

    @property
    def tag(self) -> str:
        return f"{self.dataset}_{self.backbone}_{self.horizon}_seed{self.seed}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full HalluGuard-LRBN Table A.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--backbones", default=",".join(DEFAULT_BACKBONES))
    parser.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--methods", default=",".join(TABLEA_METHODS))
    parser.add_argument("--data-root", type=Path, default=Path("external/ETDataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/halluguard/results/tablea_full"))
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--tail-len", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--san-period-len", type=int, default=24)
    parser.add_argument("--san-station-lr", type=float, default=1e-4)
    parser.add_argument("--san-pretrain-epochs", type=int, default=5)
    parser.add_argument("--sop-plug-epochs", type=int, default=10)
    parser.add_argument("--sop-plug-lr", type=float, default=1e-3)
    parser.add_argument("--sop-step-cseg-len", type=int, default=1)
    parser.add_argument("--sop-variable-cseg-len", type=int, default=1)
    parser.add_argument("--sra-policy-dir", type=Path, default=Path("experiments/halluguard/results/lrbn_sra_bp_stage5"))
    parser.add_argument("--max-train-windows", type=int, default=0, help="<=0 means all train windows.")
    parser.add_argument("--max-eval-windows", type=int, default=0, help="<=0 means all eval windows.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fetch-data", action="store_true")
    parser.add_argument("--fetch-plugin-repos", action="store_true")
    parser.add_argument("--fetch-datasets", default="", help="Dataset list for fetch_core_datasets.py; defaults to requested datasets.")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny smoke subset with same schema.")
    args = parser.parse_args()

    datasets = base.parse_list(args.datasets)
    backbones = base.parse_list(args.backbones)
    horizons = [int(x) for x in base.parse_list(args.horizons)]
    seeds = [int(x) for x in base.parse_list(args.seeds)]
    methods = base.parse_list(args.methods)

    if args.smoke:
        datasets = ["ETTm1"]
        backbones = ["DLinear", "PatchTST"]
        horizons = [96]
        seeds = [seeds[0]]
        methods = [
            m
            for m in methods
            if m
            in (
                "raw_no_correction",
                "HalluGuard-LRBN",
                "Safe-SRA",
                "Balanced-SRA",
                "RevIN",
                "SoP-step-wise",
                "naive_smoothing",
            )
        ]
        args.epochs = min(args.epochs, 1)
        args.sop_plug_epochs = min(args.sop_plug_epochs, 1)
        args.max_train_windows = 128 if args.max_train_windows <= 0 else min(args.max_train_windows, 128)
        args.max_eval_windows = 32 if args.max_eval_windows <= 0 else min(args.max_eval_windows, 32)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("predictions", "manifests", "logs"):
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)

    preflight_runtime(args)

    if args.fetch_plugin_repos:
        base.print_progress(f"fetch-plugin-repos log={args.output_dir / 'logs' / 'fetch_plugin_repos.log'}")
        base.run_command(["bash", str(FETCH_PLUGIN_REPOS_SCRIPT)], args.output_dir / "logs" / "fetch_plugin_repos.log")

    if args.fetch_data:
        fetch_datasets = args.fetch_datasets.strip() or ",".join(datasets)
        base.print_progress(f"fetch-data datasets={fetch_datasets} log={args.output_dir / 'logs' / 'fetch_data.log'}")
        base.run_command([sys.executable, str(FETCH_DATA_SCRIPT), "--datasets", fetch_datasets], args.output_dir / "logs" / "fetch_data.log")

    configs = [Config(d, b, h, s) for d in datasets for b in backbones for h in horizons for s in seeds]
    rows: List[Dict[str, object]] = []
    write_run_contract(args, datasets, backbones, horizons, seeds, methods)

    for idx, cfg in enumerate(configs, start=1):
        base.print_progress(f"TableA config {idx}/{len(configs)} {cfg.tag}")
        try:
            cfg_rows = run_config(cfg, methods, args)
        except Exception as exc:
            cfg_rows = [base.blocked_row(cfg, method, f"{type(exc).__name__}: {exc}") for method in methods]
            if not args.continue_on_error:
                raise
        annotate_rows(cfg_rows, args)
        rows.extend(cfg_rows)
        write_outputs(rows, args.output_dir)

    write_outputs(rows, args.output_dir)
    base.print_progress(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "configs": len(configs),
                "rows": len(rows),
                "completed": sum(r["status"] == "completed" for r in rows),
                "blocked": sum(r["status"] != "completed" for r in rows),
            }
        )
    )


def run_config(cfg: Config, methods: Sequence[str], args: argparse.Namespace) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    lrbn_methods = [m for m in methods if m in ("raw_no_correction", "HalluGuard-LRBN")]
    sra_methods = [m for m in methods if m in SRA_METHODS]
    adapter_methods = [m for m in methods if m in ADAPTER_METHODS]
    smoothing_methods = [m for m in methods if m in SMOOTHING_METHODS]
    need_raw_predictions = bool(lrbn_methods or sra_methods or smoothing_methods)

    if need_raw_predictions:
        try:
            lrbn_dir = args.output_dir / "predictions" / "halluguard_lrbn" / cfg.tag
            raw_dir = args.output_dir / "predictions" / "raw" / cfg.tag
            out_dir = args.output_dir / "runs" / "halluguard_lrbn" / cfg.tag
            if not (args.skip_existing and (out_dir / "lrbn_metrics.csv").exists()):
                base.print_progress(f"run LRBN/raw {cfg.tag} log={args.output_dir / 'logs' / f'{cfg.tag}_lrbn.log'}")
                cmd = [
                    sys.executable,
                    str(LRBN_SCRIPT),
                    "--datasets",
                    cfg.dataset,
                    "--models",
                    cfg.backbone,
                    "--horizons",
                    str(cfg.horizon),
                    "--variants",
                    "unified_revin_rdn_hybrid",
                    "--data-root",
                    str(args.data_root),
                    "--prediction-dir",
                    str(lrbn_dir),
                    "--raw-prediction-dir",
                    str(raw_dir),
                    "--output-dir",
                    str(out_dir),
                    "--seq-len",
                    str(args.seq_len),
                    "--tail-len",
                    str(args.tail_len),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--max-train-windows",
                    str(args.max_train_windows),
                    "--max-eval-windows",
                    str(args.max_eval_windows),
                    "--seed",
                    str(cfg.seed),
                    "--device",
                    args.device,
                    "--continue-on-error",
                ]
                base.run_command(cmd, args.output_dir / "logs" / f"{cfg.tag}_lrbn.log")
            else:
                base.print_progress(f"skip existing LRBN/raw {cfg.tag}")
            rows.extend(lrbn_rows_from_metrics(cfg, out_dir, lrbn_methods))
        except Exception as exc:
            reason = f"LRBN/raw group failed: {type(exc).__name__}: {exc}"
            rows.extend(base.blocked_row(cfg, method, reason) for method in lrbn_methods)

    if sra_methods:
        rows.extend(sra_rows_for_config(cfg, args, sra_methods))

    if adapter_methods:
        try:
            adapter_dir = args.output_dir / "predictions" / "tablea_adapters" / cfg.tag
            if not (args.skip_existing and (adapter_dir / "manifest.csv").exists()):
                base.print_progress(f"run TableA adapters {cfg.tag} methods={','.join(adapter_methods)}")
                cmd = [
                    sys.executable,
                    str(CORE12_SCRIPT),
                    "--datasets",
                    cfg.dataset,
                    "--models",
                    cfg.backbone,
                    "--horizons",
                    str(cfg.horizon),
                    "--methods",
                    ",".join(adapter_methods),
                    "--data-root",
                    str(args.data_root),
                    "--output-dir",
                    str(adapter_dir),
                    "--seq-len",
                    str(args.seq_len),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--learning-rate",
                    str(args.learning_rate),
                    "--san-period-len",
                    str(args.san_period_len),
                    "--san-station-lr",
                    str(args.san_station_lr),
                    "--san-pretrain-epochs",
                    str(args.san_pretrain_epochs),
                    "--sop-plug-epochs",
                    str(args.sop_plug_epochs),
                    "--sop-plug-lr",
                    str(args.sop_plug_lr),
                    "--sop-step-cseg-len",
                    str(args.sop_step_cseg_len),
                    "--sop-variable-cseg-len",
                    str(args.sop_variable_cseg_len),
                    "--max-train-windows",
                    str(args.max_train_windows),
                    "--max-eval-windows",
                    str(args.max_eval_windows),
                    "--seed",
                    str(cfg.seed),
                    "--device",
                    args.device,
                    "--continue-on-error",
                ]
                base.run_command(cmd, args.output_dir / "logs" / f"{cfg.tag}_tablea_adapters.log")
            else:
                base.print_progress(f"skip existing TableA adapters {cfg.tag}")
            rows.extend(base.adapter_rows_from_predictions(cfg, adapter_dir, adapter_methods))
        except Exception as exc:
            reason = f"adapter group failed: {type(exc).__name__}: {exc}"
            rows.extend(base.blocked_row(cfg, method, reason) for method in adapter_methods)

    if smoothing_methods:
        raw_path = base.raw_prediction_path_for_cfg(args, cfg)
        smoothing_dir = args.output_dir / "predictions" / "smoothing_controls" / cfg.tag
        smoothing_dir.mkdir(parents=True, exist_ok=True)
        for method in smoothing_methods:
            try:
                out_path = smoothing_dir / f"{cfg.dataset}_{cfg.backbone}_{cfg.horizon}_{method}.jsonl"
                if not raw_path.exists():
                    rows.append(base.blocked_row(cfg, method, raw_prediction_blocker(args, cfg)))
                    continue
                if not (args.skip_existing and out_path.exists()):
                    base.print_progress(f"write smoothing {cfg.tag} method={method}")
                    base.write_smoothing_predictions(raw_path, out_path, method)
                mse, mae = base.prediction_metrics(out_path)
                rows.append(base.metric_row(cfg, method, mse, mae, "", "", str(out_path), str(smoothing_dir), "completed", ""))
            except Exception as exc:
                rows.append(base.blocked_row(cfg, method, f"{type(exc).__name__}: {exc}"))

    unknown_methods = [
        m
        for m in methods
        if m not in ("raw_no_correction", "HalluGuard-LRBN")
        and m not in SRA_METHODS
        and m not in ADAPTER_METHODS
        and m not in SMOOTHING_METHODS
    ]
    for method in unknown_methods:
        rows.append(base.blocked_row(cfg, method, f"unknown TableA method: {method}"))
    return rows


def preflight_runtime(args: argparse.Namespace) -> None:
    if args.device != "cuda":
        return
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "torch.cuda.is_available() is False. Install a CUDA wheel compatible with the server driver "
                "or rerun with DEVICE=cpu for a slow wiring test."
            )
        torch.empty(1, device="cuda")
    except Exception as exc:
        raise SystemExit(
            "CUDA preflight failed before launching TableA. This prevents a full matrix of misleading blocked rows.\n"
            f"Reason: {type(exc).__name__}: {exc}\n"
            "Recommended check: python -c \"import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())\"\n"
            "If the driver reports CUDA 12.2, install a compatible wheel, for example:\n"
            "  python -m pip uninstall -y torch torchvision torchaudio\n"
            "  python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio\n"
        ) from exc


def lrbn_rows_from_metrics(cfg: Config, out_dir: Path, requested_methods: Sequence[str]) -> List[Dict[str, object]]:
    path = out_dir / "lrbn_metrics.csv"
    if not path.exists():
        return [
            base.blocked_row(cfg, m, f"missing LRBN metrics: {path}")
            for m in requested_methods
            if m in ("raw_no_correction", "HalluGuard-LRBN")
        ]
    raw_rows = base.read_csv(path)
    out = []
    for method in ("raw_no_correction", "HalluGuard-LRBN"):
        if method not in requested_methods:
            continue
        variant = "raw_no_correction" if method == "raw_no_correction" else "unified_revin_rdn_hybrid"
        completed = [r for r in raw_rows if r.get("variant") == variant and r.get("status") == "completed"]
        if completed:
            r = completed[0]
            out.append(
                base.metric_row(
                    cfg,
                    method,
                    r.get("mse"),
                    r.get("mae"),
                    r.get("mse_delta_pct_vs_raw", ""),
                    r.get("mae_delta_pct_vs_raw", ""),
                    r.get("prediction_path", ""),
                    str(out_dir),
                    "completed",
                    "",
                )
            )
            continue
        blocked = [r for r in raw_rows if r.get("variant") == variant and r.get("status") != "completed"]
        if blocked:
            reason = blocked[0].get("blocker_reason", "") or f"variant {variant} did not complete in {path}"
            out.append(base.blocked_row(cfg, method, reason))
        else:
            out.append(base.blocked_row(cfg, method, f"missing completed variant {variant} in {path}"))
    return out


def sra_rows_for_config(cfg: Config, args: argparse.Namespace, requested_methods: Sequence[str]) -> List[Dict[str, object]]:
    raw_path = base.raw_prediction_path_for_cfg(args, cfg)
    lrbn_path = (
        args.output_dir
        / "predictions"
        / "halluguard_lrbn"
        / cfg.tag
        / f"{cfg.dataset}_{cfg.backbone}_{cfg.horizon}_unified_revin_rdn_hybrid.jsonl"
    )
    out_dir = args.output_dir / "predictions" / "sra_bp" / cfg.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    if not raw_path.exists():
        reason = raw_prediction_blocker(args, cfg)
        return [base.blocked_row(cfg, method, reason) for method in requested_methods]
    if not lrbn_path.exists():
        reason = f"LRBN prediction unavailable for SRA-BP: missing {lrbn_path}"
        return [base.blocked_row(cfg, method, reason) for method in requested_methods]

    try:
        raw_samples, lrbn_samples = matched_raw_lrbn_samples(raw_path, lrbn_path)
    except Exception as exc:
        reason = f"SRA-BP sample alignment failed: {type(exc).__name__}: {exc}"
        return [base.blocked_row(cfg, method, reason) for method in requested_methods]

    for method in requested_methods:
        try:
            canonical = canonical_sra_method(method)
            out_path = out_dir / f"{cfg.dataset}_{cfg.backbone}_{cfg.horizon}_{canonical}.jsonl"
            if not (args.skip_existing and out_path.exists()):
                params = load_sra_params(method, args.sra_policy_dir)
                base.print_progress(f"write SRA-BP {cfg.tag} method={canonical}")
                write_sra_predictions(cfg, raw_samples, lrbn_samples, out_path, canonical, params)
            mse, mae = base.prediction_metrics(out_path)
            rows.append(base.metric_row(cfg, canonical, mse, mae, "", "", str(out_path), str(out_dir), "completed", ""))
        except Exception as exc:
            rows.append(base.blocked_row(cfg, method, f"SRA-BP failed: {type(exc).__name__}: {exc}"))
    return rows


def canonical_sra_method(method: str) -> str:
    if method == "LRBN-SRA-BP-safe":
        return "Safe-SRA"
    if method == "LRBN-SRA-BP-balanced":
        return "Balanced-SRA"
    return method


def load_sra_params(method: str, policy_dir: Path) -> Dict[str, object]:
    canonical = canonical_sra_method(method)
    param_file = policy_dir / SRA_PARAM_FILES.get(method, SRA_PARAM_FILES.get(canonical, ""))
    if param_file.exists():
        return json.loads(param_file.read_text(encoding="utf-8"))
    fallback = SRA_FALLBACK_PARAMS[canonical]
    return dict(fallback)


def matched_raw_lrbn_samples(raw_path: Path, lrbn_path: Path) -> tuple[List[dict], List[dict]]:
    raw_by_key = {sample_key(s): s for s in base.read_jsonl(raw_path)}
    lrbn_by_key = {sample_key(s): s for s in base.read_jsonl(lrbn_path)}
    keys = sorted(set(raw_by_key) & set(lrbn_by_key))
    if not keys:
        raise ValueError(f"no matched sample keys between {raw_path} and {lrbn_path}")
    missing_raw = sorted(set(lrbn_by_key) - set(raw_by_key))
    missing_lrbn = sorted(set(raw_by_key) - set(lrbn_by_key))
    if missing_raw or missing_lrbn:
        raise ValueError(f"raw/LRBN key mismatch: missing_raw={len(missing_raw)}, missing_lrbn={len(missing_lrbn)}")
    return [raw_by_key[k] for k in keys], [lrbn_by_key[k] for k in keys]


def sample_key(sample: dict) -> str:
    return f"{sample.get('split')}::{sample.get('sample_id')}"


def write_sra_predictions(
    cfg: Config,
    raw_samples: Sequence[dict],
    lrbn_samples: Sequence[dict],
    out_path: Path,
    method: str,
    params: Dict[str, object],
) -> None:
    context = stack_trajs([s["context"] for s in raw_samples])
    raw_pred = stack_trajs([s["prediction"] for s in raw_samples])
    lrbn_pred = stack_trajs([s["prediction"] for s in lrbn_samples])
    horizons = np.full(len(raw_samples), int(cfg.horizon), dtype=int)
    corrected, aux = sra_bp.apply_sra_bp(context, raw_pred, lrbn_pred, horizons, params)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, (raw_sample, lrbn_sample) in enumerate(zip(raw_samples, lrbn_samples)):
            horizon_len = len(lrbn_sample["prediction"])
            updated = dict(lrbn_sample)
            updated["prediction"] = round_prediction_like(corrected[idx, :horizon_len, :])
            updated["method"] = method
            updated["variant"] = method
            updated["model"] = f"{cfg.backbone}+{method}"
            updated["adapter_mode"] = "frozen_sparse_repair_aware_boundary_projection"
            updated["sra_policy_source"] = "stage5_validation_selected_frozen_params"
            updated["sra_params"] = params
            updated["sra_strength"] = round(float(np.asarray(aux["strength"])[idx]), 8)
            updated["test_threshold_leakage"] = False
            handle.write(json.dumps(updated, ensure_ascii=False) + "\n")


def stack_trajs(values: Sequence[object]) -> np.ndarray:
    arrays = [one_traj(v) for v in values]
    max_t = max(a.shape[0] for a in arrays)
    max_c = max(a.shape[1] for a in arrays)
    out = np.full((len(arrays), max_t, max_c), np.nan, dtype=float)
    for idx, arr in enumerate(arrays):
        out[idx, : arr.shape[0], : arr.shape[1]] = arr
    return out


def one_traj(value: object) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"expected trajectory [T] or [T,C], got shape={arr.shape}")


def round_prediction_like(arr: np.ndarray) -> list:
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"expected corrected trajectory [T,C], got shape={arr.shape}")
    if arr.shape[1] == 1:
        return [round(float(v), 6) for v in arr[:, 0].tolist()]
    return [[round(float(v), 6) for v in row] for row in arr.tolist()]


def raw_prediction_blocker(args: argparse.Namespace, cfg: Config) -> str:
    out_dir = args.output_dir / "runs" / "halluguard_lrbn" / cfg.tag
    path = out_dir / "lrbn_metrics.csv"
    if path.exists():
        for row in base.read_csv(path):
            if row.get("variant") == "raw_no_correction" and row.get("status") != "completed":
                reason = row.get("blocker_reason", "")
                if reason:
                    return f"raw prediction unavailable because raw_no_correction failed: {reason}"
    raw_path = base.raw_prediction_path_for_cfg(args, cfg)
    return f"raw prediction unavailable for smoothing controls: missing {raw_path}"


def annotate_rows(rows: List[Dict[str, object]], args: argparse.Namespace) -> None:
    for row in rows:
        row.setdefault("table", "TableA")
        row.setdefault("seq_len", args.seq_len)
        row.setdefault("tail_len", args.tail_len)
        row.setdefault("epochs", args.epochs)
        row.setdefault("batch_size", args.batch_size)
        row.setdefault("learning_rate", args.learning_rate)
        row.setdefault("sop_plug_epochs", args.sop_plug_epochs)
        row.setdefault("sop_plug_lr", args.sop_plug_lr)
        row.setdefault("max_train_windows", args.max_train_windows)
        row.setdefault("max_eval_windows", args.max_eval_windows)
        row.setdefault("test_threshold_leakage", False)


def write_outputs(rows: List[Dict[str, object]], output_dir: Path) -> None:
    base.write_outputs(rows, output_dir)
    summary = base.summarize(rows)
    base.atomic_write_text(output_dir / "summary.md", summary_md(rows, summary))


def summary_md(rows: List[Dict[str, object]], summary: List[Dict[str, object]]) -> str:
    completed = sum(r["status"] == "completed" for r in rows)
    blocked = [r for r in rows if r["status"] != "completed"]
    lines = [
        "# HalluGuard-LRBN TableA Full Run",
        "",
        f"- Completed rows: {completed} / {len(rows)}",
        "- HalluGuard parent: `HalluGuard-LRBN unified_revin_rdn_hybrid`",
        "- SRA mainlines: `Safe-SRA`, `Balanced-SRA`",
        "- Protocol: offline TableA, no partially observed test target feedback",
        "- Test threshold leakage: False",
        "",
        "## Method Summary",
        "",
    ]
    for row in summary:
        lines.append(
            f"- `{row['method']}`: completed {row['completed_rows']} / {row['total_rows']}, "
            f"mean MSE {row['mean_mse']}, mean MAE {row['mean_mae']}, "
            f"local MSE delta vs raw {row['mean_mse_delta_pct_vs_raw_local']}, blocked {row['blocked_rows']}"
        )
    if blocked:
        lines.extend(["", "## Blocked Rows", ""])
        for row in blocked[:300]:
            lines.append(
                f"- {row['dataset']} {row['backbone']} h{row['horizon']} seed{row['seed']} "
                f"`{row['method']}`: {row['blocker_reason']}"
            )
        if len(blocked) > 300:
            lines.append(f"- ... {len(blocked) - 300} additional blocked rows omitted from markdown; see `combined_metrics.csv`.")
    return "\n".join(lines) + "\n"


def write_run_contract(
    args: argparse.Namespace,
    datasets: Sequence[str],
    backbones: Sequence[str],
    horizons: Sequence[int],
    seeds: Sequence[int],
    methods: Sequence[str],
) -> None:
    contract = {
        "table": "TableA",
        "claim_method": "HalluGuard-LRBN parent with Safe-SRA and Balanced-SRA post-LRBN mainlines",
        "datasets": list(datasets),
        "backbones": list(backbones),
        "horizons": list(horizons),
        "seeds": list(seeds),
        "methods": list(methods),
        "seq_len": args.seq_len,
        "tail_len": args.tail_len,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "san_pretrain_epochs": args.san_pretrain_epochs,
        "sop_plug_epochs": args.sop_plug_epochs,
        "sra_policy_dir": str(args.sra_policy_dir),
        "max_train_windows": args.max_train_windows,
        "max_eval_windows": args.max_eval_windows,
        "val_test_contract": "validation split may calibrate policies; test split is evaluation-only",
        "test_threshold_leakage": False,
        "notes": [
            "TAFAS-online is excluded from TableA by default because it uses partially observed target feedback.",
            "Safe-SRA and Balanced-SRA are frozen LRBN post-processors using Stage5 validation-selected SRA-BP policies; test rows are evaluation-only.",
            "SOLID-official-supported rows are recorded in the matrix; unsupported official adapter cells are explicit blocked rows.",
            "max_train_windows/max_eval_windows <= 0 means use all available windows.",
        ],
    }
    (args.output_dir / "run_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
