#!/usr/bin/env python
"""Run Stage 16 learned patch / teacher projector compact validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from halluguard_stage16_learned_patch_teacher import Stage16Config, build_all_artifacts


DEFAULT_METRICS = Path("experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv")
DEFAULT_STAGE5 = Path("experiments/halluguard/results/lrbn_sra_bp_stage5")
DEFAULT_STAGE7 = Path("experiments/halluguard/results/stage7_safe_tae")
DEFAULT_STAGE14 = Path("experiments/halluguard/results/stage14_selector_mechanism")
DEFAULT_STAGE15 = Path("experiments/halluguard/results/stage15_endogenous_editors")
DEFAULT_STAGE3 = Path("experiments/halluguard/results/lrbn_bp_stage3")
DEFAULT_OUTPUT = Path("experiments/halluguard/results/stage16_learned_patch_teacher")


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    return str(obj)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--stage5-dir", type=Path, default=DEFAULT_STAGE5)
    parser.add_argument("--stage7-dir", type=Path, default=DEFAULT_STAGE7)
    parser.add_argument("--stage14-dir", type=Path, default=DEFAULT_STAGE14)
    parser.add_argument("--stage15-dir", type=Path, default=DEFAULT_STAGE15)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    cfg = Stage16Config(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        device=args.device,
    )
    result = build_all_artifacts(
        metrics_csv=args.metrics_csv,
        stage5_dir=args.stage5_dir,
        stage7_dir=args.stage7_dir,
        stage14_dir=args.stage14_dir,
        stage15_dir=args.stage15_dir,
        stage3_dir=args.stage3_dir,
        output_dir=args.output_dir,
        cfg=cfg,
        n_bootstrap=args.n_bootstrap,
    )
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
