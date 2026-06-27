#!/usr/bin/env python
"""Run Stage 7 Safe-TAE compact validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from halluguard_stage7_safe_tae import run_stage7_safe_tae


DEFAULT_METRICS = Path("experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv")
DEFAULT_STAGE5 = Path("experiments/halluguard/results/lrbn_sra_bp_stage5")
DEFAULT_STAGE6 = Path("experiments/halluguard/results/stage6_mechanism")
DEFAULT_STAGE3 = Path("experiments/halluguard/results/lrbn_bp_stage3")
DEFAULT_OUTPUT = Path("experiments/halluguard/results/stage7_safe_tae")


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
    parser.add_argument("--stage6-dir", type=Path, default=DEFAULT_STAGE6)
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    result = run_stage7_safe_tae(
        metrics_csv=args.metrics_csv,
        stage5_dir=args.stage5_dir,
        stage6_dir=args.stage6_dir,
        stage3_dir=args.stage3_dir,
        output_dir=args.output_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    verdict = result["verdict"]
    print(json.dumps(verdict, ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
