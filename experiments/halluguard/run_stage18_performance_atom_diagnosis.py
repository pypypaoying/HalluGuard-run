#!/usr/bin/env python
"""Run Stage18 performance atom extraction compact diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from halluguard_stage18_performance_atom_diagnosis import Stage18Config, build_all_artifacts


DEFAULT_METRICS = Path("experiments/halluguard/results/research_direction_validation/forecast_inputs/combined_metrics.csv")
DEFAULT_STAGE5 = Path("experiments/halluguard/results/lrbn_sra_bp_stage5")
DEFAULT_STAGE3 = Path("experiments/halluguard/results/lrbn_bp_stage3")
DEFAULT_OUTPUT = Path("experiments/halluguard/results/stage18_performance_atom_diagnosis")


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
    parser.add_argument("--stage3-dir", type=Path, default=DEFAULT_STAGE3)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-atoms", type=int, default=5)
    parser.add_argument("--pca-components", type=int, default=8)
    args = parser.parse_args()
    cfg = Stage18Config(
        seed=args.seed,
        bootstrap=args.n_bootstrap,
        output_dir=str(args.output_dir),
        n_atoms=args.n_atoms,
        pca_components=args.pca_components,
    )
    result = build_all_artifacts(
        metrics_csv=args.metrics_csv,
        stage5_dir=args.stage5_dir,
        stage3_dir=args.stage3_dir,
        output_dir=args.output_dir,
        cfg=cfg,
        n_bootstrap=args.n_bootstrap,
    )
    print(json.dumps(result["verdict"], ensure_ascii=False, indent=2, default=json_default))


if __name__ == "__main__":
    main()
