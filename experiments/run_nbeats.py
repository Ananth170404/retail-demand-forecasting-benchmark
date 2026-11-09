"""Step B2: N-BEATS, the univariate deep-learning entry.

Same splits, metrics and reporting as the baselines and GBM, so the numbers sit
in the same table. N-BEATS sees only the target history — no covariates — which
makes the gap to GBM a measure of what feature engineering buys.

Usage:
    python experiments/run_nbeats.py --dataset m5 --mode official
    python experiments/run_nbeats.py --dataset walmart --max-steps 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import DATASET_CONFIG, load_panel, report, resolve_mode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd  # noqa: E402

from forecasting.evaluation.backtest import run_backtest  # noqa: E402
from forecasting.models.nbeats import build_nbeats, build_nbeatsx  # noqa: E402


def main(
    dataset: str,
    mode: str = "rolling",
    max_steps: int = 500,
    n_windows: int | None = None,
    max_series_per_state: int | None = 200,
    trim_zeros: bool = True,
    variants: tuple[str, ...] = ("nbeats", "nbeatsx"),
) -> None:
    cfg = dict(DATASET_CONFIG[dataset])
    if n_windows:
        cfg["n_windows"] = n_windows
    window_mode, horizon = resolve_mode(cfg, mode)

    needs_exog = "nbeatsx" in variants
    print(f"Loading {dataset}...")
    panel, roles = load_panel(
        dataset,
        trim_zeros=trim_zeros,
        max_series_per_state=max_series_per_state,
        with_exog=needs_exog,
    )

    import torch

    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"  mode={mode} | h={horizon} | input_size={2 * horizon} | steps={max_steps} | {device}")

    # Every covariate in these datasets is known ahead of time, so the dynamic
    # ones are future-exogenous and the per-series ones are static.
    futr_exog = roles.dynamic_cat_cols + roles.dynamic_real_cols
    stat_exog = roles.static_cat_cols

    factories = {
        "nbeats": lambda: build_nbeats(freq=cfg["freq"], h=horizon, max_steps=max_steps),
        "nbeatsx": lambda: build_nbeatsx(
            freq=cfg["freq"],
            h=horizon,
            futr_exog=futr_exog,
            stat_exog=stat_exog,
            max_steps=max_steps,
        ),
    }

    all_results = []
    for variant in variants:
        label = "N-BEATS" if variant == "nbeats" else "N-BEATSx"
        extra = "" if variant == "nbeats" else f" (+{len(futr_exog)} futr, {len(stat_exog)} stat)"
        print(f"  training {label}{extra}...", flush=True)
        results = run_backtest(
            model_factory=factories[variant](),
            panel=panel,
            h=horizon,
            season_length=cfg["season_length"],
            test_fraction=cfg["test_fraction"],
            n_windows=cfg["n_windows"],
            min_train_periods=cfg["min_train_periods"],
            mode=window_mode,
        )
        all_results.append(results)

    report(pd.concat(all_results, ignore_index=True), dataset, mode, tag="nbeats")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG), required=True)
    parser.add_argument("--mode", choices=["rolling", "official"], default="rolling")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--n-windows", type=int, default=None)
    parser.add_argument(
        "--max-series-per-state", type=int, default=200, help="M5 sample size; 0 loads all series"
    )
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument(
        "--variant",
        action="append",
        choices=["nbeats", "nbeatsx"],
        default=None,
        help="repeat to run both; defaults to both",
    )
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        mode=args.mode,
        max_steps=args.max_steps,
        n_windows=args.n_windows,
        max_series_per_state=args.max_series_per_state,
        trim_zeros=not args.no_trim,
        variants=tuple(args.variant) if args.variant else ("nbeats", "nbeatsx"),
    )
