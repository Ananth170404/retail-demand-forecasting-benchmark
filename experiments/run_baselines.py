"""Step 1: statistical baselines on both datasets.

Establishes the accuracy floor every later model must beat.

Usage:
    python experiments/run_baselines.py --dataset m5
    python experiments/run_baselines.py --dataset walmart
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from _common import DATASET_CONFIG, load_panel, report, resolve_mode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecasting.evaluation.backtest import run_backtest  # noqa: E402
from forecasting.models.baselines import build_baselines  # noqa: E402


def main(
    dataset: str,
    n_windows: int | None = None,
    mode: str = "rolling",
    max_series_per_state: int | None = 200,
    trim_zeros: bool = True,
    include_slow: bool = True,
    n_jobs: int = 4,
    chunk_size: int | None = None,
) -> None:
    cfg = dict(DATASET_CONFIG[dataset])
    if n_windows:
        cfg["n_windows"] = n_windows

    # "official" is the holdout window sized to the competition's own horizon.
    window_mode, horizon = resolve_mode(cfg, mode)

    print(f"Loading {dataset}...")
    panel, _ = load_panel(dataset, trim_zeros=trim_zeros, max_series_per_state=max_series_per_state)
    print(
        f"  mode={mode} | h={horizon} | test_fraction={cfg['test_fraction']} "
        f"| season_length={cfg['season_length']}"
    )

    all_results = []
    for name, factory in build_baselines(
        cfg["freq"],
        cfg["season_length"],
        include_slow=include_slow,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    ).items():
        print(f"  running {name}...", flush=True)
        result = run_backtest(
            model_factory=factory,
            panel=panel,
            h=horizon,
            season_length=cfg["season_length"],
            test_fraction=cfg["test_fraction"],
            n_windows=cfg["n_windows"],
            min_train_periods=cfg["min_train_periods"],
            mode=window_mode,
        )
        all_results.append(result)

    results = pd.concat(all_results, ignore_index=True)

    # Every model must be scored on an identical set of series for the ranking
    # to mean anything; assert rather than trust.
    evaluated = results.groupby("model")["unique_id"].nunique()
    assert evaluated.nunique() == 1, f"models scored on different series counts:\n{evaluated}"
    assert not results["MAE"].isna().any(), "NaN forecasts leaked into the metrics"

    report(results, dataset, mode, tag="baselines")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG), required=True)
    parser.add_argument("--n-windows", type=int, default=None)
    parser.add_argument("--mode", choices=["rolling", "official"], default="rolling")
    parser.add_argument(
        "--max-series-per-state", type=int, default=200, help="M5 sample size; 0 loads all series"
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="keep pre-launch zeros in training data (for measuring the trim's effect)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="skip AutoETS/AutoTheta, whose optimisers dominate runtime",
    )
    # statsforecast forks one process per job and copies the panel into each,
    # so on a 16-core laptop n_jobs=-1 was what exhausted RAM at 6,000 series.
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="fit this many series at a time; caps peak memory on large panels",
    )
    args = parser.parse_args()
    main(
        args.dataset,
        args.n_windows,
        args.mode,
        args.max_series_per_state,
        trim_zeros=not args.no_trim,
        include_slow=not args.fast,
        n_jobs=args.n_jobs,
        chunk_size=args.chunk_size,
    )
