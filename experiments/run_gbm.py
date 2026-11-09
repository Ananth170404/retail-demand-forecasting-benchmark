"""Step B1: gradient-boosted trees as a global model.

The first model in the benchmark that sees the covariates. Runs the same
splits, metrics and reporting as the baselines so the numbers are comparable.

Usage:
    python experiments/run_gbm.py --dataset m5
    python experiments/run_gbm.py --dataset walmart --mode official
    python experiments/run_gbm.py --dataset m5 --objective l2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from _common import DATASET_CONFIG, load_panel, report, resolve_mode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecasting.data.features import get_feature_config  # noqa: E402
from forecasting.evaluation.backtest import run_backtest  # noqa: E402
from forecasting.models.gbm import build_gbm  # noqa: E402


def main(
    dataset: str,
    mode: str = "rolling",
    objectives: tuple[str, ...] = ("tweedie", "l2"),
    max_series_per_state: int | None = 200,
    trim_zeros: bool = True,
    no_exog: bool = False,
) -> None:
    cfg = DATASET_CONFIG[dataset]
    window_mode, horizon = resolve_mode(cfg, mode)

    print(f"Loading {dataset}...")
    panel, roles = load_panel(
        dataset,
        trim_zeros=trim_zeros,
        max_series_per_state=max_series_per_state,
        with_exog=not no_exog,
    )

    feature_config = get_feature_config(dataset)
    if no_exog:
        feature_config = feature_config.without_exog()

    exog_cols = roles.static_cat_cols + roles.dynamic_cat_cols + roles.dynamic_real_cols
    print(
        f"  mode={mode} | h={horizon} | lags={feature_config.lags} "
        f"| exog={'off' if no_exog else len(exog_cols)}"
    )

    all_results = []
    for objective in objectives:
        print(f"  training LightGBM ({objective})...", flush=True)
        factory = build_gbm(
            freq=cfg["freq"],
            feature_config=feature_config,
            static_features=roles.static_cat_cols,
            exog_cols=exog_cols,
            objective=objective,
        )
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
    report(results, dataset, mode, tag="gbm")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG), required=True)
    parser.add_argument("--mode", choices=["rolling", "official"], default="rolling")
    parser.add_argument("--objective", action="append", default=None)
    parser.add_argument(
        "--max-series-per-state", type=int, default=200, help="M5 sample size; 0 loads all series"
    )
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--no-exog", action="store_true")
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        mode=args.mode,
        objectives=tuple(args.objective) if args.objective else ("tweedie", "l2"),
        max_series_per_state=args.max_series_per_state,
        trim_zeros=not args.no_trim,
        no_exog=args.no_exog,
    )
