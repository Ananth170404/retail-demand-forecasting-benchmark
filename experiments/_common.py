"""Shared experiment plumbing, so every model is driven identically.

Keeping loading, splitting and reporting in one place is what makes the results
table a comparison of models rather than a comparison of harnesses.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecasting.data.loaders import load_dataset  # noqa: E402
from forecasting.evaluation.metrics import summarize, summarize_by_class  # noqa: E402
from forecasting.models.base import to_panel, trim_leading_zeros_panel  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "tables"

# `official_h` mimics each competition's holdout from the tail of the labelled
# training data: 28 days for M5 (d_1914-d_1941, the public-leaderboard period)
# and 39 weeks for Walmart.
#
# `competition_h` (M5 only) is the competition's *actual* final test: train on
# d_1-d_1941, forecast d_1942-d_1969, using the released test labels
# (scripts/download_data.py --dataset m5-labels). Scored with the official
# 12-level WRMSSE on all 30,490 series. Walmart never released its labels.
#
# `season_cutoff` / `season_h` (Walmart only): a 39-week test window on the same
# calendar as Kaggle's unreleased test (Nov-Jul), one year earlier: train to
# 2011-10-28, test 2011-11-04 to 2012-07-27. The other windows contain no
# Thanksgiving or Christmas, so they cannot show holiday effects, and comparing
# them with the winner's score (set over Christmas) would not be fair.
#
# `validation_*`: the window every choice (feature set, hyperparameters,
# ensemble weights) is made on, strictly before the matching test window, so
# the test is scored once with settings it never influenced.
#   M5: train d_1-d_1913, forecast d_1914-d_1941 (the competition's own
#       public-leaderboard period), scored with WRMSSE using the official
#       validation-phase weights. Test: the `competition` mode.
#   Walmart: train to 2011-07-29, forecast 13 weeks (Aug-Oct 2011). Test: the
#       `season` window. A Christmas-covering validation window would leave
#       52 weeks of training, too few for 52-week lags or differencing, so
#       holiday-specific choices cannot be validated here.
#
# `min_train_periods` keeps every model on the same series: seasonal methods
# need at least one full cycle of history, and without the filter their
# short-series NaNs would quietly shrink their own evaluation set.
#
# `headline_metric` ranks the results. M5 uses RMSSE, matching the competition's
# own (W)RMSSE: on a panel that is ~60% zeros, MASE is minimised by shading
# forecasts toward zero — an all-zero forecast beats several baselines on it —
# so a MASE ranking rewards under-forecasting. Walmart is only ~4% zeros, where
# MASE does not have that failure mode.
DATASET_CONFIG = {
    "m5": {
        "freq": "D",
        "season_length": 7,
        "h": 28,
        "official_h": 28,
        "test_fraction": 0.2,
        "n_windows": None,
        "min_train_periods": 8,
        "headline_metric": "RMSSE",
        "competition_h": 28,
        "validation_h": 28,
    },
    "walmart": {
        "freq": "W-FRI",
        "season_length": 52,
        "h": 8,
        "official_h": 39,
        "test_fraction": 0.2,
        "n_windows": None,
        "min_train_periods": 53,
        "headline_metric": "MASE",
        "season_cutoff": "2011-10-28",
        "season_h": 39,
        "validation_cutoff": "2011-07-29",
        "validation_h": 13,
    },
}


MODES = ("rolling", "official", "competition", "season", "validation")


def resolve_mode(cfg: dict, mode: str) -> tuple[str, int]:
    """Map the user-facing mode onto the window mode and horizon.

    "competition" is a holdout over a panel that includes the test period, so
    the final `competition_h` days are exactly the competition's test window.
    """
    if mode == "competition":
        if "competition_h" not in cfg:
            raise ValueError("competition mode needs released test labels (M5 only)")
        return "holdout", cfg["competition_h"]
    if mode == "season":
        if "season_cutoff" not in cfg:
            raise ValueError("season mode is defined for Walmart only")
        return "cutoff", cfg["season_h"]
    if mode == "validation":
        return ("cutoff" if "validation_cutoff" in cfg else "holdout"), cfg["validation_h"]
    if mode == "official":
        return "holdout", cfg["official_h"]
    if mode == "rolling":
        return "rolling", cfg["h"]
    raise ValueError(f"unknown mode '{mode}'; use one of {MODES}")


def mode_cutoff(cfg: dict, mode: str) -> str | None:
    """The fixed training cutoff for cutoff-based modes, else None."""
    return cfg.get(f"{mode}_cutoff")


def load_panel(
    dataset: str,
    trim_zeros: bool = True,
    max_series_per_state: int | None = 200,
    with_exog: bool = False,
    include_test: bool = False,
):
    """Load a dataset as a modelling panel, optionally carrying its covariates.

    `max_series_per_state` (M5 only): None or 0 loads all 30,490 series.
    `include_test` (M5 only): append the competition's test period.
    """
    load_kwargs = {}
    if dataset == "m5":
        load_kwargs["max_series_per_state"] = max_series_per_state or None
        load_kwargs["include_test"] = include_test
    elif include_test:
        raise ValueError(f"{dataset} has no released test labels")
    product_df, roles = load_dataset(dataset, **load_kwargs)

    exog_cols = None
    if with_exog:
        exog_cols = (
            roles.static_cat_cols
            + roles.dynamic_cat_cols
            + roles.dynamic_real_cols
            + roles.holiday_cols
        )

    panel = to_panel(product_df, exog_cols=exog_cols)
    raw_rows = len(panel)

    if trim_zeros:
        panel = trim_leading_zeros_panel(panel)
        dropped = raw_rows - len(panel)
        print(f"  trimmed pre-launch zeros: -{dropped:,} rows ({dropped / raw_rows * 100:.1f}%)")

    print(f"  {len(panel):,} rows | {panel['unique_id'].nunique():,} series")
    return panel, roles


def report(
    results: pd.DataFrame,
    dataset: str,
    mode: str,
    tag: str,
    results_dir: Path | None = None,
    headline: str | None = None,
) -> None:
    """Write per-series / summary / by-class tables and print the headline.

    `results_dir` redirects output (e.g. a smoke run that must not overwrite the
    committed tables); `headline` overrides the dataset's default ranking metric.
    """
    if results.empty:
        print("no results produced")
        return

    out_dir = Path(results_dir) if results_dir else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"{dataset}_{mode}"

    # Gzipped: per-series tables reach ~100MB at full M5 scale, GitHub's
    # per-file limit. pandas reads .csv.gz transparently.
    detail_path = out_dir / f"{tag}_{suffix}_per_series.csv.gz"
    results.to_csv(detail_path, index=False)

    headline = headline or DATASET_CONFIG[dataset]["headline_metric"]
    summary = summarize(results, sort_by=headline)
    summary_path = out_dir / f"{tag}_{suffix}_summary.csv"
    summary.to_csv(summary_path)

    by_class = summarize_by_class(results, metric=headline, stat="median")
    class_path = out_dir / f"{tag}_{suffix}_by_class.csv"
    by_class.to_csv(class_path)

    print(f"\n=== {dataset.upper()} {tag} [{mode}] ===")
    print(
        f"windows={results['window'].nunique()} | horizon={results['horizon'].iloc[0]} "
        f"| series={results['unique_id'].nunique():,}"
    )
    print(f"\n-- headline (sorted by trimmed-mean {headline}) --")
    cols = [c for c in summary.columns if c.startswith(("RMSSE", "MASE", "WAPE", "Volume_ratio"))]
    print(summary[["n_series"] + cols].to_string())
    print(f"\n-- median {headline} by demand class --")
    print(by_class.to_string())
    for path in (detail_path, summary_path, class_path):
        print(f"wrote {path}")
