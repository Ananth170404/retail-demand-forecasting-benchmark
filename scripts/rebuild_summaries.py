"""Rebuild every summary and by-class table from the committed per-series results.

Summaries are pure functions of the per-series tables, so a change to how
results are aggregated or ranked should not require re-training anything. Run
this after changing `summarize`, `summarize_by_class` or a dataset's
`headline_metric`.

Per-series tables written before Actual_Sum/Forecast_Sum existed are backfilled
from columns they already carry. The reconstruction is exact:
    WAPE = sum|e| / sum|y| and MAE = sum|e| / n   =>   sum y = n * MAE / WAPE
    Bias = mean(y - y_hat)                        =>   sum y_hat = sum y - n * Bias
(y is non-negative, so sum|y| = sum y.) WAPE is NaN only when sum y is 0. The
one case it cannot recover is a perfect forecast (MAE = WAPE = 0), which is left
NaN and counted in the output.

Usage:
    python scripts/rebuild_summaries.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

from _common import DATASET_CONFIG, RESULTS_DIR  # noqa: E402

from forecasting.evaluation.metrics import summarize, summarize_by_class  # noqa: E402

PER_SERIES = re.compile(
    r"(?P<tag>[a-z]+)_(?P<dataset>m5|walmart)_(?P<mode>rolling|official|competition|season)_per_series\.csv\.gz"
)


def backfill_volume(results: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill Actual_Sum/Forecast_Sum on rows missing them; return the unrecoverable count.

    Row-wise rather than per table: a table can mix rows from before and after
    the columns existed, e.g. when a new model is appended to old results.
    """
    results = results.copy()
    for col in ("Actual_Sum", "Forecast_Sum"):
        if col not in results.columns:
            results[col] = np.nan
    missing = results["Actual_Sum"].isna()
    if not missing.any():
        return results, 0

    old = results.loc[missing]
    n = old["n_test"]
    actual = pd.Series(
        np.where(old["WAPE"].isna(), 0.0, n * old["MAE"] / old["WAPE"]), index=old.index
    )
    results.loc[missing, "Actual_Sum"] = actual
    results.loc[missing, "Forecast_Sum"] = actual - n * old["Bias"]
    return results, int(results["Actual_Sum"].isna().sum())


def main() -> None:
    for path in sorted(RESULTS_DIR.glob("*_per_series.csv.gz")):
        match = PER_SERIES.fullmatch(path.name)
        # "combined_*" tables are hand-assembled across tags and have no summary.
        if not match or match["tag"] == "combined":
            continue
        dataset, mode, tag = match["dataset"], match["mode"], match["tag"]
        headline = DATASET_CONFIG[dataset]["headline_metric"]

        results, unrecovered = backfill_volume(pd.read_csv(path))
        results.to_csv(path, index=False)

        stem = f"{tag}_{dataset}_{mode}"
        summarize(results, sort_by=headline).to_csv(RESULTS_DIR / f"{stem}_summary.csv")
        summarize_by_class(results, metric=headline, stat="median").to_csv(
            RESULTS_DIR / f"{stem}_by_class.csv"
        )
        note = f" ({unrecovered} rows with unrecoverable volume)" if unrecovered else ""
        print(f"rebuilt {stem} [{headline}]{note}")


if __name__ == "__main__":
    main()
