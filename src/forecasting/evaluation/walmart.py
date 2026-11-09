"""The Walmart Store Sales competition metric: holiday-weighted MAE (WMAE).

    WMAE = sum(w * |y - y_hat|) / sum(w),  w = 5 in holiday weeks, else 1,

pooled over every store-department-week in the test set, in dollars. The four
holidays (Super Bowl, Labor Day, Thanksgiving, Christmas) are flagged in
features.csv's IsHoliday column.

Two deliberate differences from Kaggle's scoring, both from how the loader
prepares the target: negative weekly sales (returns, 0.3% of rows) are clipped
to zero, and internal gap weeks are filled with zero sales. Neither changes the
ranking materially, but absolute WMAE values are not leaderboard numbers; the
comparable figure is WMAE relative to seasonal naive on the same window.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HOLIDAY_WEIGHT = 5.0


def holiday_weeks(raw_dir: Path | str) -> pd.DatetimeIndex:
    """Week-ending dates flagged as holidays in features.csv."""
    features = pd.read_csv(Path(raw_dir) / "features.csv", parse_dates=["Date"])
    return pd.DatetimeIndex(sorted(features.loc[features["IsHoliday"], "Date"].unique()))


def wmae(
    forecast: pd.DataFrame,
    actuals: pd.DataFrame,
    holidays: pd.DatetimeIndex,
    pred_col: str = "y_pred",
) -> tuple[float, int]:
    """WMAE over the rows present in both frames; returns (score, rows scored).

    `forecast` has [unique_id, ds, pred_col]; `actuals` has [unique_id, ds, y].
    """
    merged = forecast[["unique_id", "ds", pred_col]].merge(
        actuals[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="inner"
    )
    weight = np.where(merged["ds"].isin(holidays), HOLIDAY_WEIGHT, 1.0)
    error = np.abs(merged["y"].to_numpy() - merged[pred_col].to_numpy())
    return float(np.sum(weight * error) / np.sum(weight)), len(merged)
