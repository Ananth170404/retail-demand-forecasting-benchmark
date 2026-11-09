"""Holiday-position features for weekly retail data.

A 52-week seasonal lag looks back 364 days, but the calendar year is 365 or
366, so "the same week last year" drifts by a day or two each year. For
holidays tied to a weekday (Thanksgiving, Super Bowl, Labor Day) that is
harmless: they stay in the same week. Christmas is a fixed date, so the number
of pre-Christmas shopping days in each week shifts, and Easter moves by weeks.
Every top-eight Walmart Kaggle team corrected for this.

These features describe where each week sits relative to those holidays, so a
model can learn holiday effects from position rather than from a lag that may
be misaligned. All are known in advance, so they are valid future covariates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dateutil.easter import easter

HOLIDAY_FEATURES = [
    "xmas_shop_days",
    "weeks_to_xmas",
    "weeks_after_xmas",
    "weeks_to_thanksgiving",
    "weeks_to_easter",
]


def _thanksgiving(year: int) -> pd.Timestamp:
    """Fourth Thursday of November."""
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    first_thursday = nov1 + pd.Timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thursday + pd.Timedelta(weeks=3)


def holiday_features(week_end: pd.Series, week_days: int = 7) -> pd.DataFrame:
    """Holiday-position features for weeks ending on `week_end` (inclusive).

    xmas_shop_days         days of the final pre-Christmas week (Dec 18-24)
                           falling inside this week: 0-7
    weeks_to_xmas          whole weeks until the week containing Dec 25, 0-8
                           (8 = further away)
    weeks_after_xmas       weeks since the week containing Dec 25, 0-4
                           (4 = further away): the January slump
    weeks_to_thanksgiving  signed weeks to the week containing Thanksgiving,
                           clipped to -2..6
    weeks_to_easter        signed weeks to the week containing Easter Sunday,
                           clipped to -3..3
    """
    end = pd.to_datetime(week_end).reset_index(drop=True)
    start = end - pd.Timedelta(days=week_days - 1)
    out = pd.DataFrame(index=end.index)

    def weeks_between(target: pd.Series) -> np.ndarray:
        # Week offset of the week containing `target`: 0 when inside this week.
        return np.floor((target - start).dt.days / week_days).to_numpy()

    year = end.dt.year
    # Christmas this season: the next Dec 25 on or after the week's start.
    xmas_this = pd.to_datetime(dict(year=start.dt.year, month=12, day=25))
    xmas_next = xmas_this.where(xmas_this >= start, xmas_this + pd.DateOffset(years=1))
    xmas_prev = pd.to_datetime(dict(year=end.dt.year, month=12, day=25))
    xmas_prev = xmas_prev.where(xmas_prev <= end, xmas_prev - pd.DateOffset(years=1))

    shop_lo = xmas_next - pd.Timedelta(days=7)
    shop_hi = xmas_next - pd.Timedelta(days=1)
    overlap = (np.minimum(end, shop_hi) - np.maximum(start, shop_lo)).dt.days + 1
    out["xmas_shop_days"] = overlap.clip(lower=0, upper=7)
    out["weeks_to_xmas"] = np.clip(weeks_between(xmas_next), 0, 8)
    out["weeks_after_xmas"] = np.clip(-weeks_between(xmas_prev), 0, 4) + 0.0  # no -0.0

    years = range(int(year.min()) - 1, int(year.max()) + 2)
    thanks = {y: _thanksgiving(y) for y in years}
    easters = {y: pd.Timestamp(easter(y)) for y in years}

    def nearest(dates: dict[int, pd.Timestamp]) -> pd.Series:
        # The holiday closest to each week, whichever year it falls in.
        candidates = np.array(sorted(dates.values()), dtype="datetime64[ns]")
        mid = (start + (end - start) / 2).to_numpy()
        idx = np.abs(mid[:, None] - candidates[None, :]).argmin(axis=1)
        return pd.Series(candidates[idx])

    out["weeks_to_thanksgiving"] = np.clip(weeks_between(nearest(thanks)), -2, 6)
    out["weeks_to_easter"] = np.clip(weeks_between(nearest(easters)), -3, 3)
    return out.astype(np.float32)
