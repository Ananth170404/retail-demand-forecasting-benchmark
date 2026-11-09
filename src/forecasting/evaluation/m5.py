"""The official M5 Accuracy metric: WRMSSE over the 12-level sales hierarchy.

M5 asked for forecasts of the 30,490 product-store series, but scored them after
summing up the hierarchy (total, state, store, category, department, item, and
their crossings: 42,840 series in all). Each aggregated series gets an RMSSE,
each level a dollar-weighted average of those, and the final score is the plain
mean of the 12 level scores. Aggregate levels are smooth and forecastable, so
this rewards getting totals right, not just individual items: the winner's
edge over simple benchmarks came mostly from levels 1-9.

Two details differ from this repo's general-purpose RMSSE in metrics.py and
must, for the score to be comparable with the leaderboard:

    scale    mean squared *one-step* difference of the in-sample history,
             from each series' first non-zero sale onward. metrics.py scales
             by the seasonal (7-day) difference instead.
    weights  each series' dollar sales over the final 28 days of training,
             normalised to sum to 1 within each level.

Validated against the official release (tests/test_m5_wrmsse.py): the weights
computed here match weights_evaluation.csv, and seasonal naive reproduces the
published WRMSSE of 0.847 overall and Table B's per-level scores.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

HIERARCHY_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Level -> grouping columns, in the order the M5 guide numbers them.
LEVELS: dict[int, list[str]] = {
    1: [],
    2: ["state_id"],
    3: ["store_id"],
    4: ["cat_id"],
    5: ["dept_id"],
    6: ["state_id", "cat_id"],
    7: ["state_id", "dept_id"],
    8: ["store_id", "cat_id"],
    9: ["store_id", "dept_id"],
    10: ["item_id"],
    11: ["state_id", "item_id"],
    12: ["item_id", "store_id"],
}

WEIGHT_WINDOW = 28


def _summing_matrix(keys: pd.DataFrame, cols: list[str]) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    """Sparse 0/1 matrix mapping bottom series onto one level's groups.

    Returns the matrix and the group labels for its rows (sorted by `cols`),
    so per-level outputs can be matched to series by name, never by position.
    """
    n = len(keys)
    if not cols:
        return sparse.csr_matrix(np.ones((1, n), dtype=np.float32)), pd.DataFrame(index=[0])
    codes = keys.groupby(cols, sort=True).ngroup().to_numpy()
    labels = keys[cols].assign(_code=codes).drop_duplicates("_code").sort_values("_code")
    matrix = sparse.csr_matrix(
        (np.ones(n, dtype=np.float32), (codes, np.arange(n))), shape=(codes.max() + 1, n)
    )
    return matrix, labels.drop(columns="_code").reset_index(drop=True)


def _active_scale(history: np.ndarray) -> np.ndarray:
    """Mean squared one-step difference from each row's first non-zero onward.

    Vectorised over rows. A difference at position j (between t=j and t=j+1)
    counts once t=j is on or after the first sale, matching the reference
    implementation that slices each series at its first non-zero value.
    """
    has_sale = history > 0
    first = np.where(has_sale.any(axis=1), has_sale.argmax(axis=1), history.shape[1])
    diffs = np.diff(history, axis=1).astype(np.float64) ** 2
    active = np.arange(diffs.shape[1])[None, :] >= first[:, None]
    count = active.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(count > 0, (diffs * active).sum(axis=1) / count, np.nan)
    # A history that never changes has no error scale; undefined, not infinite.
    return np.where(scale > 0, scale, np.nan)


def dollar_sales(
    sales: pd.DataFrame, train_days: list[str], calendar: pd.DataFrame, prices: pd.DataFrame
) -> np.ndarray:
    """Each bottom series' revenue over the final 28 training days.

    Days with no listed price contribute nothing: the item was not on sale,
    and its recorded units are zero there anyway.
    """
    window = train_days[-WEIGHT_WINDOW:]
    weeks = calendar.set_index("d").loc[window, "wm_yr_wk"].to_numpy()
    price_table = prices.pivot_table(
        index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price"
    )
    idx = pd.MultiIndex.from_frame(sales[["store_id", "item_id"]])
    price = price_table.reindex(index=idx, columns=weeks).to_numpy()
    units = sales[window].to_numpy(dtype=np.float64)
    return np.nansum(units * price, axis=1)


class WRMSSEEvaluator:
    """Precomputes scales and weights once, then scores any number of forecasts.

    `sales` holds one row per bottom series with HIERARCHY_COLS and one column
    per training day (d_1 ... d_N), as in sales_train_evaluation.csv.
    Forecasts are scored over the next len(forecast columns) days.
    `bottom_dollars` overrides the per-series revenue the weights derive from.
    """

    def __init__(
        self,
        sales: pd.DataFrame,
        calendar: pd.DataFrame,
        prices: pd.DataFrame,
        bottom_dollars: np.ndarray | None = None,
    ):
        self.keys = sales[HIERARCHY_COLS].reset_index(drop=True)
        self.train_days = [c for c in sales.columns if c.startswith("d_")]
        history = sales[self.train_days].to_numpy(dtype=np.float32)
        if bottom_dollars is None:
            bottom_dollars = dollar_sales(sales, self.train_days, calendar, prices)

        self.matrices: dict[int, sparse.csr_matrix] = {}
        self.groups: dict[int, pd.DataFrame] = {}
        self.scales: dict[int, np.ndarray] = {}
        self.weights: dict[int, np.ndarray] = {}
        for level, cols in LEVELS.items():
            S, self.groups[level] = _summing_matrix(self.keys, cols)
            self.matrices[level] = S
            self.scales[level] = _active_scale(np.asarray(S @ history))
            level_dollars = S @ bottom_dollars
            self.weights[level] = level_dollars / level_dollars.sum()

    def _align(self, frame: pd.DataFrame) -> np.ndarray:
        """Order a (item_id, store_id)-indexed wide frame like the training rows."""
        idx = pd.MultiIndex.from_frame(self.keys[["item_id", "store_id"]])
        aligned = frame.reindex(idx)
        if aligned.isna().any().any():
            missing = int(aligned.isna().any(axis=1).sum())
            raise ValueError(f"{missing} bottom series have missing values")
        return aligned.to_numpy(dtype=np.float64)

    def score(self, actuals: pd.DataFrame, forecast: pd.DataFrame) -> pd.Series:
        """Per-level WRMSSE plus the overall mean, as a Series indexed 1..12, 'WRMSSE'.

        Both frames are wide: index (item_id, store_id), one column per test day.
        """
        y = self._align(actuals)
        f = self._align(forecast[actuals.columns])
        out = {}
        for level, S in self.matrices.items():
            mse = np.asarray(np.square(S @ y - S @ f)).mean(axis=1)
            rmsse = np.sqrt(mse / self.scales[level])
            # A series with no defined scale (never sold) carries zero weight
            # in the official computation; nan_to_num keeps it out of the sum.
            out[level] = float(np.nansum(self.weights[level] * np.nan_to_num(rmsse)))
        scores = pd.Series(out, name="WRMSSE")
        scores["WRMSSE"] = scores.mean()
        return scores


def panel_to_wide(panel: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Long [unique_id, ds, value] -> wide (item_id, store_id) x day frame.

    unique_id is the loader's product_id, "<item_id>_<store_id>", e.g.
    "FOODS_1_001_CA_1"; the store is always the last two underscore parts.
    """
    wide = panel.pivot(index="unique_id", columns="ds", values=value_col)
    parts = wide.index.str.rsplit("_", n=2)
    wide.index = pd.MultiIndex.from_arrays(
        [parts.str[0], parts.str[1] + "_" + parts.str[2]], names=["item_id", "store_id"]
    )
    return wide


def load_release(raw_dir, phase: str = "evaluation") -> tuple[WRMSSEEvaluator, pd.DataFrame]:
    """Evaluator plus actuals for one competition phase, from data/raw/m5.

    evaluation  train d_1-d_1941, score d_1942-d_1969 (the final, private test)
    validation  train d_1-d_1913, score d_1914-d_1941 (the public leaderboard)

    Weights and scales come from that phase's training days, as officially.
    """
    from pathlib import Path

    if phase not in ("evaluation", "validation"):
        raise ValueError(f"phase must be 'evaluation' or 'validation', got {phase!r}")
    raw_dir = Path(raw_dir)
    sales = pd.read_csv(raw_dir / f"sales_train_{phase}.csv")
    calendar = pd.read_csv(raw_dir / "calendar.csv")
    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    evaluator = WRMSSEEvaluator(sales, calendar, prices)
    test = pd.read_csv(raw_dir / f"sales_test_{phase}.csv")
    days = [c for c in test.columns if c.startswith("d_")]
    actuals = test.set_index(["item_id", "store_id"])[days]
    return evaluator, actuals


def load_window(
    raw_dir, end_day: int, h: int = 28, sales: pd.DataFrame | None = None
) -> tuple[WRMSSEEvaluator, pd.DataFrame]:
    """Evaluator plus actuals for any window: train d_1-d_{end_day}, score the
    next `h` days, with weights and scales from the training days as in the
    official phases (end_day=1913 reproduces the validation phase).

    Earlier windows let choices be validated on several months rather than
    one; `sales` (sales_train_evaluation.csv) can be passed to load it once.
    """
    from pathlib import Path

    raw_dir = Path(raw_dir)
    if sales is None:
        sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    days = [f"d_{i}" for i in range(end_day + 1, end_day + h + 1)]
    missing = [d for d in days if d not in sales.columns]
    if missing:
        raise ValueError(f"window ends after the labelled data: {missing[0]}")
    train = sales[HIERARCHY_COLS + [f"d_{i}" for i in range(1, end_day + 1)]]
    calendar = pd.read_csv(raw_dir / "calendar.csv")
    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    evaluator = WRMSSEEvaluator(train, calendar, prices)
    actuals = sales.set_index(["item_id", "store_id"])[days]
    return evaluator, actuals


def score_forecast(
    evaluator: WRMSSEEvaluator, actuals: pd.DataFrame, forecast: pd.DataFrame
) -> tuple[pd.Series, int]:
    """WRMSSE for a long [unique_id, ds, y_pred] forecast over the test period.

    Series the forecast omits (e.g. never-sold items dropped by the zero trim,
    or series below a model's minimum history) are forecast as zero, the
    natural forecast for an item with no sales; the count is returned so it
    can be reported rather than hidden.
    """
    wide = panel_to_wide(forecast, "y_pred")
    if wide.shape[1] != actuals.shape[1]:
        raise ValueError(f"forecast covers {wide.shape[1]} days, test has {actuals.shape[1]}")
    wide.columns = actuals.columns
    idx = pd.MultiIndex.from_frame(evaluator.keys[["item_id", "store_id"]])
    filled = int((~idx.isin(wide.index)).sum())
    wide = wide.reindex(idx).fillna(0.0)
    return evaluator.score(actuals, wide), filled
