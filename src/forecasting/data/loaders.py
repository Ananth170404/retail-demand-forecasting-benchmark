"""Generic loaders that map raw dataset files onto a single shared schema.

Every loader returns ``(product_df, roles)`` where ``product_df`` has:
    demand_date, product_id, total_demand_qty
    plus one integer-encoded "<name>_cat" column per static/dynamic categorical
    plus one float column per dynamic real feature

and ``roles`` (a ColumnRoles) records which of those columns play which role,
so downstream model code never hardcodes column names.

Run `python scripts/download_data.py --dataset all` first to populate
data/raw/{m5,walmart}/.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .calendar import HOLIDAY_FEATURES, holiday_features

DATA_ROOT = Path(os.environ.get("DEMAND_DATA_ROOT", "data/raw"))


@dataclass
class ColumnRoles:
    target_col: str = "total_demand_qty"
    date_col: str = "demand_date"
    id_col: str = "product_id"
    static_cat_cols: list[str] = field(default_factory=list)
    dynamic_cat_cols: list[str] = field(default_factory=list)
    dynamic_real_cols: list[str] = field(default_factory=list)
    # Holiday-position features (data/calendar.py). Kept apart from the other
    # covariates so models opt in explicitly, and results without them stay
    # comparable with earlier runs.
    holiday_cols: list[str] = field(default_factory=list)


def _encode_categoricals(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Label-encode each column in place; return the new '<col>_cat' names.

    Codes are stored as int16 rather than the int64 LabelEncoder returns. These
    columns hold at most a few dozen distinct values, and at M5 scale the
    difference is ~500MB across six columns — enough to decide whether the panel
    loads at all on a 16GB laptop.
    """
    encoded_cols = []
    for col in cols:
        encoded_col = f"{col}_cat"
        df[encoded_col] = LabelEncoder().fit_transform(df[col].astype(str)).astype(np.int16)
        encoded_cols.append(encoded_col)
    return encoded_cols


def _codes(values: np.ndarray) -> np.ndarray:
    """Integer codes in sorted order of the values present, as LabelEncoder assigns.

    Only short per-series or per-day arrays come through here; the full-length
    SNAP column is coded directly, since sorting 60M values to recover 0/1 would
    dominate the loader's peak memory.
    """
    return np.unique(values.astype(str), return_inverse=True)[1].astype(np.int16)


def load_m5(
    raw_dir: Path | str | None = None,
    states: list[str] | None = None,
    max_series_per_state: int | None = 200,
    include_test: bool = False,
) -> tuple[pd.DataFrame, ColumnRoles]:
    """Load M5 Forecasting - Accuracy into the shared product_df schema.

    The full dataset is 30,490 series x 1,941 days (~59M rows long). Set
    `max_series_per_state` to None to load everything; the default keeps a
    manageable subsample (item x store series) for a laptop-scale run.

    `include_test` appends the 28 days of the competition's final test period
    (d_1942-d_1969) from sales_test_evaluation.csv, so a holdout of the last 28
    days reproduces the competition split exactly. Fetch the labels with
    `python scripts/download_data.py --dataset m5-labels`.

    The long frame is assembled from numpy arrays rather than DataFrame.melt:
    melting repeats every id column as Python strings across all rows, which at
    full scale needs more memory than a 16GB laptop has free.
    """
    raw_dir = Path(raw_dir or DATA_ROOT / "m5")
    sales_path = raw_dir / "sales_train_evaluation.csv"
    if not sales_path.exists():
        if include_test:
            raise FileNotFoundError("include_test needs sales_train_evaluation.csv (d_1-d_1941)")
        sales_path = raw_dir / "sales_train_validation.csv"

    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(raw_dir / "calendar.csv")
    prices = pd.read_csv(raw_dir / "sell_prices.csv")

    if states:
        sales = sales[sales["state_id"].isin(states)]
    if max_series_per_state:
        sales = sales.groupby("state_id", group_keys=False).apply(
            lambda g: g.sample(n=min(max_series_per_state, len(g)), random_state=42),
            include_groups=False,
        )
        # include_groups=False drops state_id (the group key) from the sampled
        # frame; store_id (e.g. "CA_1") still carries it, so recover it from there.
        sales["state_id"] = sales["store_id"].str.split("_").str[0]

    if include_test:
        test = pd.read_csv(raw_dir / "sales_test_evaluation.csv")
        test_days = [c for c in test.columns if c.startswith("d_")]
        sales = sales.merge(
            test[["item_id", "store_id", *test_days]], on=["item_id", "store_id"], how="left"
        )
        if sales[test_days].isna().any().any():
            raise ValueError("sales_test_evaluation.csv is missing some series")

    sales["product_id"] = (
        sales["id"]
        .str.replace("_evaluation", "", regex=False)
        .str.replace("_validation", "", regex=False)
    )
    sales = sales.sort_values("product_id", kind="stable").reset_index(drop=True)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    n_series, n_days = len(sales), len(day_cols)

    cal = calendar.set_index("d").loc[day_cols]
    dates = pd.to_datetime(cal["date"]).to_numpy()
    weeks = cal["wm_yr_wk"].to_numpy()

    # Prices are weekly per store-item: lay them out series x day, then fill
    # each series' gaps from its own neighbouring weeks (forward, then back).
    # pivot, not pivot_table: each store-item-week has exactly one price, and
    # pivot_table's aggregation machinery more than doubles peak memory here.
    price_table = prices.pivot(
        index=["store_id", "item_id"], columns="wm_yr_wk", values="sell_price"
    )
    series_keys = pd.MultiIndex.from_frame(sales[["store_id", "item_id"]])
    price = price_table.reindex(index=series_keys, columns=weeks)
    price = price.ffill(axis=1).bfill(axis=1).fillna(0.0).to_numpy(np.float32)

    # SNAP days are per state: pick each series' state row of the calendar.
    snap_by_state = cal[["snap_CA", "snap_TX", "snap_WI"]].to_numpy().T
    state_row = sales["state_id"].map({"CA": 0, "TX": 1, "WI": 2}).to_numpy()
    snap = snap_by_state[state_row]

    product_df = pd.DataFrame(
        {
            "demand_date": np.tile(dates, n_series),
            "product_id": np.repeat(sales["product_id"].to_numpy(), n_days),
            "total_demand_qty": sales[day_cols].to_numpy(np.int32).ravel(),
            "sell_price": price.ravel(),
        }
    )
    static_cat_cols = []
    for col in ["dept_id", "cat_id", "store_id", "state_id"]:
        product_df[f"{col}_cat"] = np.repeat(_codes(sales[col].to_numpy()), n_days)
        static_cat_cols.append(f"{col}_cat")
    event = cal["event_type_1"].fillna("none").to_numpy()
    product_df["event_type_1_cat"] = np.tile(_codes(event), n_series)
    # SNAP flags are 0/1, so their sorted codes are the flags themselves.
    product_df["snap_cat"] = (
        snap.ravel() if len(np.unique(snap)) > 1 else 0 * snap.ravel()
    ).astype(np.int16)
    dynamic_cat_cols = ["event_type_1_cat", "snap_cat"]

    del sales, price, snap
    gc.collect()

    roles = ColumnRoles(
        static_cat_cols=static_cat_cols,
        dynamic_cat_cols=dynamic_cat_cols,
        dynamic_real_cols=["sell_price"],
    )
    return product_df, roles


def _reindex_walmart_weeks(train: pd.DataFrame) -> pd.DataFrame:
    """Fill internal gaps in each store-department series with zero sales.

    Walmart records no row for a week with no transactions, so 605 of 3,331
    series have holes. Left unfilled, seasonal models index by row position and
    "52 weeks ago" silently points at the wrong calendar week.

    Filling with zero is supported by the data: gapped series have a median
    weekly revenue of $78 against $9,058 for complete series, and the weeks
    immediately bracketing a gap have a median of ~$11, with 80% under $50.
    These are near-dormant departments whose missing weeks are genuinely empty,
    not unobserved — a rolling-average fill would invent demand that never
    happened and erase the intermittency Croston/TSB are meant to model.

    Only *internal* gaps are filled. A series starting late or ending early
    reflects a department opening or closing, so extending it would fabricate
    history for a department that did not yet (or no longer) exist.
    """
    grid = pd.Index(sorted(train["Date"].unique()), name="Date")
    filled = []

    for (store, dept), g in train.groupby(["Store", "Dept"], sort=False):
        g = g.sort_values("Date")
        span = grid[grid.get_loc(g["Date"].min()) : grid.get_loc(g["Date"].max()) + 1]
        if len(span) == len(g):
            filled.append(g)
            continue

        reindexed = g.set_index("Date").reindex(span)
        reindexed["Weekly_Sales"] = reindexed["Weekly_Sales"].fillna(0.0)
        reindexed["Store"] = store
        reindexed["Dept"] = dept
        filled.append(reindexed.reset_index())

    out = pd.concat(filled, ignore_index=True)
    # Set here rather than only on the reindexed branch, so the helper does not
    # depend on the caller having assigned product_id first.
    out["product_id"] = out["Store"].astype(str) + "_" + out["Dept"].astype(str)
    return out


def load_walmart(
    raw_dir: Path | str | None = None,
    fill_internal_gaps: bool = True,
) -> tuple[pd.DataFrame, ColumnRoles]:
    """Load the Walmart Store Sales dataset into the shared product_df schema.

    Weekly frequency out of the box (no resampling). Used primarily for the
    macro-covariate ablation experiment (CPI, Unemployment, Fuel_Price).
    """
    raw_dir = Path(raw_dir or DATA_ROOT / "walmart")
    train = pd.read_csv(raw_dir / "train.csv", parse_dates=["Date"])
    features = pd.read_csv(raw_dir / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(raw_dir / "stores.csv")

    train["product_id"] = train["Store"].astype(str) + "_" + train["Dept"].astype(str)
    if fill_internal_gaps:
        train = _reindex_walmart_weeks(train)

    # IsHoliday is taken from features (defined for every store-week) rather than
    # train, so gap-filled rows get the correct flag instead of NaN.
    df = (
        train.drop(columns=["IsHoliday"], errors="ignore")
        .merge(features, on=["Store", "Date"], how="left")
        .merge(stores, on="Store", how="left")
    )

    markdown_cols = [f"MarkDown{i}" for i in range(1, 6)]
    df[markdown_cols] = df[markdown_cols].fillna(0.0)
    macro_cols = ["CPI", "Unemployment", "Fuel_Price", "Temperature"]
    df[macro_cols] = df.groupby("Store")[macro_cols].transform(lambda s: s.ffill().bfill())

    df["demand_date"] = df["Date"]
    # Weekly_Sales is dollar revenue and can be negative when returns exceed
    # sales (0.3% of rows). Clipped to 0 because the task is demand forecasting.
    df["total_demand_qty"] = df["Weekly_Sales"].clip(lower=0)
    df["is_holiday"] = df["IsHoliday"].astype(str)

    static_cat_cols = _encode_categoricals(df, ["Type"])
    dynamic_cat_cols = _encode_categoricals(df, ["is_holiday"])
    dynamic_real_cols = macro_cols + markdown_cols

    # Computed per distinct week (143 of them), then joined onto the rows.
    weeks = pd.Series(sorted(df["demand_date"].unique()))
    calendar = holiday_features(weeks).assign(demand_date=weeks.to_numpy())
    df = df.merge(calendar, on="demand_date", how="left")

    keep = (
        ["demand_date", "product_id", "total_demand_qty"]
        + static_cat_cols
        + dynamic_cat_cols
        + dynamic_real_cols
        + HOLIDAY_FEATURES
    )
    product_df = df[keep].sort_values(["product_id", "demand_date"]).reset_index(drop=True)

    roles = ColumnRoles(
        static_cat_cols=static_cat_cols,
        dynamic_cat_cols=dynamic_cat_cols,
        dynamic_real_cols=dynamic_real_cols,
        holiday_cols=list(HOLIDAY_FEATURES),
    )
    return product_df, roles


LOADERS = {"m5": load_m5, "walmart": load_walmart}


def load_dataset(name: str, **kwargs) -> tuple[pd.DataFrame, ColumnRoles]:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(LOADERS)}.")
    return LOADERS[name](**kwargs)
