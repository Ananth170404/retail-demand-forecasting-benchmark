"""The single contract every model in this benchmark implements.

Keeping one interface is what makes the comparison trustworthy: each model sees
exactly the same training frame and is asked for exactly the same horizon, so
differences in the results table come from the models rather than from
differences in how they were driven.

Panel format used throughout (Nixtla convention):
    unique_id : series identifier
    ds        : timestamp
    y         : target value
Exogenous features, when a model uses them, ride along as extra columns.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class Forecaster(Protocol):
    name: str

    def fit(self, train_df: pd.DataFrame) -> Forecaster:
        """Fit on a long panel with columns [unique_id, ds, y, *exog]."""
        ...

    def predict(self, h: int, futr_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Return [unique_id, ds, y_pred] for h steps past the training cutoff.

        Models that produce quantiles add y_pred_q{level} columns alongside.
        `futr_df` carries known-future exogenous values for models that use them.
        """
        ...


def to_panel(
    df: pd.DataFrame,
    id_col: str = "product_id",
    date_col: str = "demand_date",
    target_col: str = "total_demand_qty",
    exog_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Convert a loader's product_df into the [unique_id, ds, y, *exog] panel."""
    cols = {id_col: "unique_id", date_col: "ds", target_col: "y"}
    keep = list(cols) + (exog_cols or [])
    panel = df[keep].rename(columns=cols)
    if _is_sorted(panel):
        # The loaders already sort; re-sorting full M5 (60M rows) would copy
        # ~2GB for nothing.
        return panel.reset_index(drop=True)
    return panel.sort_values(["unique_id", "ds"]).reset_index(drop=True)


def _is_sorted(panel: pd.DataFrame) -> bool:
    """True when rows are ordered by unique_id, then strictly by ds within it."""
    ids = panel["unique_id"].to_numpy()
    ds = panel["ds"].to_numpy()
    if len(ids) < 2:
        return True
    same = ids[1:] == ids[:-1]
    return bool(np.all(ids[1:][~same] > ids[:-1][~same]) and np.all(ds[1:][same] > ds[:-1][same]))


def varying_columns(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """The subset of `cols` present in `df` that take more than one value.

    A covariate that never varies in the training window carries no learnable
    signal, but a neural model will still multiply it by weights at predict
    time. Walmart's MarkDown columns are the live case: absent (zero) before
    Nov 2011 and large afterwards, so a model trained before then saw a
    constant and was then fed values it had never seen, inflating errors up to
    46x. Trees are immune (a constant column is never split on); networks
    must drop such columns explicitly.
    """
    return [c for c in cols if c in df.columns and df[c].nunique(dropna=False) > 1]


def trim_leading_zeros_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Start each series at its first non-zero observation.

    A product that has not launched yet records zeros, but those zeros are "not
    on the shelf", not "nobody bought it". Training on them teaches a model that
    a live product is dead, and it corrupts lag features: a 28-day lag reaching
    back across a launch date is meaningless. In M5 this is not a rare edge case
    — 76% of series open with a zero run, median 464 days.

    Series that never sell anything are dropped: they carry no demand signal and
    have no defined error scale.
    """
    first_sale = panel.loc[panel["y"] > 0].groupby("unique_id")["ds"].min()
    trimmed = panel.join(first_sale.rename("_first_sale"), on="unique_id")
    trimmed = trimmed[trimmed["ds"] >= trimmed["_first_sale"]]
    return trimmed.drop(columns="_first_sale").reset_index(drop=True)
