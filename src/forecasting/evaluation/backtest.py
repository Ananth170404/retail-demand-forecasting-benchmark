"""Backtesting over an 80/20 temporal split.

The last `test_fraction` of the timeline is reserved as the test region and is
never trained on before it is forecast. Two evaluation modes run over it:

    "rolling" (headline)  h-step forecasts stepped through the test region, so
                          each model is scored on many origins instead of one.
                          The origin expands as it advances, mirroring how a
                          model is retrained as new data arrives in production.

    "holdout" (secondary) one window whose test set is the final `h` periods.
                          Used to reproduce each competition's own split — M5
                          held out 28 days (d_1914-d_1941) and Walmart 39 weeks
                          — so results line up with published work.

Note that neither competition's official test labels were ever released, so the
split has to be carved from the labelled data either way.

A single cut gives one error number per series with no spread; rolling origins
are what let us say a difference between models is consistent rather than lucky.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import evaluate_panel


@dataclass
class Window:
    index: int
    cutoff: pd.Timestamp
    horizon: int
    train: pd.DataFrame
    test: pd.DataFrame


def _apply_min_history(
    train: pd.DataFrame, test: pd.DataFrame, min_train_periods: int | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict both splits to series with enough history in this window.

    Without this, seasonal models emit NaN on short series and those series drop
    out of that model's average only, so models end up compared over different
    subsets of the panel.
    """
    if not min_train_periods:
        return train, test
    counts = train.groupby("unique_id").size()
    eligible = set(counts[counts >= min_train_periods].index)
    return train[train["unique_id"].isin(eligible)], test[test["unique_id"].isin(eligible)]


def make_windows(
    panel: pd.DataFrame,
    h: int,
    test_fraction: float = 0.2,
    n_windows: int | None = None,
    min_train_periods: int | None = None,
    mode: str = "rolling",
    cutoff: str | pd.Timestamp | None = None,
) -> list[Window]:
    """Build evaluation windows over the final `test_fraction` of the timeline.

    In "rolling" mode the test region is divided into non-overlapping h-step
    windows. If `n_windows` is fewer than fit, they are spread evenly across the
    region rather than clustered at its start, so the whole holdout is sampled.

    In "cutoff" mode there is one window: train on everything up to `cutoff`,
    test on the next `h` periods. Used to place the test window on a specific
    stretch of the calendar, e.g. one containing Thanksgiving and Christmas.
    """
    dates = pd.Index(sorted(panel["ds"].unique()))
    n_periods = len(dates)

    if mode == "cutoff":
        if cutoff is None:
            raise ValueError("cutoff mode needs a cutoff date")
        cutoff = pd.Timestamp(cutoff)
        if cutoff not in dates:
            raise ValueError(f"cutoff {cutoff.date()} is not a date in the panel")
        pos = dates.get_loc(cutoff)
        if pos + h >= n_periods:
            raise ValueError(f"only {n_periods - pos - 1} periods after {cutoff.date()}, need {h}")
        test_end = dates[pos + h]
        train = panel[panel["ds"] <= cutoff]
        test = panel[(panel["ds"] > cutoff) & (panel["ds"] <= test_end)]
        train, test = _apply_min_history(train, test, min_train_periods)
        if train.empty or test.empty:
            return []
        return [Window(index=0, cutoff=cutoff, horizon=h, train=train, test=test)]

    if mode == "holdout":
        # Test is the final h periods, matching how each competition defined its
        # own holdout; test_fraction plays no part here.
        split_pos = n_periods - h
        if split_pos < 1:
            return []
        cutoff = dates[split_pos - 1]
        train = panel[panel["ds"] <= cutoff]
        test = panel[panel["ds"] > cutoff]
        train, test = _apply_min_history(train, test, min_train_periods)
        if train.empty or test.empty:
            return []
        return [Window(index=0, cutoff=cutoff, horizon=h, train=train, test=test)]

    if mode != "rolling":
        raise ValueError(f"Unknown mode '{mode}'. Use 'rolling' or 'holdout'.")

    split_pos = int(round((1 - test_fraction) * n_periods))
    if split_pos < 1 or split_pos >= n_periods:
        return []

    max_windows = (n_periods - split_pos) // h
    if max_windows < 1:
        return []

    starts = list(range(split_pos, split_pos + max_windows * h, h))
    if n_windows and n_windows < len(starts):
        picks = np.linspace(0, len(starts) - 1, n_windows).round().astype(int)
        starts = [starts[i] for i in dict.fromkeys(picks)]

    windows = []
    for i, start_pos in enumerate(starts):
        cutoff = dates[start_pos - 1]
        test_end = dates[start_pos + h - 1]
        train = panel[panel["ds"] <= cutoff]
        test = panel[(panel["ds"] > cutoff) & (panel["ds"] <= test_end)]
        train, test = _apply_min_history(train, test, min_train_periods)
        if train.empty or test.empty:
            continue
        windows.append(Window(index=i, cutoff=cutoff, horizon=h, train=train, test=test))

    return windows


def run_backtest(
    model_factory,
    panel: pd.DataFrame,
    h: int,
    season_length: int,
    test_fraction: float = 0.2,
    n_windows: int | None = None,
    min_train_periods: int | None = None,
    mode: str = "rolling",
    return_forecasts: bool = False,
    cutoff: str | pd.Timestamp | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Fit/predict a model across evaluation windows; return per-series metrics.

    `model_factory` is a zero-arg callable returning a fresh Forecaster, so each
    window trains from scratch rather than carrying state across cutoffs.

    With `return_forecasts`, also returns the raw forecasts (tagged with their
    window): needed for metrics computed across series, such as M5's
    hierarchical WRMSSE, and for building ensembles after the fact.
    """
    results = []
    forecasts = []
    windows = make_windows(
        panel,
        h=h,
        test_fraction=test_fraction,
        n_windows=n_windows,
        min_train_periods=min_train_periods,
        mode=mode,
        cutoff=cutoff,
    )

    for window in windows:
        model = model_factory()

        t0 = time.perf_counter()
        model.fit(window.train)
        fit_seconds = time.perf_counter() - t0

        # Covariates for the forecast period are known in advance (calendar,
        # price, SNAP), so handing them to the model is legitimate. `y` is
        # dropped: that is the thing being predicted.
        futr_df = window.test.drop(columns=["y"])

        t1 = time.perf_counter()
        forecast = model.predict(h=window.horizon, futr_df=futr_df)
        predict_seconds = time.perf_counter() - t1

        metrics = evaluate_panel(
            forecast_df=forecast,
            actuals_df=window.test,
            train_df=window.train,
            season_length=season_length,
        )
        metrics["model"] = model.name
        metrics["window"] = window.index
        metrics["cutoff"] = window.cutoff
        metrics["horizon"] = window.horizon
        # Wall-clock for the whole panel, not per series — divide only when
        # comparing per-series cost, and say so when reporting.
        metrics["fit_seconds"] = fit_seconds
        metrics["predict_seconds"] = predict_seconds
        results.append(metrics)
        if return_forecasts:
            forecasts.append(forecast.assign(model=model.name, window=window.index))

        # Each window trains a fresh model. GPU-backed models hold onto device
        # memory until explicitly released, and across a dozen windows that
        # accumulates into an out-of-memory failure partway through a run.
        teardown = getattr(model, "teardown", None)
        if callable(teardown):
            teardown()
        del model

    metrics_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    if return_forecasts:
        forecasts_df = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()
        return metrics_df, forecasts_df
    return metrics_df
