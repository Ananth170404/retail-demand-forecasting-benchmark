import numpy as np
import pandas as pd
import pytest

from forecasting.evaluation.backtest import make_windows
from forecasting.models.base import to_panel


@pytest.fixture
def panel():
    """100 periods, two series — 80/20 puts the split at period 80."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    return pd.DataFrame(
        {
            "unique_id": ["A"] * 100 + ["B"] * 100,
            "ds": list(dates) * 2,
            "y": np.arange(200, dtype=float),
        }
    )


def test_no_leakage_train_strictly_precedes_test(panel):
    """The property that makes every downstream metric trustworthy."""
    for window in make_windows(panel, h=5, test_fraction=0.2):
        assert window.train["ds"].max() < window.test["ds"].min()


def test_rolling_windows_fill_the_test_region(panel):
    """20 held-out periods at h=5 gives four non-overlapping windows."""
    windows = make_windows(panel, h=5, test_fraction=0.2)
    assert len(windows) == 4
    for window in windows:
        assert window.horizon == 5
        assert window.test.groupby("unique_id").size().eq(5).all()


def test_first_cutoff_sits_on_the_80_20_boundary(panel):
    windows = make_windows(panel, h=5, test_fraction=0.2)
    # 80 training periods before the first forecast origin.
    assert windows[0].train.groupby("unique_id").size().eq(80).all()


def test_no_test_period_precedes_the_split(panel):
    """Nothing in the holdout region may appear in any window's training data."""
    split_date = sorted(panel["ds"].unique())[79]
    for window in make_windows(panel, h=5, test_fraction=0.2):
        assert window.test["ds"].min() > split_date


def test_origins_expand_monotonically(panel):
    windows = make_windows(panel, h=5, test_fraction=0.2)
    cutoffs = [w.cutoff for w in windows]
    assert cutoffs == sorted(cutoffs)
    assert [len(w.train) for w in windows] == sorted(len(w.train) for w in windows)


def test_holdout_mode_tests_on_final_h_periods(panel):
    """Reproduces a competition's own split: everything before the last h periods."""
    windows = make_windows(panel, h=20, mode="holdout")
    assert len(windows) == 1
    assert windows[0].horizon == 20
    assert windows[0].test.groupby("unique_id").size().eq(20).all()
    assert windows[0].train.groupby("unique_id").size().eq(80).all()


def test_holdout_mode_ignores_test_fraction(panel):
    """h alone defines the holdout here, so test_fraction must not shift it."""
    a = make_windows(panel, h=20, test_fraction=0.2, mode="holdout")[0]
    b = make_windows(panel, h=20, test_fraction=0.5, mode="holdout")[0]
    assert a.cutoff == b.cutoff
    assert a.horizon == b.horizon == 20


def test_holdout_mode_has_no_leakage(panel):
    window = make_windows(panel, h=20, mode="holdout")[0]
    assert window.train["ds"].max() < window.test["ds"].min()


def test_n_windows_subsamples_across_the_region(panel):
    """Fewer windows than fit should sample the whole holdout, not just its start."""
    windows = make_windows(panel, h=5, test_fraction=0.2, n_windows=2)
    assert len(windows) == 2
    all_windows = make_windows(panel, h=5, test_fraction=0.2)
    assert windows[-1].cutoff == all_windows[-1].cutoff


def test_min_train_periods_drops_short_series_from_train_and_test():
    """Regression: seasonal models emit NaN on series shorter than one cycle.

    If short series are left in, those NaNs drop out of the seasonal model's
    average only, so it gets scored on an easier subset than its competitors.
    """
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    long_series = pd.DataFrame({"unique_id": "long", "ds": dates, "y": np.arange(30.0)})
    short_series = pd.DataFrame({"unique_id": "short", "ds": dates[-8:], "y": np.arange(8.0)})
    panel = pd.concat([long_series, short_series], ignore_index=True)

    windows = make_windows(panel, h=3, test_fraction=0.2, min_train_periods=20)
    assert windows
    for window in windows:
        assert set(window.train["unique_id"]) == {"long"}
        assert set(window.test["unique_id"]) == {"long"}


def test_horizon_larger_than_test_region_yields_no_windows(panel):
    assert make_windows(panel, h=50, test_fraction=0.2) == []


def test_unknown_mode_raises(panel):
    with pytest.raises(ValueError, match="Unknown mode"):
        make_windows(panel, h=5, mode="expanding")


def test_to_panel_renames_and_sorts():
    raw = pd.DataFrame(
        {
            "product_id": ["B", "A", "A"],
            "demand_date": pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-01"]),
            "total_demand_qty": [3.0, 2.0, 1.0],
        }
    )
    panel = to_panel(raw)
    assert list(panel.columns) == ["unique_id", "ds", "y"]
    assert panel["unique_id"].tolist() == ["A", "A", "B"]
    assert panel["y"].tolist() == [1.0, 2.0, 3.0]


def test_to_panel_keeps_sorted_input_and_sorts_unsorted():
    from forecasting.models.base import _is_sorted, to_panel

    df = pd.DataFrame(
        {
            "product_id": ["a", "a", "b", "b"],
            "demand_date": pd.to_datetime(["2020-01-01", "2020-01-02"] * 2),
            "total_demand_qty": [1, 2, 3, 4],
        }
    )
    assert _is_sorted(to_panel(df))
    shuffled = to_panel(df.iloc[[3, 0, 2, 1]])
    assert _is_sorted(shuffled)
    assert shuffled["y"].tolist() == [1, 2, 3, 4]
    # Same id twice on one date is not "sorted": ds must strictly increase.
    dup = df.assign(demand_date=pd.to_datetime(["2020-01-01"] * 4))
    assert not _is_sorted(dup.rename(columns={"product_id": "unique_id", "demand_date": "ds"}))


def test_run_backtest_can_return_forecasts():
    from forecasting.evaluation.backtest import run_backtest
    from forecasting.models.baselines import ZeroForecaster

    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    panel = pd.DataFrame({"unique_id": "a", "ds": dates, "y": range(1, 21)})
    metrics, forecasts = run_backtest(
        lambda: ZeroForecaster("D"),
        panel,
        h=5,
        season_length=1,
        mode="holdout",
        return_forecasts=True,
    )
    assert len(metrics) == 1
    assert len(forecasts) == 5 and set(forecasts.columns) >= {"unique_id", "ds", "y_pred", "window"}


def test_varying_columns_drops_constants_and_missing():
    from forecasting.models.base import varying_columns

    df = pd.DataFrame({"a": [1, 2, 3], "markdown": [0.0, 0.0, 0.0], "c": ["x", "x", "y"]})
    assert varying_columns(df, ["a", "markdown", "c", "absent"]) == ["a", "c"]
