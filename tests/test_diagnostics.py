import numpy as np
import pandas as pd
import pytest

from forecasting.evaluation.metrics import (
    bias,
    mdae,
    series_diagnostics,
    smape,
    summarize,
    trimmed_mean,
    wape,
)
from forecasting.models.base import trim_leading_zeros_panel

# --- new accuracy metrics -------------------------------------------------


def test_bias_sign_marks_under_forecasting():
    y = np.array([10.0, 10.0])
    assert bias(y, np.array([8.0, 8.0])) == pytest.approx(2.0)  # under-forecast
    assert bias(y, np.array([12.0, 12.0])) == pytest.approx(-2.0)  # over-forecast


def test_bias_cancels_where_mae_does_not():
    """Two symmetric errors net to zero bias but non-zero MAE — the point of tracking both."""
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([5.0, 15.0])
    assert bias(y_true, y_pred) == pytest.approx(0.0)
    assert mdae(y_true, y_pred) == pytest.approx(5.0)


def test_wape_is_total_error_over_total_demand():
    y_true = np.array([0.0, 10.0, 30.0])
    y_pred = np.array([1.0, 12.0, 27.0])
    # |1| + |2| + |3| = 6 over 40
    assert wape(y_true, y_pred) == pytest.approx(6 / 40)


def test_wape_survives_zeros_that_would_break_mape():
    y_true = np.array([0.0, 0.0, 5.0])
    assert np.isfinite(wape(y_true, np.array([0.0, 1.0, 4.0])))


def test_wape_undefined_when_no_demand():
    assert np.isnan(wape(np.zeros(3), np.ones(3)))


def test_smape_bounds_and_zero_case():
    assert smape(np.array([10.0]), np.array([10.0])) == pytest.approx(0.0)
    # both zero counts as agreement, not a divide-by-zero
    assert smape(np.array([0.0]), np.array([0.0])) == pytest.approx(0.0)
    # maximal disagreement approaches 2
    assert smape(np.array([0.0]), np.array([10.0])) == pytest.approx(2.0)


# --- series diagnostics ---------------------------------------------------


def test_smooth_series_classified_smooth():
    y = np.array([10.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0])
    d = series_diagnostics(y)
    assert d["ADI"] == pytest.approx(1.0)
    assert d["demand_class"] == "smooth"


def test_intermittent_series_detected():
    """Sells rarely, but a steady amount when it does.

    Starts on a sale so ADI is unambiguous: 9 periods / 3 demands = 3.0.
    """
    y = np.array([5.0, 0.0, 0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    d = series_diagnostics(y)
    assert d["ADI"] == pytest.approx(3.0)
    assert d["demand_class"] == "intermittent"


def test_lumpy_series_detected():
    """Sells rarely and in wildly varying amounts — the hardest class."""
    y = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 60.0, 0.0, 0.0, 3.0, 0.0, 0.0, 90.0])
    d = series_diagnostics(y)
    assert d["ADI"] >= 1.32
    assert d["CV2"] >= 0.49
    assert d["demand_class"] == "lumpy"


def test_diagnostics_ignore_pre_launch_zeros():
    """ADI must describe the selling period, not the shelf-empty period."""
    launched = np.array([5.0, 5.0, 5.0, 5.0])
    with_preamble = np.concatenate([np.zeros(20), launched])
    assert series_diagnostics(with_preamble)["ADI"] == pytest.approx(
        series_diagnostics(launched)["ADI"]
    )


def test_never_sold_series_flagged():
    d = series_diagnostics(np.zeros(10))
    assert d["demand_class"] == "no_demand"
    assert np.isnan(d["ADI"])


# --- robust aggregation ---------------------------------------------------


def test_trimmed_mean_ignores_extremes():
    values = pd.Series([1.0] * 98 + [1000.0, 2000.0])
    assert trimmed_mean(values, proportion=0.05) == pytest.approx(1.0)
    assert values.mean() > 30  # the plain mean is dragged far away


def test_trimmed_mean_handles_all_nan():
    assert np.isnan(trimmed_mean(pd.Series([np.nan, np.nan])))


def test_summarize_reports_mean_median_and_trimmed():
    results = pd.DataFrame(
        {
            "model": ["A"] * 5 + ["B"] * 5,
            "unique_id": list(range(5)) * 2,
            "MASE": [1.0, 1.0, 1.0, 1.0, 500.0] + [2.0] * 5,
            "RMSSE": [1.0] * 10,
        }
    )
    out = summarize(results, metrics=("MASE", "RMSSE"))
    # A's mean is wrecked by one outlier; its median and trimmed mean are not.
    assert out.loc["A", "MASE_mean"] > out.loc["B", "MASE_mean"]
    assert out.loc["A", "MASE_median"] < out.loc["B", "MASE_median"]
    assert out.loc["A", "MASE_trimmed"] < out.loc["B", "MASE_trimmed"]


def test_summarize_pools_volume_ratio_across_series():
    """Pooled, not averaged: B's zero-demand series has no ratio of its own."""
    results = pd.DataFrame(
        {
            "model": ["A", "A", "B", "B"],
            "unique_id": [0, 1, 0, 1],
            "MASE": [1.0] * 4,
            "Actual_Sum": [10.0, 30.0, 10.0, 0.0],
            "Forecast_Sum": [5.0, 15.0, 10.0, 3.0],
        }
    )
    out = summarize(results, metrics=("MASE",))
    assert out.loc["A", "Volume_ratio"] == pytest.approx(0.5)
    assert out.loc["B", "Volume_ratio"] == pytest.approx(1.3)


def test_summarize_omits_volume_ratio_without_sums():
    results = pd.DataFrame({"model": ["A"], "unique_id": [0], "MASE": [1.0]})
    assert "Volume_ratio" not in summarize(results, metrics=("MASE",)).columns


# --- panel trimming -------------------------------------------------------


def test_panel_trim_starts_each_series_at_first_sale():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    panel = pd.DataFrame(
        {
            "unique_id": ["A"] * 6 + ["B"] * 6,
            "ds": list(dates) * 2,
            "y": [0.0, 0.0, 3.0, 0.0, 1.0, 2.0] + [5.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        }
    )
    out = trim_leading_zeros_panel(panel)
    assert len(out[out.unique_id == "A"]) == 4  # first sale on day 3
    assert len(out[out.unique_id == "B"]) == 6  # sells immediately
    assert out[out.unique_id == "A"]["y"].iloc[0] == 3.0


def test_panel_trim_preserves_interior_zeros():
    """Only the pre-launch run goes; zeros after launch are real no-sale periods."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    panel = pd.DataFrame({"unique_id": ["A"] * 5, "ds": dates, "y": [0.0, 4.0, 0.0, 0.0, 2.0]})
    out = trim_leading_zeros_panel(panel)
    assert out["y"].tolist() == [4.0, 0.0, 0.0, 2.0]


def test_panel_trim_drops_series_that_never_sell():
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    panel = pd.DataFrame(
        {
            "unique_id": ["dead"] * 4 + ["live"] * 4,
            "ds": list(dates) * 2,
            "y": [0.0] * 4 + [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = trim_leading_zeros_panel(panel)
    assert set(out["unique_id"]) == {"live"}
