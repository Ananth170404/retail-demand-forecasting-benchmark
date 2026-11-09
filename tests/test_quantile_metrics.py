import numpy as np
import pandas as pd
import pytest

from forecasting.evaluation.metrics import (
    evaluate_panel,
    interval_coverage,
    pinball_loss,
    quantile_metrics,
)


def test_coverage_counts_actuals_inside_the_interval():
    y = np.array([1.0, 5.0, 9.0, 20.0])
    lo = np.array([0.0, 0.0, 0.0, 0.0])
    hi = np.array([10.0, 10.0, 10.0, 10.0])
    assert interval_coverage(y, lo, hi) == pytest.approx(0.75)


def test_coverage_includes_the_boundaries():
    y = np.array([0.0, 10.0])
    assert interval_coverage(y, np.zeros(2), np.full(2, 10.0)) == pytest.approx(1.0)


def test_quantile_metrics_reports_each_quantile_and_the_mean():
    y = np.array([5.0, 5.0])
    preds = {0.05: np.array([1.0, 1.0]), 0.5: np.array([5.0, 5.0]), 0.95: np.array([9.0, 9.0])}
    out = quantile_metrics(y, preds)

    assert out["pinball_q50"] == pytest.approx(0.0)  # median is exact
    assert out["pinball_mean"] == pytest.approx(
        np.mean([out["pinball_q05"], out["pinball_q50"], out["pinball_q95"]])
    )


def test_quantile_metrics_derives_coverage_from_paired_quantiles():
    """q05/q95 form the 90% interval; q10/q90 the 80% interval."""
    y = np.array([5.0, 5.0, 100.0, 100.0])
    preds = {
        0.05: np.zeros(4),
        0.10: np.zeros(4),
        0.50: np.full(4, 5.0),
        0.90: np.full(4, 10.0),
        0.95: np.full(4, 10.0),
    }
    out = quantile_metrics(y, preds)
    # Two of four actuals sit inside [0, 10].
    assert out["coverage_80"] == pytest.approx(0.5)
    assert out["coverage_90"] == pytest.approx(0.5)


def test_quantile_metrics_empty_for_point_forecasts():
    assert quantile_metrics(np.array([1.0]), {}) == {}


def test_wider_intervals_raise_coverage():
    """Sanity: an overconfident model covers less than a conservative one."""
    y = np.array([1.0, 5.0, 9.0])
    narrow = interval_coverage(y, np.full(3, 4.0), np.full(3, 6.0))
    wide = interval_coverage(y, np.zeros(3), np.full(3, 10.0))
    assert wide > narrow


def test_perfect_median_still_penalised_at_other_quantiles():
    """A point forecast repeated at every quantile is not a good distribution."""
    y = np.array([0.0, 10.0])
    flat = {q: np.full(2, 5.0) for q in (0.05, 0.5, 0.95)}
    out = quantile_metrics(y, flat)
    assert out["pinball_q05"] > 0
    assert out["pinball_q95"] > 0


def test_evaluate_panel_picks_up_quantile_columns():
    """Probabilistic columns are detected by name; point models skip the scoring."""
    ds = pd.date_range("2020-01-10", periods=2, freq="D")
    actuals = pd.DataFrame({"unique_id": ["A", "A"], "ds": ds, "y": [4.0, 6.0]})
    forecast = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": ds,
            "y_pred": [5.0, 5.0],
            "y_pred_q05": [1.0, 1.0],
            "y_pred_q50": [5.0, 5.0],
            "y_pred_q95": [9.0, 9.0],
        }
    )
    train = pd.DataFrame(
        {
            "unique_id": ["A"] * 8,
            "ds": pd.date_range("2020-01-01", periods=8, freq="D"),
            "y": [1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0],
        }
    )
    out = evaluate_panel(forecast, actuals, train, season_length=1)
    assert "pinball_mean" in out.columns
    assert "coverage_90" in out.columns
    assert out["coverage_90"].iloc[0] == pytest.approx(1.0)


def test_evaluate_panel_omits_quantile_metrics_for_point_models():
    ds = pd.date_range("2020-01-10", periods=2, freq="D")
    actuals = pd.DataFrame({"unique_id": ["A", "A"], "ds": ds, "y": [4.0, 6.0]})
    forecast = pd.DataFrame({"unique_id": ["A", "A"], "ds": ds, "y_pred": [5.0, 5.0]})
    train = pd.DataFrame(
        {
            "unique_id": ["A"] * 8,
            "ds": pd.date_range("2020-01-01", periods=8, freq="D"),
            "y": [1.0, 3.0] * 4,
        }
    )
    out = evaluate_panel(forecast, actuals, train, season_length=1)
    assert not [c for c in out.columns if c.startswith(("pinball", "coverage"))]


def test_pinball_is_asymmetric_at_upper_quantiles():
    """A high quantile should punish under-forecasting harder than over."""
    y = np.array([10.0])
    assert pinball_loss(y, np.array([5.0]), 0.95) > pinball_loss(y, np.array([15.0]), 0.95)
