import pandas as pd
import pytest

from forecasting.evaluation.backtest import make_windows
from forecasting.evaluation.walmart import wmae


def test_wmae_weights_holiday_weeks_five_times():
    dates = pd.to_datetime(["2012-02-03", "2012-02-10"])  # second is a holiday week
    actuals = pd.DataFrame({"unique_id": "1_1", "ds": dates, "y": [100.0, 100.0]})
    forecast = pd.DataFrame({"unique_id": "1_1", "ds": dates, "y_pred": [90.0, 70.0]})
    score, rows = wmae(forecast, actuals, pd.DatetimeIndex(["2012-02-10"]))
    # (1 * 10 + 5 * 30) / (1 + 5)
    assert score == pytest.approx(160 / 6)
    assert rows == 2


def test_wmae_scores_only_rows_present_in_both():
    actuals = pd.DataFrame(
        {"unique_id": ["a", "b"], "ds": pd.to_datetime(["2012-01-06"] * 2), "y": [1.0, 2.0]}
    )
    forecast = actuals.rename(columns={"y": "y_pred"}).iloc[:1]
    assert wmae(forecast, actuals, pd.DatetimeIndex([])) == (0.0, 1)


def test_cutoff_window_tests_the_next_h_periods():
    dates = pd.date_range("2011-01-07", periods=60, freq="W-FRI")
    panel = pd.DataFrame({"unique_id": "a", "ds": dates, "y": range(60)})
    [window] = make_windows(panel, h=10, mode="cutoff", cutoff=dates[39])
    assert window.train["ds"].max() == dates[39]
    assert window.test["ds"].tolist() == list(dates[40:50])


def test_cutoff_window_rejects_bad_cutoffs():
    dates = pd.date_range("2011-01-07", periods=20, freq="W-FRI")
    panel = pd.DataFrame({"unique_id": "a", "ds": dates, "y": range(20)})
    with pytest.raises(ValueError, match="not a date"):
        make_windows(panel, h=5, mode="cutoff", cutoff="2011-01-08")
    with pytest.raises(ValueError, match="need 5"):
        make_windows(panel, h=5, mode="cutoff", cutoff=dates[17])
