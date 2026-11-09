"""Model-level unit tests: forecast dates, training windows and LightGBM options."""

import pandas as pd
import pytest

from forecasting.models.base import Forecaster
from forecasting.models.baselines import ZeroForecaster


def test_zero_forecaster_continues_each_series_from_its_own_end():
    """A series that ended early must be forecast from its own last date."""
    train = pd.DataFrame(
        {
            "unique_id": ["A"] * 3 + ["B"] * 2,
            "ds": pd.to_datetime(
                ["2020-01-03", "2020-01-10", "2020-01-17", "2020-01-03", "2020-01-10"]
            ),
            "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    model = ZeroForecaster(freq="W-FRI").fit(train)
    out = model.predict(h=2)

    assert isinstance(model, Forecaster)
    assert (out["y_pred"] == 0).all()
    assert out[out.unique_id == "A"]["ds"].tolist() == list(
        pd.to_datetime(["2020-01-24", "2020-01-31"])
    )
    assert out[out.unique_id == "B"]["ds"].tolist() == list(
        pd.to_datetime(["2020-01-17", "2020-01-24"])
    )


def test_gbm_warmup_covers_longest_feature():
    from forecasting.data.features import M5_FEATURES
    from forecasting.models.gbm import GBMForecaster

    # Lag 28 with a 4-week seasonal rolling mean reaches back 28 + 4 * 7 days.
    assert GBMForecaster("D", M5_FEATURES, train_days=365)._warmup() == 56


def test_gbm_train_days_limits_training_history(monkeypatch):
    import numpy as np
    from mlforecast import MLForecast

    from forecasting.data.features import FeatureConfig
    from forecasting.models.gbm import GBMForecaster

    seen = []
    original_fit = MLForecast.fit

    def spy(self, df, *args, **kwargs):
        seen.append(len(df))
        return original_fit(self, df, *args, **kwargs)

    monkeypatch.setattr(MLForecast, "fit", spy)

    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    rng = np.random.default_rng(0)
    panel = pd.concat(
        [pd.DataFrame({"unique_id": uid, "ds": dates, "y": rng.poisson(3, 200)}) for uid in "ab"]
    )
    fc = FeatureConfig(lags=[1, 7])
    model = GBMForecaster("D", fc, params={"n_estimators": 5}, train_days=30).fit(panel)
    # 30 training days plus the 7-day warm-up for lag 7, per series.
    assert seen == [2 * (30 + 7)]
    assert len(model.predict(h=3)) == 6


def test_christmas_shift_blends_december_and_leaves_other_weeks_alone():
    import numpy as np

    from forecasting.models.baselines import ChristmasShiftedSeasonalNaive

    dates = pd.date_range("2010-01-01", "2011-11-25", freq="W-FRI")
    y = np.arange(len(dates), dtype=float)  # value = week index, so lags are traceable
    panel = pd.DataFrame({"unique_id": "s", "ds": dates, "y": y})
    out = ChristmasShiftedSeasonalNaive().fit(panel).predict(h=6).set_index("ds")["y_pred"]
    index = {d: i for i, d in enumerate(dates)}

    # Early December: plain 52-week lag.
    assert out[pd.Timestamp("2011-12-02")] == index[pd.Timestamp("2010-12-03")]
    # Week ending 2011-12-23: 52 weeks back is 2010-12-24, but the same calendar
    # week last year ends a day earlier, so 1/7 comes from the week before.
    lag52, lag53 = index[pd.Timestamp("2010-12-24")], index[pd.Timestamp("2010-12-17")]
    assert out[pd.Timestamp("2011-12-23")] == pytest.approx(6 / 7 * lag52 + 1 / 7 * lag53)


@pytest.mark.parametrize("keep, expected", [(False, 2 * (50 - 7)), (True, 2 * 50)])
def test_gbm_keep_incomplete_rows_trains_on_early_history(monkeypatch, keep, expected):
    """Rows before the longest lag is defined are dropped unless asked to keep them."""
    import lightgbm as lgb
    import numpy as np

    from forecasting.data.features import FeatureConfig
    from forecasting.models.gbm import GBMForecaster

    rows = []
    original_fit = lgb.LGBMRegressor.fit

    def spy(self, X, y, *args, **kwargs):
        rows.append(len(X))
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(lgb.LGBMRegressor, "fit", spy)
    dates = pd.date_range("2020-01-01", periods=50, freq="D")
    rng = np.random.default_rng(0)
    panel = pd.concat(
        [pd.DataFrame({"unique_id": uid, "ds": dates, "y": rng.poisson(3, 50)}) for uid in "ab"]
    )
    params = {"n_estimators": 5, "min_child_samples": 1}
    GBMForecaster("D", FeatureConfig(lags=[1, 7]), params=params, keep_incomplete_rows=keep).fit(
        panel
    )
    assert rows == [expected]


def test_gbm_seasonal_difference_rejects_tweedie():
    from forecasting.data.features import FeatureConfig
    from forecasting.models.gbm import GBMForecaster

    with pytest.raises(ValueError, match="tweedie"):
        GBMForecaster("W-FRI", FeatureConfig(lags=[1]), seasonal_difference=52)


def test_gbm_seasonal_difference_reproduces_a_repeating_season():
    """On a series that repeats exactly every 52 weeks, the change is always 0."""
    import numpy as np

    from forecasting.data.features import FeatureConfig
    from forecasting.models.gbm import GBMForecaster

    dates = pd.date_range("2010-01-01", periods=130, freq="W-FRI")
    season = 100 + 50 * np.sin(np.arange(52) / 52 * 2 * np.pi)
    panel = pd.concat(
        pd.DataFrame({"unique_id": uid, "ds": dates, "y": np.resize(season, 130) * k})
        for uid, k in [("a", 1.0), ("b", 2.0)]
    )
    model = GBMForecaster(
        "W-FRI",
        FeatureConfig(lags=[1, 2, 52]),
        params={"objective": "l2", "n_estimators": 20, "min_child_samples": 1},
        seasonal_difference=52,
    ).fit(panel)
    out = model.predict(h=8)
    expected = np.concatenate([np.resize(season, 138)[130:] * k for k in (1.0, 2.0)])
    np.testing.assert_allclose(out["y_pred"].to_numpy(), expected, atol=1e-6)
