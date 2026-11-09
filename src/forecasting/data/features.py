"""Feature configuration shared by the tabular models.

Deliberately modest. The M5 winners stacked dozens of target encodings across
many aggregation levels, which wins competitions but produces a feature set too
tangled to reason about. The point here is a clean covariate ablation, so the
features are grouped into named blocks that can be switched on one at a time.

Blocks:
    lags      recent history at seasonally meaningful offsets
    rolling   rolling/expanding statistics over those lags
    calendar  date parts derived from the timestamp
    exog      dataset covariates (price, SNAP, CPI, markdowns, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from mlforecast.lag_transforms import ExpandingMean, RollingMean, RollingStd, SeasonalRollingMean


@dataclass
class FeatureConfig:
    """Which feature blocks a model gets, and how each is parameterised."""

    lags: list[int] = field(default_factory=list)
    lag_transforms: dict = field(default_factory=dict)
    date_features: list[str] = field(default_factory=list)
    use_exog: bool = True

    def without_exog(self) -> FeatureConfig:
        return FeatureConfig(self.lags, self.lag_transforms, self.date_features, use_exog=False)

    def lags_only(self) -> FeatureConfig:
        return FeatureConfig(lags=self.lags, use_exog=False)

    def with_calendar(self) -> FeatureConfig:
        return FeatureConfig(lags=self.lags, date_features=self.date_features, use_exog=False)


# M5: daily data with strong weekly seasonality. Lags cover last week, two
# weeks and the 28-day horizon; the seasonal rolling mean captures "same
# weekday, recent weeks", which matters more than raw recency in retail.
M5_FEATURES = FeatureConfig(
    lags=[1, 7, 14, 28],
    lag_transforms={
        1: [RollingMean(window_size=7), RollingStd(window_size=7), ExpandingMean()],
        7: [RollingMean(window_size=28)],
        28: [SeasonalRollingMean(season_length=7, window_size=4)],
    },
    date_features=["dayofweek", "day", "month"],
)

# M5 with nothing newer than the 28-day horizon: every feature is known for the
# whole forecast, so no prediction is ever fed back in and errors cannot compound.
# The "non-recursive" half of the M5 winner's models.
M5_DIRECT_FEATURES = FeatureConfig(
    lags=[28, 35, 42, 56],
    lag_transforms={
        28: [
            RollingMean(window_size=7),
            RollingStd(window_size=7),
            RollingMean(window_size=28),
            SeasonalRollingMean(season_length=7, window_size=4),
            ExpandingMean(),
        ],
    },
    date_features=["dayofweek", "day", "month"],
)

# Walmart: weekly data with yearly seasonality, but only ~114 training weeks.
# A lag of 52 is the single most informative feature and also the most
# expensive — it costs a full year of usable rows per series.
WALMART_FEATURES = FeatureConfig(
    lags=[1, 2, 4, 52],
    lag_transforms={
        1: [RollingMean(window_size=4), RollingStd(window_size=4), ExpandingMean()],
        4: [RollingMean(window_size=13)],
    },
    date_features=["month", "quarter", "week"],
)

FEATURE_CONFIGS = {"m5": M5_FEATURES, "walmart": WALMART_FEATURES}


def get_feature_config(dataset: str) -> FeatureConfig:
    if dataset not in FEATURE_CONFIGS:
        raise ValueError(f"No feature config for '{dataset}'. Have {list(FEATURE_CONFIGS)}.")
    return FEATURE_CONFIGS[dataset]
