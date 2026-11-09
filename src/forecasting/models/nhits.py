"""NHITS with multi-quantile loss.

NHITS is N-BEATS' successor: the same residual-stack idea, but each stack pools
the input at a different rate and interpolates its output back up. Low-frequency
stacks capture trend cheaply while high-frequency stacks handle short-run
detail, which cuts the parameter count relative to N-BEATS and usually trains
faster.

It earns its place here for two reasons. N-BEATS is the strongest family in this
benchmark so far, and NHITS extends it in the two directions N-BEATS cannot go:
it accepts covariates, and it emits quantiles.

The pooling and downsampling schedules below are the paper's defaults. They are
expressed per stack, coarsest first, so `n_freq_downsample=[24, 12, 1]` means
the first stack models a 24-period rhythm and the last models every period.
"""

from __future__ import annotations

from neuralforecast.models import NHITS

from .neural_quantile import QuantileNeuralForecaster


class NHITSForecaster(QuantileNeuralForecaster):
    MODEL_CLS = NHITS


def build_nhits(
    freq: str,
    h: int,
    futr_exog: list[str] | None = None,
    stat_exog: list[str] | None = None,
    name: str = "NHITS",
    season_length: int | None = None,
    n_freq_downsample: list[int] | None = None,
    **kwargs,
):
    """Zero-arg factory for the backtest harness.

    The pooling schedule is derived from the data's seasonality rather than
    taken from the paper's [24, 12, 1]. That default assumes hourly data, where
    24 is one daily cycle; applied to daily data it pools 24 days into one
    bucket and destroys the weekly structure the model needs. Deriving from
    season_length gives the coarsest stack exactly one seasonal cycle.

    Rates are clipped to the horizon: a stack cannot interpolate a cycle longer
    than the output it produces.
    """
    if n_freq_downsample is None:
        if season_length and season_length > 1:
            n_freq_downsample = [season_length, max(1, season_length // 2), 1]
        else:
            n_freq_downsample = [24, 12, 1]
    schedule = [min(rate, max(1, h)) for rate in n_freq_downsample]

    def factory():
        return NHITSForecaster(
            freq=freq,
            h=h,
            futr_exog=futr_exog,
            stat_exog=stat_exog,
            name=name,
            n_freq_downsample=schedule,
            **kwargs,
        )

    return factory
