"""DeepAR — probabilistic forecasting by distribution, not by quantile loss.

DeepAR is the classic autoregressive demand-forecasting model from Amazon, and
it is probabilistic in a different way from TFT/NHITS/BiTCN. Rather than
regressing a fixed set of quantiles, it learns the parameters of a distribution
at each step and draws sample trajectories; quantiles are read off those
samples. That makes it the natural counterpart in this benchmark: two routes to
a predictive distribution, scored with the same pinball loss and coverage.

The distribution family matters and is not one-size-fits-all here:

    NegativeBinomial  counts, over-dispersed, large mass at zero — M5 unit sales
    StudentT          continuous with heavy tails — Walmart dollar revenue

Using a count distribution on Walmart's dollar revenue would be a category
error, so the family is chosen per dataset rather than defaulted globally.

Note DeepAR declares EXOGENOUS_HIST = False: it accepts future-known and static
covariates only. Every covariate in this benchmark is known ahead of time, so
nothing is lost.
"""

from __future__ import annotations

from neuralforecast.models import DeepAR

from .neural_quantile import LEVELS, QuantileNeuralForecaster, distribution_loss


class DeepARForecaster(QuantileNeuralForecaster):
    MODEL_CLS = DeepAR

    def __init__(self, *args, distribution: str = "StudentT", **kwargs):
        super().__init__(*args, **kwargs)
        self._distribution = distribution

    def _make_loss(self):
        """DeepAR rejects MQLoss; it needs a distribution to parameterise.

        `level` is passed so predictions come back with the same interval column
        names the quantile models produce, which keeps the shared prediction
        path in the base class working unchanged.
        """
        return distribution_loss(self._distribution, LEVELS)


def build_deepar(
    freq: str,
    h: int,
    futr_exog: list[str] | None = None,
    stat_exog: list[str] | None = None,
    name: str = "DeepAR",
    distribution: str = "StudentT",
    lstm_hidden_size: int = 128,
    **kwargs,
):
    """Zero-arg factory for the backtest harness."""

    def factory():
        return DeepARForecaster(
            freq=freq,
            h=h,
            futr_exog=futr_exog,
            stat_exog=stat_exog,
            name=name,
            distribution=distribution,
            lstm_hidden_size=lstm_hidden_size,
            **kwargs,
        )

    return factory
