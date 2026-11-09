"""BiTCN with multi-quantile loss.

Bidirectional temporal convolutions: dilated causal convolutions run forward
over the history and backward over the known future covariates, so the receptive
field grows exponentially with depth at a fraction of attention's cost.

This is the closest architecture in a maintained library to MQCNN, Amazon's
convolutional multi-quantile retail forecaster. MQCNN was a CNN encoder feeding
an MLP decoder that emitted multi-horizon quantiles; BiTCN is a convolutional
encoder emitting multi-horizon quantiles. It represents that design without
depending on MXNet, which cannot run against NumPy 2.

Unlike TFT and NHITS it also accepts historical exogenous features, though this
benchmark only supplies covariates that are known ahead of time.
"""

from __future__ import annotations

from neuralforecast.models import BiTCN

from .neural_quantile import QuantileNeuralForecaster


class BiTCNForecaster(QuantileNeuralForecaster):
    MODEL_CLS = BiTCN


def build_bitcn(
    freq: str,
    h: int,
    futr_exog: list[str] | None = None,
    stat_exog: list[str] | None = None,
    name: str = "BiTCN",
    hidden_size: int = 16,
    dropout: float = 0.1,
    **kwargs,
):
    """Zero-arg factory for the backtest harness.

    hidden_size defaults to 16 (the library default). BiTCN's capacity comes
    from convolution depth rather than width, so it is far smaller than TFT at
    comparable accuracy.
    """

    def factory():
        return BiTCNForecaster(
            freq=freq,
            h=h,
            futr_exog=futr_exog,
            stat_exog=stat_exog,
            name=name,
            hidden_size=hidden_size,
            dropout=dropout,
            **kwargs,
        )

    return factory
