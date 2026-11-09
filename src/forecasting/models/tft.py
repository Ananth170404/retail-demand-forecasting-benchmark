"""Temporal Fusion Transformer with multi-quantile loss.

This is the benchmark's stand-in for MQCNN. MQCNN exists only in `gluonts.mx`;
MXNet was retired by Apache in 2023 and calls `np.PZERO`, removed in NumPy 2.0,
so it cannot run in this environment at all (verified, not assumed). TFT with
MQLoss provides the capability that made MQCNN worth including — multi-horizon
quantile forecasts conditioned on static and future-known covariates — on a
maintained PyTorch stack.

TFT combines an LSTM encoder with variable-selection networks and interpretable
multi-head attention, so it is the heaviest model in the benchmark by some
margin. That cost is itself a data point for the accuracy-vs-compute analysis.

Shared prediction and covariate handling lives in `neural_quantile.py`.
"""

from __future__ import annotations

from neuralforecast.models import TFT

from .neural_quantile import COLUMN_QUANTILES, LEVELS, QuantileNeuralForecaster  # noqa: F401


class TFTForecaster(QuantileNeuralForecaster):
    MODEL_CLS = TFT


def build_tft(
    freq: str,
    h: int,
    futr_exog: list[str] | None = None,
    stat_exog: list[str] | None = None,
    name: str = "TFT",
    hidden_size: int = 64,
    **kwargs,
):
    """Zero-arg factory for the backtest harness."""

    def factory():
        return TFTForecaster(
            freq=freq,
            h=h,
            futr_exog=futr_exog,
            stat_exog=stat_exog,
            name=name,
            hidden_size=hidden_size,
            **kwargs,
        )

    return factory
