"""Shared machinery for probabilistic neural forecasters.

TFT and NHITS differ only in architecture internals — every part that touches
this benchmark (quantile loss, covariate wiring, the ragged-future-frame fix,
GPU teardown) is identical. Keeping it in one place means adding another
probabilistic model is a subclass, and means a fix to the prediction path
cannot silently apply to one model but not another.

Quantile output rather than a point forecast is the reason these models are
here: safety stock is set from an upper quantile, so on intermittent demand the
shape of the tail is the decision-relevant output.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss, HuberMQLoss, MQLoss

from .base import varying_columns

# neuralforecast names MQLoss outputs by interval, not by quantile. This maps
# each emitted column suffix onto the quantile it actually represents.
LEVELS = [80, 90]
COLUMN_QUANTILES = {
    "median": 0.50,
    "lo-80": 0.10,
    "hi-80": 0.90,
    "lo-90": 0.05,
    "hi-90": 0.95,
}


def _clamped_nbinomial_decouple(decouple):
    """neuralforecast's negative binomial computes probs = r / (1 + r) + 1e-8,
    which reaches 1 (outside the distribution's support) when the mean of a
    high-volume series is large; full M5 hits that within the first epoch.
    Clamping just below 1 changes nothing elsewhere."""

    def wrapped(output, loc=None, scale=None):
        total_count, probs = decouple(output, loc=loc, scale=scale)
        return total_count, probs.clamp(max=1.0 - 1e-6)

    return wrapped


def distribution_loss(distribution: str, level: list[int]) -> DistributionLoss:
    loss = DistributionLoss(distribution=distribution, level=level)
    if distribution == "NegativeBinomial":
        loss.scale_decouple = _clamped_nbinomial_decouple(loss.scale_decouple)
    return loss


LOSSES = {
    "mq": MQLoss,
    "huber_mq": HuberMQLoss,
    # A count distribution: the quantiles come from it, and its mean is the
    # point forecast (see `point`).
    "negbin": lambda level: distribution_loss("NegativeBinomial", level),
}

# Losses that fit a distribution, so a mean forecast exists alongside the median.
DISTRIBUTION_LOSSES = {"negbin"}


class QuantileNeuralForecaster:
    """Base class: a global neuralforecast model emitting median + quantiles.

    Subclasses set MODEL_CLS and may pass architecture-specific keyword
    arguments through `model_kwargs`.
    """

    MODEL_CLS = None

    def __init__(
        self,
        freq: str,
        h: int,
        futr_exog: list[str] | None = None,
        stat_exog: list[str] | None = None,
        input_size: int | None = None,
        max_steps: int = 500,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        val_size: int = 0,
        scaler_type: str = "standard",
        loss_type: str = "mq",
        random_seed: int = 42,
        name: str | None = None,
        point: str | None = None,
        **model_kwargs,
    ):
        if self.MODEL_CLS is None:
            raise NotImplementedError("subclass must set MODEL_CLS")
        if loss_type not in LOSSES:
            raise ValueError(f"loss_type must be one of {list(LOSSES)}, got '{loss_type}'")
        # The point forecast: the median, or the distribution's mean. On a
        # mostly-zero panel the median sits far below the mean, so a median
        # point under-forecasts every aggregate; squared-error metrics such as
        # M5's WRMSSE are minimised by the mean. A distribution loss defaults
        # to its mean, a quantile loss can only give the median.
        self._point = point or ("mean" if loss_type in DISTRIBUTION_LOSSES else "median")
        if self._point not in ("median", "mean"):
            raise ValueError(f"point must be 'median' or 'mean', got '{point}'")

        self.name = name or self.MODEL_CLS.__name__
        self._freq = freq
        self._h = h
        # Twice the horizon is the usual starting point for the context length.
        self._input_size = input_size or 2 * h
        self._futr_exog = futr_exog or []
        self._stat_exog = stat_exog or []
        self._fit_futr: list[str] = []
        self._max_steps = max_steps
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        # Off by default: a validation tail of h periods exceeds the shortest
        # series in early rolling windows, and raising min_train_periods would
        # score this model on an easier panel than its competitors.
        self._val_size = val_size
        self._scaler_type = scaler_type
        self._loss_type = loss_type
        self._random_seed = random_seed
        self._model_kwargs = model_kwargs
        self._nf: NeuralForecast | None = None

    def _make_loss(self):
        """Build the training loss.

        Overridden by models that are probabilistic through a different
        mechanism — DeepAR parameterises a distribution and samples from it,
        so it rejects a quantile loss and needs DistributionLoss instead.
        """
        return LOSSES[self._loss_type](level=LEVELS)

    def fit(self, train_df: pd.DataFrame) -> QuantileNeuralForecaster:
        torch.manual_seed(self._random_seed)
        np.random.seed(self._random_seed)

        futr = varying_columns(train_df, self._futr_exog)
        stat = varying_columns(train_df, self._stat_exog)
        self._fit_futr = futr

        df = train_df[["unique_id", "ds", "y"] + futr].copy()
        # Static features are constant within a series, so the first row is
        # representative.
        static_df = train_df.groupby("unique_id")[stat].first().reset_index() if stat else None

        model = self.MODEL_CLS(
            h=self._h,
            input_size=self._input_size,
            loss=self._make_loss(),
            futr_exog_list=futr or None,
            stat_exog_list=stat or None,
            max_steps=self._max_steps,
            learning_rate=self._learning_rate,
            batch_size=self._batch_size,
            early_stop_patience_steps=3 if self._val_size > 0 else -1,
            val_check_steps=50,
            start_padding_enabled=True,
            scaler_type=self._scaler_type,
            random_seed=self._random_seed,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
            **self._model_kwargs,
        )

        self._nf = NeuralForecast(models=[model], freq=self._freq)
        self._nf.fit(df, static_df=static_df, val_size=self._val_size, verbose=False)
        return self

    def predict(self, h: int, futr_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if h != self._h:
            raise ValueError(
                f"{self.name} was compiled for h={self._h}, got h={h}. "
                "neuralforecast models are horizon-specific; build a fresh one."
            )

        # Exactly the covariates the model was fitted with.
        futr = [c for c in self._fit_futr if futr_df is not None and c in futr_df.columns]
        X = None
        if futr:
            # The test split is ragged where series end early, but neuralforecast
            # needs a full h x n_series frame, so build the grid explicitly and
            # merge the known covariates onto it.
            future = self._nf.make_future_dataframe()
            X = future.merge(
                futr_df[["unique_id", "ds"] + futr], on=["unique_id", "ds"], how="left"
            )
            X[futr] = X.groupby("unique_id")[futr].ffill().fillna(0)

        raw = self._nf.predict(futr_df=X, verbose=False)
        # neuralforecast already returns unique_id as a column; an unconditional
        # reset_index() would add an "index" column that could be mistaken for
        # the prediction.
        if "unique_id" not in raw.columns:
            raw = raw.reset_index()

        alias = self._nf.models[0].alias or type(self._nf.models[0]).__name__
        out = raw[["unique_id", "ds"]].copy()

        for suffix, q in COLUMN_QUANTILES.items():
            col = f"{alias}-{suffix}"
            if col not in raw.columns:
                raise KeyError(f"expected '{col}' in {list(raw.columns)}")
            values = raw[col].clip(lower=0)
            if suffix == "median" and self._point == "median":
                out["y_pred"] = values
            out[f"y_pred_q{int(round(q * 100)):02d}"] = values
        if self._point == "mean":
            # Distribution losses return the mean under the bare alias.
            if alias not in raw.columns:
                raise KeyError(f"point='mean' needs '{alias}' in {list(raw.columns)}")
            out["y_pred"] = raw[alias].clip(lower=0)

        # Clipping at zero can leave quantiles out of order; enforce monotonicity
        # so the intervals stay valid and coverage stays meaningful.
        qcols = sorted(
            (c for c in out.columns if c.startswith("y_pred_q")),
            key=lambda c: int(c[-2:]),
        )
        out[qcols] = np.maximum.accumulate(out[qcols].to_numpy(), axis=1)
        return out

    def teardown(self) -> None:
        """Release GPU memory between backtest windows.

        The harness trains a fresh model per window; without this, thirteen
        un-freed models exhaust a 6GB card partway through a rolling backtest.
        """
        self._nf = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
