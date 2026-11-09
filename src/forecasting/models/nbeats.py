"""N-BEATS via neuralforecast (PyTorch).

Uses neuralforecast's maintained PyTorch implementation. The GluonTS version
depends on MXNet, which Apache retired in 2023 and which calls `np.PZERO`,
removed in NumPy 2.0, so it cannot run alongside the rest of this stack.

N-BEATS is strictly univariate — `EXOGENOUS_FUTR`, `EXOGENOUS_HIST` and
`EXOGENOUS_STAT` are all False on the class. That makes it the clean control in
this benchmark: whatever it achieves comes from the target history alone, so the
gap to GBM measures what the covariates and engineered features are worth.

Unlike the statistical baselines and GBM, a neuralforecast model is compiled for
one specific horizon, so the horizon is fixed at construction rather than passed
to predict().
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE, MSE
from neuralforecast.models import NBEATS, NBEATSx

from .base import varying_columns

# MAE targets the median, MSE the mean. On a mostly-zero panel the two differ
# a lot, and squared-error metrics (M5's WRMSSE) reward the mean.
POINT_LOSSES = {"mae": MAE, "mse": MSE}


class NBEATSForecaster:
    """Global N-BEATS trained across every series at once."""

    def __init__(
        self,
        freq: str,
        h: int,
        input_size: int | None = None,
        max_steps: int = 500,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        val_size: int = 0,
        scaler_type: str = "standard",
        random_seed: int = 42,
        name: str = "N-BEATS",
        loss: str = "mae",
    ):
        if loss not in POINT_LOSSES:
            raise ValueError(f"loss must be one of {list(POINT_LOSSES)}, got '{loss}'")
        self.name = name
        self._loss = loss
        self._freq = freq
        self._h = h
        # Twice the horizon is the usual starting point for the context length.
        self._input_size = input_size or 2 * h
        self._max_steps = max_steps
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        # Validation is off by default. Carving a validation tail of h periods
        # would exceed the length of the shortest series in the early rolling
        # windows, and neuralforecast refuses to fit. Raising min_train_periods
        # instead would drop those series and leave N-BEATS scored on an easier
        # panel than the models it is being compared against; a fixed step
        # budget keeps the panel identical, which matters more here than early
        # stopping.
        self._val_size = val_size
        self._scaler_type = scaler_type
        self._random_seed = random_seed
        self._nf: NeuralForecast | None = None

    def fit(self, train_df: pd.DataFrame) -> NBEATSForecaster:
        torch.manual_seed(self._random_seed)
        np.random.seed(self._random_seed)

        model = NBEATS(
            h=self._h,
            input_size=self._input_size,
            loss=POINT_LOSSES[self._loss](),
            max_steps=self._max_steps,
            learning_rate=self._learning_rate,
            batch_size=self._batch_size,
            # -1 disables early stopping, which has no meaning without a
            # validation split.
            early_stop_patience_steps=3 if self._val_size > 0 else -1,
            val_check_steps=50,
            # Series shorter than the context window are padded rather than
            # dropped, so the model is scored on the same panel as every other
            # model in the benchmark.
            start_padding_enabled=True,
            scaler_type=self._scaler_type,
            # Must be passed explicitly: neuralforecast re-seeds at fit time
            # with its own default (1), overriding torch.manual_seed above.
            random_seed=self._random_seed,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )

        self._nf = NeuralForecast(models=[model], freq=self._freq)
        # Validation is carved from the tail of training data, which sits before
        # the test window — no leakage.
        self._nf.fit(train_df[["unique_id", "ds", "y"]], val_size=self._val_size, verbose=False)
        return self

    def teardown(self) -> None:
        """Release GPU memory. The harness trains a fresh model per window, and
        without this the allocations accumulate until CUDA runs out."""
        self._nf = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, h: int, futr_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if h != self._h:
            raise ValueError(
                f"{self.name} was compiled for h={self._h} but asked for h={h}. "
                "Build a fresh model per horizon."
            )
        raw = self._nf.predict(verbose=False)
        # neuralforecast already returns unique_id as a column; calling
        # reset_index() unconditionally adds an "index" column that then gets
        # mistaken for the prediction, yielding row numbers as "forecasts".
        if "unique_id" not in raw.columns:
            raw = raw.reset_index()

        model_col = self._nf.models[0].alias or type(self._nf.models[0]).__name__
        if model_col not in raw.columns:
            raise KeyError(f"expected prediction column '{model_col}' in {list(raw.columns)}")
        out = raw.rename(columns={model_col: "y_pred"})[["unique_id", "ds", "y_pred"]]
        out["y_pred"] = out["y_pred"].clip(lower=0)
        return out


class NBEATSxForecaster(NBEATSForecaster):
    """N-BEATS with covariates — the controlled counterpart to plain N-BEATS.

    Architecture, optimiser and training budget are held identical to
    NBEATSForecaster, so the difference between the two isolates what the
    covariates are worth. Comparing N-BEATS against GBM cannot do that: there
    the architecture and the feature set both change at once.

    All dataset covariates here (calendar events, SNAP, prices, CPI, markdowns)
    are known ahead of time, so they are declared as future exogenous rather
    than historical.
    """

    def __init__(
        self,
        freq: str,
        h: int,
        futr_exog: list[str] | None = None,
        stat_exog: list[str] | None = None,
        name: str = "N-BEATSx",
        **kwargs,
    ):
        super().__init__(freq=freq, h=h, name=name, **kwargs)
        self._futr_exog = futr_exog or []
        self._stat_exog = stat_exog or []
        self._fit_futr: list[str] = []
        self._static_df: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> NBEATSxForecaster:
        torch.manual_seed(self._random_seed)
        np.random.seed(self._random_seed)

        futr = varying_columns(train_df, self._futr_exog)
        stat = varying_columns(train_df, self._stat_exog)
        self._fit_futr = futr

        df = train_df[["unique_id", "ds", "y"] + futr].copy()
        # Static features are constant within a series, so the first row is
        # representative.
        self._static_df = (
            train_df.groupby("unique_id")[stat].first().reset_index() if stat else None
        )

        model = NBEATSx(
            h=self._h,
            input_size=self._input_size,
            futr_exog_list=futr or None,
            stat_exog_list=stat or None,
            loss=POINT_LOSSES[self._loss](),
            max_steps=self._max_steps,
            learning_rate=self._learning_rate,
            batch_size=self._batch_size,
            early_stop_patience_steps=3 if self._val_size > 0 else -1,
            val_check_steps=50,
            start_padding_enabled=True,
            scaler_type=self._scaler_type,
            # Must be passed explicitly: neuralforecast re-seeds at fit time
            # with its own default (1), overriding torch.manual_seed above.
            random_seed=self._random_seed,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
        )

        self._nf = NeuralForecast(models=[model], freq=self._freq)
        self._nf.fit(df, static_df=self._static_df, val_size=self._val_size, verbose=False)
        return self

    def predict(self, h: int, futr_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if h != self._h:
            raise ValueError(f"{self.name} was compiled for h={self._h}, got h={h}.")

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
        if "unique_id" not in raw.columns:
            raw = raw.reset_index()

        model_col = self._nf.models[0].alias or type(self._nf.models[0]).__name__
        if model_col not in raw.columns:
            raise KeyError(f"expected '{model_col}' in {list(raw.columns)}")
        out = raw.rename(columns={model_col: "y_pred"})[["unique_id", "ds", "y_pred"]]
        out["y_pred"] = out["y_pred"].clip(lower=0)
        return out


def build_nbeats(freq: str, h: int, name: str = "N-BEATS", **kwargs):
    """Zero-arg factory for the backtest harness."""

    def factory():
        return NBEATSForecaster(freq=freq, h=h, name=name, **kwargs)

    return factory


def build_nbeatsx(
    freq: str,
    h: int,
    futr_exog: list[str] | None = None,
    stat_exog: list[str] | None = None,
    name: str = "N-BEATSx",
    **kwargs,
):
    """Zero-arg factory for the covariate-aware variant."""

    def factory():
        return NBEATSxForecaster(
            freq=freq, h=h, futr_exog=futr_exog, stat_exog=stat_exog, name=name, **kwargs
        )

    return factory
