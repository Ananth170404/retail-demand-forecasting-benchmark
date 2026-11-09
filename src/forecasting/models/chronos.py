"""Chronos-2 — a pretrained time-series foundation model, used zero-shot.

Nothing is trained on this data: `fit` only stores each series' history, and
`predict` hands the most recent `context_length` values to the pretrained
model. That makes it the benchmark's reference point for how far a model that
has never seen M5 or Walmart gets against models trained on them.

The point forecast is the mean of the predicted distribution, computed by
integrating its 21 quantiles. Chronos-2's own "mean" output equals its median,
and on intermittent series the median sits far below the mean: on simulated
Poisson(0.3) demand the median forecast is 0.004 while the integrated mean is
0.319 (true 0.301). Squared-error metrics such as M5's WRMSSE reward the mean,
so the median would under-forecast every aggregate.
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import torch

from .neural_quantile import COLUMN_QUANTILES


class ChronosForecaster:
    def __init__(
        self,
        freq: str,
        model_id: str = "amazon/chronos-2",
        context_length: int | None = None,
        batch_size: int = 256,
        point: str = "mean",
        name: str = "Chronos-2",
    ):
        if point not in ("mean", "median"):
            raise ValueError(f"point must be 'mean' or 'median', got '{point}'")
        self.name = name
        self._freq = freq
        self._model_id = model_id
        self._context_length = context_length
        self._batch_size = batch_size
        self._point = point
        self._pipeline = None
        self._history: list[np.ndarray] = []
        self._ids: np.ndarray | None = None
        self._last: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame) -> ChronosForecaster:
        from chronos import BaseChronosPipeline

        groups = train_df.sort_values(["unique_id", "ds"]).groupby("unique_id", sort=True)
        self._ids = np.array(list(groups.groups))
        self._history = [g["y"].to_numpy(dtype=np.float32) for _, g in groups]
        if self._context_length:
            self._history = [h[-self._context_length :] for h in self._history]
        self._last = groups["ds"].max()
        self._pipeline = BaseChronosPipeline.from_pretrained(
            self._model_id, device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        return self

    def predict(self, h: int, futr_df: pd.DataFrame | None = None) -> pd.DataFrame:
        levels = list(self._pipeline.quantiles)
        quantiles, _ = self._pipeline.predict_quantiles(
            [torch.from_numpy(x) for x in self._history],
            prediction_length=h,
            quantile_levels=levels,
            batch_size=self._batch_size,
        )
        # series x horizon x level; demand is non-negative.
        q = torch.stack([t.squeeze(0) for t in quantiles]).clamp(min=0).cpu().numpy()

        offset = pd.tseries.frequencies.to_offset(self._freq)
        out = pd.DataFrame(
            {
                "unique_id": np.repeat(self._ids, h),
                "ds": np.concatenate(
                    [
                        pd.date_range(last + offset, periods=h, freq=self._freq)
                        for last in self._last
                    ]
                ),
            }
        )
        if self._point == "mean":
            out["y_pred"] = integrated_mean(q, levels).reshape(-1)
        else:
            out["y_pred"] = q[..., levels.index(0.5)].reshape(-1)
        for q_level in sorted(set(COLUMN_QUANTILES.values())):
            out[f"y_pred_q{int(round(q_level * 100)):02d}"] = q[..., levels.index(q_level)].reshape(
                -1
            )
        return out

    def teardown(self) -> None:
        self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def integrated_mean(quantiles: np.ndarray, levels: list[float]) -> np.ndarray:
    """E[Y] = integral of the quantile function over (0, 1), by the trapezoid
    rule; the tails beyond the outermost levels take those levels' values."""
    u = np.array([0.0, *levels, 1.0])
    padded = np.concatenate([quantiles[..., :1], quantiles, quantiles[..., -1:]], axis=-1)
    widths = np.diff(u)
    return ((padded[..., 1:] + padded[..., :-1]) / 2 * widths).sum(axis=-1)


def build_chronos(freq: str, **kwargs):
    """Zero-arg factory for the backtest harness."""

    def factory():
        return ChronosForecaster(freq=freq, **kwargs)

    return factory
