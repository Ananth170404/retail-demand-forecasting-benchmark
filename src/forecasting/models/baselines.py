"""Statistical baselines via statsforecast.

Every model added later has to beat these to justify its complexity. Croston
and TSB are included because they are the standard methods for intermittent
demand — on a panel that is 68% zeros, they are serious contenders rather than
strawmen.
"""

from __future__ import annotations

import copy
import gc

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import (
    ADIDA,
    IMAPA,
    TSB,
    AutoETS,
    AutoTheta,
    CrostonOptimized,
    CrostonSBA,
    HistoricAverage,
    Naive,
    SeasonalNaive,
    SimpleExponentialSmoothing,
    SimpleExponentialSmoothingOptimized,
)


class StatsForecastModel:
    """Adapts any statsforecast model to the Forecaster interface.

    Memory notes, learned from an OOM at 6,000 M5 series (9.2M rows):

    statsforecast parallelises across series by forking processes, and each one
    receives a copy of the panel. With `n_jobs=-1` on a 16-core machine that is
    sixteen copies, which exhausted a laptop with ~4GB free even though a single
    fit is comfortable. `n_jobs` is therefore capped rather than left at -1.

    `chunk_size` adds a second line of defence: series are fitted in batches and
    each batch's fitted state is discarded once its forecast is produced, so
    peak memory is set by the batch rather than the panel. AutoTheta in
    particular holds per-series state that does not fit at full scale.

    When chunking, fit and predict necessarily happen together — the horizon is
    not known at fit time — so the work lands in predict(). The harness's
    fit/predict split is therefore not meaningful for chunked models, though
    their total time is still correct.
    """

    def __init__(
        self,
        name: str,
        model,
        freq: str,
        season_length: int | None = None,
        n_jobs: int = 4,
        chunk_size: int | None = None,
    ):
        self.name = name
        self._model = model
        self._freq = freq
        self._season_length = season_length
        self._n_jobs = n_jobs
        self._chunk_size = chunk_size
        self._sf: StatsForecast | None = None
        self._train_df: pd.DataFrame | None = None

    def _new_sf(self) -> StatsForecast:
        # A fresh copy of the model per StatsForecast instance: fitting stores
        # per-series state on the model object, so reusing one across chunks
        # would carry state between batches.
        return StatsForecast(
            models=[copy.deepcopy(self._model)], freq=self._freq, n_jobs=self._n_jobs
        )

    def fit(self, train_df: pd.DataFrame) -> StatsForecastModel:
        self._train_df = train_df[["unique_id", "ds", "y"]]
        if self._chunk_size:
            return self  # deferred: chunked fits happen alongside prediction
        self._sf = self._new_sf()
        self._sf.fit(self._train_df)
        return self

    def _predict_chunked(self, h: int) -> pd.DataFrame:
        ids = self._train_df["unique_id"].unique()
        parts = []
        for start in range(0, len(ids), self._chunk_size):
            batch = ids[start : start + self._chunk_size]
            sub = self._train_df[self._train_df["unique_id"].isin(batch)]
            sf = self._new_sf()
            # forecast() fits and predicts in one pass without keeping every
            # fitted model alive: AutoETS state for 2,000 series held across a
            # fit/predict pair exhausted RAM on full M5.
            parts.append(sf.forecast(df=sub, h=h))
            del sf
            gc.collect()
        return pd.concat(parts, ignore_index=True)

    def predict(self, h: int, futr_df: pd.DataFrame | None = None) -> pd.DataFrame:
        raw = self._predict_chunked(h) if self._chunk_size else self._sf.predict(h=h)
        model_col = [c for c in raw.columns if c not in ("unique_id", "ds")][0]
        return raw.rename(columns={model_col: "y_pred"})[["unique_id", "ds", "y_pred"]]


class ZeroForecaster:
    """Forecasts zero demand everywhere — the floor for a metric, not a model.

    On a panel where most periods are zero, absolute-error metrics reward
    predicting nothing: on M5 this beats LightGBM, Croston and SeasonalNaive on
    MASE. A model that cannot clearly beat it on a given metric has not shown
    skill on that metric, so it runs alongside every baseline set.
    """

    def __init__(self, freq: str, name: str = "Zero"):
        self.name = name
        self._offset = pd.tseries.frequencies.to_offset(freq)
        self._last_ds: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame) -> ZeroForecaster:
        self._last_ds = train_df.groupby("unique_id")["ds"].max()
        return self

    def predict(self, h: int, futr_df: pd.DataFrame | None = None) -> pd.DataFrame:
        # Each series continues from its own last date, as statsforecast does,
        # so series that ended early line up with their (absent) test rows.
        last = pd.DatetimeIndex(self._last_ds)
        steps = [
            pd.DataFrame({"unique_id": self._last_ds.index, "ds": last + k * self._offset})
            for k in range(1, h + 1)
        ]
        out = pd.concat(steps, ignore_index=True).sort_values(["unique_id", "ds"])
        out["y_pred"] = 0.0
        return out.reset_index(drop=True)


class ChristmasShiftedSeasonalNaive:
    """Weekly seasonal naive, with December realigned to the calendar date.

    Plain seasonal naive copies the week 52 weeks (364 days) back. The calendar
    year is 365 or 366 days, so that week sits one or two days later in the
    calendar than "the same week last year". Weekday holidays are unaffected,
    but Christmas is a fixed date: pre-Christmas shopping days slide between
    weeks, e.g. the week before Christmas held 7 such days in 2010 and 6 in
    2011. In December weeks this forecaster blends the two neighbouring weeks
    of last year by that day drift. Every other week is plain seasonal naive.
    """

    def __init__(self, freq: str = "W-FRI", season: int = 52, name: str = "SNaive-xmas"):
        self.name = name
        self._offset = pd.tseries.frequencies.to_offset(freq)
        self._season = season
        self._history: pd.DataFrame | None = None

    @staticmethod
    def in_christmas_period(dates: pd.DatetimeIndex) -> np.ndarray:
        """Week-ending dates from Dec 10 to Jan 7: the weeks Christmas drift affects."""
        md = dates.month * 100 + dates.day
        return np.asarray((md >= 1210) | (md <= 107))

    def fit(self, train_df: pd.DataFrame) -> ChristmasShiftedSeasonalNaive:
        self._history = train_df.pivot(index="unique_id", columns="ds", values="y")
        return self

    def predict(self, h: int, futr_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if h > self._season:
            raise ValueError(f"horizon {h} exceeds one season ({self._season})")
        hist = self._history
        last = hist.columns.max()
        future = pd.DatetimeIndex([last + k * self._offset for k in range(1, h + 1)])
        lag = hist.reindex(columns=future - self._season * self._offset).to_numpy()
        lag_prev = hist.reindex(columns=future - (self._season + 1) * self._offset).to_numpy()

        # Days by which last year's calendar-aligned week precedes the 52-week lag.
        drift = np.array(
            [
                ((d - self._season * self._offset) - (d - pd.DateOffset(years=1))).days
                for d in future
            ]
        )
        share = np.where(self.in_christmas_period(future), np.clip(drift, 0, 7) / 7.0, 0.0)
        blended = (1 - share) * lag + share * np.where(np.isnan(lag_prev), lag, lag_prev)

        # Too little history for a lag: fall back to the series' last value.
        last_value = hist.ffill(axis=1).iloc[:, -1].to_numpy()[:, None]
        values = np.where(np.isnan(blended), last_value, blended)
        out = pd.DataFrame(values, index=hist.index, columns=future).stack()
        out = out.rename("y_pred").reset_index()
        out.columns = ["unique_id", "ds", "y_pred"]
        return out


def build_baselines(
    freq: str,
    season_length: int,
    include_slow: bool = True,
    n_jobs: int = 4,
    chunk_size: int | None = None,
) -> dict[str, callable]:
    """Zero-arg factories for each baseline, keyed by name.

    Factories (not instances) so each backtest window trains a fresh model.

    Two families are represented deliberately. The intermittent-demand methods
    (Croston and its SBA correction, TSB, ADIDA, IMAPA) are what the literature
    actually recommends for sparse series, so beating them is the bar that
    matters on M5. The automatic statistical methods (ETS, Theta) are the
    strongest classical competitors — Theta won the M3 competition — and they
    are the reason a deep model's win cannot be waved away as "it only beat
    naive forecasts".

    On parameter fairness: AutoETS, AutoTheta, CrostonOptimized and
    SES-opt fit their own parameters per series, so comparing them against a
    model frozen at hand-picked values measures tuning as much as method. SES is
    therefore run in both forms — the fixed alpha=0.1 and the optimised variant
    — which turns that confound into a measurement of what self-tuning is worth.

    TSB is the one model left with hand-set parameters: statsforecast provides
    no optimised variant, and alpha_d/alpha_p would need a validation protocol
    of our own to fit honestly. Its numbers should be read as a lower bound on
    what TSB can do.

    `include_slow` drops ETS/Theta, whose optimisers are far more expensive
    than the rest of the set. `Zero` always runs; see ZeroForecaster.
    """

    def make(label: str, model):
        """Factory for one baseline, carrying the shared memory settings."""
        return lambda: StatsForecastModel(
            label,
            model,
            freq,
            season_length,
            n_jobs=n_jobs,
            chunk_size=chunk_size,
        )

    specs = [
        ("Naive", Naive()),
        ("SeasonalNaive", SeasonalNaive(season_length=season_length)),
        ("HistoricAverage", HistoricAverage()),
        # SES is kept alongside SES-opt so the gap between them isolates the
        # value of self-tuning rather than leaving it as a confound.
        ("SES", SimpleExponentialSmoothing(alpha=0.1)),
        ("SES-opt", SimpleExponentialSmoothingOptimized()),
        ("Croston", CrostonOptimized()),
        # Croston is known to be biased; SBA applies the standard correction.
        ("CrostonSBA", CrostonSBA()),
        ("TSB", TSB(alpha_d=0.2, alpha_p=0.2)),
        # ADIDA and IMAPA aggregate demand into coarser buckets before
        # forecasting, which is a different way of coping with sparsity than
        # Croston's separate size/interval modelling.
        ("ADIDA", ADIDA()),
        ("IMAPA", IMAPA()),
    ]
    if include_slow:
        specs += [
            ("AutoETS", AutoETS(season_length=season_length)),
            ("AutoTheta", AutoTheta(season_length=season_length)),
        ]

    factories = {"Zero": lambda: ZeroForecaster(freq)}
    # Christmas drift only exists for weekly data with a yearly season.
    if freq.startswith("W") and season_length == 52:
        factories["SNaive-xmas"] = lambda: ChristmasShiftedSeasonalNaive(freq, season_length)
    factories.update({label: make(label, model) for label, model in specs})
    return factories
