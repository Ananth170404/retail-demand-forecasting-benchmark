"""Gradient-boosted trees as a global forecaster, via mlforecast.

One model is trained across every series at once, rather than one model per
series. That is the arrangement that won M5. It also removes a classic failure
of per-series pipelines: refitting a categorical encoder inside a per-product
loop maps the same category to different integers for different products, so
the encoding at predict time silently stops matching training. Fitting globally
on columns encoded once, up front, makes that impossible.

Forecasts are produced recursively — each step feeds its own prediction back in
as the next lag. Over a 28-step horizon that compounds error, which is the known
trade-off against training a separate model per horizon.

XGBoost and CatBoost slot in through `library`, on the same features and the
equivalent settings (see _xgboost_params, _catboost_params), so a difference in
score comes from the library. CatBoost alone treats the encoded categoricals as
categories; LightGBM and XGBoost see their integer codes as numbers.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences

from ..data.features import FeatureConfig

DEFAULT_PARAMS = {
    "objective": "tweedie",  # counts with many zeros; see note in build_gbm
    "tweedie_variance_power": 1.1,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


LIBRARY_LABELS = {"lightgbm": "LightGBM", "xgboost": "XGBoost", "catboost": "CatBoost"}


def _is_tweedie(params: dict | None) -> bool:
    """Whether the effective objective (explicit or default) is tweedie."""
    return {**DEFAULT_PARAMS, **(params or {})}.get("objective") == "tweedie"


def _xgboost_params(params: dict, device: str) -> dict:
    """LightGBM settings in XGBoost's terms. Leaf-wise growth limited by leaf
    count is how LightGBM grows trees; under squared error the hessian is 1, so
    min_child_weight is exactly a minimum row count."""
    out = {
        "objective": "reg:tweedie" if params["objective"] == "tweedie" else "reg:squarederror",
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "max_depth": 0,
        "max_leaves": params["num_leaves"],
        "min_child_weight": params["min_child_samples"],
        "device": device,
        "verbosity": 0,
    }
    if params["objective"] == "tweedie":
        out["tweedie_variance_power"] = params["tweedie_variance_power"]
    for key in ("n_estimators", "learning_rate", "subsample", "colsample_bytree", "random_state"):
        out[key] = params[key]
    for key in ("reg_lambda", "n_jobs"):
        if key in params:
            out[key] = params[key]
    return out


def _catboost_params(params: dict, device: str, cat_features: list[str]) -> dict:
    """LightGBM settings in CatBoost's terms. Feature subsampling per tree has
    no GPU equivalent, so it applies on CPU only."""
    loss = (
        f"Tweedie:variance_power={params['tweedie_variance_power']}"
        if params["objective"] == "tweedie"
        else "RMSE"
    )
    out = {
        "loss_function": loss,
        "iterations": params["n_estimators"],
        "learning_rate": params["learning_rate"],
        "grow_policy": "Lossguide",
        "max_leaves": params["num_leaves"],
        "min_data_in_leaf": params["min_child_samples"],
        "bootstrap_type": "Bernoulli",
        "subsample": params["subsample"],
        "random_seed": params["random_state"],
        "cat_features": cat_features,
        # No combinations of categoricals, to save memory. Full M5 still runs out
        # of 15GB on CPU, so its config trains on the GPU.
        "max_ctr_complexity": 1,
        "task_type": "GPU" if device == "cuda" else "CPU",
        "allow_writing_files": False,
        "verbose": False,
    }
    if device != "cuda":
        out["rsm"] = params["colsample_bytree"]
        out["thread_count"] = params["n_jobs"]
    if "reg_lambda" in params:
        out["l2_leaf_reg"] = params["reg_lambda"]
    return out


def _make_regressor(library: str, params: dict, device: str, cat_features: list[str]):
    if library == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**_xgboost_params(params, device))
    if library == "catboost":
        from catboost import CatBoostRegressor

        class Regressor(CatBoostRegressor):
            # mlforecast clones its models. CatBoost rewrites cat_features in its
            # constructor, which sklearn's generic clone rejects as tampering.
            def __sklearn_clone__(self):
                return self.copy()

        return Regressor(**_catboost_params(params, device, cat_features))
    return lgb.LGBMRegressor(**params)


def _clip_non_negative(preds):
    """Floor recursive predictions at zero before they become the next lag.

    mlforecast hands this callback a raw ndarray, so np.clip is used rather than
    the DataFrame-style clip(lower=...).
    """
    return np.clip(preds, 0, None)


class GBMForecaster:
    """Boosted trees over lag/calendar/exogenous features, fit across all series."""

    def __init__(
        self,
        freq: str,
        feature_config: FeatureConfig,
        static_features: list[str] | None = None,
        exog_cols: list[str] | None = None,
        params: dict | None = None,
        name: str = "LightGBM",
        train_days: int | None = None,
        keep_incomplete_rows: bool = False,
        seasonal_difference: int | None = None,
        library: str = "lightgbm",
        device: str = "cpu",
        cat_features: list[str] | None = None,
    ):
        if library not in LIBRARY_LABELS:
            raise ValueError(f"library must be one of {sorted(LIBRARY_LABELS)}, got '{library}'")
        self.name = name
        self._library = library
        self._device = device
        self._cat_features = cat_features or []
        # Model y_t - y_{t-season} instead of y_t; forecasts are last season's
        # value plus the predicted change. The seasonal shape (holiday spikes,
        # the January slump) then comes from history directly instead of being
        # rebuilt from features, and recursive errors cannot carry a spike
        # forward. Lags reaching a full season or more are dropped: on the
        # differenced series they would need two seasons of history.
        if seasonal_difference and _is_tweedie(params):
            raise ValueError("tweedie needs non-negative targets; seasonal differences are not")
        self._seasonal_difference = seasonal_difference
        # By default rows whose lag features are not yet defined (a series'
        # first `max lag` periods) are dropped. Keeping them lets LightGBM
        # learn from that history with the missing lags as NaN, which it
        # handles natively. It matters when history is short: with a 52-week
        # lag and 91 weeks of Walmart data, dropping leaves no Christmas at all
        # among the training targets, so holiday effects cannot be learned.
        self._keep_incomplete_rows = keep_incomplete_rows
        # Train only on each series' most recent `train_days` periods (plus a
        # warm-up so lag features are complete). Full M5 history is ~47M rows of
        # features, more than a 16GB machine holds; recent history is also the
        # most relevant, and M5's leading solutions trained on recent years.
        self._train_days = train_days
        self._freq = freq
        self._fc = feature_config
        self._static_features = static_features or []
        self._exog_cols = exog_cols or []
        self._params = {**DEFAULT_PARAMS, **(params or {})}
        self._mlf: MLForecast | None = None
        self._fit_cols: list[str] = []

    def _feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Columns to hand mlforecast beyond the panel keys."""
        if not self._fc.use_exog:
            return []
        return [c for c in self._exog_cols if c in df.columns]

    def _warmup(self) -> int:
        """Periods of history the lag features reach back, e.g. lag 28 with a
        4-week seasonal window needs 28 + 28. Expanding means are the exception:
        under a train window they expand over the window, not all history."""
        spans = [0]
        for lag in self._fc.lags:
            spans.append(lag)
        for lag, transforms in self._fc.lag_transforms.items():
            for t in transforms:
                window = getattr(t, "window_size", 0) or 0
                spans.append(lag + window * (getattr(t, "season_length", 1) or 1))
        return max(spans)

    def fit(self, train_df: pd.DataFrame) -> GBMForecaster:
        exog = self._feature_columns(train_df)
        self._fit_cols = exog
        df = train_df[["unique_id", "ds", "y"] + exog]
        if self._train_days:
            offset = pd.tseries.frequencies.to_offset(self._freq)
            cutoff = df["ds"].max() - (self._train_days + self._warmup()) * offset
            df = df[df["ds"] > cutoff]
        df = df.copy()

        lags, lag_transforms, target_transforms = self._fc.lags, self._fc.lag_transforms, None
        if self._seasonal_difference:
            d = self._seasonal_difference
            lags = [lag for lag in lags if lag < d]
            lag_transforms = {lag: t for lag, t in lag_transforms.items() if lag < d}
            target_transforms = [Differences([d])]
        self._mlf = MLForecast(
            models=[
                _make_regressor(
                    self._library,
                    self._params,
                    self._device,
                    [c for c in self._cat_features if c in exog],
                )
            ],
            freq=self._freq,
            lags=lags,
            lag_transforms=lag_transforms,
            date_features=self._fc.date_features,
            target_transforms=target_transforms,
        )
        # static_features=[] tells mlforecast that no exogenous column should be
        # carried forward automatically; dynamic ones are supplied via X_df at
        # predict time. Passing statics here instead would silently freeze
        # time-varying covariates at their last training value.
        statics = [c for c in self._static_features if c in exog]
        self._mlf.fit(
            df,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            static_features=statics,
            dropna=not self._keep_incomplete_rows,
        )
        return self

    def predict(self, h: int, futr_df: pd.DataFrame | None = None) -> pd.DataFrame:
        dynamic = [c for c in self._fit_cols if c not in self._static_features]
        X_df = None
        if dynamic and futr_df is not None:
            missing = [c for c in dynamic if c not in futr_df.columns]
            if missing:
                raise ValueError(f"futr_df is missing required covariates: {missing}")

            # mlforecast needs a complete h x n_series grid. The test split is
            # ragged — Walmart departments that closed mid-window simply stop —
            # so the future frame is generated from the model's own series index
            # and the known covariates are merged onto it.
            future = self._mlf.make_future_dataframe(h=h)
            X_df = future.merge(
                futr_df[["unique_id", "ds"] + dynamic], on=["unique_id", "ds"], how="left"
            )
            # Series that end early leave gaps: carry the last known covariate
            # forward, then fall back to zero for anything still unset.
            X_df[dynamic] = X_df.groupby("unique_id")[dynamic].ffill().fillna(0)

        # Predictions feed back in as lags at the next step, so a negative
        # value would propagate through the rest of the horizon. Clipping inside
        # the recursion, not just at the end, keeps the fed-back history valid.
        # With seasonal differencing the recursion runs on year-over-year
        # changes, which are legitimately negative; clip only the final level.
        callback = None if self._seasonal_difference else _clip_non_negative
        raw = self._mlf.predict(h=h, X_df=X_df, after_predict_callback=callback)
        model_col = [c for c in raw.columns if c not in ("unique_id", "ds")][0]
        out = raw.rename(columns={model_col: "y_pred"})[["unique_id", "ds", "y_pred"]]
        # Demand cannot be negative; tweedie will not emit negatives but a
        # swapped-in objective might.
        out["y_pred"] = out["y_pred"].clip(lower=0)
        return out


class PerGroupForecaster:
    """One model per group (e.g. store) instead of one across every series.

    Stores differ in level and trend, and a model that sees only one store fits
    that store's totals directly: the arrangement behind the M5 winner's models.
    """

    def __init__(self, factory, group_col: str, name: str):
        self.name = name
        self._factory = factory
        self._group_col = group_col
        self._models: dict = {}

    def fit(self, train_df: pd.DataFrame) -> PerGroupForecaster:
        self._models = {
            key: self._factory().fit(part)
            for key, part in train_df.groupby(self._group_col, sort=True)
        }
        return self

    def predict(self, h: int, futr_df: pd.DataFrame | None = None) -> pd.DataFrame:
        parts = []
        for key, model in self._models.items():
            part = None if futr_df is None else futr_df[futr_df[self._group_col] == key]
            parts.append(model.predict(h, futr_df=part))
        return pd.concat(parts, ignore_index=True)


def build_gbm(
    freq: str,
    feature_config: FeatureConfig,
    static_features: list[str] | None = None,
    exog_cols: list[str] | None = None,
    objective: str = "tweedie",
    name: str | None = None,
    train_days: int | None = None,
    keep_incomplete_rows: bool = False,
    seasonal_difference: int | None = None,
    lgb_params: dict | None = None,
    library: str = "lightgbm",
    device: str = "cpu",
    cat_features: list[str] | None = None,
):
    """Zero-arg factory for the backtest harness.

    `lgb_params` overrides LightGBM settings (e.g. tuned num_leaves,
    learning_rate) on top of DEFAULT_PARAMS.

    Tweedie is the default objective because the target is non-negative with a
    large point mass at zero — squared error on that distribution pulls
    predictions toward the conditional mean and over-forecasts quiet series.
    Pass objective="l2" to compare against the conventional choice.
    """
    params = {**(lgb_params or {}), "objective": objective}
    if objective != "tweedie":
        params["tweedie_variance_power"] = None
    label = name or f"{LIBRARY_LABELS[library]}-{objective}"

    def factory():
        model = GBMForecaster(
            freq=freq,
            feature_config=feature_config,
            static_features=static_features,
            exog_cols=exog_cols,
            params={k: v for k, v in params.items() if v is not None},
            name=label,
            train_days=train_days,
            keep_incomplete_rows=keep_incomplete_rows,
            seasonal_difference=seasonal_difference,
            library=library,
            device=device,
            cat_features=cat_features,
        )
        return model

    return factory
