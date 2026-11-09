"""XGBoost and CatBoost share LightGBM's pipeline, features and settings."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from registry import Context, build_entry  # noqa: E402

from forecasting.data.features import M5_DIRECT_FEATURES, FeatureConfig  # noqa: E402
from forecasting.data.loaders import ColumnRoles  # noqa: E402
from forecasting.models.gbm import (  # noqa: E402
    DEFAULT_PARAMS,
    GBMForecaster,
    _catboost_params,
    _xgboost_params,
)

CTX = Context(
    dataset="m5",
    cfg={"freq": "D", "season_length": 7},
    horizon=7,
    roles=ColumnRoles(static_cat_cols=["store_cat"], dynamic_cat_cols=[], dynamic_real_cols=[]),
)
L2 = {**DEFAULT_PARAMS, "objective": "l2"}


def _panel() -> pd.DataFrame:
    days = pd.date_range("2020-01-01", periods=60)
    return pd.concat(
        pd.DataFrame(
            {
                "unique_id": f"s{i}",
                "ds": days,
                "y": (i + 1.0) * (1 + (days.dayofweek == 5)),
                "store_cat": np.int16(i),
            }
        )
        for i in range(3)
    )


def test_labels_name_the_library_and_lightgbm_labels_do_not_change():
    assert list(build_entry("gbm", {"objectives": ["l2"]}, CTX)) == ["LightGBM-l2"]
    assert list(build_entry("gbm", {"objectives": ["l2"], "library": "xgboost"}, CTX)) == [
        "XGBoost-l2"
    ]
    assert list(build_entry("gbm", {"objectives": ["l2"], "library": "catboost"}, CTX)) == [
        "CatBoost-l2"
    ]


def test_unknown_library_fails_loudly():
    with pytest.raises(ValueError, match="library"):
        GBMForecaster(freq="D", feature_config=FeatureConfig(), library="sklearn")


def test_settings_translate_from_lightgbm():
    xgb = _xgboost_params(L2, "cpu")
    assert (xgb["objective"], xgb["max_leaves"], xgb["min_child_weight"]) == (
        "reg:squarederror",
        L2["num_leaves"],
        L2["min_child_samples"],
    )
    assert "tweedie_variance_power" not in xgb
    tweedie = {**DEFAULT_PARAMS, "tweedie_variance_power": 1.3}
    assert _xgboost_params(tweedie, "cpu")["tweedie_variance_power"] == 1.3

    cat = _catboost_params(L2, "cpu", ["store_cat"])
    assert (cat["loss_function"], cat["iterations"], cat["cat_features"]) == (
        "RMSE",
        L2["n_estimators"],
        ["store_cat"],
    )
    assert cat["rsm"] == L2["colsample_bytree"]
    assert _catboost_params(tweedie, "cpu", [])["loss_function"] == "Tweedie:variance_power=1.3"
    assert "rsm" not in _catboost_params(L2, "cuda", [])


@pytest.mark.parametrize("library", ["xgboost", "catboost"])
def test_fits_and_forecasts_end_to_end(library):
    pytest.importorskip(library)
    [factory] = build_entry(
        "gbm",
        {"objectives": ["l2"], "library": library, "lgb_params": {"n_estimators": 5}},
        CTX,
    ).values()
    model = factory()
    model._params.update(min_child_samples=1)
    out = model.fit(_panel()).predict(7)
    assert len(out) == 21
    assert out["y_pred"].notna().all() and (out["y_pred"] >= 0).all()


def test_direct_features_never_look_inside_the_m5_horizon():
    assert min(M5_DIRECT_FEATURES.lags) >= 28
    assert min(M5_DIRECT_FEATURES.lag_transforms) >= 28


def test_per_group_fits_one_model_per_store_and_forecasts_every_series():
    [factory] = build_entry(
        "gbm",
        {
            "objectives": ["l2"],
            "group_by": "store_cat",
            "feature_step": "static",
            "lgb_params": {"n_estimators": 5, "min_child_samples": 1},
            "model_name": "LightGBM-store",
        },
        CTX,
    ).values()
    model = factory()
    assert model.name == "LightGBM-store"
    model.fit(_panel())
    assert sorted(model._models) == [0, 1, 2]
    out = model.predict(7)
    assert len(out) == 21
    assert out["unique_id"].nunique() == 3


def test_direct_model_uses_direct_features_and_forecasts_the_horizon():
    days = pd.date_range("2020-01-01", periods=200)
    panel = pd.concat(
        pd.DataFrame(
            {
                "unique_id": f"s{i}",
                "ds": days,
                "y": (i + 1.0) * (1 + (days.dayofweek == 5)),
                "store_cat": np.int16(i),
            }
        )
        for i in range(3)
    )
    [factory] = build_entry(
        "gbm",
        {
            "objectives": ["l2"],
            "direct": True,
            "feature_step": "static",
            "lgb_params": {"n_estimators": 5, "min_child_samples": 1},
            "model_name": "LightGBM-direct",
        },
        Context(dataset="m5", cfg=CTX.cfg, horizon=28, roles=CTX.roles),
    ).values()
    model = factory()
    assert model._fc is M5_DIRECT_FEATURES
    assert len(model.fit(panel).predict(28)) == 84
