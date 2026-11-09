import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from registry import Context, build_entry, feature_ladder  # noqa: E402

from forecasting.data.loaders import ColumnRoles  # noqa: E402

CFG = {"freq": "W-FRI", "season_length": 52}


def _ctx(holiday_cols):
    roles = ColumnRoles(
        static_cat_cols=["Type_cat"],
        dynamic_cat_cols=["is_holiday_cat"],
        dynamic_real_cols=["CPI"],
        holiday_cols=holiday_cols,
    )
    return Context(dataset="walmart", cfg=CFG, horizon=8, roles=roles)


def test_each_step_adds_exactly_one_block():
    ladder = feature_ladder(_ctx(["weeks_to_xmas"]))
    assert [step for step, _, _ in ladder] == [
        "lags",
        "rolling",
        "calendar",
        "static",
        "events",
        "covariates",
        "holiday",
    ]
    lags, rolling, calendar = (fc for _, fc, _ in ladder[:3])
    assert not lags.lag_transforms and not lags.date_features
    assert rolling.lag_transforms and not rolling.date_features
    assert calendar.lag_transforms and calendar.date_features and not calendar.use_exog
    assert [cols for _, _, cols in ladder[3:]] == [
        ["Type_cat"],
        ["Type_cat", "is_holiday_cat"],
        ["Type_cat", "is_holiday_cat", "CPI"],
        ["Type_cat", "is_holiday_cat", "CPI", "weeks_to_xmas"],
    ]


def test_ladder_skips_empty_blocks():
    assert [step for step, _, _ in feature_ladder(_ctx([]))][-1] == "covariates"


def test_ablation_entry_labels_are_unique_and_ordered():
    factories = build_entry(
        "gbm",
        {"objectives": ["l2"], "ablation_ladder": True, "seasonal_difference": 52},
        _ctx(["weeks_to_xmas"]),
    )
    assert list(factories)[0] == "LightGBM-l2-sdiff52-L1-lags"
    assert list(factories)[-1] == "LightGBM-l2-sdiff52-L7-holiday"
    assert factories["LightGBM-l2-sdiff52-L4-static"]()._exog_cols == ["Type_cat"]


def test_existing_variant_labels_are_unchanged():
    """Saved forecasts and result rows are keyed by these exact labels."""
    ctx = _ctx(["weeks_to_xmas"])
    cases = [
        (
            {"objectives": ["l2"], "holiday_features": True, "seasonal_difference": 52},
            "LightGBM-l2-hol-sdiff52",
        ),
        (
            {"objectives": ["tweedie"], "holiday_features": True, "keep_incomplete_rows": True},
            "LightGBM-tweedie-hol-allrows",
        ),
        ({"objectives": ["tweedie"], "use_exog": False}, "LightGBM-tweedie-noexog"),
        ({"objectives": ["l2"]}, "LightGBM-l2"),
    ]
    for params, label in cases:
        assert list(build_entry("gbm", params, ctx)) == [label]


def test_tuned_configuration_selects_one_step_with_params():
    params = {
        "objectives": ["l2"],
        "feature_step": "events",
        "lgb_params": {"num_leaves": 31, "learning_rate": 0.1},
        "model_name": "LightGBM-tuned",
    }
    [(label, factory)] = build_entry("gbm", params, _ctx(["weeks_to_xmas"])).items()
    model = factory()
    assert label == model.name == "LightGBM-tuned"
    assert model._exog_cols == ["Type_cat", "is_holiday_cat"]
    assert model._params["num_leaves"] == 31 and model._params["objective"] == "l2"


def test_unknown_feature_step_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="feature_step"):
        build_entry("gbm", {"objectives": ["l2"], "feature_step": "price"}, _ctx([]))
