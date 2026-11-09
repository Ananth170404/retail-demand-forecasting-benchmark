"""WRMSSE: hand-checkable synthetic cases, plus a check against the official release.

The release checks need data/raw/m5 with the Kaggle files and the test labels
(`python scripts/download_data.py --dataset m5 && ... --dataset m5-labels`) and
are skipped without them, so CI runs only the synthetic cases.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.evaluation.m5 import LEVELS, WRMSSEEvaluator, _active_scale, panel_to_wide

M5_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "m5"
HAS_RELEASE = (M5_DIR / "sales_test_evaluation.csv").exists() and (
    M5_DIR / "sales_train_evaluation.csv"
).exists()


@pytest.fixture
def tiny():
    """Two items in two stores of one state; four training days, two test days."""
    sales = pd.DataFrame(
        {
            "item_id": ["A", "A", "B", "B"],
            "dept_id": ["D"] * 4,
            "cat_id": ["C"] * 4,
            "store_id": ["S_1", "S_2", "S_1", "S_2"],
            "state_id": ["S"] * 4,
            "d_1": [0, 1, 2, 2],
            "d_2": [1, 1, 2, 4],
            "d_3": [3, 1, 2, 2],
            "d_4": [1, 1, 2, 4],
        }
    )
    actuals = pd.DataFrame(
        [[2, 0], [1, 1], [2, 2], [3, 3]],
        index=pd.MultiIndex.from_arrays([sales.item_id, sales.store_id]),
        columns=["d_5", "d_6"],
    )
    dollars = np.array([1.0, 1.0, 1.0, 1.0])
    return sales, actuals, dollars


def test_active_scale_starts_at_first_sale():
    # Leading zero dropped: active history [1, 3, 1] -> diffs [2, -2] -> mean sq 4.
    assert _active_scale(np.array([[0.0, 1.0, 3.0, 1.0]]))[0] == pytest.approx(4.0)


def test_flat_history_has_undefined_scale():
    assert np.isnan(_active_scale(np.array([[1.0, 1.0, 1.0, 1.0]]))[0])


def test_perfect_forecast_scores_zero(tiny):
    sales, actuals, dollars = tiny
    ev = WRMSSEEvaluator(sales, calendar=None, prices=None, bottom_dollars=dollars)
    scores = ev.score(actuals, actuals)
    assert scores["WRMSSE"] == pytest.approx(0.0)
    assert list(scores.index[:-1]) == list(LEVELS)


def test_total_level_against_hand_computation(tiny):
    sales, actuals, dollars = tiny
    ev = WRMSSEEvaluator(sales, calendar=None, prices=None, bottom_dollars=dollars)
    forecast = actuals * 0 + 1  # forecast 1 everywhere
    # Level 1 total history: [5, 8, 8, 8] -> diffs [3, 0, 0] -> scale 9/3 = 3.
    # Total actuals [8, 6] vs forecast [4, 4] -> mse (16 + 4) / 2 = 10.
    expected = np.sqrt(10 / 3)
    assert ev.score(actuals, forecast)[1] == pytest.approx(expected)


def test_weights_follow_dollar_sales(tiny):
    sales, _, _ = tiny
    ev = WRMSSEEvaluator(
        sales, calendar=None, prices=None, bottom_dollars=np.array([1.0, 1.0, 1.0, 5.0])
    )
    assert ev.weights[12].tolist() == pytest.approx([0.125, 0.125, 0.125, 0.625])
    # Level 10 (item): A = 2/8, B = 6/8
    assert ev.weights[10].tolist() == pytest.approx([0.25, 0.75])
    for level in LEVELS:
        assert ev.weights[level].sum() == pytest.approx(1.0)


def test_missing_bottom_series_is_an_error(tiny):
    sales, actuals, dollars = tiny
    ev = WRMSSEEvaluator(sales, calendar=None, prices=None, bottom_dollars=dollars)
    with pytest.raises(ValueError, match="missing"):
        ev.score(actuals, actuals.iloc[:3])


def test_panel_to_wide_splits_product_id():
    panel = pd.DataFrame(
        {"unique_id": ["FOODS_1_001_CA_1"] * 2, "ds": ["d_1", "d_2"], "y": [1.0, 2.0]}
    )
    wide = panel_to_wide(panel, "y")
    assert wide.index.tolist() == [("FOODS_1_001", "CA_1")]


# --- against the official release ----------------------------------------------

# Paper Table B (Makridakis et al., 2022), levels 1-12 then overall.
PUBLISHED = {
    "sNaive": [0.560, 0.673, 0.718, 0.623, 0.708, 0.760, 0.829, 0.801, 0.888, 1.223, 1.205, 1.176]
    + [0.847],
    "Naive": [1.967, 1.904, 1.880, 1.947, 1.914, 1.881, 1.878, 1.798, 1.764, 1.479, 1.360, 1.253]
    + [1.752],
}


@pytest.fixture(scope="module")
def release():
    sales = pd.read_csv(M5_DIR / "sales_train_evaluation.csv")
    calendar = pd.read_csv(M5_DIR / "calendar.csv")
    prices = pd.read_csv(M5_DIR / "sell_prices.csv")
    test = pd.read_csv(M5_DIR / "sales_test_evaluation.csv")
    actuals = test.set_index(["item_id", "store_id"])[[f"d_{d}" for d in range(1942, 1970)]]
    ev = WRMSSEEvaluator(sales, calendar, prices)
    return sales, ev, actuals


@pytest.mark.skipif(not HAS_RELEASE, reason="M5 data and test labels not downloaded")
def test_computed_weights_match_official_file(release):
    _, ev, _ = release
    official = pd.read_csv(M5_DIR / "weights_evaluation.csv")
    # Agg_Level_1/2 per level, as the official file names them.
    file_cols = {
        12: ["item_id", "store_id"],
        11: ["state_id", "item_id"],
        9: ["store_id", "dept_id"],
    }
    for level, cols in file_cols.items():
        rows = official[official.Level_id == f"Level{level}"]
        by_name = rows.set_index(["Agg_Level_1", "Agg_Level_2"])["weight"]
        keys = ev.groups[level][cols].itertuples(index=False, name=None)
        np.testing.assert_allclose(ev.weights[level], by_name.loc[list(keys)], atol=1e-6)


@pytest.mark.skipif(not HAS_RELEASE, reason="M5 data and test labels not downloaded")
@pytest.mark.parametrize("method", ["sNaive", "Naive"])
def test_benchmarks_reproduce_published_scores(release, method):
    sales, ev, actuals = release
    history = sales.set_index(["item_id", "store_id"])[ev.train_days]
    if method == "sNaive":
        values = np.tile(history.iloc[:, -7:].to_numpy(), 4)
    else:
        values = np.repeat(history.iloc[:, -1:].to_numpy(), 28, axis=1)
    forecast = pd.DataFrame(values, index=history.index, columns=actuals.columns)
    scores = ev.score(actuals, forecast).round(3).tolist()
    assert scores == pytest.approx(PUBLISHED[method], abs=0.0011)


@pytest.mark.skipif(
    not (M5_DIR / "sales_test_validation.csv").exists(), reason="M5 validation labels missing"
)
def test_validation_phase_weights_match_official_file():
    from forecasting.evaluation.m5 import load_release

    ev, actuals = load_release(M5_DIR, phase="validation")
    assert ev.train_days[-1] == "d_1913"
    assert list(actuals.columns[[0, -1]]) == ["d_1914", "d_1941"]
    official = pd.read_csv(M5_DIR / "weights_validation.csv")
    rows = official[official.Level_id == "Level12"].set_index(["Agg_Level_1", "Agg_Level_2"])
    keys = ev.groups[12][["item_id", "store_id"]].itertuples(index=False, name=None)
    np.testing.assert_allclose(ev.weights[12], rows.loc[list(keys), "weight"], atol=1e-6)


def test_load_release_rejects_unknown_phase():
    from forecasting.evaluation.m5 import load_release

    with pytest.raises(ValueError, match="phase"):
        load_release(M5_DIR, phase="test")


@pytest.mark.skipif(
    not (M5_DIR / "sales_train_evaluation.csv").exists(), reason="M5 evaluation data missing"
)
def test_window_ending_d1913_is_the_validation_phase():
    from forecasting.evaluation.m5 import load_release, load_window

    ev, actuals = load_window(M5_DIR, end_day=1913)
    official_ev, official_actuals = load_release(M5_DIR, phase="validation")
    assert ev.train_days == official_ev.train_days
    np.testing.assert_allclose(ev.weights[12], official_ev.weights[12])
    np.testing.assert_allclose(actuals.to_numpy(), official_actuals.to_numpy())

    earlier, earlier_actuals = load_window(M5_DIR, end_day=1857)
    assert earlier.train_days[-1] == "d_1857"
    assert list(earlier_actuals.columns[[0, -1]]) == ["d_1858", "d_1885"]
