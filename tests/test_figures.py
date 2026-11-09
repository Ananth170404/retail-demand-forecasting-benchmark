"""Every README figure draws from the committed result tables."""

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_figures as mf  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from forecasting.viz import plots  # noqa: E402

M5 = pd.read_csv(mf.TABLES / "wrmsse_m5_competition.csv")
WALMART = pd.read_csv(mf.TABLES / "wmae_walmart_season.csv")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_every_named_model_is_in_its_leaderboard():
    assert set(mf.M5_NAMES) | set(mf.LEVEL_MODELS) <= set(M5["model"])
    assert set(mf.WALMART_NAMES) <= set(WALMART["model"])


def test_table_figures_draw():
    plots.m5_leaderboard(M5, mf.M5_NAMES)
    plots.m5_levels(M5, mf.LEVEL_MODELS)
    plots.walmart_season(WALMART, mf.WALMART_NAMES)
    m5_ladder = mf.ladder(M5, "LightGBM-l2", "WRMSSE")
    walmart_ladder = mf.ladder(WALMART, "LightGBM-l2-sdiff52", "WMAE_vs_SeasonalNaive")
    assert m5_ladder.index[0] == "lags" and len(m5_ladder) == 6
    assert len(walmart_ladder) == 7
    plots.feature_ladder(m5_ladder, walmart_ladder)


def test_validation_points_pair_each_model_with_its_test_score():
    points = mf.validation_points(M5)
    assert points["model"].is_unique
    headline = points.set_index("model").loc["Ensemble-3w-winner"]
    assert headline["test"] == pytest.approx(0.5899, abs=1e-4)
    assert headline["validation"] == pytest.approx(0.5935, abs=1e-4)
    plots.validation_vs_test(points, {"Ensemble-3w-winner": (8, 0)})
