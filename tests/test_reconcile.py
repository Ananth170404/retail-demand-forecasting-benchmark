import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from reconcile import scale_to_totals  # noqa: E402

DAY = pd.Timestamp("2020-01-01")
ITEMS = pd.DataFrame(
    {
        "unique_id": ["a", "b", "c"],
        "ds": [DAY, DAY, DAY],
        "y_pred": [1.0, 3.0, 0.0],
    }
)
GROUP_OF = pd.Series({"a": "g1", "b": "g1", "c": "g2"})
TOTALS = pd.DataFrame(
    {"unique_id": ["g1", "g2"], "ds": [DAY, DAY], "y_pred": [8.0, 5.0]},
)


def test_alpha_one_matches_totals_and_keeps_shares():
    out = scale_to_totals(ITEMS, TOTALS, GROUP_OF, 1.0)
    assert out["y_pred"].tolist()[:2] == [2.0, 6.0]


def test_alpha_zero_changes_nothing():
    out = scale_to_totals(ITEMS, TOTALS, GROUP_OF, 0.0)
    assert out["y_pred"].tolist() == ITEMS["y_pred"].tolist()


def test_alpha_moves_the_group_total_linearly():
    out = scale_to_totals(ITEMS, TOTALS, GROUP_OF, 0.5)
    assert out["y_pred"].tolist()[:2] == pytest.approx([1.5, 4.5])  # total 6: halfway 4 to 8


def test_a_group_forecast_at_zero_is_left_alone():
    out = scale_to_totals(ITEMS, TOTALS, GROUP_OF, 1.0)
    assert out["y_pred"].tolist()[2] == 0.0


def test_missing_totals_fail_loudly():
    with pytest.raises(ValueError, match="totals"):
        scale_to_totals(ITEMS, TOTALS.iloc[:1], GROUP_OF, 1.0)
