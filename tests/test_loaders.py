import pandas as pd
import pytest

from forecasting.data.loaders import _reindex_walmart_weeks


@pytest.fixture
def gapped_train():
    """Two series: one complete, one missing two internal weeks and starting late."""
    weeks = pd.date_range("2010-02-05", periods=6, freq="W-FRI")
    complete = pd.DataFrame(
        {"Store": 1, "Dept": 1, "Date": weeks, "Weekly_Sales": [10.0] * 6, "IsHoliday": False}
    )
    # Starts one week late, and is missing weeks[3] and weeks[4] internally.
    gapped = pd.DataFrame(
        {
            "Store": 1,
            "Dept": 2,
            "Date": [weeks[1], weeks[2], weeks[5]],
            "Weekly_Sales": [5.0, 6.0, 7.0],
            "IsHoliday": False,
        }
    )
    return pd.concat([complete, gapped], ignore_index=True)


def test_internal_gaps_filled_with_zero(gapped_train):
    out = _reindex_walmart_weeks(gapped_train)
    dept2 = out[out["Dept"] == 2].sort_values("Date")
    # Late start is preserved (5 weeks, not 6); the two internal holes are filled.
    assert len(dept2) == 5
    assert dept2["Weekly_Sales"].tolist() == [5.0, 6.0, 0.0, 0.0, 7.0]


def test_leading_absence_not_backfilled(gapped_train):
    """A department that opened late must not have history fabricated before it existed."""
    out = _reindex_walmart_weeks(gapped_train)
    dept2 = out[out["Dept"] == 2]
    assert dept2["Date"].min() == gapped_train["Date"].min() + pd.Timedelta(weeks=1)


def test_complete_series_untouched(gapped_train):
    out = _reindex_walmart_weeks(gapped_train)
    dept1 = out[out["Dept"] == 1]
    assert len(dept1) == 6
    assert dept1["Weekly_Sales"].tolist() == [10.0] * 6


def test_grid_is_regular_after_reindex(gapped_train):
    """The property that makes seasonal lags calendar-correct rather than row-correct."""
    out = _reindex_walmart_weeks(gapped_train)
    for _, g in out.groupby(["Store", "Dept"]):
        deltas = g.sort_values("Date")["Date"].diff().dropna().dt.days.unique()
        assert set(deltas) <= {7}


def test_identifiers_preserved_on_filled_rows(gapped_train):
    out = _reindex_walmart_weeks(gapped_train)
    assert out["Store"].notna().all()
    assert out["Dept"].notna().all()
    assert out["product_id"].notna().all()
