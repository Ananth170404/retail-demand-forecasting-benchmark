import numpy as np
import pandas as pd
import pytest

from forecasting.evaluation.significance import (
    compare_to_reference,
    holm,
    mean_ranks,
    nemenyi_cd,
    per_series_matrix,
)


@pytest.fixture
def wide():
    """'good' is uniformly 0.1 better than 'base'; 'twin' is base plus pure noise."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.5, 1.5, size=400)
    return pd.DataFrame(
        {
            "base": base,
            "good": base - 0.1,
            "twin": base + rng.normal(0, 0.05, size=400),
        }
    )


def test_per_series_matrix_averages_windows_and_keeps_common_series():
    results = pd.DataFrame(
        {
            "unique_id": ["a", "a", "b", "b", "c"],
            "window": [0, 1, 0, 0, 0],
            "model": ["M", "M", "M", "N", "M"],
            "MASE": [1.0, 3.0, 5.0, 6.0, 7.0],
        }
    )
    # add N's score for "a" so "a" and "b" are common; "c" lacks N and is dropped
    results.loc[len(results)] = ["a", 0, "N", 4.0]
    wide = per_series_matrix(results, "MASE")
    assert list(wide.index) == ["a", "b"]
    assert wide.loc["a", "M"] == 2.0  # mean over the two windows


def test_holm_matches_hand_computation():
    p = pd.Series({"x": 0.01, "y": 0.04, "z": 0.03})
    # sorted 0.01, 0.03, 0.04 -> 3*0.01, 2*0.03, max(0.06, 1*0.04)
    assert holm(p).to_dict() == pytest.approx({"x": 0.03, "z": 0.06, "y": 0.06})


def test_reference_defaults_to_best_and_detects_real_gap(wide):
    out = compare_to_reference(wide, n_boot=500)
    assert out["reference"].iloc[0] == "good"
    assert out.loc["base", "diff"] == pytest.approx(0.1)
    assert out.loc["base", "ci_low"] > 0
    assert out.loc["base", "significant"]
    assert out.loc["base", "win_rate"] == 0.0


def test_noise_alone_is_not_called_significant(wide):
    out = compare_to_reference(wide, reference="base", n_boot=500)
    assert not out.loc["twin", "significant"]
    assert out.loc["twin", "ci_low"] < 0 < out.loc["twin", "ci_high"]


def test_mean_ranks_and_critical_difference(wide):
    ranks = mean_ranks(wide)
    assert ranks.index[0] == "good"
    # Not exactly 1: twin's noise occasionally drops it below good.
    assert ranks.loc["good", "mean_rank"] < 1.05
    assert not ranks.loc["base", "tied_with_best"]
    # Demšar (2006) Table 5: q_0.05 for k=3 is 2.343
    assert nemenyi_cd(3, 400) == pytest.approx(2.343 * np.sqrt(3 * 4 / (6 * 400)), rel=1e-3)
