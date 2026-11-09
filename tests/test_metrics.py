import numpy as np
import pytest

from forecasting.evaluation.metrics import (
    mae,
    mase,
    pinball_loss,
    rmse,
    rmsse,
    trim_leading_zeros,
    zero_counts,
)


@pytest.fixture
def alternating_train():
    """Seasonal-naive(1) absolute diff is always 10, so the MASE scale is exactly 10."""
    return np.array([10.0, 20.0, 10.0, 20.0, 10.0, 20.0])


def test_mae_and_rmse():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([15.0, 15.0])
    assert mae(y_true, y_pred) == 5.0
    assert rmse(y_true, y_pred) == 5.0


def test_mase_against_hand_computed(alternating_train):
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([15.0, 15.0])
    assert mase(y_true, y_pred, alternating_train, season_length=1) == pytest.approx(0.5)


def test_rmsse_against_hand_computed(alternating_train):
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([15.0, 15.0])
    assert rmsse(y_true, y_pred, alternating_train, season_length=1) == pytest.approx(0.5)


def test_perfect_forecast_scores_zero(alternating_train):
    y = np.array([10.0, 20.0])
    assert mase(y, y, alternating_train, season_length=1) == 0.0
    assert rmsse(y, y, alternating_train, season_length=1) == 0.0


def test_flat_train_series_gives_nan_not_inf():
    """A constant history has zero seasonal scale; MASE is undefined, not infinite."""
    flat = np.array([5.0, 5.0, 5.0, 5.0])
    result = mase(np.array([1.0]), np.array([2.0]), flat, season_length=1)
    assert np.isnan(result)


def test_train_shorter_than_season_gives_nan():
    short = np.array([1.0, 2.0])
    assert np.isnan(mase(np.array([1.0]), np.array([2.0]), short, season_length=7))


def test_pinball_median_is_half_mae():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([15.0, 15.0])
    assert pinball_loss(y_true, y_pred, 0.5) == pytest.approx(mae(y_true, y_pred) / 2)


def test_pinball_penalises_asymmetrically():
    y_true = np.array([10.0])
    under, over = np.array([8.0]), np.array([12.0])
    # q=0.9 should punish under-forecasting more than over-forecasting.
    assert pinball_loss(y_true, under, 0.9) > pinball_loss(y_true, over, 0.9)


def test_trim_leading_zeros_drops_pre_launch_period():
    assert trim_leading_zeros(np.array([0.0, 0.0, 3.0, 0.0, 5.0])).tolist() == [3.0, 0.0, 5.0]


def test_trim_leading_zeros_keeps_series_starting_nonzero():
    y = np.array([1.0, 0.0, 2.0])
    assert trim_leading_zeros(y).tolist() == y.tolist()


def test_trim_leading_zeros_on_all_zero_series():
    assert len(trim_leading_zeros(np.zeros(5))) == 0


def test_mase_excludes_pre_launch_zeros_by_default():
    """Official M5 rule: the scale covers only actively-sold periods.

    Leading zeros are 'not stocked yet', and counting them produces zero
    seasonal differences that shrink the denominator and inflate MASE.
    """
    y_train = np.array([0.0] * 6 + [10.0, 20.0, 10.0, 20.0, 10.0, 20.0])
    y_true, y_pred = np.array([10.0, 20.0]), np.array([15.0, 15.0])

    active = mase(y_true, y_pred, y_train, season_length=1, active_only=True)
    naive = mase(y_true, y_pred, y_train, season_length=1, active_only=False)

    assert active == pytest.approx(0.5)
    assert naive > active


def test_rmsse_excludes_pre_launch_zeros_by_default():
    y_train = np.array([0.0] * 6 + [10.0, 20.0, 10.0, 20.0, 10.0, 20.0])
    y_true, y_pred = np.array([10.0, 20.0]), np.array([15.0, 15.0])
    assert rmsse(y_true, y_pred, y_train, season_length=1) == pytest.approx(0.5)


def test_zero_counts_floors_predictions():
    y_true = np.array([0.0, 0.0, 5.0])
    y_pred = np.array([0.4, 1.2, 5.0])
    actual, predicted = zero_counts(y_true, y_pred)
    assert actual == 2
    assert predicted == 1  # 0.4 floors to 0; 1.2 does not
