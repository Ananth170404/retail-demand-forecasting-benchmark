import sys
from pathlib import Path

import optuna
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from tune import DEFAULTS, default_trial, suggest  # noqa: E402

STEPS = ["lags", "rolling", "calendar", "static", "events", "covariates", "holiday"]


@pytest.mark.parametrize("dataset", ["m5", "walmart"])
def test_trial_zero_reproduces_the_untuned_configuration(dataset):
    """Trial 0 must be the pre-tuning model, or 'tuned vs default' means nothing."""
    params = suggest(optuna.trial.FixedTrial(default_trial(dataset)), dataset, STEPS)
    expected = DEFAULTS[dataset]
    assert params["feature_step"] == expected["feature_step"]
    assert params["objectives"] == [expected["objective"]]
    assert params.get("seasonal_difference") == expected.get("seasonal_difference")
    assert params.get("train_days") == expected.get("train_days")
    # LightGBM settings equal the defaults the untuned models ran with.
    from forecasting.models.gbm import DEFAULT_PARAMS

    for key in ("num_leaves", "learning_rate", "n_estimators", "min_child_samples"):
        assert params["lgb_params"][key] == DEFAULT_PARAMS[key]


def test_neural_trial_zero_reproduces_the_untuned_configuration():
    from tune import NEURAL_DEFAULT, suggest_neural

    params = suggest_neural(optuna.trial.FixedTrial(NEURAL_DEFAULT), "nhits")
    # configs/m5_competition.yaml: {model: nhits, max_steps: 2000, loss: negbin}
    assert params["max_steps"] == 2000 and params["loss"] == "negbin"
    import inspect

    pytest.importorskip("neuralforecast")
    from forecasting.models.neural_quantile import QuantileNeuralForecaster

    signature = inspect.signature(QuantileNeuralForecaster.__init__).parameters
    for key in ("learning_rate", "batch_size", "scaler_type"):
        assert params["hparams"][key] == signature[key].default
    assert params["hparams"]["input_size"] == 2 * 28  # the default, 2h


def test_validation_windows_step_back_a_month_and_end_before_the_test():
    from tune import m5_windows

    assert m5_windows(1) == [1913]
    assert m5_windows(3) == [1913, 1885, 1857]  # each scores the 28 days after it
    assert max(m5_windows(3)) + 28 < 1942  # the competition test starts at d_1942


def test_study_names_keep_the_single_window_study():
    from tune import study_name

    assert study_name("m5", "lightgbm", 1) == "m5_lightgbm"  # the committed results
    assert study_name("m5", "nhits", 3) == "m5_nhits_3win"


def test_nbeats_trial_zero_reproduces_the_untuned_configuration():
    import inspect

    from tune import NEURAL_DEFAULT, suggest_neural

    params = suggest_neural(optuna.trial.FixedTrial(NEURAL_DEFAULT), "nbeats")
    # configs/m5_competition.yaml: {variants: [nbeats, ...], max_steps: 2000, loss: mse}
    assert params["variants"] == ["nbeats"]
    assert params["max_steps"] == 2000 and params["loss"] == "mse"
    pytest.importorskip("neuralforecast")
    from forecasting.models.nbeats import NBEATSForecaster

    signature = inspect.signature(NBEATSForecaster.__init__).parameters
    for key in ("learning_rate", "batch_size", "scaler_type"):
        assert params["hparams"][key] == signature[key].default
