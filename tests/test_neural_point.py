"""Neural point forecasts: median by default, mean where a distribution exists."""

import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("neuralforecast")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from registry import Context, build_entry  # noqa: E402

from forecasting.data.loaders import ColumnRoles  # noqa: E402
from forecasting.models.nbeats import NBEATSForecaster  # noqa: E402
from forecasting.models.nhits import NHITSForecaster  # noqa: E402

CTX = Context(
    dataset="m5",
    cfg={"freq": "D", "season_length": 7},
    horizon=28,
    roles=ColumnRoles(static_cat_cols=[], dynamic_cat_cols=[], dynamic_real_cols=[]),
)


def _labels(kind, params):
    return list(build_entry(kind, params, CTX))


def test_labels_name_the_point_forecast():
    # Existing labels stay byte-identical; mean-forecast variants are suffixed.
    assert _labels("neural", {"model": "nhits"}) == ["NHITS"]
    assert _labels("neural", {"model": "nhits", "loss": "negbin"}) == ["NHITS-negbin"]
    assert _labels("neural", {"model": "deepar", "point": "mean"}) == ["DeepAR-mean"]
    assert _labels("nbeats", {"variants": ["nbeats", "nbeatsx"]}) == ["N-BEATS", "N-BEATSx"]
    assert _labels("nbeats", {"variants": ["nbeats", "nbeatsx"], "loss": "mse"}) == [
        "N-BEATS-mse",
        "N-BEATSx-mse",
    ]


def test_point_defaults_follow_the_loss():
    assert NHITSForecaster(freq="D", h=7)._point == "median"
    assert NHITSForecaster(freq="D", h=7, loss_type="negbin")._point == "mean"


def test_bad_point_and_loss_fail_loudly():
    with pytest.raises(ValueError, match="point"):
        NHITSForecaster(freq="D", h=7, point="mode")
    with pytest.raises(ValueError, match="loss"):
        NBEATSForecaster(freq="D", h=7, loss="huber")


def test_mean_point_reads_the_bare_alias_column():
    model = NHITSForecaster(freq="D", h=2, loss_type="negbin", name="NHITS")
    raw = pd.DataFrame(
        {
            "unique_id": ["a", "a"],
            "ds": pd.date_range("2020-01-01", periods=2),
            "NHITS": [0.9, -0.2],
            "NHITS-median": [0.0, 0.0],
            "NHITS-lo-80": [0.0, 0.0],
            "NHITS-hi-80": [3.0, 3.0],
            "NHITS-lo-90": [0.0, 0.0],
            "NHITS-hi-90": [4.0, 4.0],
        }
    )

    class _NF:
        models = [type("M", (), {"alias": "NHITS"})()]

        def predict(self, **_):
            return raw

    model._nf = _NF()
    out = model.predict(2)
    assert out["y_pred"].tolist() == [0.9, 0.0]  # mean, clipped at zero
    assert out["y_pred_q50"].tolist() == [0.0, 0.0]


def test_negative_binomial_probs_stay_inside_support():
    import torch

    from forecasting.models.neural_quantile import distribution_loss

    loss = distribution_loss("NegativeBinomial", [80, 90])
    # A huge mean on a high-volume series pushes the unclamped probs to >= 1.
    output = (torch.full((2, 3, 1), 50.0), torch.full((2, 3, 1), 50.0))
    scale = torch.full((2, 3, 1), 1e4)
    _, probs = loss.scale_decouple(output, loc=torch.zeros_like(scale), scale=scale)
    assert (probs < 1).all()
    torch.distributions.NegativeBinomial(total_count=1.0, probs=probs)  # validates


def test_tuned_nbeats_takes_hyperparameters_and_a_name():
    [(label, factory)] = build_entry(
        "nbeats",
        {
            "variants": ["nbeats"],
            "loss": "mse",
            "hparams": {"input_size": 112, "batch_size": 128},
            "model_name": "N-BEATS-tuned",
        },
        CTX,
    ).items()
    model = factory()
    assert label == model.name == "N-BEATS-tuned"
    assert (model._input_size, model._batch_size, model._loss) == (112, 128, "mse")
    with pytest.raises(ValueError, match="single variant"):
        build_entry("nbeats", {"model_name": "x"}, CTX)
