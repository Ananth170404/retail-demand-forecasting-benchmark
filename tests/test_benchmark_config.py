import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_benchmark import load_config  # noqa: E402

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def _write(tmp_path, config: dict) -> Path:
    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


@pytest.fixture
def minimal():
    return {"dataset": "m5", "models": [{"tag": "baselines", "kind": "baselines"}]}


@pytest.mark.parametrize("name", ["m5.yaml", "walmart.yaml", "smoke.yaml"])
def test_committed_configs_are_valid(name):
    config = load_config(CONFIGS / name)
    assert config["models"]


def test_defaults_are_filled(tmp_path, minimal):
    config = load_config(_write(tmp_path, minimal))
    assert config["name"] == "bench"
    assert config["modes"] == ["rolling"]
    assert config["models"][0]["params"] == {}


@pytest.mark.parametrize(
    "patch, message",
    [
        ({"modles": []}, "unknown top-level keys"),
        ({"dataset": "m6"}, "dataset must be one of"),
        ({"modes": ["holdout"]}, "unknown modes"),
        ({"data": {"max_series": 10}}, "unknown data keys"),
        ({"backtest": {"horizon": 7}}, "unknown backtest keys"),
        ({"models": [{"tag": "a", "kind": "gbm"}, {"tag": "a", "kind": "gbm"}]}, "duplicate"),
        ({"models": [{"kind": "gbm"}]}, "needs a 'tag'"),
        ({"models": [{"tag": "a"}]}, "has no 'kind'"),
    ],
)
def test_typos_fail_loudly(tmp_path, minimal, patch, message):
    """A misspelt key must not silently fall back to a default."""
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, {**minimal, **patch}))


def test_competition_mode_must_run_alone(tmp_path, minimal):
    config = {
        **minimal,
        "modes": ["competition", "rolling"],
        "data": {"max_series_per_state": None},
    }
    with pytest.raises(ValueError, match="only mode"):
        load_config(_write(tmp_path, config))


def test_competition_mode_needs_every_series(tmp_path, minimal):
    with pytest.raises(ValueError, match="max_series_per_state: null"):
        load_config(_write(tmp_path, {**minimal, "modes": ["competition"]}))


def test_m5_validation_mode_needs_every_series(tmp_path, minimal):
    with pytest.raises(ValueError, match="max_series_per_state: null"):
        load_config(_write(tmp_path, {**minimal, "modes": ["validation"]}))


def test_validation_windows_precede_their_test_windows():
    from _common import DATASET_CONFIG, mode_cutoff, resolve_mode

    m5, wm = DATASET_CONFIG["m5"], DATASET_CONFIG["walmart"]
    assert resolve_mode(m5, "validation") == ("holdout", 28)
    assert resolve_mode(wm, "validation") == ("cutoff", 13)
    # Walmart: validation ends (cutoff + 13 weeks) on or before the season cutoff.
    import pandas as pd

    end = pd.Timestamp(mode_cutoff(wm, "validation")) + pd.Timedelta(weeks=wm["validation_h"])
    assert end <= pd.Timestamp(mode_cutoff(wm, "season"))
