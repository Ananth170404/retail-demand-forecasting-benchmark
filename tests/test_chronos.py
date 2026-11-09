import numpy as np
import pytest

pytest.importorskip("torch")

from forecasting.models.chronos import integrated_mean  # noqa: E402


def test_integrated_mean_recovers_a_known_mean():
    # Quantiles of Uniform(0, 10): Q(u) = 10u, mean 5.
    levels = [0.01] + [round(0.05 * i, 2) for i in range(1, 20)] + [0.99]
    q = np.array(levels) * 10
    assert integrated_mean(q[None, None, :], levels)[0, 0] == pytest.approx(5.0, abs=0.01)


def test_mean_exceeds_median_on_intermittent_demand():
    # 70% zeros, otherwise 1-3 units: the median is 0, the mean is not.
    levels = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
    q = np.array([[[0, 0, 0, 0.5, 1, 3]]], dtype=float)
    assert integrated_mean(q, levels)[0, 0] > 0.5 > q[0, 0, 2]


def test_registry_labels_the_point_forecast():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
    from registry import Context, build_entry

    from forecasting.data.loaders import ColumnRoles

    ctx = Context("m5", {"freq": "D"}, 28, ColumnRoles())
    assert list(build_entry("chronos", {}, ctx)) == ["Chronos-2"]
    assert list(build_entry("chronos", {"point": "median"}, ctx)) == ["Chronos-2-median"]
