import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from ensemble import BlendScorer, weight_grid  # noqa: E402


def test_grid_covers_the_simplex_including_single_members():
    grid = weight_grid(3, step=0.05)
    assert len(grid) == 231  # C(20 + 2, 2)
    assert all(sum(w) == pytest.approx(1) and min(w) >= 0 for w in grid)
    for solo in [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]:
        assert solo in grid  # each member alone is a candidate
    assert (0.5, 0.5, 0.0) in grid


def test_batch_scores_match_scoring_each_blend_directly():
    rng = np.random.default_rng(0)
    y = rng.poisson(3.0, (4, 28)).astype(float)
    mats = [y + rng.normal(0, s, y.shape) for s in (0.5, 1.0, 2.0)]
    scales = {"L1": np.full(1, 2.0), "L2": np.array([1.0, 2.0, 0.5, 4.0])}
    weights = {"L1": np.ones(1), "L2": np.array([0.1, 0.2, 0.3, 0.4])}
    matrices = {"L1": np.ones((1, 4)), "L2": np.eye(4)}

    class Evaluator:
        pass

    ev = Evaluator()
    ev.matrices, ev.scales, ev.weights = matrices, scales, weights
    ev._align = lambda actuals: actuals
    scorer = BlendScorer(ev, y, mats)

    def direct(w):
        blend = sum(a * m for a, m in zip(w, mats, strict=True))
        per_level = []
        for k, S in matrices.items():
            mse = np.square(S @ (y - blend)).mean(axis=1)
            per_level.append((weights[k] * np.sqrt(mse / scales[k])).sum())
        return np.mean(per_level)

    grid = np.array(weight_grid(3, step=0.1))
    assert scorer.score_many(grid) == pytest.approx([direct(w) for w in grid])
    assert scorer.score(grid[5]) == pytest.approx(direct(grid[5]))
