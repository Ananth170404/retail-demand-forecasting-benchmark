"""Score saved M5 test forecasts with the official WRMSSE, without retraining.

Each file is a long parquet of [unique_id, ds, y_pred] for d_1942-d_1969, as in
results/forecasts/m5_competition/ (published as test_forecasts.zip on the
releases page). Needs the raw data and test labels:
    python scripts/download_data.py --dataset m5
    python scripts/download_data.py --dataset m5-labels

Usage:
    python scripts/score_m5_forecast.py results/forecasts/m5_competition/Ensemble-3w-winner.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecasting.data.loaders import DATA_ROOT  # noqa: E402
from forecasting.evaluation.m5 import load_release, score_forecast  # noqa: E402


def main(paths: list[Path]) -> None:
    evaluator, actuals = load_release(DATA_ROOT / "m5", phase="evaluation")
    for path in paths:
        forecast = pd.read_parquet(path, columns=["unique_id", "ds", "y_pred"])
        scores, filled = score_forecast(evaluator, actuals, forecast)
        levels = "  ".join(f"L{i} {scores[i]:.3f}" for i in (1, 9, 12))
        print(f"{path.stem:<28} WRMSSE {scores['WRMSSE']:.4f}   {levels}   zero-filled {filled}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("forecasts", type=Path, nargs="+")
    main(parser.parse_args().forecasts)
