r"""A weighted ensemble with weights chosen on validation windows, never on test.

Each member is a model entry of the test config (tag + label). The script

1. forecasts every validation window with every member (the same M5 windows
   tune.py uses: train to d_1913 / d_1885 / d_1857, score the next 28 days),
   caching each forecast so a rerun resumes;
2. scores every weight vector on a simplex grid (step 0.05) by mean WRMSSE
   over the windows -- a member's weight may be 0, so choosing *which*
   members to combine is part of the validation choice too;
3. applies the best weights to the members' saved test forecasts and scores
   the result once, adding it to the competition leaderboard.

Outputs:
    results/tuning/m5_ensemble_3win[_<name>].csv   every weight vector, best first
    results/forecasts/m5_competition/<name>.parquet, a leaderboard row

Usage:
    python experiments/ensemble.py      # LightGBM x2 + N-BEATS: Ensemble-3w
    python experiments/ensemble.py --name Ensemble-3w-chronos \
        --members LightGBM-l2,LightGBM-tuned-3w,N-BEATS-tuned-3w,Chronos-2
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from _common import DATASET_CONFIG, load_panel, resolve_mode
from registry import NEEDS_EXOG, Context, build_entry
from run_benchmark import _write_leaderboard, load_config
from tune import M5_DAY_ONE, TUNING_DIR, m5_windows

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecasting.data.loaders import DATA_ROOT  # noqa: E402
from forecasting.evaluation.backtest import run_backtest  # noqa: E402
from forecasting.evaluation.m5 import (  # noqa: E402
    load_release,
    load_window,
    panel_to_wide,
    score_forecast,
)

ROOT = Path(__file__).resolve().parent.parent
FORECASTS = ROOT / "results" / "forecasts"
TABLES = ROOT / "results" / "tables"

# Model label -> config tag. Both LightGBMs are candidates: which one to use
# is decided by the validation windows like everything else.
CANDIDATES = {
    "LightGBM-l2": "gbm",
    "LightGBM-tuned-3w": "gbmtuned3w",
    "N-BEATS-tuned-3w": "nbeatstuned3w",
    "Chronos-2": "chronos",
    "XGBoost-l2": "xgboost",
    "CatBoost-l2": "catboost",
    "LightGBM-store": "gbmstore",
    "LightGBM-direct": "gbmdirect",
    "LightGBM-store-direct": "gbmstoredirect",
}
DEFAULT_MEMBERS = ["LightGBM-l2", "LightGBM-tuned-3w", "N-BEATS-tuned-3w"]
STEP = 0.05


def weight_grid(n: int, step: float = STEP) -> list[tuple[float, ...]]:
    """Every weight vector of n non-negative multiples of `step` summing to 1."""
    k = round(1 / step)
    grid = []
    # Stars and bars: n-1 dividers among k+n-1 slots mark out each split of k.
    for bars in itertools.combinations(range(k + n - 1), n - 1):
        edges = (-1, *bars, k + n - 1)
        grid.append(tuple((b - a - 1) / k for a, b in zip(edges, edges[1:], strict=False)))
    return grid


def to_matrix(forecast: pd.DataFrame, keys: pd.MultiIndex, days: pd.Index) -> np.ndarray:
    """Long forecast -> (series x day) array in the evaluator's row order;
    series a model omits are forecast as zero, as score_forecast does."""
    wide = panel_to_wide(forecast, "y_pred")
    wide.columns = days
    return wide.reindex(keys).fillna(0.0).to_numpy()


class BlendScorer:
    """WRMSSE of weighted blends of fixed member forecasts, fast.

    With weights summing to 1, a blend's error is the weighted sum of the members'
    errors, so its mean squared error on a series is w'Gw, where G holds the mean
    product of each pair of members' errors. G is computed once per level (after
    summing up the hierarchy), and scoring a weight vector never touches the
    42,840 series x 28 days again. Matches WRMSSEEvaluator.score to float32
    precision (checked on the first blend).
    """

    BATCH = 1000  # weight vectors at a time: bounds memory at (series x BATCH)

    def __init__(self, evaluator, actuals: pd.DataFrame, mats: list[np.ndarray]):
        y = evaluator._align(actuals)
        self.pairs = np.triu_indices(len(mats))
        self.levels = []
        for level, S in evaluator.matrices.items():
            agg_y = np.asarray(S @ y, dtype=np.float64)
            errors = [agg_y - np.asarray(S @ m, dtype=np.float64) for m in mats]
            gram = np.column_stack(
                [(errors[a] * errors[b]).mean(axis=1) for a, b in zip(*self.pairs, strict=True)]
            )
            self.levels.append((gram, evaluator.scales[level], evaluator.weights[level]))

    def score_many(self, weights: np.ndarray) -> np.ndarray:
        """WRMSSE of each row of a (vectors x members) array of weights summing to 1."""
        weights = np.asarray(weights, dtype=np.float64)
        a, b = self.pairs
        # An off-diagonal pair appears twice in w'Gw.
        terms = weights[:, a] * weights[:, b] * np.where(a == b, 1.0, 2.0)
        out = np.zeros(len(weights))
        for start in range(0, len(weights), self.BATCH):
            batch = terms[start : start + self.BATCH]
            for gram, scale, level_weights in self.levels:
                mse = np.clip(gram @ batch.T, 0, None)
                rmsse = np.sqrt(mse / np.asarray(scale)[:, None])
                out[start : start + self.BATCH] += np.nansum(
                    np.asarray(level_weights)[:, None] * np.nan_to_num(rmsse), axis=0
                )
        return out / len(self.levels)

    def score(self, weights) -> float:
        return float(self.score_many(np.array([weights]))[0])


def validation_forecasts(
    config: dict, windows: list[int], members: list[str]
) -> dict[str, list[pd.DataFrame]]:
    cfg = DATASET_CONFIG["m5"]
    _, horizon = resolve_mode(cfg, "validation")
    cache = FORECASTS / "m5_validation"
    cache.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[pd.DataFrame]] = {}
    todo = [
        (CANDIDATES[label], label, end)
        for label in members
        for end in windows
        if not (cache / f"{label}_d{end}.parquet").exists()
    ]
    if todo:
        entries = {e["tag"]: e for e in config["models"]}
        # Covariates roughly double the full-scale panel's memory; load them
        # only if a member still to forecast uses them (Chronos-2 does not).
        with_exog = any(entries[tag]["kind"] in NEEDS_EXOG for tag, _, _ in todo)
        panel, roles = load_panel("m5", max_series_per_state=None, with_exog=with_exog)
        ctx = Context(dataset="m5", cfg=cfg, horizon=horizon, roles=roles)
        for tag, label, end in todo:
            entry = entries[tag]
            factory = build_entry(entry["kind"], entry.get("params", {}), ctx)[label]
            cutoff = M5_DAY_ONE + pd.Timedelta(days=end - 1)
            print(f"  {label}: train to d_{end}...", flush=True)
            _, forecast = run_backtest(
                model_factory=factory,
                panel=panel,
                h=horizon,
                season_length=cfg["season_length"],
                min_train_periods=cfg["min_train_periods"],
                mode="cutoff",
                cutoff=cutoff,
                return_forecasts=True,
            )
            forecast.to_parquet(cache / f"{label}_d{end}.parquet", index=False)
    for label in members:
        out[label] = [pd.read_parquet(cache / f"{label}_d{end}.parquet") for end in windows]
    return out


def main(config_path: Path, windows: int, members: list[str], name: str) -> None:
    unknown = set(members) - set(CANDIDATES)
    if unknown:
        raise ValueError(f"unknown members {sorted(unknown)}; have {list(CANDIDATES)}")
    config = load_config(config_path)
    ends = m5_windows(windows)
    forecasts = validation_forecasts(config, ends, members)
    labels = members

    sales = pd.read_csv(DATA_ROOT / "m5" / "sales_train_evaluation.csv")
    scorers = []
    for k, end in enumerate(ends):
        evaluator, actuals = load_window(DATA_ROOT / "m5", end, sales=sales)
        keys = pd.MultiIndex.from_frame(evaluator.keys[["item_id", "store_id"]])
        mats = [to_matrix(forecasts[label][k], keys, actuals.columns) for label in labels]
        scorer = BlendScorer(evaluator, actuals, mats)
        if k == 0:
            equal = [1 / len(mats)] * len(mats)
            frame = pd.DataFrame(
                sum(w * m for w, m in zip(equal, mats, strict=True)),
                index=keys,
                columns=actuals.columns,
            )
            reference = float(evaluator.score(actuals, frame)["WRMSSE"])
            # Forecasts are float32, so the reference blends in float32.
            if not np.isclose(scorer.score(equal), reference, rtol=1e-6):
                raise AssertionError(f"fast scorer {scorer.score(equal)} != {reference}")
        scorers.append(scorer)
    del sales

    grid = np.array(weight_grid(len(labels)))
    print(f"scoring {len(grid)} weight vectors on {len(ends)} windows...", flush=True)
    window_scores = np.column_stack([scorer.score_many(grid) for scorer in scorers])
    table = pd.DataFrame(grid, columns=labels)
    table["score"] = window_scores.mean(axis=1)
    table["window_scores"] = [[round(float(s), 5) for s in row] for row in window_scores]
    table = table.sort_values("score").reset_index(drop=True)
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    suffix = name.removeprefix(f"Ensemble-{windows}w").lower().replace("-", "_")
    # The best blends and each member alone: the full grid can pass 100MB.
    solo = table[table[labels].max(axis=1) == 1.0]
    kept = pd.concat([table.head(1000), solo]).drop_duplicates(subset=labels)
    kept.to_csv(TUNING_DIR / f"m5_ensemble_{windows}win{suffix}.csv", index=False)

    best = table.iloc[0]
    print("\nvalidation (mean WRMSSE over windows):")
    for label in labels:
        solo = table[table[label] == 1.0].iloc[0]
        print(f"  {label:<20} {solo['score']:.4f} {solo['window_scores']}")
    weights = {label: float(best[label]) for label in labels}
    print(f"  {'ensemble':<20} {best['score']:.4f} {best['window_scores']}  weights {weights}")

    # The test, once: the members' saved competition forecasts, blended.
    members = {
        label: pd.read_parquet(FORECASTS / "m5_competition" / f"{label}.parquet")
        for label, w in weights.items()
        if w > 0
    }
    base = next(iter(members.values()))[["unique_id", "ds"]]
    blend = base.copy()
    blend["y_pred"] = 0.0
    for label, forecast in members.items():
        aligned = base.merge(forecast[["unique_id", "ds", "y_pred"]], on=["unique_id", "ds"])
        if len(aligned) != len(base):
            raise ValueError(f"{label} covers different series/days than the other members")
        blend["y_pred"] += weights[label] * aligned["y_pred"].to_numpy()
    blend["model"], blend["window"] = name, 0
    blend.to_parquet(FORECASTS / "m5_competition" / f"{name}.parquet", index=False)

    evaluator, actuals = load_release(DATA_ROOT / "m5", phase="evaluation")
    scores, zero_filled = score_forecast(evaluator, actuals, blend)
    row = {"model": name, **scores, "zero_filled_series": zero_filled}
    _write_leaderboard(pd.DataFrame([row]), TABLES / "wrmsse_m5_competition.csv", "WRMSSE")
    print(f"\ntest WRMSSE {scores['WRMSSE']:.4f} ({name}, weights {weights})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "m5_competition.yaml")
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--members", default=",".join(DEFAULT_MEMBERS))
    parser.add_argument("--name", default=None, help="defaults to Ensemble-<windows>w")
    args = parser.parse_args()
    main(
        args.config,
        args.windows,
        args.members.split(","),
        args.name or f"Ensemble-{args.windows}w",
    )
