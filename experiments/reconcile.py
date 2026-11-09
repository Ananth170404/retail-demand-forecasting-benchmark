"""Scale an ensemble's item forecasts toward separately forecast store x department totals.

Totals are smoother than items and easier to forecast, and M5's WRMSSE rewards
getting them right at nine of its twelve levels. The second-placed M5 team used
this idea. The script

1. forecasts the 70 store x department series directly, with LightGBM and AutoETS;
2. on each validation window, scales the ensemble's item forecasts toward each
   choice of totals (LightGBM, AutoETS, their mean) by a strength alpha, where
   alpha 0 leaves the ensemble unchanged and alpha 1 matches the totals exactly;
3. applies the choice with the best mean WRMSSE to the test forecast, once.

Outputs:
    results/tuning/m5_reconcile_3win.csv   every (totals, alpha), best first
    results/forecasts/m5_competition/<name>.parquet, a leaderboard row

Usage:
    python experiments/reconcile.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from _common import DATASET_CONFIG, load_panel
from ensemble import FORECASTS, TABLES
from registry import Context, build_entry
from run_benchmark import _write_leaderboard
from tune import M5_DAY_ONE, TUNING_DIR, m5_windows

from forecasting.data.loaders import DATA_ROOT, ColumnRoles
from forecasting.evaluation.backtest import run_backtest
from forecasting.evaluation.m5 import load_release, load_window, score_forecast

TEST_END = 1941
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
STORE, DEPT = "store_id_cat", "dept_id_cat"
EVENTS = ["event_type_1_cat", "snap_cat"]


def scale_to_totals(
    items: pd.DataFrame, totals: pd.DataFrame, group_of: pd.Series, alpha: float
) -> pd.DataFrame:
    """Move item forecasts toward group totals, each day, keeping their shares.

    `group_of` maps unique_id to group; `totals` holds [unique_id (the group),
    ds, y_pred]. A group forecast at zero on a day has no shares to keep, so it
    is left as it is.
    """
    out = items.copy()
    out["group"] = out["unique_id"].map(group_of)
    current = out.groupby(["group", "ds"])["y_pred"].transform("sum")
    target = out[["group", "ds"]].merge(
        totals.rename(columns={"unique_id": "group", "y_pred": "total"})[["group", "ds", "total"]],
        on=["group", "ds"],
        how="left",
    )["total"]
    if target.isna().any():
        raise ValueError("totals do not cover every item forecast")
    factor = np.where(current > 0, target.to_numpy() / current.to_numpy().clip(1e-12), 1.0)
    out["y_pred"] = out["y_pred"].to_numpy() * (1 - alpha + alpha * factor)
    return out.drop(columns="group")


def store_dept_panel() -> tuple[pd.DataFrame, pd.Series]:
    """Daily store x department sales with event flags, and each item's group."""
    panel, _ = load_panel("m5", max_series_per_state=None, with_exog=True, include_test=True)
    panel = panel[["unique_id", "ds", "y", STORE, DEPT, *EVENTS]].copy()
    panel["group"] = panel[STORE].astype(str) + "_" + panel[DEPT].astype(str)
    group_of = panel.drop_duplicates("unique_id").set_index("unique_id")["group"]
    totals = panel.groupby(["group", "ds"], sort=True, as_index=False).agg(
        y=("y", "sum"), **{c: (c, "first") for c in [STORE, DEPT, *EVENTS]}
    )
    return totals.rename(columns={"group": "unique_id"}), group_of


def forecast_totals(panel: pd.DataFrame, end: int) -> pd.DataFrame:
    """[unique_id, ds, LightGBM, AutoETS, mean] store x department forecasts."""
    cfg = DATASET_CONFIG["m5"]
    roles = ColumnRoles(static_cat_cols=[STORE, DEPT], dynamic_cat_cols=EVENTS)
    ctx = Context(dataset="m5", cfg=cfg, horizon=cfg["validation_h"], roles=roles)
    factories = {
        "LightGBM": build_entry(
            "gbm",
            {"objectives": ["l2"], "feature_step": "events", "model_name": "LightGBM"},
            ctx,
        )["LightGBM"],
        "AutoETS": build_entry("baselines", {"only": ["AutoETS"], "n_jobs": 1}, ctx)["AutoETS"],
    }
    out = None
    for label, factory in factories.items():
        print(f"  totals, {label}: train to d_{end}...", flush=True)
        _, forecast = run_backtest(
            model_factory=factory,
            panel=panel,
            h=ctx.horizon,
            season_length=cfg["season_length"],
            min_train_periods=cfg["min_train_periods"],
            mode="cutoff",
            cutoff=M5_DAY_ONE + pd.Timedelta(days=end - 1),
            return_forecasts=True,
        )
        forecast = forecast[["unique_id", "ds", "y_pred"]].rename(columns={"y_pred": label})
        out = forecast if out is None else out.merge(forecast, on=["unique_id", "ds"])
    out["mean"] = out[list(factories)].mean(axis=1)
    return out


def blend(weights: dict[str, float], end: int) -> pd.DataFrame:
    """The ensemble's item forecast on a validation window."""
    cache = FORECASTS / "m5_validation"
    out = None
    for label, weight in weights.items():
        part = pd.read_parquet(cache / f"{label}_d{end}.parquet")[["unique_id", "ds", "y_pred"]]
        part["y_pred"] = part["y_pred"].astype(np.float64) * weight
        if out is None:
            out = part
        else:
            merged = out.merge(part, on=["unique_id", "ds"], suffixes=("", "_add"))
            if len(merged) != len(out):
                raise ValueError(f"{label} covers different series/days than the other members")
            merged["y_pred"] += merged.pop("y_pred_add")
            out = merged
    return out


def main(base_name: str, weights_csv: Path, name: str, windows: int) -> None:
    table = pd.read_csv(weights_csv)
    meta = {"score", "window_scores"}
    best = table.iloc[0]
    weights = {c: float(best[c]) for c in table.columns if c not in meta and best[c] > 0}
    print(f"base {base_name}, weights {weights}", flush=True)

    panel, group_of = store_dept_panel()
    ends = m5_windows(windows)
    totals = {end: forecast_totals(panel, end) for end in [*ends, TEST_END]}

    sales = pd.read_csv(DATA_ROOT / "m5" / "sales_train_evaluation.csv")
    rows = []
    bases = {end: blend(weights, end) for end in ends}
    scorers = {end: load_window(DATA_ROOT / "m5", end, sales=sales) for end in ends}
    del sales
    choices = [("none", 0.0)] + [
        (top, a) for top in ("LightGBM", "AutoETS", "mean") for a in ALPHAS if a > 0
    ]
    for top, alpha in choices:
        scores = []
        for end in ends:
            target = totals[end][["unique_id", "ds"]].assign(
                y_pred=totals[end]["mean" if top == "none" else top]
            )
            forecast = scale_to_totals(bases[end], target, group_of, alpha)
            evaluator, actuals = scorers[end]
            scores.append(float(score_forecast(evaluator, actuals, forecast)[0]["WRMSSE"]))
        rows.append(
            {
                "totals": top,
                "alpha": alpha,
                "score": float(np.mean(scores)),
                "window_scores": [round(s, 5) for s in scores],
            }
        )
        print(f"  {top:<9} alpha {alpha:.2f}  {rows[-1]['score']:.4f} {scores}", flush=True)
    results = pd.DataFrame(rows).sort_values("score").reset_index(drop=True)
    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(TUNING_DIR / f"m5_reconcile_{windows}win.csv", index=False)

    # The test, once.
    chosen = results.iloc[0]
    print(f"\nvalidation chose totals={chosen['totals']}, alpha={chosen['alpha']}", flush=True)
    base = pd.read_parquet(FORECASTS / "m5_competition" / f"{base_name}.parquet")
    column = "mean" if chosen["totals"] == "none" else chosen["totals"]
    target = totals[TEST_END][["unique_id", "ds"]].assign(y_pred=totals[TEST_END][column])
    forecast = scale_to_totals(
        base[["unique_id", "ds", "y_pred"]], target, group_of, float(chosen["alpha"])
    )
    forecast["model"], forecast["window"] = name, 0
    forecast.to_parquet(FORECASTS / "m5_competition" / f"{name}.parquet", index=False)

    evaluator, actuals = load_release(DATA_ROOT / "m5", phase="evaluation")
    scores, zero_filled = score_forecast(evaluator, actuals, forecast)
    row = {"model": name, **scores, "zero_filled_series": zero_filled}
    _write_leaderboard(pd.DataFrame([row]), TABLES / "wrmsse_m5_competition.csv", "WRMSSE")
    print(f"\ntest WRMSSE {scores['WRMSSE']:.4f} ({name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Ensemble-3w-winner")
    parser.add_argument("--weights", type=Path, default=TUNING_DIR / "m5_ensemble_3win_winner.csv")
    parser.add_argument("--name", default="Ensemble-3w-reconciled")
    parser.add_argument("--windows", type=int, default=3)
    args = parser.parse_args()
    main(args.base, args.weights, args.name, args.windows)
