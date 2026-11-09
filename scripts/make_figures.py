"""Draw the README figures into results/figures/ from the committed result tables.

The total-sales figure also needs the raw M5 data and the saved test forecasts
(both git-ignored); it is skipped, with a message, when they are missing.

Usage:
    python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from forecasting.viz import plots  # noqa: E402

TABLES = ROOT / "results" / "tables"
TUNING = ROOT / "results" / "tuning"
FORECASTS = ROOT / "results" / "forecasts" / "m5_competition"
FIGURES = ROOT / "results" / "figures"
M5_RAW = ROOT / "data" / "raw" / "m5"

M5_NAMES = {
    "Ensemble-3w-winner": "Ensemble (headline)",
    "Ensemble-3w-reconciled": "Ensemble, scaled to totals*",
    "LightGBM-store": "LightGBM, one per store",
    "Ensemble-3w": "Ensemble, LightGBM + N-BEATS",
    "LightGBM-l2-L5-events": "LightGBM",
    "CatBoost-l2": "CatBoost",
    "N-BEATS-tuned-3w": "N-BEATS, tuned",
    "Chronos-2": "Chronos-2, zero-shot",
    "LightGBM-tuned-3w": "LightGBM, tuned",
    "AutoETS": "AutoETS",
    "AutoTheta": "AutoTheta",
    "XGBoost-l2": "XGBoost",
    "N-BEATS-mse": "N-BEATS, untuned",
    "SeasonalNaive": "Seasonal naive",
}
LEVEL_MODELS = {
    "Ensemble-3w-winner": "Ensemble (headline)",
    "LightGBM-store": "LightGBM, one per store",
    "N-BEATS-tuned-3w": "N-BEATS, tuned",
    "SeasonalNaive": "Seasonal naive",
}
WALMART_NAMES = {
    "LightGBM-tuned": "LightGBM, tuned",
    "LightGBM-l2-sdiff52-L1-lags": "LightGBM, lags only",
    "SNaive-xmas": "Seasonal naive, Christmas-aligned",
    "SeasonalNaive": "Seasonal naive",
    "LightGBM-tweedie": "LightGBM, no differencing",
    "N-BEATS": "N-BEATS",
    "TFT": "TFT",
    "DeepAR": "DeepAR",
    "SES": "SES",
    "AutoETS": "AutoETS",
}
LADDER_STEPS = {
    "L1-lags": "lags",
    "L2-rolling": "+ rolling",
    "L3-calendar": "+ calendar",
    "L4-static": "+ store/item",
    "L5-events": "+ events, SNAP",
    "L6-covariates": "+ price / macro",
    "L7-holiday": "+ holidays",
}


def ladder(board: pd.DataFrame, prefix: str, column: str) -> pd.Series:
    out = {}
    for step, label in LADDER_STEPS.items():
        row = board[board["model"] == f"{prefix}-{step}"]
        if len(row):
            out[label] = float(row[column].iloc[0])
    return pd.Series(out)


def validation_points(board: pd.DataFrame) -> pd.DataFrame:
    """Each blend member's validation score alone, and each ensemble's, against its test."""
    rows = []
    for csv in ("m5_ensemble_3win_winner.csv", "m5_ensemble_3win_chronos.csv"):
        table = pd.read_csv(TUNING / csv)
        members = [c for c in table.columns if c not in ("score", "window_scores")]
        for member in members:
            solo = table[table[member] == 1.0]
            if len(solo):
                rows.append((member, float(solo["score"].iloc[0]), "Single model"))
    for name, csv in (
        ("Ensemble-3w", "m5_ensemble_3win.csv"),
        ("Ensemble-3w-trees", "m5_ensemble_3win_trees.csv"),
        ("Ensemble-3w-winner", "m5_ensemble_3win_winner.csv"),
    ):
        rows.append((name, float(pd.read_csv(TUNING / csv)["score"].iloc[0]), "Ensemble"))
    reconciled = pd.read_csv(TUNING / "m5_reconcile_3win.csv").iloc[0]
    rows.append(("Ensemble-3w-reconciled", float(reconciled["score"]), "Ensemble"))
    points = pd.DataFrame(rows, columns=["model", "validation", "kind"]).drop_duplicates("model")
    points = points.merge(board[["model", "WRMSSE"]], on="model").rename(columns={"WRMSSE": "test"})
    points["label"] = (
        points["model"]
        .map(
            {
                **M5_NAMES,
                # Chronos-2 lands on the same point (0.7225 / 0.642): mark both.
                "N-BEATS-tuned-3w": "N-BEATS, tuned and Chronos-2",
                "LightGBM-store-direct": "LightGBM, per store, direct",
                "LightGBM-direct": "LightGBM, direct",
            }
        )
        .fillna(points["model"])
    )
    return points


def total_sales() -> tuple | None:
    needed = [
        M5_RAW / "sales_train_evaluation.csv",
        M5_RAW / "sales_test_evaluation.csv",
        M5_RAW / "calendar.csv",
        FORECASTS / "Ensemble-3w-winner.parquet",
        FORECASTS / "SeasonalNaive.parquet",
    ]
    missing = [p for p in needed if not p.exists()]
    if missing:
        print(f"skipping m5_total_sales.png: missing {missing[0]}")
        return None
    dates = pd.read_csv(M5_RAW / "calendar.csv").set_index("d")["date"]
    days = [f"d_{i}" for i in range(1886, 1942)]
    history = pd.read_csv(M5_RAW / "sales_train_evaluation.csv", usecols=days).sum()
    test = pd.read_csv(M5_RAW / "sales_test_evaluation.csv")
    actual = test.filter(like="d_").sum()
    history.index = pd.to_datetime(dates.loc[history.index])
    actual.index = pd.to_datetime(dates.loc[actual.index])
    forecasts = {}
    for name, label in (("Ensemble-3w-winner", "Ensemble"), ("SeasonalNaive", "Seasonal naive")):
        f = pd.read_parquet(FORECASTS / f"{name}.parquet")
        forecasts[label] = f.groupby("ds")["y_pred"].sum()
    # Join the last training day to the test so the actual line is continuous.
    actual = pd.concat([history.iloc[-1:], actual])
    board = pd.read_csv(TABLES / "wrmsse_m5_competition.csv").set_index("model")["L1"]
    scores = {"Ensemble": board["Ensemble-3w-winner"], "Seasonal naive": board["SeasonalNaive"]}
    return history, actual, forecasts, scores


def summary_rows(dataset: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in TABLES.glob(f"*_{dataset}_summary.csv")]
    return pd.concat(frames).drop_duplicates("model").set_index("model")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    m5 = pd.read_csv(TABLES / "wrmsse_m5_competition.csv")
    walmart = pd.read_csv(TABLES / "wmae_walmart_season.csv")
    m5_summary = summary_rows("m5_competition")

    volume = pd.DataFrame(
        [
            ("NHITS, median (quantile loss)", "NHITS", "Median forecast"),
            ("NHITS, mean (negative binomial)", "NHITS-negbin", "Mean forecast"),
            ("Chronos-2, its default (median)", "Chronos-2-median", "Median forecast"),
            ("Chronos-2, mean of quantiles", "Chronos-2", "Mean forecast"),
        ],
        columns=["label", "model", "kind"],
    )
    volume["Volume_ratio"] = volume["model"].map(m5_summary["Volume_ratio"])

    figures = {
        "m5_leaderboard.png": plots.m5_leaderboard(m5, M5_NAMES),
        "m5_levels.png": plots.m5_levels(m5, LEVEL_MODELS),
        "feature_ladder.png": plots.feature_ladder(
            ladder(m5, "LightGBM-l2", "WRMSSE"),
            ladder(walmart, "LightGBM-l2-sdiff52", "WMAE_vs_SeasonalNaive"),
        ),
        "walmart_season.png": plots.walmart_season(walmart, WALMART_NAMES),
        "m5_validation_vs_test.png": plots.validation_vs_test(
            validation_points(m5),
            {
                "Ensemble-3w-winner": (8, -4),
                "Ensemble-3w-reconciled": (0, 14),
                "LightGBM-store": (8, -4),
                "LightGBM-tuned-3w": (-8, 4),
                "N-BEATS-tuned-3w": (8, 0),
                "XGBoost-l2": (-8, 0),
                "LightGBM-direct": (8, 0),
                "LightGBM-store-direct": (-6, -12),
            },
        ),
        "volume_ratio.png": plots.volume_ratio(volume),
    }
    sales = total_sales()
    if sales is not None:
        figures["m5_total_sales.png"] = plots.m5_total_sales(*sales)
    for name, fig in figures.items():
        fig.savefig(FIGURES / name, dpi=150, facecolor=plots.SURFACE)
        print(f"wrote results/figures/{name}")


if __name__ == "__main__":
    main()
