"""Forecast accuracy metrics, series diagnostics, and robust aggregation.

Three groups live here:

1. Accuracy metrics — scale-free (MASE, RMSSE), absolute (MAE, RMSE), relative
   (WAPE, sMAPE), directional (bias), and robust (MdAE) — plus pinball loss for
   models that emit quantiles.
2. Series diagnostics — ADI, CV², zero rate and the Syntetos-Boylan demand
   class. Attached to every result row so later analysis is a groupby rather
   than a re-computation.
3. Robust aggregation — mean alongside median and trimmed mean, because a few
   near-flat series can produce enormous scale-free ratios and drag a plain
   mean around.

MAPE is deliberately absent as a headline: both datasets contain zero demand
(68% of M5 rows), where MAPE is undefined and any epsilon guard makes the number
an artifact of the guard. WAPE and sMAPE cover the same intuition safely.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Syntetos-Boylan cutoffs for classifying demand patterns.
ADI_CUTOFF = 1.32
CV2_CUTOFF = 0.49


# --------------------------------------------------------------------------
# scale helpers
# --------------------------------------------------------------------------


def trim_leading_zeros(y: np.ndarray) -> np.ndarray:
    """Drop the run of zeros before a series' first non-zero observation.

    Follows the official M5 rule: the error scale is computed only over periods
    where the product was actively sold. Leading zeros usually mean "not stocked
    yet" rather than "nobody bought it", and they shrink the MASE denominator,
    inflating the metric for every model.
    """
    nonzero = np.flatnonzero(y > 0)
    return y[nonzero[0] :] if len(nonzero) else y[:0]


def _seasonal_scale(y_train: np.ndarray, season_length: int, active_only: bool = True) -> float:
    """Mean absolute seasonal-naive error on the training series (MASE denominator).

    Computed on train, never on test — using test values here would leak.
    """
    if active_only:
        y_train = trim_leading_zeros(y_train)
    if len(y_train) <= season_length:
        return np.nan
    scale = np.abs(y_train[season_length:] - y_train[:-season_length]).mean()
    # A flat series has zero scale; MASE is undefined there rather than infinite.
    return scale if scale > 0 else np.nan


def _squared_seasonal_scale(
    y_train: np.ndarray, season_length: int, active_only: bool = True
) -> float:
    if active_only:
        y_train = trim_leading_zeros(y_train)
    if len(y_train) <= season_length:
        return np.nan
    scale = ((y_train[season_length:] - y_train[:-season_length]) ** 2).mean()
    return scale if scale > 0 else np.nan


# --------------------------------------------------------------------------
# accuracy metrics
# --------------------------------------------------------------------------


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mdae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median absolute error — unmoved by the odd catastrophic period."""
    return float(np.median(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean error, signed. Positive means the model under-forecasts.

    Direction matters for inventory in a way absolute error hides: two models
    with equal MAE can differ between chronic stockouts and chronic overstock.
    """
    return float(np.mean(y_true - y_pred))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error, sum|e| / sum|y|. Also called ND.

    The retail standard for intermittent demand: percentage-like but zero-safe,
    since only the total needs to be non-zero.
    """
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else np.nan


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE in [0, 2]. Defined at zero when both sides are zero."""
    denom = np.abs(y_true) + np.abs(y_pred)
    ratio = np.divide(
        2 * np.abs(y_true - y_pred), denom, out=np.zeros_like(denom, dtype=float), where=denom > 0
    )
    return float(np.mean(ratio))


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season_length: int,
    active_only: bool = True,
) -> float:
    scale = _seasonal_scale(y_train, season_length, active_only)
    return float(np.mean(np.abs(y_true - y_pred)) / scale) if scale == scale else np.nan


def rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season_length: int,
    active_only: bool = True,
) -> float:
    scale = _squared_seasonal_scale(y_train, season_length, active_only)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2) / scale)) if scale == scale else np.nan


def pinball_loss(y_true: np.ndarray, y_pred_q: np.ndarray, q: float) -> float:
    """Quantile (pinball) loss — scores probabilistic forecasts, not just the median."""
    delta = y_true - y_pred_q
    return float(np.mean(np.maximum(q * delta, (q - 1) * delta)))


def zero_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int]:
    """Actual vs. predicted zero counts — a sparsity sanity check."""
    return int(np.sum(y_true == 0)), int(np.sum(np.floor(np.clip(y_pred, 0, None)) == 0))


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actuals falling inside a prediction interval.

    A well-calibrated 80% interval should contain ~80% of outcomes. Lower means
    overconfident, higher means the interval is uselessly wide — accuracy
    metrics alone cannot tell the two apart.
    """
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def quantile_metrics(
    y_true: np.ndarray, quantile_preds: dict[float, np.ndarray]
) -> dict[str, float]:
    """Score a probabilistic forecast: pinball loss per quantile plus coverage.

    The mean pinball loss over a quantile grid approximates CRPS, which is the
    standard proper scoring rule for probabilistic forecasts. Inventory planning
    cares about the upper tail — how bad demand could get — and a point forecast
    cannot answer that at all.
    """
    if not quantile_preds:
        return {}

    out: dict[str, float] = {}
    losses = []
    for q, preds in sorted(quantile_preds.items()):
        loss = pinball_loss(y_true, preds, q)
        out[f"pinball_q{int(round(q * 100)):02d}"] = loss
        losses.append(loss)
    out["pinball_mean"] = float(np.mean(losses))

    # Pair quantiles symmetrically around the median into intervals.
    for level in (80, 90):
        lo_q, hi_q = (100 - level) / 200, 1 - (100 - level) / 200
        lo = _closest_quantile(quantile_preds, lo_q)
        hi = _closest_quantile(quantile_preds, hi_q)
        if lo is not None and hi is not None:
            out[f"coverage_{level}"] = interval_coverage(y_true, lo, hi)
    return out


def _closest_quantile(quantile_preds: dict[float, np.ndarray], target: float, tol: float = 0.01):
    for q, preds in quantile_preds.items():
        if abs(q - target) <= tol:
            return preds
    return None


# --------------------------------------------------------------------------
# series diagnostics
# --------------------------------------------------------------------------


def series_diagnostics(y_train: np.ndarray) -> dict[str, float | str]:
    """Describe a series' demand pattern, for stratified analysis later.

    ADI  — average interval between non-zero demands (how often it sells).
    CV²  — squared coefficient of variation of non-zero demand (how erratic
           the size is when it does sell).

    Together they give the Syntetos-Boylan classes, which is how the literature
    decides whether a series is a candidate for Croston-style methods at all.
    """
    active = trim_leading_zeros(y_train)
    nonzero = active[active > 0]

    if len(active) == 0 or len(nonzero) == 0:
        return {
            "ADI": np.nan,
            "CV2": np.nan,
            "demand_class": "no_demand",
            "zero_rate": np.nan,
            "mean_demand": np.nan,
            "n_train": int(len(active)),
        }

    adi = len(active) / len(nonzero)
    mean_nz = nonzero.mean()
    cv2 = float((nonzero.std() / mean_nz) ** 2) if mean_nz > 0 else np.nan

    if np.isnan(cv2):
        demand_class = "unknown"
    elif adi < ADI_CUTOFF and cv2 < CV2_CUTOFF:
        demand_class = "smooth"
    elif adi >= ADI_CUTOFF and cv2 < CV2_CUTOFF:
        demand_class = "intermittent"
    elif adi < ADI_CUTOFF and cv2 >= CV2_CUTOFF:
        demand_class = "erratic"
    else:
        demand_class = "lumpy"

    return {
        "ADI": float(adi),
        "CV2": cv2,
        "demand_class": demand_class,
        "zero_rate": float((active == 0).mean()),
        "mean_demand": float(active.mean()),
        "n_train": int(len(active)),
    }


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def evaluate_series(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    season_length: int,
) -> dict[str, float]:
    actual_zeros, predicted_zeros = zero_counts(y_true, y_pred)
    return {
        "MAE": mae(y_true, y_pred),
        "MdAE": mdae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, season_length),
        "RMSSE": rmsse(y_true, y_pred, y_train, season_length),
        "WAPE": wape(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "Bias": bias(y_true, y_pred),
        "Actual_Zero_Count": actual_zeros,
        "Predicted_Zero_Count": predicted_zeros,
        # Raw totals, so volume can be pooled across series at summary time.
        "Actual_Sum": float(np.sum(y_true)),
        "Forecast_Sum": float(np.sum(y_pred)),
        "n_test": int(len(y_true)),
    }


def evaluate_panel(
    forecast_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    train_df: pd.DataFrame,
    season_length: int,
    pred_col: str = "y_pred",
    with_diagnostics: bool = True,
) -> pd.DataFrame:
    """Per-series metrics for one model over one window, plus series diagnostics."""
    train_by_id = {uid: g["y"].to_numpy() for uid, g in train_df.groupby("unique_id")}
    merged = actuals_df.merge(forecast_df, on=["unique_id", "ds"], how="inner")

    # Probabilistic models emit y_pred_q05, y_pred_q90, ... alongside the point
    # forecast; point models emit none and simply skip the quantile scoring.
    quantile_cols = {}
    for col in forecast_df.columns:
        match = re.fullmatch(r"y_pred_q(\d{2})", str(col))
        if match:
            quantile_cols[int(match.group(1)) / 100] = col

    rows = []
    for uid, g in merged.groupby("unique_id"):
        y_train = train_by_id.get(uid)
        if y_train is None or len(g) == 0:
            continue
        y_true = g["y"].to_numpy()
        row = {
            "unique_id": uid,
            **evaluate_series(y_true, g[pred_col].to_numpy(), y_train, season_length),
        }
        if quantile_cols:
            row.update(
                quantile_metrics(y_true, {q: g[c].to_numpy() for q, c in quantile_cols.items()})
            )
        if with_diagnostics:
            row.update(series_diagnostics(y_train))
        rows.append(row)

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def trimmed_mean(values: pd.Series, proportion: float = 0.05) -> float:
    """Mean after dropping the top and bottom `proportion` of values.

    Scale-free metrics explode on near-flat series, so a handful of extreme
    ratios can decide a ranking that the other thousands of series contradict.
    """
    clean = values.dropna()
    if clean.empty:
        return np.nan
    lo, hi = clean.quantile(proportion), clean.quantile(1 - proportion)
    kept = clean[(clean >= lo) & (clean <= hi)]
    return float(kept.mean()) if len(kept) else float(clean.mean())


def volume_ratio(results: pd.DataFrame) -> float:
    """Total forecast volume over total actual volume, pooled across series.

    Below 1 means the model under-forecasts in aggregate. This is the check
    MASE cannot make on intermittent data: absolute error is minimised by the
    median, which is zero for a sparse series, so MASE rewards shading
    forecasts down. On M5 the deep models forecast ~65% of actual volume and
    lead on MASE; an all-zero forecast beats several baselines on MASE.

    Pooled rather than averaged per series, since a series with no test demand
    has an undefined ratio.
    """
    actual = results["Actual_Sum"].sum()
    return float(results["Forecast_Sum"].sum() / actual) if actual > 0 else np.nan


def summarize(
    results: pd.DataFrame,
    by: str = "model",
    metrics: tuple[str, ...] = ("MASE", "RMSSE", "WAPE", "sMAPE", "MAE", "RMSE", "Bias"),
    sort_by: str = "MASE",
) -> pd.DataFrame:
    """Aggregate per-series results with mean, median and trimmed mean side by side.

    Divergence between the three is itself the signal: if the ranking flips
    between mean and median, the mean is being driven by outliers.

    `sort_by` should be the dataset's headline metric — RMSSE for M5, where
    MASE rewards under-forecasting (see `volume_ratio`).
    """
    rows = []
    for key, g in results.groupby(by):
        row = {by: key, "n_series": g["unique_id"].nunique(), "n_obs": len(g)}
        for metric in metrics:
            if metric not in g.columns:
                continue
            row[f"{metric}_mean"] = g[metric].mean()
            row[f"{metric}_median"] = g[metric].median()
            row[f"{metric}_trimmed"] = trimmed_mean(g[metric])
        if {"Actual_Sum", "Forecast_Sum"} <= set(g.columns):
            row["Volume_ratio"] = volume_ratio(g)
        row["MASE_nan"] = int(g["MASE"].isna().sum()) if "MASE" in g else 0
        rows.append(row)

    out = pd.DataFrame(rows)
    sort_col = f"{sort_by}_trimmed"
    if sort_col in out.columns:
        out = out.sort_values(sort_col)
    return out.set_index(by).round(4)


def summarize_by_class(
    results: pd.DataFrame,
    metric: str = "MASE",
    stat: str = "median",
) -> pd.DataFrame:
    """Model x demand-class table — the evidence a regime router is built from.

    Defaults to the median rather than the trimmed mean. Some classes hold only
    a few dozen series, and at that size a 5% trim removes too little: one
    near-dormant series with a degenerate error scale can still dominate the
    cell. The median is robust regardless of group size.
    """
    agg = {"mean": "mean", "median": "median", "trimmed": trimmed_mean}[stat]
    table = results.pivot_table(index="model", columns="demand_class", values=metric, aggfunc=agg)
    counts = results.groupby("demand_class")["unique_id"].nunique()
    table.columns = [f"{c} (n={counts.get(c, 0)})" for c in table.columns]
    return table.round(4)
