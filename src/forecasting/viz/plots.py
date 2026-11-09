"""Shared plotting helpers for the README figures and the notebooks.

Every function takes plain DataFrames (the committed result tables) and returns
a matplotlib Figure, so the notebooks and scripts/make_figures.py draw the same
charts. Figures render on a solid light surface: GitHub shows README images on
a dark page in dark mode, and a transparent chart would put dark ink on it.

matplotlib is an optional extra (`pip install -e ".[viz]"`), imported here only.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categorical slots in fixed order (validated for colour-vision deficiency).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

# Official M5 accuracy results (Makridakis et al. 2022, Table 3 and Table B).
M5_BENCHMARK = {"official ES_bu benchmark": 0.671}

M5_LEVELS = [
    "Total",
    "State",
    "Store",
    "Category",
    "Dept",
    "State×Cat",
    "State×Dept",
    "Store×Cat",
    "Store×Dept",
    "Item",
    "Item×State",
    "Item×Store",
]

FAMILIES = {
    "Ensemble": BLUE,
    "Boosted trees": ORANGE,
    "Neural and pretrained": AQUA,
    "Statistical and naive": YELLOW,
}


def family(model: str) -> str:
    if model.startswith("Ensemble"):
        return "Ensemble"
    if model.startswith(("LightGBM", "XGBoost", "CatBoost")):
        return "Boosted trees"
    if model.startswith(("N-BEATS", "NHITS", "TFT", "DeepAR", "BiTCN", "Chronos")):
        return "Neural and pretrained"
    return "Statistical and naive"


def style_axes(ax: plt.Axes, grid_axis: str = "x") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def _figure(width: float, height: float, ncols: int = 1):
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]
    return plt.subplots(1, ncols, figsize=(width, height), facecolor=SURFACE)


def _header(fig: plt.Figure, title: str, subtitle: str, height: float) -> float:
    """Title, subtitle and a legend row above the plot, at fixed sizes in inches.
    Returns the figure fraction where the plot area may start."""
    inch = 1 / height
    fig.text(
        0.01, 1 - 0.12 * inch, title, ha="left", va="top", fontsize=13, weight="bold", color=INK
    )
    fig.text(0.01, 1 - 0.42 * inch, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)
    return 1 - 0.95 * inch


def _legend_row(fig: plt.Figure, handles, labels, height: float) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.005, 1 - 0.62 / height),
        ncol=len(labels),
        frameon=False,
        fontsize=9,
        labelcolor=INK_2,
        handlelength=1.2,
        columnspacing=1.6,
    )


def _swatches(colors) -> list:
    return [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]


def _hbars(ax, labels, values, colors, fmt="{:.3f}") -> None:
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, height=0.5, color=colors)
    for yi, v in zip(y, values, strict=True):
        ax.text(v + 0.008, yi, fmt.format(v), va="center", fontsize=8.5, color=INK_2)
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.6, len(labels) - 0.4)


def _reference_lines(ax, lines: dict[str, float]) -> None:
    """Vertical reference lines, labelled above the plot; close labels are staggered."""
    last = -np.inf
    row = 0
    for label, x in sorted(lines.items(), key=lambda kv: kv[1]):
        row = 1 - row if x - last < 0.1 * ax.get_xlim()[1] else 0
        last = x
        ax.axvline(x, color=MUTED, linewidth=1, zorder=0)
        ax.text(
            x,
            1.012 + 0.045 * row,
            f"{label} {x:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=MUTED,
            transform=ax.get_xaxis_transform(),
        )


def m5_leaderboard(board: pd.DataFrame, names: dict[str, str]) -> plt.Figure:
    """Test WRMSSE for the models in `names` (label for each), with the official
    ES_bu benchmark as a reference line."""
    rows = board.set_index("model").loc[list(names)].sort_values("WRMSSE")
    height = 0.34 * len(rows) + 2.0
    fig, ax = _figure(9, height)
    style_axes(ax)
    _hbars(
        ax,
        [names[m] for m in rows.index],
        rows["WRMSSE"],
        [FAMILIES[family(m)] for m in rows.index],
    )
    ax.set_xlim(0, max(0.95, rows["WRMSSE"].max() + 0.08))
    _reference_lines(ax, M5_BENCHMARK)
    ax.set_xlabel("WRMSSE on the competition test (lower is better)", color=INK_2, fontsize=9)
    top = _header(
        fig,
        "M5: test WRMSSE against the official benchmark",
        "Same 28-day test as the 2020 competition (d_1942–d_1969), official "
        "12-level metric. * validation's choice; see README.",
        height,
    )
    present = [f for f in FAMILIES if f in {family(m) for m in rows.index}]
    _legend_row(fig, _swatches(FAMILIES[f] for f in present), present, height)
    fig.subplots_adjust(left=0.25, right=0.97, top=top - 0.45 / height, bottom=0.65 / height)
    return fig


def m5_levels(board: pd.DataFrame, names: dict[str, str]) -> plt.Figure:
    """WRMSSE at each of the 12 aggregation levels."""
    height = 4.6
    fig, ax = _figure(9, height)
    style_axes(ax, grid_axis="y")
    x = np.arange(1, 13)
    for (model, label), color in zip(names.items(), (BLUE, ORANGE, AQUA, YELLOW), strict=False):
        row = board.set_index("model").loc[model]
        ax.plot(
            x,
            [row[f"L{i}"] for i in x],
            color=color,
            linewidth=2,
            marker="o",
            markersize=5,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
            label=label,
        )
    ax.set_xticks(x, [f"L{i}\n{n}" for i, n in zip(x, M5_LEVELS, strict=True)], fontsize=8)
    ax.set_ylim(0, None)
    ax.set_ylabel("RMSSE at this level", color=INK_2, fontsize=9)
    top = _header(
        fig,
        "M5: where the error is, level by level",
        "Models differ most on totals (L1–L9); at item level (L10–L12) the good "
        "models are nearly tied",
        height,
    )
    _legend_row(fig, *ax.get_legend_handles_labels(), height)
    fig.subplots_adjust(left=0.08, right=0.98, top=top, bottom=0.17)
    return fig


def feature_ladder(m5: pd.Series, walmart: pd.Series) -> plt.Figure:
    """Score after each feature group is added, one panel per dataset (different metrics)."""
    height = 4.4
    fig, axes = _figure(10, height, ncols=2)
    panels = [
        (axes[0], m5, "M5 — WRMSSE"),
        (axes[1], walmart, "Walmart — WMAE ÷ seasonal naive"),
    ]
    for ax, series, label in panels:
        style_axes(ax, grid_axis="y")
        x = np.arange(len(series))
        ax.plot(
            x,
            series.to_numpy(),
            color=BLUE,
            linewidth=2,
            marker="o",
            markersize=6,
            markeredgecolor=SURFACE,
            markeredgewidth=1.5,
        )
        best = int(np.argmin(series.to_numpy()))
        ax.plot(
            x[best],
            series.iloc[best],
            "o",
            color=BLUE,
            markersize=10,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
        ax.annotate(
            f"best {series.iloc[best]:.3f}",
            (x[best], series.iloc[best]),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
            color=INK,
        )
        ax.set_xticks(x, series.index, rotation=30, ha="right", fontsize=8.5)
        ax.set_title(label, loc="left", fontsize=10, color=INK)
        lo, hi = series.min(), series.max()
        ax.set_ylim(lo - 0.12 * (hi - lo) - 0.02, hi + 0.08 * (hi - lo))
    top = _header(
        fig,
        "Feature ablation: each step adds one group of features to LightGBM",
        "The datasets disagree: M5 needs calendar and event features, Walmart does "
        "best with lags alone. Best steps were seen on the test, so this is a study, "
        "not a pick.",
        height,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=top - 0.05, bottom=0.22, wspace=0.18)
    return fig


def m5_total_sales(
    history: pd.Series,
    actual: pd.Series,
    forecasts: dict[str, pd.Series],
    level_scores: dict[str, float],
) -> plt.Figure:
    """Daily total units sold: recent history, the test month, and forecasts of it."""
    height = 4.4
    fig, ax = _figure(10, height)
    style_axes(ax, grid_axis="y")
    ax.axvspan(actual.index[1], actual.index.max(), color=GRID, alpha=0.45, linewidth=0)
    ax.plot(history.index, history.to_numpy(), color=INK, linewidth=1.5)
    ax.plot(actual.index, actual.to_numpy(), color=INK, linewidth=2)
    handles = [plt.Line2D([], [], color=INK, linewidth=2)]
    labels = ["Actual"]
    for (label, series), color in zip(forecasts.items(), (BLUE, ORANGE), strict=False):
        ax.plot(series.index, series.to_numpy(), color=color, linewidth=2)
        handles.append(plt.Line2D([], [], color=color, linewidth=2))
        labels.append(f"{label} (L1 error {level_scores[label]:.3f})")
    ax.text(
        actual.index[1],
        1.0,
        " test period",
        va="top",
        fontsize=8.5,
        color=MUTED,
        transform=ax.get_xaxis_transform(),
    )
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e3:.0f}K"))
    ax.set_ylabel("Units sold per day, all 30,490 series", color=INK_2, fontsize=9)
    top = _header(
        fig,
        "M5: total daily sales in the test month",
        "Total sales is level 1 of WRMSSE; the ensemble's error there is about half "
        "of seasonal naive's",
        height,
    )
    _legend_row(fig, handles, labels, height)
    fig.subplots_adjust(left=0.08, right=0.98, top=top, bottom=0.10)
    return fig


def walmart_season(board: pd.DataFrame, names: dict[str, str]) -> plt.Figure:
    """WMAE relative to seasonal naive on the Walmart holiday window."""
    rows = board.set_index("model").loc[list(names)].sort_values("WMAE_vs_SeasonalNaive")
    height = 0.34 * len(rows) + 2.0
    fig, ax = _figure(9, height)
    style_axes(ax)
    _hbars(
        ax,
        [names[m] for m in rows.index],
        rows["WMAE_vs_SeasonalNaive"],
        [FAMILIES[family(m)] for m in rows.index],
    )
    ax.set_xlim(0, rows["WMAE_vs_SeasonalNaive"].max() + 0.15)
    _reference_lines(ax, {"seasonal naive": 1.0})
    ax.set_xlabel(
        "WMAE ÷ seasonal naive (below 1 beats it; lower is better)", color=INK_2, fontsize=9
    )
    top = _header(
        fig,
        "Walmart: holiday-season test, Nov 2011 – Jul 2012",
        "Kaggle's test calendar one year earlier (its answers were never released); "
        "left of the line beats seasonal naive",
        height,
    )
    present = [f for f in FAMILIES if f in {family(m) for m in rows.index}]
    _legend_row(fig, _swatches(FAMILIES[f] for f in present), present, height)
    fig.subplots_adjust(left=0.27, right=0.97, top=top - 0.45 / height, bottom=0.65 / height)
    return fig


def validation_vs_test(points: pd.DataFrame, offsets: dict[str, tuple[int, int]]) -> plt.Figure:
    """[model, validation, test, kind, label] — does the validation score predict the test?
    Points in `offsets` are labelled, the label moved by that many points."""
    height = 6.0
    fig, ax = _figure(7.5, height)
    style_axes(ax, grid_axis="both")
    lo = min(points["validation"].min(), points["test"].min()) - 0.01
    hi = max(points["validation"].max(), points["test"].max()) + 0.01
    ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1, zorder=0)
    ax.text(
        hi - 0.004, hi - 0.012, "test = validation", ha="right", va="top", fontsize=8, color=MUTED
    )
    handles = []
    for kind, color in (("Ensemble", BLUE), ("Single model", ORANGE)):
        part = points[points["kind"] == kind]
        ax.scatter(
            part["validation"],
            part["test"],
            s=55,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
        )
        handles.append(plt.Line2D([], [], marker="o", linestyle="", color=color, markersize=7))
    for _, p in points[points["model"].isin(offsets)].iterrows():
        dx, dy = offsets[p["model"]]
        ax.annotate(
            p["label"],
            (p["validation"], p["test"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            color=INK_2,
            ha="right" if dx < 0 else "left",
            va="center",
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Validation WRMSSE (mean of 3 earlier 28-day windows)", color=INK_2, fontsize=9)
    ax.set_ylabel("Test WRMSSE", color=INK_2, fontsize=9)
    top = _header(
        fig,
        "M5: validation against test",
        "Below the line, the test was easier. Scaling to totals: best on "
        "validation, worse on test.",
        height,
    )
    _legend_row(fig, handles, ["Ensemble", "Single model"], height)
    fig.subplots_adjust(left=0.12, right=0.97, top=top, bottom=0.10)
    return fig


def volume_ratio(rows: pd.DataFrame) -> plt.Figure:
    """[label, Volume_ratio, kind] — total forecast ÷ total actual, median vs mean forecasts."""
    height = 0.4 * len(rows) + 2.0
    fig, ax = _figure(8, height)
    style_axes(ax)
    colors = [ORANGE if k == "Median forecast" else BLUE for k in rows["kind"]]
    _hbars(ax, rows["label"].tolist(), rows["Volume_ratio"], colors, fmt="{:.2f}")
    ax.set_xlim(0, 1.3)
    _reference_lines(ax, {"unbiased": 1.0})
    ax.set_xlabel("Total forecast ÷ total actual sales on the M5 test", color=INK_2, fontsize=9)
    top = _header(
        fig,
        "Median forecasts under-forecast sparse demand",
        "On M5 most days sell zero, so the median sits low; WRMSSE rewards the mean",
        height,
    )
    _legend_row(fig, _swatches((ORANGE, BLUE)), ["Median forecast", "Mean forecast"], height)
    fig.subplots_adjust(left=0.3, right=0.97, top=top - 0.35 / height, bottom=0.65 / height)
    return fig
