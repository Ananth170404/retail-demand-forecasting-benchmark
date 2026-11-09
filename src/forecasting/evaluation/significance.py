"""Is one model's lead real, or within the noise of which series were sampled?

Two complementary answers, both cross-sectional — the series is the unit of
observation:

1. Pairwise tests against a reference model (usually the leader): a paired
   bootstrap CI on the difference in trimmed means, a Wilcoxon signed-rank
   test on per-series differences, and Holm's correction because a dozen
   comparisons at alpha=0.05 would otherwise produce false wins by chance.

2. Mean ranks with the Nemenyi critical difference (Demšar 2006), the format
   the M4 competition used: models whose mean ranks differ by less than the CD
   are statistically indistinguishable.

Why not Diebold-Mariano: DM tests a loss differential over time and needs
per-timestep errors, which the results tables do not store — each row is
already one aggregated error per series per window. With hundreds to thousands
of series, the cross-section is also where the statistical power is.

Rolling backtests score each series in many windows. Those rows are not
independent (same series, overlapping history), so metrics are first averaged
per series and the tests run over series. Point estimates here can therefore
differ slightly from the summary tables, which trim over (series, window) rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import studentized_range, trim_mean, wilcoxon

TRIM = 0.05


def per_series_matrix(results: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Series x model table of `metric`, averaged over windows.

    Only series every model scored are kept: a test between two models is
    meaningless unless both were scored on the same series. NaNs (flat series
    with no defined scale) are dropped the same way.
    """
    wide = results.pivot_table(index="unique_id", columns="model", values=metric, aggfunc="mean")
    return wide.dropna()


def holm(p_values: pd.Series) -> pd.Series:
    """Holm-Bonferroni adjusted p-values: controls family-wise error, less
    conservative than plain Bonferroni."""
    order = p_values.sort_values()
    m = len(order)
    adjusted = np.maximum.accumulate([(m - i) * p for i, p in enumerate(order.to_numpy())])
    return pd.Series(np.minimum(adjusted, 1.0), index=order.index).reindex(p_values.index)


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float, float]:
    """Difference in trimmed means (a - b) with a 95% CI and two-sided p-value.

    Series are resampled jointly, so each draw keeps the pairing between the two
    models' errors on the same series — that pairing is what makes the test
    sensitive to differences far smaller than the spread across series.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = trim_mean(a[idx], TRIM, axis=1) - trim_mean(b[idx], TRIM, axis=1)
    point = trim_mean(a, TRIM) - trim_mean(b, TRIM)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return float(point), float(lo), float(hi), float(p)


def compare_to_reference(
    wide: pd.DataFrame, reference: str | None = None, n_boot: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """Test every model against `reference` (default: lowest trimmed mean).

    `diff` is model minus reference, so with lower-is-better metrics a positive
    diff means the reference is ahead. `win_rate` is the share of series on
    which the model beats the reference — a ranking driven by a few series
    shows up as a lead in the mean alongside a win rate near 0.5.
    """
    if reference is None:
        reference = wide.apply(lambda col: trim_mean(col, TRIM)).idxmin()
    ref = wide[reference].to_numpy()

    rows = []
    for model in wide.columns:
        if model == reference:
            continue
        x = wide[model].to_numpy()
        diff, lo, hi, p_boot = paired_bootstrap(x, ref, n_boot=n_boot, seed=seed)
        # Wilcoxon is undefined when every difference is zero (identical models).
        p_wil = wilcoxon(x, ref).pvalue if np.any(x != ref) else 1.0
        rows.append(
            {
                "model": model,
                "reference": reference,
                "diff": diff,
                "ci_low": lo,
                "ci_high": hi,
                "p_bootstrap": p_boot,
                "p_wilcoxon": float(p_wil),
                "win_rate": float(np.mean(x < ref)),
            }
        )

    out = pd.DataFrame(rows).set_index("model")
    out["p_wilcoxon_holm"] = holm(out["p_wilcoxon"])
    out["p_bootstrap_holm"] = holm(out["p_bootstrap"])
    # Called significant only if both tests agree after correction: the
    # bootstrap tests the trimmed mean, Wilcoxon the typical series, and a lead
    # that holds on one but not the other is not one to report as settled.
    out["significant"] = (out["p_wilcoxon_holm"] < 0.05) & (out["p_bootstrap_holm"] < 0.05)
    out["n_series"] = len(wide)
    return out.sort_values("diff")


def nemenyi_cd(n_models: int, n_series: int, alpha: float = 0.05) -> float:
    """Nemenyi critical difference for mean ranks (Demšar 2006, eq. 5)."""
    q = studentized_range.ppf(1 - alpha, n_models, np.inf) / np.sqrt(2)
    return float(q * np.sqrt(n_models * (n_models + 1) / (6 * n_series)))


def mean_ranks(wide: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Mean rank per model (1 = best), plus which models tie with the leader.

    Ranks are immune to the scale explosions that make means fragile, at the
    cost of ignoring by how much a model wins on each series.
    """
    ranks = wide.rank(axis=1, method="average").mean().sort_values()
    cd = nemenyi_cd(wide.shape[1], wide.shape[0], alpha)
    out = ranks.rename("mean_rank").to_frame()
    out["gap_to_best"] = out["mean_rank"] - out["mean_rank"].iloc[0]
    out["tied_with_best"] = out["gap_to_best"] < cd
    out["critical_difference"] = cd
    out["n_series"] = wide.shape[0]
    return out
