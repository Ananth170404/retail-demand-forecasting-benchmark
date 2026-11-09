"""Which leads in the results tables are real?

Reads the committed per-series tables — no retraining — and tests every model
against the leader on the dataset's headline metric, plus mean ranks with the
Nemenyi critical difference.

Usage:
    python experiments/run_significance.py --dataset m5 --mode rolling
    python experiments/run_significance.py --dataset walmart --mode official --metric RMSSE
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from _common import DATASET_CONFIG, MODES, RESULTS_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecasting.evaluation.significance import (  # noqa: E402
    compare_to_reference,
    mean_ranks,
    per_series_matrix,
)

# "combined_*" tables are hand-assembled copies of the per-model tables.
SKIP_TAGS = {"combined"}


def load_results(dataset: str, mode: str, results_dir: Path = RESULTS_DIR) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(results_dir).glob(f"*_{dataset}_{mode}_per_series.csv.gz")):
        tag = path.name.split("_")[0]
        if tag in SKIP_TAGS:
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"no per-series tables for {dataset}/{mode} in {results_dir}")
    return pd.concat(frames, ignore_index=True)


def main(
    dataset: str,
    mode: str,
    metric: str | None = None,
    reference: str | None = None,
    n_boot: int = 2000,
    results_dir: Path = RESULTS_DIR,
) -> None:
    metric = metric or DATASET_CONFIG[dataset]["headline_metric"]
    results = load_results(dataset, mode, results_dir)
    wide = per_series_matrix(results, metric)
    dropped = results["unique_id"].nunique() - len(wide)

    pairwise = compare_to_reference(wide, reference=reference, n_boot=n_boot)
    ranks = mean_ranks(wide)

    suffix = f"{dataset}_{mode}_{metric}"
    pairwise_path = Path(results_dir) / f"significance_{suffix}.csv"
    ranks_path = Path(results_dir) / f"ranks_{suffix}.csv"
    pairwise.round(4).to_csv(pairwise_path)
    ranks.round(4).to_csv(ranks_path)

    ref = pairwise["reference"].iloc[0]
    print(f"\n=== {dataset.upper()} [{mode}] {metric} - vs {ref} ===")
    print(
        f"series={len(wide):,} (dropped {dropped} with a NaN for some model) "
        f"| models={wide.shape[1]}"
    )
    cols = [
        "diff",
        "ci_low",
        "ci_high",
        "win_rate",
        "p_wilcoxon_holm",
        "p_bootstrap_holm",
        "significant",
    ]
    print(pairwise[cols].round(4).to_string())
    print(f"\n-- mean ranks (Nemenyi CD = {ranks['critical_difference'].iloc[0]:.3f}) --")
    print(ranks[["mean_rank", "gap_to_best", "tied_with_best"]].round(3).to_string())
    print(f"\nwrote {pairwise_path}\nwrote {ranks_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG), required=True)
    parser.add_argument("--mode", choices=list(MODES), default="rolling")
    parser.add_argument("--metric", default=None, help="defaults to the dataset's headline metric")
    parser.add_argument("--reference", default=None, help="defaults to the best model")
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    main(args.dataset, args.mode, args.metric, args.reference, args.n_boot)
