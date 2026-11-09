"""Unified benchmark: every model in a config, on one panel, under one protocol.

The per-model scripts each load data and build windows themselves; a benchmark
run loads the panel once, drives every model through identical windows, and
then refuses to report if any two models were scored on different series —
the comparison is only fair if that holds, so it is checked rather than
assumed.

Results go to the same per-tag tables the per-model scripts write, so
run_significance.py and scripts/rebuild_summaries.py work on either. Each run
also writes a manifest recording the resolved config and git commit.

Usage:
    python experiments/run_benchmark.py --config configs/m5.yaml
    python experiments/run_benchmark.py --config configs/m5.yaml --dry-run
    python experiments/run_benchmark.py --config configs/m5.yaml --only baselines,gbm
    python experiments/run_benchmark.py --config configs/m5.yaml --skip-existing
    python experiments/run_benchmark.py --config configs/smoke.yaml --output-dir /tmp/smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from _common import (
    DATASET_CONFIG,
    MODES,
    RESULTS_DIR,
    load_panel,
    mode_cutoff,
    report,
    resolve_mode,
)
from registry import NEEDS_EXOG, Context, build_entry

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import run_significance  # noqa: E402

from forecasting.data.loaders import DATA_ROOT  # noqa: E402
from forecasting.evaluation.backtest import make_windows, run_backtest  # noqa: E402
from forecasting.evaluation.m5 import load_release, score_forecast  # noqa: E402
from forecasting.evaluation.metrics import evaluate_panel  # noqa: E402
from forecasting.evaluation.walmart import holiday_weeks, wmae  # noqa: E402

TOP_LEVEL_KEYS = {"name", "dataset", "modes", "data", "backtest", "significance", "models"}
DATA_KEYS = {"max_series_per_state", "trim_zeros"}


def load_config(path: Path) -> dict:
    """Read and validate a config, failing on typos rather than ignoring them.

    A misspelt key silently falling back to a default is exactly the kind of
    error that produces a plausible-looking but wrong results table.
    """
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    unknown = set(config) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"unknown top-level keys {sorted(unknown)}; allowed {sorted(TOP_LEVEL_KEYS)}"
        )
    dataset = config.get("dataset")
    if dataset not in DATASET_CONFIG:
        raise ValueError(f"dataset must be one of {list(DATASET_CONFIG)}, got {dataset!r}")

    config.setdefault("name", Path(path).stem)
    config.setdefault("modes", ["rolling"])
    config.setdefault("data", {})
    config.setdefault("backtest", {})
    config.setdefault("significance", True)

    bad_modes = set(config["modes"]) - set(MODES)
    if bad_modes:
        raise ValueError(f"unknown modes {sorted(bad_modes)}")
    if "competition" in config["modes"]:
        # The competition panel includes the test period, which would shift the
        # windows of any other mode sharing it.
        if config["modes"] != ["competition"]:
            raise ValueError("competition mode must be the only mode in its config")
    full_scale = config["data"].get("max_series_per_state", 200) is None
    if dataset == "m5" and {"competition", "validation"} & set(config["modes"]) and not full_scale:
        # WRMSSE aggregates all 30,490 series; a sample would zero-fill the rest.
        raise ValueError("M5 competition/validation modes need data.max_series_per_state: null")
    bad_data = set(config["data"]) - DATA_KEYS
    if bad_data:
        raise ValueError(f"unknown data keys {sorted(bad_data)}; allowed {sorted(DATA_KEYS)}")
    bad_backtest = set(config["backtest"]) - set(DATASET_CONFIG[dataset])
    if bad_backtest:
        raise ValueError(
            f"unknown backtest keys {sorted(bad_backtest)}; "
            f"allowed {sorted(DATASET_CONFIG[dataset])}"
        )

    tags = [m.get("tag") for m in config.get("models", [])]
    if not tags or None in tags:
        raise ValueError("every model entry needs a 'tag' (and there must be at least one)")
    if len(set(tags)) != len(tags):
        raise ValueError(f"duplicate model tags: {tags}")
    for entry in config["models"]:
        if "kind" not in entry:
            raise ValueError(f"model entry '{entry['tag']}' has no 'kind'")
        entry.setdefault("params", {})
    return config


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def _check_same_series(scored: dict[str, set], mode: str) -> None:
    """Every model must be scored on the same (series, window) pairs."""
    reference_model, reference = next(iter(scored.items()))
    mismatched = {m: len(keys ^ reference) for m, keys in scored.items() if keys != reference}
    if mismatched:
        raise RuntimeError(
            f"[{mode}] models scored on different series/windows than {reference_model}: "
            f"{mismatched} (count of differing pairs). Rankings would not be comparable."
        )


def score_saved(forecast: pd.DataFrame, windows, season_length: int, label: str) -> pd.DataFrame:
    """Per-series metrics for saved forecasts, as run_backtest would produce them.

    Fit and predict times are unknown for a resumed model and left empty.
    """
    parts = []
    for window in windows:
        metrics = evaluate_panel(
            forecast_df=forecast[forecast["window"] == window.index].drop(
                columns=["model", "window"], errors="ignore"
            ),
            actuals_df=window.test,
            train_df=window.train,
            season_length=season_length,
        )
        metrics["model"] = label
        metrics["window"] = window.index
        metrics["cutoff"] = window.cutoff
        metrics["horizon"] = window.horizon
        metrics["fit_seconds"] = float("nan")
        metrics["predict_seconds"] = float("nan")
        parts.append(metrics)
    return pd.concat(parts, ignore_index=True)


def _write_leaderboard(new: pd.DataFrame, path: Path, score: str) -> None:
    """Merge this run's competition-metric rows into the table, replacing re-run models.

    Runs with --only add to the table rather than overwrite it, so a full
    leaderboard can be assembled from separate runs. When seasonal naive is
    present, each score is also given relative to it: the form in which the
    published competition results are comparable across test periods.
    """
    new.columns = [f"L{c}" if isinstance(c, int) else c for c in new.columns]
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["model"].isin(new["model"])], new], ignore_index=True)
    new = new.drop(columns=[f"{score}_vs_SeasonalNaive"], errors="ignore")
    if "SeasonalNaive" in set(new["model"]):
        ref = new.loc[new["model"] == "SeasonalNaive", score].iloc[0]
        new[f"{score}_vs_SeasonalNaive"] = new[score] / ref
    new = new.sort_values(score).reset_index(drop=True)
    new.round(4).to_csv(path, index=False)
    print(f"    -> {path.name} ({len(new)} models)", end=" ")


def run_mode(
    config: dict,
    mode: str,
    panel: pd.DataFrame,
    roles,
    out_dir: Path,
    only: set[str] | None,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    dataset = config["dataset"]
    cfg = {**DATASET_CONFIG[dataset], **config["backtest"]}
    window_mode, horizon = resolve_mode(cfg, mode)
    ctx = Context(dataset=dataset, cfg=cfg, horizon=horizon, roles=roles)
    entries = [e for e in config["models"] if not only or e["tag"] in only]

    print(f"\n##### {dataset} [{mode}] h={horizon} | headline={cfg['headline_metric']} #####")

    window_kwargs = dict(
        h=horizon,
        test_fraction=cfg["test_fraction"],
        n_windows=cfg["n_windows"],
        min_train_periods=cfg["min_train_periods"],
        mode=window_mode,
        cutoff=mode_cutoff(cfg, mode) if window_mode == "cutoff" else None,
    )

    if dry_run:
        windows = make_windows(panel, **window_kwargs)
        n_series = windows[0].test["unique_id"].nunique() if windows else 0
        print(
            f"  {len(windows)} window(s); first cutoff "
            f"{windows[0].cutoff.date() if windows else '-'}; ~{n_series:,} series scored"
        )
        for entry in entries:
            labels = list(build_entry(entry["kind"], entry["params"], ctx))
            print(f"  [{entry['tag']}] {', '.join(labels)}")
        return

    # M5's official metric scores both phases: validation (d_1914-d_1941)
    # and the final test (d_1942-d_1969), each with its own official weights.
    competition = dataset == "m5" and mode in ("competition", "validation")
    if competition:
        phase = "evaluation" if mode == "competition" else "validation"
        print(f"  building official WRMSSE evaluator ({phase} phase)...", flush=True)
        evaluator, actuals = load_release(DATA_ROOT / "m5", phase=phase)
    # Walmart's own competition metric is scored in every mode.
    holidays = holiday_weeks(DATA_ROOT / "walmart") if dataset == "walmart" else None
    # Every model's forecasts are saved as soon as it finishes, so an
    # interrupted run loses at most the model in progress: with
    # --skip-existing, saved forecasts are re-scored instead of retrained.
    forecast_dir = out_dir.parent / "forecasts" / f"{dataset}_{mode}"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)  # leaderboards are written per model
    windows = None  # built lazily, only if a saved forecast needs re-scoring

    scored: dict[str, set] = {}
    for entry in entries:
        tag = entry["tag"]

        results = []
        for label, factory in build_entry(entry["kind"], entry["params"], ctx).items():
            print(f"  [{tag}] {label}...", end=" ", flush=True)
            t0 = time.perf_counter()
            saved = forecast_dir / f"{label}.parquet"
            if skip_existing and saved.exists():
                if windows is None:
                    windows = make_windows(panel, **window_kwargs)
                forecast = pd.read_parquet(saved)
                result = score_saved(forecast, windows, cfg["season_length"], label)
                elapsed = "resumed from saved forecast"
            else:
                result, forecast = run_backtest(
                    model_factory=factory,
                    panel=panel,
                    season_length=cfg["season_length"],
                    return_forecasts=True,
                    **window_kwargs,
                )
                forecast.to_parquet(saved, index=False)
                elapsed = f"{time.perf_counter() - t0:,.0f}s"
            if competition:
                scores, zero_filled = score_forecast(evaluator, actuals, forecast)
                row = {"model": label, **scores, "zero_filled_series": zero_filled}
                path = out_dir / f"wrmsse_{dataset}_{mode}.csv"
                _write_leaderboard(pd.DataFrame([row]), path, "WRMSSE")
                elapsed += f" | WRMSSE {scores['WRMSSE']:.4f}"
            if holidays is not None:
                score, n_rows = wmae(forecast, panel, holidays)
                row = {"model": label, "WMAE": score, "rows": n_rows}
                path = out_dir / f"wmae_{dataset}_{mode}.csv"
                _write_leaderboard(pd.DataFrame([row]), path, "WMAE")
                elapsed += f" | WMAE {score:,.0f}"
            print(elapsed, flush=True)
            if result["MAE"].isna().any():
                raise RuntimeError(f"{label}: NaN forecasts leaked into the metrics")
            scored[label] = set(zip(result["unique_id"].astype(str), result["window"], strict=True))
            results.append(result)

        report(
            pd.concat(results, ignore_index=True),
            dataset,
            mode,
            tag,
            results_dir=out_dir,
            headline=cfg["headline_metric"],
        )

    if not scored:
        return
    _check_same_series(scored, mode)
    print(f"\n  all {len(scored)} models scored on identical series/windows")

    if config["significance"] and len(scored) > 1:
        run_significance.main(dataset, mode, metric=cfg["headline_metric"], results_dir=out_dir)


def main(
    config_path: Path, out_dir: Path, only: set[str] | None, skip_existing: bool, dry_run: bool
) -> None:
    config = load_config(config_path)
    dataset = config["dataset"]

    tags = {e["tag"] for e in config["models"]}
    if only and only - tags:
        raise ValueError(
            f"--only names unknown tags {sorted(only - tags)}; config has {sorted(tags)}"
        )

    selected = [e for e in config["models"] if not only or e["tag"] in only]
    with_exog = any(e["kind"] in NEEDS_EXOG for e in selected)

    print(f"Benchmark '{config['name']}' -> {out_dir}")
    print(f"Loading {dataset}...")
    panel, roles = load_panel(
        dataset,
        trim_zeros=config["data"].get("trim_zeros", True),
        # Absent means the 200-per-state sample; an explicit null loads all.
        max_series_per_state=config["data"].get("max_series_per_state", 200),
        with_exog=with_exog,
        include_test=config["modes"] == ["competition"],
    )

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for mode in config["modes"]:
        run_mode(config, mode, panel, roles, out_dir, only, skip_existing, dry_run)

    if not dry_run:
        manifest = {
            "config_path": str(config_path),
            "config": config,
            "only": sorted(only) if only else None,
            "git_commit": _git_commit(),
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path = out_dir / f"benchmark_{config['name']}_manifest.json"
        path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--only", default=None, help="comma-separated model tags to run")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="re-score models whose forecasts are saved instead of retraining them",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="load data and list windows and models without training",
    )
    args = parser.parse_args()
    main(
        args.config,
        args.output_dir,
        set(args.only.split(",")) if args.only else None,
        args.skip_existing,
        args.dry_run,
    )
