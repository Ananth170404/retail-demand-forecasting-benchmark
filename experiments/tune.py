"""Tune a model on validation windows, never on the test window.

Every trial trains on data before a validation cutoff and is scored on the
window after it with the competition's own metric:

    M5       train to d_N, score the next 28 days, official WRMSSE; window k
             ends at d_N = 1913 - 28k, so --windows 3 averages d_1858-d_1941
    Walmart  train to 2011-07-29, score the next 13 weeks, WMAE

Several M5 windows matter: a single 28-day window cannot pin down a few
percent of aggregate bias, which dominates WRMSSE's top levels. Tuning
LightGBM on one window improved validation 0.614 -> 0.519 but the test got
worse (0.638 -> 0.683) through a +5.8% over-forecast. With several windows the
score is their mean, and a trial far behind the median after its first window
is pruned rather than finished.

The LightGBM search covers the feature set (every rung of the feature ladder,
so no test-window ablation result narrows it) and LightGBM's main settings;
the neural search covers history length, training budget, learning rate, batch
and scaling. Trial 0 is the configuration used before tuning, so "tuned vs
default" is measured on the same windows. Studies persist to SQLite: rerunning
resumes, and --trials is the total budget, not an increment.

Output, for the test configs to consume:
    results/tuning/<study>_trials.csv   every completed trial
    results/tuning/<study>_best.yaml    a ready-to-use model entry

Usage:
    python experiments/tune.py --dataset walmart --trials 60
    python experiments/tune.py --dataset m5 --trials 30 --windows 3
    python experiments/tune.py --dataset m5 --model nhits --trials 30 --windows 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import optuna
import pandas as pd
import yaml
from _common import DATASET_CONFIG, load_panel, mode_cutoff, resolve_mode
from registry import Context, build_entry, feature_ladder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecasting.data.loaders import DATA_ROOT  # noqa: E402
from forecasting.evaluation.backtest import run_backtest  # noqa: E402
from forecasting.evaluation.m5 import load_window, score_forecast  # noqa: E402
from forecasting.evaluation.walmart import holiday_weeks, wmae  # noqa: E402

TUNING_DIR = Path(__file__).resolve().parent.parent / "results" / "tuning"

MODELS = ("lightgbm", "nhits", "nbeats")

# The configuration each dataset used before tuning: trial 0.
DEFAULTS = {
    "m5": {"feature_step": "covariates", "objective": "l2", "train_days": 730},
    "walmart": {"feature_step": "holiday", "objective": "l2", "seasonal_difference": 52},
}

# Untuned neural settings (configs/m5_competition.yaml), in study names.
NEURAL_DEFAULT = {
    "input_size": 56,
    "max_steps": 2000,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "scaler_type": "standard",
}

# M5 day d_1 is 2011-01-29.
M5_DAY_ONE = pd.Timestamp("2011-01-29")
M5_LAST_VALIDATION_DAY = 1913


def study_name(dataset: str, model: str, windows: int) -> str:
    return f"{dataset}_{model}" + (f"_{windows}win" if windows > 1 else "")


def suggest(trial: optuna.Trial, dataset: str, steps: list[str]) -> dict:
    """Sample one LightGBM configuration as registry `gbm` params."""
    params: dict = {"feature_step": trial.suggest_categorical("feature_step", steps)}
    if dataset == "walmart":
        sdiff = trial.suggest_categorical("seasonal_difference", [0, 52])
        params["seasonal_difference"] = sdiff or None
        # Tweedie cannot fit year-over-year changes, which can be negative.
        objectives = ["l2"] if sdiff else ["tweedie", "l2"]
    else:
        params["train_days"] = trial.suggest_categorical("train_days", [365, 730])
        objectives = ["tweedie", "l2"]
    objective = trial.suggest_categorical(f"objective_{'-'.join(objectives)}", objectives)

    lgb = {
        "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 300, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    if objective == "tweedie":
        lgb["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.05, 1.6)
    params.update({"objectives": [objective], "lgb_params": lgb})
    return params


def suggest_neural(trial: optuna.Trial, model: str) -> dict:
    """Sample one neural configuration as registry params (`neural`, or
    `nbeats` for N-BEATS).

    The loss is fixed to one whose point forecast is the mean (a negative
    binomial; MSE for N-BEATS): that is a property of the metric (squared error
    on mostly-zero counts), not a setting to search. Compute is set by steps x
    windows_batch_size, so a larger batch_size covers more series per step at
    almost no extra cost.
    """
    hparams = {
        "input_size": trial.suggest_categorical("input_size", [56, 112, 224, 364]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 128, 256]),
        "scaler_type": trial.suggest_categorical("scaler_type", ["standard", "robust"]),
    }
    max_steps = trial.suggest_categorical("max_steps", [1000, 2000, 3000])
    if model == "nbeats":
        return {"variants": ["nbeats"], "loss": "mse", "max_steps": max_steps, "hparams": hparams}
    return {"model": model, "loss": "negbin", "max_steps": max_steps, "hparams": hparams}


def default_trial(dataset: str) -> dict:
    """Trial-0 values in the study's parameter names (LightGBM defaults)."""
    d = DEFAULTS[dataset]
    objectives = ["l2"] if d.get("seasonal_difference") else ["tweedie", "l2"]
    values = {
        "feature_step": d["feature_step"],
        f"objective_{'-'.join(objectives)}": d["objective"],
        "num_leaves": 63,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "min_child_samples": 20,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "reg_lambda": 1e-3,  # LightGBM default is 0; 1e-3 is the search floor
    }
    if dataset == "walmart":
        values["seasonal_difference"] = d.get("seasonal_difference") or 0
    else:
        values["train_days"] = d["train_days"]
    return values


def m5_windows(windows: int) -> list[int]:
    """Last training day of each window, most recent first."""
    return [M5_LAST_VALIDATION_DAY - 28 * k for k in range(windows)]


def main(
    dataset: str, trials: int, model: str = "lightgbm", windows: int = 1, timeout_hours=None
) -> None:
    if dataset != "m5" and windows != 1:
        raise ValueError("--windows applies to M5 only")
    cfg = DATASET_CONFIG[dataset]
    window_mode, horizon = resolve_mode(cfg, "validation")
    name = study_name(dataset, model, windows)
    print(f"Loading {dataset} for study {name}...")
    panel, roles = load_panel(
        dataset, max_series_per_state=None if dataset == "m5" else 200, with_exog=True
    )
    ctx = Context(dataset=dataset, cfg=cfg, horizon=horizon, roles=roles)
    steps = [step for step, _, _ in feature_ladder(ctx)]

    # (cutoff, scorer) per validation window.
    if dataset == "m5":
        sales = pd.read_csv(DATA_ROOT / "m5" / "sales_train_evaluation.csv")
        plan = []
        for end in m5_windows(windows):
            evaluator, actuals = load_window(DATA_ROOT / "m5", end, h=horizon, sales=sales)
            cutoff = M5_DAY_ONE + pd.Timedelta(days=end - 1)

            def score(forecast, evaluator=evaluator, actuals=actuals) -> float:
                return float(score_forecast(evaluator, actuals, forecast)[0]["WRMSSE"])

            plan.append((cutoff, score))
            print(f"  window: train to d_{end} ({cutoff.date()}), score next {horizon} days")
        del sales
    else:
        holidays = holiday_weeks(DATA_ROOT / "walmart")

        def score(forecast) -> float:
            return wmae(forecast, panel, holidays)[0]

        plan = [(mode_cutoff(cfg, "validation") if window_mode == "cutoff" else None, score)]

    def objective(trial: optuna.Trial) -> float:
        if model == "lightgbm":
            kind, params = "gbm", suggest(trial, dataset, steps)
        else:
            kind, params = TAGS[model][0], suggest_neural(trial, model)
        [(_, factory)] = build_entry(kind, {**params, "model_name": "trial"}, ctx).items()
        trial.set_user_attr("params", params)
        t0 = time.perf_counter()
        scores = []
        for k, (cutoff, scorer) in enumerate(plan):
            _, forecast = run_backtest(
                model_factory=factory,
                panel=panel,
                h=horizon,
                season_length=cfg["season_length"],
                min_train_periods=cfg["min_train_periods"],
                mode="cutoff" if cutoff is not None else window_mode,
                cutoff=cutoff,
                return_forecasts=True,
            )
            scores.append(scorer(forecast))
            trial.set_user_attr("window_scores", [round(s, 5) for s in scores])
            trial.report(sum(scores) / len(scores), step=k)
            if trial.should_prune():
                print(f"  trial {trial.number}: pruned after window {k + 1} ({scores})")
                raise optuna.TrialPruned()
        trial.set_user_attr("seconds", round(time.perf_counter() - t0, 1))
        value = sum(scores) / len(scores)
        label = params.get("feature_step") or params.get("hparams")
        print(f"  trial {trial.number}: {value:.4f} {scores} ({label})", flush=True)
        return value

    TUNING_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=name,
        storage=f"sqlite:///{TUNING_DIR / f'{name}.db'}",
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        # Prunes a trial whose running mean is behind the median of earlier
        # trials at the same window, from the first window on; nothing is
        # pruned until 5 trials have finished, so the median means something.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
        load_if_exists=True,
    )
    if not study.trials:
        if model == "lightgbm":
            study.enqueue_trial(default_trial(dataset))
            previous = _single_window_best(dataset) if windows > 1 else None
            if previous:
                # The configuration that won on one window, so the gain from
                # more windows is visible inside the same study.
                study.enqueue_trial(previous)
        else:
            study.enqueue_trial(NEURAL_DEFAULT)
    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, trials - done)
    print(f"{done} trials done; running {remaining} more")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(
        objective,
        n_trials=remaining,
        timeout=timeout_hours * 3600 if timeout_hours else None,
    )

    write_outputs(study, dataset, model, name)


def _single_window_best(dataset: str) -> dict | None:
    path = TUNING_DIR / f"{study_name(dataset, 'lightgbm', 1)}.db"
    if not path.exists():
        return None
    old = optuna.load_study(
        study_name=study_name(dataset, "lightgbm", 1), storage=f"sqlite:///{path}"
    )
    return old.best_trial.params


TAGS = {
    "lightgbm": ("gbm", "gbmtuned", "LightGBM-tuned"),
    "nhits": ("neural", "nhitstuned", "NHITS-tuned"),
    "nbeats": ("nbeats", "nbeatstuned", "N-BEATS-tuned"),
}


def write_outputs(study: optuna.Study, dataset: str, model: str, name: str) -> None:
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    table = pd.DataFrame(
        [
            {
                "trial": t.number,
                "score": t.value,
                "window_scores": t.user_attrs.get("window_scores"),
                "seconds": t.user_attrs.get("seconds"),
                **t.params,
            }
            for t in complete
        ]
    ).sort_values("score")
    table.to_csv(TUNING_DIR / f"{name}_trials.csv", index=False)

    best = study.best_trial
    default = next(t for t in complete if t.number == 0)
    kind, tag, label = TAGS[model]
    suffix = name.removeprefix(f"{dataset}_{model}").replace("_", "-").replace("win", "w")
    entry = {
        "tag": tag + suffix.replace("-", ""),
        "kind": kind,
        "params": {**best.user_attrs["params"], "model_name": label + suffix},
    }
    header = (
        f"# Best of {len(complete)} trials ({pruned} pruned) on the {dataset} validation "
        f"window(s) (score {best.value:.4f}; untuned default {default.value:.4f}).\n"
        "# Paste into the test config's `models:` list.\n"
    )
    path = TUNING_DIR / f"{name}_best.yaml"
    path.write_text(header + yaml.safe_dump([entry], sort_keys=False), encoding="utf-8")
    print(f"\nbest {best.value:.4f} (trial {best.number}) vs default {default.value:.4f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG), required=True)
    parser.add_argument("--model", choices=MODELS, default="lightgbm")
    parser.add_argument("--trials", type=int, default=30, help="total trial budget")
    parser.add_argument(
        "--windows", type=int, default=1, help="M5: number of 28-day validation windows"
    )
    parser.add_argument("--timeout-hours", type=float, default=None, help="stop after this long")
    args = parser.parse_args()
    main(args.dataset, args.trials, args.model, args.windows, args.timeout_hours)
