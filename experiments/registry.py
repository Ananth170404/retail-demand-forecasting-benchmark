"""Turn a config's model entries into backtest factories.

Each `kind` maps a YAML entry onto the same builders, covariate wiring and
labels the per-model scripts (run_baselines, run_gbm, run_nbeats,
run_neural_quantile) use, so a benchmark run and a script run produce results
that line up row for row in the same tables.

A builder takes the entry's params plus a `Context` and returns
{label: zero-arg factory}. Neural imports are deferred to their builders, so a
config with only baselines and GBM runs without torch installed.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecasting.data.loaders import ColumnRoles  # noqa: E402


@dataclass
class Context:
    dataset: str
    cfg: dict
    horizon: int
    roles: ColumnRoles

    @property
    def futr_exog(self) -> list[str]:
        # Every covariate in these datasets is known ahead of time, so the
        # dynamic ones are future-exogenous and the per-series ones are static.
        return self.roles.dynamic_cat_cols + self.roles.dynamic_real_cols

    @property
    def stat_exog(self) -> list[str]:
        return self.roles.static_cat_cols


def _baselines(
    ctx: Context,
    include_slow: bool = True,
    n_jobs: int = 4,
    chunk_size: int | None = None,
    only: list[str] | None = None,
) -> dict:
    from forecasting.models.baselines import build_baselines

    factories = build_baselines(
        ctx.cfg["freq"],
        ctx.cfg["season_length"],
        include_slow=include_slow,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
    )
    if only:
        unknown = set(only) - set(factories)
        if unknown:
            raise ValueError(f"unknown baselines {sorted(unknown)}; have {list(factories)}")
        factories = {k: v for k, v in factories.items() if k in only}
    return factories


def _gbm(
    ctx: Context,
    objectives: list[str] = ("tweedie", "l2"),
    use_exog: bool = True,
    train_days: int | None = None,
    holiday_features: bool = False,
    keep_incomplete_rows: bool = False,
    seasonal_difference: int | None = None,
    ablation_ladder: bool = False,
    feature_step: str | None = None,
    lgb_params: dict | None = None,
    model_name: str | None = None,
    library: str = "lightgbm",
    device: str = "cpu",
    group_by: str | None = None,
    direct: bool = False,
) -> dict:
    """Boosted-tree variants. `feature_step` picks one rung of feature_ladder (e.g.
    "events"), and `lgb_params` sets hyperparameters (LightGBM's names, translated
    for the other libraries): together they reproduce a tuned configuration,
    labelled `model_name`. `library` swaps in XGBoost or CatBoost. `group_by` fits
    one model per value of that column; `direct` uses features no newer than the
    M5 horizon, so nothing is forecast recursively."""
    from forecasting.data.features import M5_DIRECT_FEATURES, get_feature_config
    from forecasting.models.gbm import LIBRARY_LABELS, PerGroupForecaster, build_gbm

    prefix = LIBRARY_LABELS[library]
    feature_config = get_feature_config(ctx.dataset)
    if not use_exog:
        feature_config = feature_config.without_exog()
    exog_cols = ctx.roles.static_cat_cols + ctx.roles.dynamic_cat_cols + ctx.roles.dynamic_real_cols
    if holiday_features:
        exog_cols = exog_cols + ctx.roles.holiday_cols

    # Each variant: (label, feature config, covariate columns). Labels for the
    # ordinary variants must stay byte-identical: saved forecasts and results
    # are matched by them.
    stem = f"{'-allrows' if keep_incomplete_rows else ''}" + (
        f"-sdiff{seasonal_difference}" if seasonal_difference else ""
    )
    factories = {}
    for objective in objectives:
        if feature_step:
            ladder = {step: (fc, cols) for step, fc, cols in feature_ladder(ctx)}
            if feature_step not in ladder:
                raise ValueError(f"feature_step {feature_step!r} not in {list(ladder)}")
            fc, cols = ladder[feature_step]
            fc = M5_DIRECT_FEATURES if direct else fc
            variants = [(model_name or f"{prefix}-{objective}-{feature_step}", fc, cols)]
        elif ablation_ladder:
            variants = [
                (f"{prefix}-{objective}{stem}-L{i}-{step}", fc, cols)
                for i, (step, fc, cols) in enumerate(feature_ladder(ctx), start=1)
            ]
        else:
            label = (
                f"{prefix}-{objective}"
                + ("" if use_exog else "-noexog")
                + ("-hol" if holiday_features else "")
                + stem
            )
            variants = [(label, feature_config, exog_cols)]
        for label, fc, cols in variants:
            factory = build_gbm(
                freq=ctx.cfg["freq"],
                feature_config=fc,
                static_features=ctx.roles.static_cat_cols,
                exog_cols=cols,
                objective=objective,
                train_days=train_days,
                keep_incomplete_rows=keep_incomplete_rows,
                seasonal_difference=seasonal_difference,
                lgb_params=lgb_params,
                name=label,
                library=library,
                device=device,
                cat_features=ctx.roles.static_cat_cols + ctx.roles.dynamic_cat_cols,
            )
            if group_by:
                factory = partial(PerGroupForecaster, factory, group_by, label)
            factories[label] = factory
    return factories


def feature_ladder(ctx: Context) -> list[tuple[str, object, list[str]]]:
    """Feature sets that each add one block to the last, for an ablation.

    lags -> +rolling stats -> +calendar -> +static ids -> +event flags
    -> +real-valued covariates (price; CPI, fuel, markdowns) -> +holiday
    position (Walmart only). Comparing neighbours isolates what each block
    is worth, which a with/without-everything comparison cannot.
    """
    from forecasting.data.features import FeatureConfig, get_feature_config

    full = get_feature_config(ctx.dataset)
    roles = ctx.roles
    ladder = [
        ("lags", FeatureConfig(lags=full.lags, use_exog=False), []),
        (
            "rolling",
            FeatureConfig(lags=full.lags, lag_transforms=full.lag_transforms, use_exog=False),
            [],
        ),
        ("calendar", full.without_exog(), []),
    ]
    cols: list[str] = []
    for step, block in [
        ("static", roles.static_cat_cols),
        ("events", roles.dynamic_cat_cols),
        ("covariates", roles.dynamic_real_cols),
        ("holiday", roles.holiday_cols),
    ]:
        if not block:
            continue
        cols = cols + block
        ladder.append((step, full, cols))
    return ladder


def _nbeats(
    ctx: Context,
    variants: list[str] = ("nbeats", "nbeatsx"),
    max_steps: int = 500,
    loss: str = "mae",
    hparams: dict | None = None,
    model_name: str | None = None,
) -> dict:
    """`loss: mse` forecasts the mean rather than the median (labelled -mse).
    `hparams` (input_size, learning_rate, batch_size, scaler_type) reproduce a
    tuned configuration of a single variant, labelled `model_name`."""
    from forecasting.models.nbeats import build_nbeats, build_nbeatsx

    if model_name and len(variants) != 1:
        raise ValueError("model_name labels a single variant")
    suffix = "" if loss == "mae" else f"-{loss}"
    common = dict(
        freq=ctx.cfg["freq"], h=ctx.horizon, max_steps=max_steps, loss=loss, **(hparams or {})
    )
    builders = {
        "nbeats": (
            "N-BEATS" + suffix,
            lambda name: build_nbeats(name=name, **common),
        ),
        "nbeatsx": (
            "N-BEATSx" + suffix,
            lambda name: build_nbeatsx(
                futr_exog=ctx.futr_exog, stat_exog=ctx.stat_exog, name=name, **common
            ),
        ),
    }
    labels = {v: model_name or builders[v][0] for v in variants}
    return {labels[v]: builders[v][1](labels[v]) for v in variants}


def _neural(
    ctx: Context,
    model: str,
    max_steps: int = 500,
    loss: str = "mq",
    use_exog: bool = True,
    holiday_features: bool = False,
    point: str | None = None,
    hparams: dict | None = None,
    model_name: str | None = None,
) -> dict:
    """`loss: negbin` fits a negative binomial and forecasts its mean; `point:
    mean` does the same for DeepAR's own distribution. `hparams` (input_size,
    learning_rate, batch_size, scaler_type, ...) reproduce a tuned
    configuration, labelled `model_name`."""
    from run_neural_quantile import BUILDERS, DEEPAR_DISTRIBUTION, LABELS

    # Labelled after configuration, as run_neural_quantile does, so variants do
    # not overwrite one another in the results tables.
    label = model_name or (
        LABELS[model]
        + ("-huber" if loss == "huber_mq" else "")
        + ("-negbin" if loss == "negbin" else "")
        + ("-mean" if model == "deepar" and point == "mean" else "")
        + ("" if use_exog else "-noexog")
        + ("-hol" if holiday_features else "")
    )
    futr_exog = (ctx.futr_exog if use_exog else []) + (
        ctx.roles.holiday_cols if holiday_features else []
    )
    kwargs = dict(
        freq=ctx.cfg["freq"],
        h=ctx.horizon,
        futr_exog=futr_exog,
        stat_exog=ctx.stat_exog if use_exog else [],
        name=label,
        max_steps=max_steps,
        loss_type=loss,
    )
    if point:
        kwargs["point"] = point
    kwargs.update(hparams or {})
    if model == "nhits":
        kwargs["season_length"] = ctx.cfg["season_length"]
    if model == "deepar":
        kwargs.pop("loss_type")
        kwargs["distribution"] = DEEPAR_DISTRIBUTION[ctx.dataset]
    return {label: BUILDERS[model](**kwargs)}


def _chronos(
    ctx: Context,
    model_id: str = "amazon/chronos-2",
    context_length: int | None = None,
    batch_size: int = 256,
    point: str = "mean",
) -> dict:
    """Zero-shot foundation model; `point: median` is labelled -median."""
    from forecasting.models.chronos import build_chronos

    label = "Chronos-2" + ("-median" if point == "median" else "")
    return {
        label: build_chronos(
            freq=ctx.cfg["freq"],
            model_id=model_id,
            context_length=context_length,
            batch_size=batch_size,
            point=point,
            name=label,
        )
    }


KINDS = {
    "baselines": _baselines,
    "gbm": _gbm,
    "nbeats": _nbeats,
    "neural": _neural,
    "chronos": _chronos,
}

# Which kinds need covariates loaded into the panel.
NEEDS_EXOG = {"gbm", "nbeats", "neural"}


def build_entry(kind: str, params: dict, ctx: Context) -> dict:
    if kind not in KINDS:
        raise ValueError(f"unknown model kind '{kind}'; have {list(KINDS)}")
    return KINDS[kind](ctx, **params)
