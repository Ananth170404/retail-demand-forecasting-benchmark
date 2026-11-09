"""Probabilistic neural forecasters: TFT and NHITS.

Both emit quantiles rather than a point, so results carry pinball loss and
interval coverage alongside the usual accuracy metrics. Supersedes run_tft.py.

Usage:
    python experiments/run_neural_quantile.py --dataset m5 --model nhits --mode official
    python experiments/run_neural_quantile.py --dataset m5 --model tft --loss huber_mq
    python experiments/run_neural_quantile.py --dataset walmart --model nhits --no-exog
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import DATASET_CONFIG, load_panel, report, resolve_mode

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forecasting.evaluation.backtest import run_backtest  # noqa: E402
from forecasting.models.bitcn import build_bitcn  # noqa: E402
from forecasting.models.deepar import build_deepar  # noqa: E402
from forecasting.models.nhits import build_nhits  # noqa: E402
from forecasting.models.tft import build_tft  # noqa: E402

BUILDERS = {
    "tft": build_tft,
    "nhits": build_nhits,
    "bitcn": build_bitcn,
    "deepar": build_deepar,
}
LABELS = {"tft": "TFT", "nhits": "NHITS", "bitcn": "BiTCN", "deepar": "DeepAR"}

# DeepAR parameterises a distribution rather than regressing quantiles, so the
# family has to suit the target: M5 is over-dispersed counts with a large mass
# at zero, Walmart is continuous dollar revenue with heavy tails.
DEEPAR_DISTRIBUTION = {"m5": "NegativeBinomial", "walmart": "StudentT"}


def main(
    dataset: str,
    model: str = "nhits",
    mode: str = "rolling",
    max_steps: int = 500,
    loss_type: str = "mq",
    n_windows: int | None = None,
    max_series_per_state: int | None = 200,
    trim_zeros: bool = True,
    no_exog: bool = False,
) -> None:
    cfg = dict(DATASET_CONFIG[dataset])
    if n_windows:
        cfg["n_windows"] = n_windows
    window_mode, horizon = resolve_mode(cfg, mode)

    print(f"Loading {dataset}...")
    panel, roles = load_panel(
        dataset,
        trim_zeros=trim_zeros,
        max_series_per_state=max_series_per_state,
        with_exog=not no_exog,
    )

    import torch

    futr_exog = [] if no_exog else roles.dynamic_cat_cols + roles.dynamic_real_cols
    stat_exog = [] if no_exog else roles.static_cat_cols
    device = "GPU" if torch.cuda.is_available() else "CPU"

    # Name the run after its configuration so a HuberMQLoss variant does not
    # overwrite the MQLoss one in the results table.
    label = LABELS[model] + ("-huber" if loss_type == "huber_mq" else "")
    if no_exog:
        label += "-noexog"

    print(
        f"  model={label} | mode={mode} | h={horizon} | steps={max_steps} "
        f"| {device} | futr={len(futr_exog)} stat={len(stat_exog)}"
    )

    build_kwargs = dict(
        freq=cfg["freq"],
        h=horizon,
        futr_exog=futr_exog,
        stat_exog=stat_exog,
        name=label,
        max_steps=max_steps,
        loss_type=loss_type,
    )
    # NHITS derives its pooling schedule from the seasonality; the others have
    # no such parameter.
    if model == "nhits":
        build_kwargs["season_length"] = cfg["season_length"]
    # DeepAR takes a distribution family instead of a quantile loss.
    if model == "deepar":
        build_kwargs.pop("loss_type", None)
        build_kwargs["distribution"] = DEEPAR_DISTRIBUTION[dataset]
        print(f"  distribution={build_kwargs['distribution']}")

    factory = BUILDERS[model](**build_kwargs)
    results = run_backtest(
        model_factory=factory,
        panel=panel,
        h=horizon,
        season_length=cfg["season_length"],
        test_fraction=cfg["test_fraction"],
        n_windows=cfg["n_windows"],
        min_train_periods=cfg["min_train_periods"],
        mode=window_mode,
    )
    report(results, dataset, mode, tag=model)

    qcols = [c for c in results.columns if c.startswith(("pinball", "coverage"))]
    if qcols:
        print("\n-- probabilistic metrics (mean) --")
        print(results[qcols].mean().round(4).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASET_CONFIG), required=True)
    parser.add_argument("--model", choices=list(BUILDERS), default="nhits")
    parser.add_argument("--mode", choices=["rolling", "official"], default="rolling")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--loss", choices=["mq", "huber_mq"], default="mq")
    parser.add_argument("--n-windows", type=int, default=None)
    parser.add_argument(
        "--max-series-per-state", type=int, default=200, help="M5 sample size; 0 loads all series"
    )
    parser.add_argument("--no-trim", action="store_true")
    parser.add_argument("--no-exog", action="store_true")
    args = parser.parse_args()

    main(
        dataset=args.dataset,
        model=args.model,
        mode=args.mode,
        max_steps=args.max_steps,
        loss_type=args.loss,
        n_windows=args.n_windows,
        max_series_per_state=args.max_series_per_state,
        trim_zeros=not args.no_trim,
        no_exog=args.no_exog,
    )
