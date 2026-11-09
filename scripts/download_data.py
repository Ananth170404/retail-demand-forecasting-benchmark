"""Download the M5 and Walmart Store Sales datasets.

Kaggle hosts the training data; the M5 test labels and official scoring weights
were never added to Kaggle, so `m5-labels` fetches them from Nixtla's public
mirror of the M5 release (verified: its training data is value-identical to
Kaggle's, and seasonal naive scored on its labels and weights reproduces the
published level-12 WRMSSE of 1.176).

Prerequisites for the Kaggle downloads:
    pip install kaggle
    Place your Kaggle API token at ~/.kaggle/kaggle.json (Account -> Create New API Token).
    Accept the competition rules on kaggle.com for both competitions below
    (the API returns 403 until you do this manually in the browser once):
        https://www.kaggle.com/c/m5-forecasting-accuracy/rules
        https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/rules

Usage:
    python scripts/download_data.py --dataset m5
    python scripts/download_data.py --dataset m5-labels
    python scripts/download_data.py --dataset walmart
    python scripts/download_data.py --dataset all
"""

import argparse
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

COMPETITIONS = {
    "m5": "m5-forecasting-accuracy",
    "walmart": "walmart-recruiting-store-sales-forecasting",
}

M5_LABELS_URL = "https://github.com/Nixtla/m5-forecasts/raw/main/datasets/m5.zip"
# Test labels for the public (validation) and private (evaluation) leaderboard
# periods, and the official per-series weights used by WRMSSE for each.
M5_LABEL_FILES = [
    "sales_test_validation.csv",
    "sales_test_evaluation.csv",
    "weights_validation.csv",
    "weights_evaluation.csv",
]


def _extract(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)


def download(dataset_key: str) -> None:
    competition = COMPETITIONS[dataset_key]
    dest = DATA_ROOT / dataset_key
    dest.mkdir(parents=True, exist_ok=True)

    cmd = ["kaggle", "competitions", "download", "-c", competition, "-p", str(dest)]
    subprocess.run(cmd, check=True)

    # Walmart ships nested per-file zips; M5 ships one flat zip.
    for _ in range(2):
        for zip_path in dest.glob("*.zip"):
            _extract(zip_path, dest)
            zip_path.unlink()

    print(f"[{dataset_key}] downloaded and extracted to {dest}")


def download_m5_labels() -> None:
    dest = DATA_ROOT / "m5"
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "m5.zip"
        print(f"[m5-labels] fetching {M5_LABELS_URL}")
        urllib.request.urlretrieve(M5_LABELS_URL, archive)
        with zipfile.ZipFile(archive) as zf:
            for name in M5_LABEL_FILES:
                (dest / name).write_bytes(zf.read(name))

    # Cheap sanity check that the mirror still serves what we validated.
    test = pd.read_csv(dest / "sales_test_evaluation.csv")
    days = [c for c in test.columns if c.startswith("d_")]
    if len(test) != 30_490 or (days[0], days[-1]) != ("d_1942", "d_1969"):
        raise RuntimeError(
            f"unexpected sales_test_evaluation.csv: {len(test)} rows, days {days[0]}..{days[-1]}"
        )
    print(f"[m5-labels] wrote {', '.join(M5_LABEL_FILES)} to {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["m5", "m5-labels", "walmart", "all"], required=True)
    args = parser.parse_args()

    if args.dataset in ("m5-labels", "all"):
        download_m5_labels()
    if args.dataset == "m5-labels":
        sys.exit(0)

    targets = list(COMPETITIONS) if args.dataset == "all" else [args.dataset]
    for key in targets:
        try:
            download(key)
        except subprocess.CalledProcessError as e:
            print(
                f"[{key}] download failed ({e}). Confirm you accepted the competition "
                f"rules at https://www.kaggle.com/c/{COMPETITIONS[key]}/rules and that "
                f"~/.kaggle/kaggle.json is present.",
                file=sys.stderr,
            )
            sys.exit(1)
