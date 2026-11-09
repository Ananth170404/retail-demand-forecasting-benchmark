# Data

Two Kaggle datasets of Walmart sales, at very different grains:

| Dataset | Role | Series | Frequency | Key covariates |
|---|---|---|---|---|
| [M5 Forecasting - Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy) | Primary benchmark, scored on the competition's own test | 30,490 item x store (42,840 counting all 12 aggregation levels) | Daily | sell price, calendar events, SNAP |
| [Walmart Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting) | Second benchmark: weekly, holiday-driven | 3,331 store x dept | Weekly | holidays, MarkDowns, CPI, Unemployment, Fuel_Price |

We passed on the plain "Store Item Demand Forecasting" Kaggle dataset (10 stores x 50
items, date and sales only): too simple to exercise covariate-aware models such as TFT
or N-BEATSx, and too small for the demand-class analysis to mean much.

## Setup

1. `pip install kaggle`
2. Create an API token at kaggle.com -> Account -> "Create New API Token", save as `~/.kaggle/kaggle.json`
3. Accept the competition rules in-browser (the API 403s until you do, even with a valid token):
   - https://www.kaggle.com/c/m5-forecasting-accuracy/rules
   - https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/rules
4. `python scripts/download_data.py --dataset all`

`--dataset all` also fetches the M5 test labels and scoring weights (`--dataset m5-labels`
alone), which Kaggle never published; they come from Nixtla's public mirror of the
official M5 release.

This populates `data/raw/m5/` and `data/raw/walmart/` (both git-ignored — see `.gitignore`).

## Loading

```python
from forecasting.data.loaders import load_dataset

product_df, roles = load_dataset("m5")       # or "walmart"
```

Both loaders return the same shape: `product_df` with `demand_date`, `product_id`,
`total_demand_qty`, plus encoded static/dynamic categoricals and dynamic real columns;
`roles` (a `ColumnRoles`) records which columns play which part. See
[src/forecasting/data/loaders.py](../src/forecasting/data/loaders.py).

`load_m5` defaults to a subsample (`max_series_per_state=200`) since the full dataset is
~58M rows after melting to long format — pass `max_series_per_state=None` for the full run.
