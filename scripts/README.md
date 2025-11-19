
---

## Scripts overview (what each scripts does)

### `scripts/compat_from_clean_parquet.py`
**Purpose:** Convert `data/clean_zori.parquet` to the canonical, model-ready parquet used by all other scripts (datetime index, monthly frequency, value column set to `zori_smoothed_seasonal`).

- **Inputs:** `data/clean_zori.parquet`
- **Outputs:** `data/processed/zori_smoothed_seasonal.parquet`
- **Key args:** `--in`, `--out`

---

### `scripts/make_forecasts.py`
**Purpose:** Run a conservative SARIMAX and write a **per-ZIP** CSV of the next 12 months. Skips ZIPs with too little data, logs a one-line summary per ZIP.

- **Reads:** `data/processed/zori_smoothed_seasonal.parquet`
- **Writes:** `data/processed/forecasts/zip-<ZIP>/forecast.csv`
- **Forecast horizon:** default 12 months; confidence interval columns `ci95_lo`, `ci95_hi`
- **Key args:**  
  - `--parquet <path>`: model-ready parquet  
  - `--geo-type zip|state` (we use `zip`)  
  - `--geos "01002 01103 …"` (omit to process all zips)  
  - `--steps 12`  
  - `--out-dir data/processed/forecasts`  
  - `--no-figures` to suppress plots

---

### `scripts/volatility_index.py`
**Purpose:** Compute a “volatility index” per ZIP = rolling std of monthly % change (12-month window by default). Used for the final forecasts.

- **Reads:** `data/processed/zori_smoothed_seasonal.parquet`
- **Writes:** `data/processed/volatility/volatility_index.csv`
- **Key args:** `--parquet`, `--out`, `--window` (default 12)

---

### `scripts/export_forecast_table.py`
**Purpose:** Combine all the **per-ZIP forecast.csv** files into a **single handoff CSV** with columns:
`ZipCode, Forecast_Month, Forecast_Rent, Volatility`.

- **Reads:**  
  - Forecasts under `data/processed/forecasts/zip-*/forecast.csv`  
  - Optional `data/processed/volatility/volatility_index.csv`
- **Writes:** `data/Sarimax_Forecasts.csv`
- **Notes:**  
  - `Forecast_Month` ∈ {3, 6, 9, 12}  
  - `Forecast_Rent` = mean forecast at that horizon  
  - `Volatility` is looked up per ZIP (blank if not provided)

---

### `scripts/evaluate_rmse.py`
**Purpose:** Backtest SARIMAX on a **sample** of ZIPs and write per-ZIP RMSE; prints overall mean RMSE for quick comparison (e.g., vs Exponential Smoothing).

- **Reads:** `data/processed/zori_smoothed_seasonal.parquet`
- **Writes:** `data/processed/metrics/rmse_sarimax_sample100.csv` (path may vary)
- **Key args:**  
  - `--sample 100` (number of ZIPs)  
  - `--horizons 3 6 9 12`  
  - `--out <csv>`

---


---

## Typical workflow (one paragraph)

1) Build the model-ready parquet with `compat_from_clean_parquet.py`.  
2) Run `make_forecasts.py` to produce per-ZIP 12-month forecasts (skips ZIPs with no usable series).  
3) (Optional) Generate `volatility_index.csv`.  
4) Export a single handoff file with `export_forecast_table.py`.  
5) (Optional) Evaluate SARIMAX error with `evaluate_rmse.py` on a 100-ZIP sample.

---

## Repro tips / troubleshooting

- “**no data**” messages mean a ZIP had an ultra-short or missing series and was skipped.  
- Non-stationary start warnings are normal; the script falls back to safe baselines if CIs are non-finite.  
- All scripts are idempotent: re-running safely overwrites outputs under `data/processed/…`.

---


