# Multi-Asset Volatility Forecasting

A leakage-controlled, offline-reproducible Week 1 research baseline for forecasting the next five trading days of realized volatility for six liquid ETFs.

## Research question

Can volatility information available at the close of day `t` improve forecasts of realized volatility over trading days `t+1` through `t+5`, and can later improvements translate into better risk control?

This first milestone implements the historical-volatility and EWMA baselines. It covers SPY, QQQ, IWM, TLT, GLD, and USO. `^VIX` is downloaded and quality-checked as a context series but is not included in baseline metrics.

## Definitions

Daily log returns use adjusted close:

```text
r_t = log(AdjClose_t / AdjClose_{t-1})
```

The annualized future five-day realized-volatility label is:

```text
RV_t = sqrt((252 / 5) * sum(r_{t+i}^2, i=1..5))
```

The label at `t` therefore uses exactly `t+1` through `t+5`; it excludes `r_t`. The historical baseline uses 21 valid returns available through `t`:

```text
HistoricalRV_t = sqrt(252 * mean(r_i^2 over the last 21 valid returns through t))
```

The EWMA baseline uses the recursive variance estimate with `lambda = 0.94`, initialized from the first 21 valid squared returns:

```text
EWMA variance_t = lambda * EWMA variance_{t-1} + (1 - lambda) * r_t^2
EWMA RV_t = sqrt(252 * EWMA variance_t)
```

Both baselines use only information available through `t`.

MAE and RMSE are evaluated on volatility. QLIKE is deliberately evaluated on variance:

```text
QLIKE = log(forecast_variance) + actual_variance / forecast_variance
```

Forecast variance is floored at the configured epsilon, and the number of floored observations is reported.

## Project layout

```text
configs/default.yaml       Central asset, file, horizon, window, and output settings
data/raw/                  User-provided or downloaded raw CSV files
data/quality/              Per-asset CSV and summary JSON quality reports
data/processed/            Baseline prediction detail in Parquet
outputs/tables/            Metrics and run metadata
outputs/figures/           SPY actual-vs-baseline comparison
scripts/download_data.py   Online Yahoo Finance acquisition only
scripts/build_dataset.py   Fully offline raw-to-results pipeline
src/                       Tested calculation and reporting modules
tests/                     Synthetic, network-free tests
```

## Environment

Python 3.10 or later is required. Install the declared dependencies in an environment of your choice:

```bash
python -m pip install -r requirements.txt
```

The implementation was tested with the already-installed global Python packages; no project virtual environment or package installation is required merely to run the checked-in tests in the current development environment.

## Data acquisition (online, separate command)

```bash
python scripts/download_data.py --config configs/default.yaml
```

The downloader requests each ticker separately through `yfinance` with `auto_adjust=False`, writes the original OHLCV and `Adj Close` fields, and updates `data/raw/manifest.json` after every asset. Retries are bounded. Successful files remain available if another ticker fails.

Expected file names are configured in `configs/default.yaml`:

```text
SPY.csv QQQ.csv IWM.csv TLT.csv GLD.csv USO.csv VIX.csv
```

Do not run the downloader when an offline reproduction is required. The build script never imports `yfinance` and never accesses the network.

## Offline build

After the raw files exist, run:

```bash
python scripts/build_dataset.py --config configs/default.yaml
```

This writes:

- `data/quality/asset_quality.csv`
- `data/quality/quality_summary.json`
- `data/processed/baseline_predictions.parquet`
- `outputs/tables/baseline_metrics.csv`
- `outputs/tables/run_metadata.json`
- `outputs/figures/spy_baseline_vs_actual.png`

The metadata records raw-file SHA-256 hashes, the strict label interval, variance-scale QLIKE, configuration, and row counts.

## Tests

```bash
python -m pytest -q
```

All tests use synthetic local data and require no network. They verify exact label timing, final-five-row missing labels, asset isolation, 21-return baseline windows, input sorting, variance-scale QLIKE, epsilon flooring, pooled `ALL` metrics, complete offline outputs, and clear failures for malformed CSVs.

## Data conventions and quality policy

- Source: Yahoo Finance through `yfinance` for the default downloader.
- Download date: recorded in `data/raw/manifest.json` by the downloader.
- Adjustment: returns use `Adj Close`; raw OHLC remain Yahoo's unadjusted fields because `auto_adjust=False`.
- Timezone/calendar: timestamps are converted to date-like, timezone-naive midnight values after parsing. Each asset retains its own observed trading dates; assets are not forced onto a common calendar.
- Missing values: no future values are filled. Missing or non-positive `Adj Close` is a hard error. Missing VIX volume is a warning.
- Missing `Adj Close`: the pipeline refuses to substitute `Close` by default. `allow_close_as_adjusted` should only be enabled after the data provider's adjustment convention has been independently confirmed.
- Duplicates and bad dates: duplicate asset-date rows and unparseable dates are hard errors.
- Ordering: unsorted input is sorted safely and recorded as a warning.
- Outliers: extreme returns, long calendar gaps, OHLC relationship anomalies, and USO-like events are reported but not automatically removed, winsorized, or corrected.

## Limitations

The future five-day RV label is a noisy proxy constructed from daily squared returns and overlaps across adjacent rows. Yahoo Finance can revise history and may impose throttling or availability limits. Adjusted close and raw OHLC are on different adjustment bases, so raw range estimators require additional care in later milestones. This baseline is descriptive rather than a walk-forward trained model and does not yet include EWMA, GARCH, HAR, VIX features, statistical inference, strategy execution, or portfolio allocation.
