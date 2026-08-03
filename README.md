# Multi-Asset Volatility Forecasting

A leakage-controlled, offline-reproducible Week 1 research baseline for forecasting the next five trading days of realized volatility for six liquid ETFs.

## Research question

Can volatility information available at the close of day `t` improve forecasts of realized volatility over trading days `t+1` through `t+5`, and can later improvements translate into better risk control?

This first milestone implements the historical-volatility and EWMA baselines plus the HAR-RV statistical baseline and an experimental HAR-VIX variant. It covers SPY, QQQ, IWM, TLT, GLD, and USO. `^VIX` is downloaded and quality-checked as a context series and is used only in the experimental VIX-incremental comparison, not in the headline baseline metrics.

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
- `outputs/tables/walk_forward_metrics.csv`
- `outputs/tables/dm_tests.csv`
- `outputs/tables/ewma_lambda_selection.csv`
- `outputs/tables/har_coefficients.csv`
- `outputs/tables/har_vix_coefficients.csv`
- `outputs/tables/test_model_comparison.csv`
- `outputs/tables/vix_incremental_comparison.csv`
- `outputs/tables/asset_robustness.csv`
- `outputs/tables/regime_robustness.csv`
- `outputs/tables/run_metadata.json`
- `outputs/figures/spy_baseline_vs_actual.png`

The metadata records raw-file SHA-256 hashes, the strict label interval, variance-scale QLIKE, configuration, and row counts.

## Walk-forward evaluation

Forecast dates are split into three configured segments: train through `2021-12-31`, validation from `2022-01-01` through `2023-12-31`, and test from `2024-01-01` onward. The last five forecast dates per asset are removed from train and validation so their `t+1` through `t+5` label windows cannot cross a segment boundary. Historical rolling and EWMA states are computed continuously from prior observations and are not restarted at validation or test boundaries. Segment, asset, forecast, and pooled `ALL` metrics are written to `outputs/tables/walk_forward_metrics.csv`.

## Diebold-Mariano comparison

The EWMA and historical baselines are compared using paired QLIKE losses. The differential is defined as `loss_ewma - loss_historical`, so a negative value favors EWMA. For pooled `ALL`, loss differences are first averaged across valid assets on each date; the resulting date series is then sorted and evaluated with Bartlett/Newey-West HAC lag 4. This avoids treating different assets as adjacent time observations. Results by segment, asset, and pooled `ALL` are written to `outputs/tables/dm_tests.csv`; a p-value indicates evidence against equal mean loss, not a guarantee of future superiority.

## Loss sensitivity

QLIKE is the primary DM loss and is evaluated on the variance scale. MAE is also evaluated on the volatility scale as a sensitivity analysis. Both use `loss_ewma - loss_historical`, the same paired observations, date-level pooled `ALL` aggregation, and HAC lag 4. In the test segment pooled `ALL` result, QLIKE has mean difference `-0.039615`, DM statistic `-2.909619`, and p-value `0.003619`; MAE has mean difference `-0.000996`, DM statistic `-0.842524`, and p-value `0.399495`.

## EWMA lambda selection

Candidate EWMA lambdas `0.90`, `0.94`, `0.97`, and `0.99` are evaluated using only validation-segment pooled QLIKE. The selected lambda is `0.94`; ties would be resolved deterministically in favor of the smaller lambda. After validation selection, the chosen EWMA forecast is locked before test evaluation. Candidate scores and the selected flag are written to `outputs/tables/ewma_lambda_selection.csv`; test observations are not used for lambda selection.

Test-regime robustness is written to `outputs/tables/regime_robustness.csv`. Regimes are defined only for the test segment using pooled future five-day realized-volatility tertiles; they are descriptive and do not affect fitting or model selection. In pooled `ALL`, the historical baseline ranks best in low volatility, while HAR-RV ranks best across MAE, RMSE, and QLIKE in medium and high volatility.

The HAR-RV baseline uses daily, five-day, and 22-day historical squared-return features available through each forecast date. It fits one OLS model per asset on the cleaned train segment only, with `future_rv_5d^2` as the variance-scale target. Coefficients are locked before validation and test evaluation; negative variance forecasts are clipped at zero before taking the square root. Predictions and metrics include `har_rv`, and coefficients are written to `outputs/tables/har_coefficients.csv`.

The experimental HAR-VIX variant adds the same-day log VIX level to the HAR features. Because a linear model on the variance scale can imply negative variance, HAR-VIX is instead fitted on `log(future_rv_5d^2)` and recovered with an exponential plus a lognormal smearing term (`0.5 * sigma^2` from training residuals), so predictions are always strictly positive and no clipping is required. Standardization and coefficients are locked from the train segment. In the pooled `ALL` comparison, VIX improves QLIKE within train and validation (train mean loss difference `-0.033712`, DM statistic `-4.373`, p `0.0000`; validation `-0.040212`, DM `-3.136`, p `0.0017`) but the advantage does not persist out of sample: test mean loss difference `+0.013485`, DM `+0.863`, p `0.388`. The test result is heterogeneous across assets — SPY improves significantly (p `0.005`) while GLD degrades significantly (p `0.007`), and the remaining assets are statistically indistinguishable. Incremental DM results are written to `outputs/tables/vix_incremental_comparison.csv` and coefficients to `outputs/tables/har_vix_coefficients.csv`. This comparison is treated as evidence about the role of implied volatility rather than as part of the headline baseline comparison.

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

The future five-day RV label is a noisy proxy constructed from daily squared returns and overlaps across adjacent rows. Yahoo Finance can revise history and may impose throttling or availability limits. Adjusted close and raw OHLC are on different adjustment bases, so raw range estimators require additional care in later milestones. The project is descriptive rather than a fully walk-forward-trained model family and does not yet include GARCH, Ridge/Lasso, tree models, range-based labels, non-overlapping-label inference, strategy execution, or portfolio allocation.
