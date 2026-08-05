# Multi-Asset Volatility Forecasting

A leakage-controlled, offline-reproducible Week 1 research baseline for forecasting the next five trading days of realized volatility for six liquid ETFs.

## Research question

Can volatility information available at the close of day `t` improve forecasts of realized volatility over trading days `t+1` through `t+5`, and can later improvements translate into better risk control?

This project implements the historical-volatility and EWMA baselines plus the HAR-RV statistical baseline, the GARCH(1,1) conditional-volatility model, a Ridge/Lasso penalized-linear model on expanded features, a LightGBM tree model on the same feature set, and an experimental HAR-VIX variant. It covers SPY, QQQ, IWM, TLT, GLD, and USO. `^VIX` is downloaded and quality-checked as a context series and is used in the experimental VIX-incremental comparison and as a feature in Ridge/LightGBM, not in the headline baselines.

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

Python 3.10 or later is required. A project virtual environment is provided (`.venv`, Python 3.12, Windows-native layout). To create it from scratch and install the pinned dependencies:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

The pins in `requirements.txt` are load-bearing: `lightgbm>=3.3,<4.0` (4.x crashes on Windows/Anaconda) and `scikit-learn>=1.0,<1.6` (lightgbm 3.3.x calls the `force_all_finite` keyword that sklearn 1.6+ removed). Run the checks with the venv interpreter:

```bash
.venv/Scripts/python.exe -m pytest -q
```

All tests use synthetic local data and require no network.

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
- `outputs/tables/yearly_metrics.csv`
- `outputs/tables/dm_tests.csv`
- `outputs/tables/ewma_lambda_selection.csv`
- `outputs/tables/har_coefficients.csv`
- `outputs/tables/har_vix_coefficients.csv`
- `outputs/tables/garch_params.csv`
- `outputs/tables/ridge_params.csv`
- `outputs/tables/ridge_lambda_selection.csv`
- `outputs/tables/lightgbm_params.csv`
- `outputs/tables/lightgbm_importance.csv`
- `outputs/tables/worst_error_dates.csv`
- `outputs/tables/test_model_comparison.csv`
- `outputs/tables/vix_incremental_comparison.csv`
- `outputs/tables/asset_robustness.csv`
- `outputs/tables/regime_robustness.csv`
- `outputs/tables/strategy_metrics.csv`
- `outputs/tables/transmission_waterfall.csv`
- `outputs/tables/portfolio_metrics.csv`
- `outputs/tables/alt_label_metrics.csv`
- `outputs/tables/alt_label_dm.csv`
- `outputs/tables/strategy_cost_sensitivity.csv`
- `outputs/tables/run_metadata.json`
- `outputs/figures/spy_baseline_vs_actual.png`

The metadata records raw-file SHA-256 hashes, the strict label interval, variance-scale QLIKE, configuration, and row counts.

## Walk-forward evaluation

Forecast dates are split into three configured segments: train through `2021-12-31`, validation from `2022-01-01` through `2023-12-31`, and test from `2024-01-01` onward. The last five forecast dates per asset are removed from train and validation so their `t+1` through `t+5` label windows cannot cross a segment boundary. Historical rolling and EWMA states are computed continuously from prior observations and are not restarted at validation or test boundaries. Segment, asset, forecast, and pooled `ALL` metrics are written to `outputs/tables/walk_forward_metrics.csv`.

## Diebold-Mariano comparison

Pairs of model forecasts are compared using paired QLIKE losses, with MAE on the volatility scale as a sensitivity check. The differential is defined as `loss_model_a - loss_model_b`, so a negative value favors model A. For pooled `ALL`, loss differences are first averaged across valid assets on each date; the resulting date series is then sorted and evaluated with Bartlett/Newey-West HAC lag 4. This avoids treating different assets as adjacent time observations. Results by segment, asset, and pooled `ALL` are written to `outputs/tables/dm_tests.csv`; a p-value indicates evidence against equal mean loss, not a guarantee of future superiority.

## Loss sensitivity

QLIKE is the primary DM loss and is evaluated on the variance scale. MAE is also evaluated on the volatility scale as a sensitivity analysis. Both use `loss_ewma - loss_historical`, the same paired observations, date-level pooled `ALL` aggregation, and HAC lag 4. In the test segment pooled `ALL` result, QLIKE has mean difference `-0.039615`, DM statistic `-2.909619`, and p-value `0.003619`; MAE has mean difference `-0.000996`, DM statistic `-0.842524`, and p-value `0.399495`.

## EWMA lambda selection

Candidate EWMA lambdas `0.90`, `0.94`, `0.97`, and `0.99` are evaluated using only validation-segment pooled QLIKE. The selected lambda is `0.94`; ties would be resolved deterministically in favor of the smaller lambda. After validation selection, the chosen EWMA forecast is locked before test evaluation. Candidate scores and the selected flag are written to `outputs/tables/ewma_lambda_selection.csv`; test observations are not used for lambda selection.

Test-regime robustness is written to `outputs/tables/regime_robustness.csv`. Regimes are defined only for the test segment using pooled future five-day realized-volatility tertiles; they are descriptive and do not affect fitting or model selection. In pooled `ALL`, the Ridge model ranks best across all three metrics in low volatility, while HAR-RV and GARCH rank best in medium and high volatility.

The HAR-RV baseline uses daily, five-day, and 22-day historical squared-return features available through each forecast date. It fits one OLS model per asset on the cleaned train segment only, with `future_rv_5d^2` as the variance-scale target. Coefficients are locked before validation and test evaluation; negative variance forecasts are clipped at zero before taking the square root. Predictions and metrics include `har_rv`, and coefficients are written to `outputs/tables/har_coefficients.csv`.

The experimental HAR-VIX variant adds the same-day log VIX level to the HAR features. Because a linear model on the variance scale can imply negative variance, HAR-VIX is instead fitted on `log(future_rv_5d^2)` and recovered with an exponential plus a lognormal smearing term (`0.5 * sigma^2` from training residuals), so predictions are always strictly positive and no clipping is required. Standardization and coefficients are locked from the train segment. In the pooled `ALL` comparison, VIX improves QLIKE within train and validation (train mean loss difference `-0.033712`, DM statistic `-4.373`, p `0.0000`; validation `-0.040212`, DM `-3.136`, p `0.0017`) but the advantage does not persist out of sample: test mean loss difference `+0.013485`, DM `+0.863`, p `0.388`. The test result is heterogeneous across assets — SPY improves significantly (p `0.005`) while GLD degrades significantly (p `0.007`), and the remaining assets are statistically indistinguishable. Incremental DM results are written to `outputs/tables/vix_incremental_comparison.csv` and coefficients to `outputs/tables/har_vix_coefficients.csv`. This comparison is treated as evidence about the role of implied volatility rather than as part of the headline baseline comparison.

The GARCH(1,1) model uses Gaussian maximum likelihood on centered returns, fitted once per asset on the train segment only. Returns are scaled by the training-sample standard deviation before estimation for optimizer conditioning, and `omega` and `mu` are rescaled back so all parameters (`omega`, `alpha`, `beta`, `mu`) are locked before validation and test evaluation. The fitted coefficients then act as a fixed conditional-variance filter: the recursive variance path continues through validation and test using only past observations, and the five-day forecast iterates `h_k = omega + (alpha + beta) * h_{k-1}` from the one-step-ahead variance. Parameters, including the `alpha + beta` persistence and the optimizer convergence flag, are written to `outputs/tables/garch_params.csv`; all six assets fit with stationary (persistence < 1) and converged parameters in the current run. In the test segment pooled `ALL` result, GARCH has MAE `0.063940`, RMSE `0.097671`, and QLIKE `-2.304942`, the lowest among the formal models. Its QLIKE advantage over the historical baseline is significant (test mean difference `-0.073286`, DM `-3.720`, p `0.0002`), while the smaller edge over HAR-RV (`-0.004014`, DM `-0.630`, p `0.529`) is not statistically significant. Per-asset first places in the test segment are split between GARCH and HAR-RV, with EWMA winning a minority.

The Ridge/Lasso model fits a penalized linear model on `log(future_rv_5d^2)` with fourteen features known through the forecast date: the three HAR variances, Parkinson 5- and 22-day ranges, Garman-Klass 22-day range, 5- and 21-day cumulative returns, the 21-day fraction of negative returns, distance to the 21-day moving average, 21-day drawdown, the 21-day volume ratio, the same-day log VIX, and the 5-day log-VIX change. Features are z-scored per asset from the train segment; the penalty is chosen per asset on the validation segment by pooled QLIKE (Ridge uses a closed-form solution, Lasso a coordinate-descent path); the target is recovered with an exponential plus the lognormal smearing term so forecasts stay positive. Selected lambdas, coefficients, and smearing offsets are written to `outputs/tables/ridge_params.csv` and the per-lambda QLIKE grid to `outputs/tables/ridge_lambda_selection.csv`. In the test segment pooled `ALL` result, Ridge has QLIKE `-2.298476`, statistically indistinguishable from HAR-RV (`+0.002451`, DM `0.142`, p `0.887`) and GARCH, and significantly better than the historical baseline (DM `-2.359`, p `0.018`). The expanded feature set therefore does not add out-of-sample QLIKE above HAR-RV, even though it was far better in-sample (train mean difference vs HAR-RV `-0.064445`, p `<0.0001`); MAE and RMSE are also worse than HAR-RV. The gains concentrate in low volatility, where Ridge ranks best on all three metrics, and reverse sharply in high volatility, where its RMSE is the worst.

The LightGBM model uses the exact same 14 features and log-variance target as Ridge, making it a direct linear-vs-nonlinear comparison. Hyperparameters (`num_leaves × learning_rate × n_estimators = {8,31} × {0.05,0.1} × {100,300}`) are selected per asset on the validation segment by pooled QLIKE and locked; `random_state=42` and `deterministic=True` make runs reproducible. All six assets select the most regularized configuration (`num_leaves=8, learning_rate=0.05, n_estimators=100`). In the test segment pooled `ALL`, LightGBM has the best MAE (`0.05992`) and RMSE (`0.09616`) of any model and is significantly better than HAR-RV and GARCH on MAE (p `0.001` / `0.011`), while its QLIKE (`-2.28483`) stays statistically indistinguishable from the top tier (vs HAR p `0.394`, vs GARCH p `0.266`). Feature importance (gain) is dominated by the same-day log VIX (`27.5%` cross-asset mean share) and the Parkinson/Garman-Klass range features (`~34%` combined), while the HAR daily/weekly features are barely used — the opposite of the Ridge finding. LightGBM is also the most robust model: it stays essentially unchanged when 2020 is excluded from training (test MAE `0.0601` vs `0.0599`) and is significantly better than HAR under both alternative labels. Diagnostics (per-asset hyperparameters, gain/split importance, worst-error dates) are written to `outputs/tables/lightgbm_params.csv`, `outputs/tables/lightgbm_importance.csv`, and `outputs/tables/worst_error_dates.csv`. In volatility targeting it is the exception to the accuracy-to-risk ordering: despite the best point MAE/RMSE it has the largest average target deviation (`1.12 pp`), consistent with shrinkage of tree forecasts toward the asset average.

## Volatility targeting and risk allocation

The forecast models are applied to a single-asset volatility-targeting strategy and to a weekly multi-asset risk allocation portfolio, both evaluated only on the test segment (2024 onward) where every forecaster is out of sample. The target strategy takes `weight_t = clip(target_vol / forecast_vol_t, 0, max_leverage)` at the close of `t` with `target_vol = 10%`, `max_leverage = 1.5`, no shorting, and the position first earns the return of `t+1`; transactions are charged at 10 bps per unit of position change. Per-asset realized volatility is close to target for every forecaster: average absolute deviation from the 10% target over the six assets is `0.38 pp` for GARCH, `0.48 pp` for HAR, `0.52 pp` for EWMA, `0.77 pp` for Ridge, and `1.00 pp` for the historical baseline. The pooled-`ALL` strategy lines have much lower realized volatility (about 6%) than the per-asset numbers because six largely uncorrelated targeted legs diversify across assets; the full 100%-invested reference has realized volatility 12.3%.

The transmission decomposition (`outputs/tables/transmission_waterfall.csv`) adds frictions in order to isolate where forecast accuracy is or is not transferred. The `1.5` leverage cap is almost never binding and changes risk outcomes by less than one basis point; adding one further day of execution lag raises the average per-asset target deviation by roughly `0.05–0.14 pp`; charging 10 bps of cost trims net annualized return by about `0.7–1.2 pp` while leaving risk control unchanged. So in this design the main gap between the forecast and realized risk outcome is small and is driven by the forecast level itself rather than by the frictions.

The weekly portfolio layer (`outputs/tables/portfolio_metrics.csv`) compares equal weighting, inverse-21-day-historical-volatility weighting, and inverse-GARCH-forecast weighting over the same test segment. Pure risk allocation, no return prediction. The inverse-forecast portfolio has the lowest realized volatility (`10.68%`), the lowest maximum drawdown (`10.7%`), the lowest volatility-of-volatility (`3.67 pp`), and the highest Sharpe (`1.69`), with the inverse-historical portfolio a close second (`10.76%`, `11.2%`, `3.88 pp`, Sharpe `1.64`) and equal weighting clearly behind (`12.35%`, `13.7%`, `5.33 pp`). The differences are modest and measured over a single test window, so they are treated as suggestive rather than definitive evidence that forecast-based risk allocation is more stable.

## Robustness checks

Four checks test whether the main conclusions survive changes to the label proxy, the training sample, the cost assumption, and the market-state split. Results are written to `outputs/tables/alt_label_metrics.csv`, `outputs/tables/alt_label_dm.csv`, `outputs/tables/strategy_cost_sensitivity.csv`, and for the 2020-free run to `outputs/robustness/ex2020/` (run `python scripts/build_dataset.py --exclude-year 2020 --output-prefix outputs/robustness/ex2020`).

- **Alternative labels.** The fixed out-of-sample forecasts are re-scored against Parkinson and Garman-Klass range-based realized-volatility labels (using the same `t+1` to `t+5` window). Under the Parkinson label the main ranking is unchanged: GARCH and Ridge remain in an indistinguishable top tier with HAR (DM vs HAR p `0.27` and `0.43`). Under the Garman-Klass label the ordering splits: GARCH and Ridge are significantly better than HAR (p `0.002` and `0.011`), while HAR, whose OLS target is specifically the squared-return label, drops to last. HAR is therefore label-sensitive in a way that the return-process model (GARCH) and the regularized feature model (Ridge) are not.
- **Training-sample sensitivity.** Re-running the pipeline with 2020 excluded from training (16,452 versus 17,970 training observations) changes the 2024+ ordering materially: Ridge keeps the best QLIKE (`-2.290`) with LightGBM close behind (`-2.281`), and both are significantly better than HAR and GARCH (DM `-3.35` to `-3.50` vs HAR and `-2.33` to `-2.76` vs GARCH, all p `< 0.02`), while HAR and GARCH degrade sharply (test MAE about `0.084-0.087` versus `0.064-0.065` when trained with 2020). LightGBM is the only model essentially unchanged by excluding 2020 (test MAE `0.0599` -> `0.0601`). The headline finding that the top tier is statistically indistinguishable therefore depends on the 2020 training regime and should be reported with that caveat.
- **Cost doubling.** Doubling the transaction cost from 10 to 20 bps leaves realized volatility unchanged and trims net annualized return by about `0.5-1.2 pp`; the risk ranking and the transmission-decomposition conclusions are unchanged, and the lower-turnover GARCH forecast loses the least to costs.
- **Market-state regimes** are already reported in `outputs/tables/regime_robustness.csv`.

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

The future five-day RV label is a noisy proxy constructed from daily squared returns and overlaps across adjacent rows. Yahoo Finance can revise history and may impose throttling or availability limits. Adjusted close and raw OHLC are on different adjustment bases, so raw range estimators require additional care in later milestones. The project is descriptive rather than a fully walk-forward-trained model family (parameters are locked once, not re-estimated on a rolling window); it does not yet include non-overlapping-label inference, model ensembles, or a rolling/expanding retraining protocol.
