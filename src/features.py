"""Leakage-safe historical features and baseline forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_historical_volatility_baseline(frame: pd.DataFrame, *, window: int = 21, annualization_factor: float = 252.0, return_column: str = "log_return", output_column: str = "historical_rv_21d") -> pd.DataFrame:
    if window <= 0:
        raise ValueError("window must be positive")
    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    rolling_variance = data.groupby("asset", sort=False)[return_column].transform(lambda returns: returns.pow(2).rolling(window=window, min_periods=window).mean())
    data[output_column] = np.sqrt(annualization_factor * rolling_variance)
    return data


def add_ewma_volatility_baseline(frame: pd.DataFrame, *, decay: float = 0.94, min_periods: int = 21, annualization_factor: float = 252.0, return_column: str = "log_return", output_column: str = "ewma_rv") -> pd.DataFrame:
    if not 0 < decay < 1:
        raise ValueError("decay must be between 0 and 1")
    if min_periods <= 0:
        raise ValueError("min_periods must be positive")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    def ewma_variance(returns: pd.Series) -> pd.Series:
        values = returns.to_numpy(dtype=float); result = np.full(len(values), np.nan); variance = np.nan; valid_count = 0
        for index, value in enumerate(values):
            if not np.isfinite(value): continue
            valid_count += 1; squared = value * value
            if valid_count == min_periods: variance = float(values[np.isfinite(values)][:valid_count].dot(values[np.isfinite(values)][:valid_count]) / min_periods)
            elif valid_count > min_periods: variance = decay * variance + (1.0 - decay) * squared
            if valid_count >= min_periods: result[index] = variance
        return pd.Series(result, index=returns.index)
    variance = data.groupby("asset", sort=False)[return_column].transform(ewma_variance)
    data[output_column] = np.sqrt(annualization_factor * variance)
    return data


def add_ewma_volatility_candidates(frame: pd.DataFrame, *, decays: tuple[float, ...] | list[float], min_periods: int = 21, annualization_factor: float = 252.0, return_column: str = "log_return", output_prefix: str = "ewma_rv_lambda_") -> pd.DataFrame:
    values = [float(decay) for decay in decays]
    if not values or len(set(values)) != len(values) or not all(np.isfinite(values)): raise ValueError("decays must be a non-empty list of unique finite values")
    if not all(0 < decay < 1 for decay in values): raise ValueError("each decay must be between 0 and 1")
    data = frame.copy()
    for decay in values:
        data = add_ewma_volatility_baseline(data, decay=decay, min_periods=min_periods, annualization_factor=annualization_factor, return_column=return_column, output_column=f"{output_prefix}{decay:g}")
    return data


def add_vix_level(frame: pd.DataFrame, vix_frame: pd.DataFrame, *, output_column: str = "log_vix") -> pd.DataFrame:
    """Exact-date left join of same-day positive VIX adjusted close."""
    vix = vix_frame.loc[:, ["date", "adj_close"]].rename(columns={"adj_close": "vix_close"}).drop_duplicates("date")
    if (vix["vix_close"].dropna() <= 0).any():
        raise ValueError("VIX adjusted close must be positive")
    data = frame.merge(vix, on="date", how="left", validate="many_to_one")
    data[output_column] = np.log(data["vix_close"])
    return data


def _train_log_variance(actual: np.ndarray, floor: float) -> np.ndarray:
    return np.log(np.maximum(np.square(actual), floor))


def fit_har_vix_by_asset(
    frame: pd.DataFrame,
    *,
    output_column: str = "har_vix_rv",
    log_floor: float = 1e-12,
    smearing: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit train-only HAR-X OLS on log variance with train-standardized log VIX.

    The target is ``log(future_rv_5d^2)``, so a linear prediction in log space
    can never imply a negative variance and no zero-clipping is required. An
    optional lognormal smearing term ``0.5 * sigma^2`` (estimated from training
    residuals) levels the recovered variance expectation before converting back
    to volatility scale. Coefficients, standardization, and the smearing term
    are all locked from the train segment only.
    """
    if log_floor <= 0:
        raise ValueError("log_floor must be positive")
    features = ["har_daily_rv", "har_weekly_rv", "har_monthly_rv", "log_vix_z"]
    data = frame.copy()
    data[output_column] = np.nan
    data["har_vix_logvar"] = np.nan
    rows: list[dict[str, float | int | str]] = []
    train_vix = data.loc[(data["segment"] == "train") & data["log_vix"].notna(), "log_vix"]
    mean, std = float(train_vix.mean()), float(train_vix.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("training log VIX standard deviation must be positive")
    data["log_vix_z"] = (data["log_vix"] - mean) / std
    for asset, group in data.groupby("asset", sort=True):
        train = group.loc[
            (group["segment"] == "train")
            & group[["future_rv_5d", *features]].notna().all(axis=1)
        ]
        if len(train) < len(features) + 1:
            raise ValueError(f"insufficient HAR-VIX training observations for {asset}")
        x = np.column_stack([np.ones(len(train)), train[features].to_numpy()])
        y = _train_log_variance(train["future_rv_5d"].to_numpy(), floor=log_floor)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        sigma2 = float(np.mean(np.square(y - x.dot(beta))))
        smearing_offset = 0.5 * sigma2 if smearing else 0.0
        valid = group[features].notna().all(axis=1)
        predictor = np.column_stack(
            [np.ones(int(valid.sum())), group.loc[valid, features].to_numpy()]
        )
        logvar = predictor.dot(beta) + smearing_offset
        variance = np.exp(logvar)
        data.loc[group.index[valid], "har_vix_logvar"] = logvar
        data.loc[group.index[valid], output_column] = np.sqrt(variance)
        rows.append({
            "asset": asset,
            "intercept": beta[0],
            "daily": beta[1],
            "weekly": beta[2],
            "monthly": beta[3],
            "log_vix_z": beta[4],
            "smearing_variance": sigma2,
            "smearing_offset": smearing_offset,
            "n_train": len(train),
            "below_floor_count": int(np.sum(variance < 1e-8)),
            "nonfinite_prediction_count": int(np.sum(~np.isfinite(variance))),
            "vix_missing_train_count": int(
                group.loc[group["segment"] == "train", "log_vix"].isna().sum()
            ),
            "vix_mean_train": mean,
            "vix_std_train": std,
        })
    return data, pd.DataFrame(rows)


def add_range_features(
    frame: pd.DataFrame,
    *,
    annualization_factor: float = 252.0,
    windows: tuple[int, ...] = (5, 22),
) -> pd.DataFrame:
    """Add Parkinson and Garman-Klass annualized variance features per asset.

    Both estimators use log(High/Low) and log(Close/Open) ratios, which are
    invariant to the unadjusted/adjusted basis, so they can be computed from the
    raw OHLC fields. The output features are on the annualized variance scale to
    match the HAR daily features. Rolling windows never cross asset boundaries.
    """
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    if not all(window >= 1 for window in windows):
        raise ValueError("windows must be positive")
    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    close = pd.to_numeric(data["close"], errors="coerce")
    open_ = pd.to_numeric(data["open"], errors="coerce")
    valid = (high > 0) & (low > 0) & (close > 0) & (open_ > 0) & (low <= high)
    parkinson_day = np.where(valid, 0.5 * np.square(np.log(high / low)), np.nan)
    garman_klass_day = np.where(
        valid,
        0.5 * np.square(np.log(high / low))
        - (2.0 * np.log(2.0) - 1.0) * np.square(np.log(close / open_)),
        np.nan,
    )
    data["_parkinson_day"] = parkinson_day
    data["_garman_klass_day"] = garman_klass_day
    for window in windows:
        data[f"parkinson_{window}d"] = annualization_factor * data.groupby(
            "asset", sort=False
        )["_parkinson_day"].transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
        data[f"garman_klass_{window}d"] = annualization_factor * data.groupby(
            "asset", sort=False
        )["_garman_klass_day"].transform(
            lambda values: values.rolling(window, min_periods=window).mean()
        )
    return data.drop(columns=["_parkinson_day", "_garman_klass_day"])


def add_market_state_features(
    frame: pd.DataFrame,
    *,
    return_column: str = "log_return",
    close_column: str = "close",
    volume_column: str = "volume",
    window: int = 21,
    short_window: int = 5,
) -> pd.DataFrame:
    """Add return, trend, drawdown, volume, and VIX-change features per asset.

    All features use only information available through the forecast date and
    rolling windows never cross asset boundaries. The VIX change requires the
    same-day ``log_vix`` column to already be present (e.g. from add_vix_level).
    """
    if window <= 0 or short_window <= 0:
        raise ValueError("windows must be positive")
    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    grouped = data.groupby("asset", sort=False)
    data["rel_return_5d"] = grouped[return_column].transform(
        lambda values: values.rolling(short_window, min_periods=short_window).sum()
    )
    data["rel_return_21d"] = grouped[return_column].transform(
        lambda values: values.rolling(window, min_periods=window).sum()
    )
    data["downside_frac_21d"] = grouped[return_column].transform(
        lambda values: (values < 0).rolling(window, min_periods=window).mean()
    )
    data["close_to_ma_21d"] = grouped[close_column].transform(
        lambda values: values / values.rolling(window, min_periods=window).mean() - 1.0
    )
    data["drawdown_21d"] = grouped[close_column].transform(
        lambda values: values / values.rolling(window, min_periods=window).max() - 1.0
    )
    volume = pd.to_numeric(data[volume_column], errors="coerce").replace(0.0, np.nan)
    volume_mean = grouped[volume_column].transform(
        lambda values: values.replace(0.0, np.nan).rolling(window, min_periods=window).mean()
    )
    data["volume_ratio_21d"] = volume / volume_mean - 1.0
    if "log_vix" in data.columns:
        data["vix_change_5d"] = grouped["log_vix"].transform(
            lambda values: values.diff(short_window)
        )
    return data
def add_har_features(frame: pd.DataFrame, *, return_column: str = "log_return", annualization_factor: float = 252.0, daily_column: str = "har_daily_rv", weekly_column: str = "har_weekly_rv", monthly_column: str = "har_monthly_rv") -> pd.DataFrame:
    if annualization_factor <= 0: raise ValueError("annualization_factor must be positive")
    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    squared = data[return_column].pow(2)
    grouped = squared.groupby(data["asset"], sort=False)
    data[daily_column] = annualization_factor * squared
    data[weekly_column] = annualization_factor * grouped.transform(lambda values: values.rolling(5, min_periods=5).mean())
    data[monthly_column] = annualization_factor * grouped.transform(lambda values: values.rolling(22, min_periods=22).mean())
    return data
