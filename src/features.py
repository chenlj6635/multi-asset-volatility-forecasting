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


def fit_har_vix_by_asset(frame: pd.DataFrame, *, output_column: str = "har_vix_rv", variance_floor: float = 1e-12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit train-only HAR-X OLS with train-standardized same-day log VIX."""
    if variance_floor <= 0: raise ValueError("variance_floor must be positive")
    features = ["har_daily_rv", "har_weekly_rv", "har_monthly_rv", "log_vix_z"]
    data = frame.copy(); data[output_column] = np.nan; data["har_vix_variance_raw"] = np.nan; rows = []
    train_vix = data.loc[(data["segment"] == "train") & data["log_vix"].notna(), "log_vix"]
    mean, std = float(train_vix.mean()), float(train_vix.std(ddof=0))
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0: raise ValueError("training log VIX standard deviation must be positive")
    data["log_vix_z"] = (data["log_vix"] - mean) / std
    for asset, group in data.groupby("asset", sort=True):
        train = group.loc[(group["segment"] == "train") & group[["future_rv_5d", *features]].notna().all(axis=1)]
        if len(train) < len(features) + 1: raise ValueError(f"insufficient HAR-VIX training observations for {asset}")
        x = np.column_stack([np.ones(len(train)), train[features].to_numpy()]); y = np.square(train["future_rv_5d"].to_numpy())
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        valid = group[features].notna().all(axis=1)
        raw = np.column_stack([np.ones(int(valid.sum())), group.loc[valid, features].to_numpy()]).dot(beta)
        clipped = np.maximum(np.where(np.isfinite(raw), raw, variance_floor), variance_floor)
        data.loc[group.index[valid], "har_vix_variance_raw"] = raw
        data.loc[group.index[valid], output_column] = np.sqrt(clipped)
        rows.append({"asset": asset, "intercept": beta[0], "daily": beta[1], "weekly": beta[2], "monthly": beta[3], "log_vix_z": beta[4], "n_train": len(train), "negative_variance_prediction_count": int(np.sum(raw < variance_floor)), "nonfinite_prediction_count": int(np.sum(~np.isfinite(raw))), "vix_missing_train_count": int(group.loc[group["segment"] == "train", "log_vix"].isna().sum()), "vix_mean_train": mean, "vix_std_train": std})
    return data, pd.DataFrame(rows)
def add_har_features(frame: pd.DataFrame, *, return_column: str = "log_return", annualization_factor: float = 252.0, daily_column: str = "har_daily_rv", weekly_column: str = "har_weekly_rv", monthly_column: str = "har_monthly_rv") -> pd.DataFrame:
    if annualization_factor <= 0: raise ValueError("annualization_factor must be positive")
    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    squared = data[return_column].pow(2)
    grouped = squared.groupby(data["asset"], sort=False)
    data[daily_column] = annualization_factor * squared
    data[weekly_column] = annualization_factor * grouped.transform(lambda values: values.rolling(5, min_periods=5).mean())
    data[monthly_column] = annualization_factor * grouped.transform(lambda values: values.rolling(22, min_periods=22).mean())
    return data
