"""Leakage-safe historical features and baseline forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_historical_volatility_baseline(
    frame: pd.DataFrame,
    *,
    window: int = 21,
    annualization_factor: float = 252.0,
    return_column: str = "log_return",
    output_column: str = "historical_rv_21d",
) -> pd.DataFrame:
    """Add annualized volatility using the last `window` returns available at t."""
    if window <= 0:
        raise ValueError("window must be positive")
    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    rolling_variance = data.groupby("asset", sort=False)[return_column].transform(
        lambda returns: returns.pow(2).rolling(window=window, min_periods=window).mean()
    )
    data[output_column] = np.sqrt(annualization_factor * rolling_variance)
    return data


def add_ewma_volatility_baseline(
    frame: pd.DataFrame,
    *,
    decay: float = 0.94,
    min_periods: int = 21,
    annualization_factor: float = 252.0,
    return_column: str = "log_return",
    output_column: str = "ewma_rv",
) -> pd.DataFrame:
    """Add annualized EWMA volatility using returns available at t."""
    if not 0 < decay < 1:
        raise ValueError("decay must be between 0 and 1")
    if min_periods <= 0:
        raise ValueError("min_periods must be positive")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")

    data = frame.sort_values(["asset", "date"], kind="stable").copy()

    def ewma_variance(returns: pd.Series) -> pd.Series:
        values = returns.to_numpy(dtype=float)
        result = np.full(len(values), np.nan, dtype=float)
        variance = np.nan
        valid_count = 0
        for index, value in enumerate(values):
            if not np.isfinite(value):
                continue
            squared = value * value
            valid_count += 1
            if valid_count == min_periods:
                valid_squared = values[np.isfinite(values)][:valid_count] ** 2
                variance = float(valid_squared.mean())
            elif valid_count > min_periods:
                variance = decay * variance + (1.0 - decay) * squared
            if valid_count >= min_periods:
                result[index] = variance
        return pd.Series(result, index=returns.index)

    variance = data.groupby("asset", sort=False)[return_column].transform(ewma_variance)
    data[output_column] = np.sqrt(annualization_factor * variance)
    return data
