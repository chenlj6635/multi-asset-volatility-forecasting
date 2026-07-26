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
