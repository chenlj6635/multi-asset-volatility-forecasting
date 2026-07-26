"""Return and future realized-volatility label construction."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_log_returns(
    frame: pd.DataFrame,
    *,
    price_column: str = "adj_close",
    output_column: str = "log_return",
) -> pd.DataFrame:
    """Compute close-to-close log returns independently for each asset."""
    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    if (data[price_column].dropna() <= 0).any():
        raise ValueError(f"{price_column} must be positive before taking logarithms")
    data[output_column] = data.groupby("asset", sort=False)[price_column].transform(
        lambda prices: np.log(prices).diff()
    )
    return data


def add_future_realized_volatility(
    frame: pd.DataFrame,
    *,
    horizon: int = 5,
    annualization_factor: float = 252.0,
    return_column: str = "log_return",
    output_column: str = "future_rv_5d",
) -> pd.DataFrame:
    """Add annualized RV using exactly returns at t+1 through t+horizon."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    data = frame.sort_values(["asset", "date"], kind="stable").copy()

    grouped_returns = data.groupby("asset", sort=False)[return_column]
    # Each column is explicit: at row t these are r²(t+1), ..., r²(t+horizon).
    future_terms = [
        grouped_returns.transform(lambda returns, step=step: returns.pow(2).shift(-step))
        for step in range(1, horizon + 1)
    ]
    future_squared_returns = pd.concat(future_terms, axis=1)
    summed = future_squared_returns.sum(axis=1, min_count=horizon)
    data[output_column] = np.sqrt((annualization_factor / horizon) * summed)
    return data
