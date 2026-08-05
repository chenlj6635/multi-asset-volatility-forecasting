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


def add_range_based_future_volatility(
    frame: pd.DataFrame,
    *,
    horizon: int = 5,
    annualization_factor: float = 252.0,
    open_column: str = "open",
    high_column: str = "high",
    low_column: str = "low",
    close_column: str = "close",
) -> pd.DataFrame:
    """Add future realized-volatility labels from price ranges.

    Two alternative labels to the squared-return ``future_rv_5d``:

    - Parkinson: ``0.5 * ln(High/Low)^2`` per day;
    - Garman-Klass: adds the ``(2*ln2 - 1) * ln(Close/Open)^2`` adjustment.

    Both use exactly ``t+1`` through ``t+horizon`` and the same annualization
    convention, so they are directly comparable to the main label as a
    robustness check on label proxy noise. Range ratios are scale-invariant to
    the raw/unadjusted OHLC basis. Non-finite or invalid ranges leave the label
    missing, so the last ``horizon`` rows of each asset are not labelled.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    high = pd.to_numeric(data[high_column], errors="coerce")
    low = pd.to_numeric(data[low_column], errors="coerce")
    close = pd.to_numeric(data[close_column], errors="coerce")
    open_ = pd.to_numeric(data[open_column], errors="coerce")
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
    for name, column in (
        ("parkinson", "_parkinson_day"),
        ("garman_klass", "_garman_klass_day"),
    ):
        future_terms = [
            data.groupby("asset", sort=False)[column].transform(
                lambda values, step=step: values.shift(-step)
            )
            for step in range(1, horizon + 1)
        ]
        summed = pd.concat(future_terms, axis=1).sum(axis=1, min_count=horizon)
        data[f"future_rv_{name}_{horizon}d"] = np.sqrt(
            (annualization_factor / horizon) * summed
        )
    return data.drop(columns=["_parkinson_day", "_garman_klass_day"])
