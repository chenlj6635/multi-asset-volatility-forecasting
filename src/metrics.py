"""Forecast metrics, including variance-scale QLIKE."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MetricResult:
    n_obs: int
    mae: float
    rmse: float
    qlike: float
    variance_floor_count: int


def evaluate_forecast(
    actual_volatility: pd.Series | np.ndarray,
    forecast_volatility: pd.Series | np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> MetricResult:
    """Compute MAE/RMSE on volatility and QLIKE on variance."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    actual = np.asarray(actual_volatility, dtype=float)
    forecast = np.asarray(forecast_volatility, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(forecast) & (actual >= 0)
    actual = actual[valid]
    forecast = forecast[valid]
    if actual.size == 0:
        return MetricResult(0, np.nan, np.nan, np.nan, 0)

    errors = forecast - actual
    forecast_variance = np.square(forecast)
    floor_mask = (~np.isfinite(forecast_variance)) | (forecast_variance < epsilon)
    forecast_variance = np.maximum(forecast_variance, epsilon)
    actual_variance = np.square(actual)
    qlike_values = np.log(forecast_variance) + actual_variance / forecast_variance
    return MetricResult(
        n_obs=int(actual.size),
        mae=float(np.mean(np.abs(errors))),
        rmse=float(np.sqrt(np.mean(np.square(errors)))),
        qlike=float(np.mean(qlike_values)),
        variance_floor_count=int(floor_mask.sum()),
    )


def metrics_by_asset(
    frame: pd.DataFrame,
    *,
    actual_column: str = "future_rv_5d",
    forecast_column: str = "historical_rv_21d",
    forecast_columns: tuple[str, ...] | None = None,
    epsilon: float = 1.0e-12,
) -> pd.DataFrame:
    """Evaluate each forecast by asset and direct pooled ALL row."""
    columns = forecast_columns or (forecast_column,)
    rows: list[dict[str, float | int | str]] = []
    for forecast in columns:
        for asset, group in frame.groupby("asset", sort=True):
            rows.append(
                {
                    "asset": asset,
                    "forecast": forecast,
                    **asdict(evaluate_forecast(group[actual_column], group[forecast], epsilon=epsilon)),
                }
            )
        rows.append(
            {
                "asset": "ALL",
                "forecast": forecast,
                **asdict(evaluate_forecast(frame[actual_column], frame[forecast], epsilon=epsilon)),
            }
        )
    return pd.DataFrame(rows)[["asset", "forecast", "n_obs", "mae", "rmse", "qlike", "variance_floor_count"]]
