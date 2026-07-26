from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import evaluate_forecast, metrics_by_asset


def test_metrics_match_manual_values_and_use_variance_qlike() -> None:
    actual = np.array([0.2, 0.4])
    forecast = np.array([0.1, 0.5])
    result = evaluate_forecast(actual, forecast)
    assert result.mae == np.mean(np.abs(forecast - actual))
    assert result.rmse == np.sqrt(np.mean(np.square(forecast - actual)))
    forecast_variance = forecast**2
    expected_qlike = np.mean(np.log(forecast_variance) + actual**2 / forecast_variance)
    assert result.qlike == expected_qlike


def test_floor_and_pairwise_missing_filter() -> None:
    result = evaluate_forecast([0.2, np.nan, 0.4], [0.0, 0.3, np.nan], epsilon=1e-6)
    assert result.n_obs == 1
    assert result.variance_floor_count == 1
    assert np.isfinite(result.qlike)


def test_all_row_is_direct_pool_not_average_of_asset_metrics() -> None:
    frame = pd.DataFrame({
        "asset": ["A", "A", "B"],
        "future_rv_5d": [1.0, 1.0, 10.0],
        "historical_rv_21d": [2.0, 2.0, 10.0],
    })
    table = metrics_by_asset(frame)
    pooled = table.set_index("asset").loc["ALL"]
    assert pooled.n_obs == 3
    assert pooled.mae == 2 / 3
