from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import (
    inverse_vol_weights,
    portfolio_metrics,
    transmission_waterfall,
    vol_target_position,
    vol_targeting_metrics,
)


def _strategy_frame(rows: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    rng = np.random.default_rng(0)
    rows_out = []
    for asset, base in (("A", 0.15), ("B", 0.25)):
        returns = rng.normal(0, base / np.sqrt(252), len(dates))
        for index, date in enumerate(dates):
            segment = "train" if index < 30 else ("validation" if index < 35 else "test")
            rows_out.append({
                "asset": asset,
                "date": date,
                "segment": segment,
                "log_return": returns[index],
                "historical_rv_21d": base,
                "forecast_rv": base,
            })
    return pd.DataFrame(rows_out)


def test_position_uses_only_previous_day_forecast() -> None:
    frame = _strategy_frame()
    positioned = vol_target_position(
        frame, forecast_column="historical_rv_21d", target_vol=0.10, max_leverage=1.5
    )
    a = positioned.loc[positioned["asset"] == "A"].reset_index(drop=True)
    weight_a = float(np.clip(0.10 / 0.15, None, 1.5))
    for index in range(2, 15):
        assert a["position"].iloc[index] == pytest.approx(a["weight"].iloc[index - 1])
        assert a["position"].iloc[index] == pytest.approx(weight_a)
    # no future information: first row has no position yet
    assert pd.isna(a["position"].iloc[0])


def test_leverage_cap_and_no_short() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="B")
    frame = pd.DataFrame({
        "asset": "A",
        "date": dates,
        "segment": ["test"] * 6,
        "log_return": [0.01] * 6,
        "forecast_rv": [0.0001, 0.05, float("nan"), 0.5, 0.2, 0.3],
    })
    positioned = vol_target_position(frame, forecast_column="forecast_rv", target_vol=0.10, max_leverage=1.5)
    weights = positioned["weight"]
    assert weights.iloc[0] == pytest.approx(1.5)  # clipped at cap
    assert weights.iloc[1] == pytest.approx(1.5)  # 0.10/0.05 = 2.0 -> 1.5
    assert weights.iloc[3] == pytest.approx(0.20)  # 0.10/0.5
    assert (weights.dropna() >= 0).all()
    assert pd.isna(weights.iloc[2])


def test_metrics_sanity_and_fixed_weight() -> None:
    frame = _strategy_frame()
    metrics = vol_targeting_metrics(frame, forecast_column="forecast_rv", target_vol=0.10, max_leverage=1.5, cost_bps=5)
    assert set(metrics["asset"]) == {"A", "B", "ALL"}
    assert metrics["n_obs"].gt(0).all()
    assert metrics["target_deviation"].notna().all()
    fixed = vol_targeting_metrics(frame, fixed_weight=1.0)
    assert fixed["total_cost"].eq(0).all()
    assert fixed["turnover"].eq(0).all()


def test_waterfall_has_four_stages_per_forecaster() -> None:
    frame = _strategy_frame()
    waterfall = transmission_waterfall(
        frame, forecast_columns=("forecast_rv",), target_vol=0.10, max_leverage=1.5, cost_bps=10
    )
    assert set(waterfall["stage"]) == {
        "no cap / no lag / no cost",
        "leverage cap",
        "+ one-day lag",
        "+ transaction cost",
    }
    assert set(waterfall["forecast"]) == {"forecast_rv"}


def test_portfolio_schemes_and_inverse_weights() -> None:
    frame = _strategy_frame()
    weights = inverse_vol_weights(pd.Series([0.10, 0.20, 0.05]))
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()
    equal = portfolio_metrics(frame, weight_scheme="equal", rebalance_every=5)
    inverse_hist = portfolio_metrics(frame, weight_scheme="inverse_historical", rebalance_every=5)
    inverse_fc = portfolio_metrics(frame, weight_scheme="inverse_forecast", forecast_column="forecast_rv", rebalance_every=5)
    assert equal["scheme"] == "equal"
    assert inverse_hist["scheme"] == "inverse_historical"
    assert inverse_fc["scheme"] == "inverse_forecast"
    for metric in (equal, inverse_hist, inverse_fc):
        assert metric["n_days"] > 0
        assert np.isfinite(metric["realized_vol"])
    with pytest.raises(ValueError, match="weight_scheme"):
        portfolio_metrics(frame, weight_scheme="mean_variance")
