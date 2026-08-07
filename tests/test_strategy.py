from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy import (
    calibrate_forecast_level,
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


def test_level_calibration_recovers_forecast_scale() -> None:
    rng = np.random.default_rng(1)
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    rows = []
    for asset, base in (("A", 0.15), ("B", 0.25)):
        actual = np.maximum(base + rng.normal(0, 0.02, len(dates)), 0.05)
        for index, date in enumerate(dates):
            segment = "train" if index < 40 else ("validation" if index < 50 else "test")
            rows.append({
                "asset": asset, "date": date, "segment": segment,
                "future_rv_5d": actual[index], "biased_rv": actual[index] * 0.8,
            })
    frame = pd.DataFrame(rows)
    expected_scale = 1 / 0.8
    for method in ("multiplicative", "variance_rms"):
        out = calibrate_forecast_level(frame, forecast_column="biased_rv", output_column="cal_rv", source_segment="validation", method=method, min_observations=5)
        scale = float(out.loc[out["asset"] == "A", "cal_rv"].iloc[0] / frame.loc[frame["asset"] == "A", "biased_rv"].iloc[0])
        np.testing.assert_allclose(scale, expected_scale, rtol=1e-6)
        np.testing.assert_allclose(out["cal_rv"], frame["future_rv_5d"], rtol=1e-6)
    out = calibrate_forecast_level(frame, forecast_column="biased_rv", output_column="cal_rv", source_segment="validation", method="loglinear", min_observations=5)
    np.testing.assert_allclose(out["cal_rv"], frame["future_rv_5d"], rtol=1e-6)


def test_level_calibration_rejects_unknown_method() -> None:
    frame = _strategy_frame()
    with pytest.raises(ValueError, match="unknown calibration method"):
        calibrate_forecast_level(frame, forecast_column="ewma_rv", output_column="cal_rv", method="bogus")
    with pytest.raises(ValueError, match="weight_scheme"):
        portfolio_metrics(frame, weight_scheme="mean_variance")


def test_per_regime_calibration_flattens_level_dependent_bias() -> None:
    """A single global scale cannot fix under-forecast in high-vol and
    over-forecast in low-vol at the same time; per-regime (by forecast level)
    should recover both. Bends the forecast by actual level."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2020-01-01", periods=90, freq="B")
    rows = []
    for index, date in enumerate(dates):
        low = index % 2 == 0
        actual = 0.10 + rng.normal(0, 0.01) if low else 0.30 + rng.normal(0, 0.02)
        bend = 1.2 if low else 0.8          # low -> over-forecast, high -> under-forecast
        segment = "train" if index < 30 else ("validation" if index < 60 else "test")
        rows.append({
            "asset": "A", "date": date, "segment": segment,
            "future_rv_5d": actual, "biased_rv": actual * bend,
        })
    frame = pd.DataFrame(rows)

    # 2 buckets split the low/high regimes cleanly -> exact recovery of both.
    per = calibrate_forecast_level(
        frame, forecast_column="biased_rv", output_column="cal_rv",
        source_segment="validation", method="per_regime", min_observations=5, n_buckets=2,
    )
    global_rms = calibrate_forecast_level(
        frame, forecast_column="biased_rv", output_column="cal_rv",
        source_segment="validation", method="variance_rms", min_observations=5,
    )

    test = frame["segment"] == "test"
    actual = frame.loc[test, "future_rv_5d"].to_numpy()
    per_test = per.loc[test, "cal_rv"].to_numpy()
    global_test = global_rms.loc[test, "cal_rv"].to_numpy()

    # per-regime recovers the actual in both the low- and high-vol halves...
    np.testing.assert_allclose(per_test[::2], actual[::2], rtol=0.05)   # low rows
    np.testing.assert_allclose(per_test[1::2], actual[1::2], rtol=0.05)  # high rows
    # ...and beats the single global scale on the test segment.
    per_rmse = float(np.sqrt(np.mean((per_test - actual) ** 2)))
    global_rmse = float(np.sqrt(np.mean((global_test - actual) ** 2)))
    assert per_rmse < global_rmse


def test_per_regime_three_buckets_improve_over_global() -> None:
    """With the report's default 3 terciles the middle bucket straddles a hard
    bimodal split, so exact recovery is not expected -- but per-regime must
    still reduce the residual level bias relative to a single global scale."""
    rng = np.random.default_rng(11)
    dates = pd.date_range("2020-01-01", periods=120, freq="B")
    rows = []
    for index, date in enumerate(dates):
        low = index % 2 == 0
        actual = 0.10 + rng.normal(0, 0.01) if low else 0.30 + rng.normal(0, 0.02)
        bend = 1.2 if low else 0.8
        segment = "train" if index < 40 else ("validation" if index < 80 else "test")
        rows.append({
            "asset": "A", "date": date, "segment": segment,
            "future_rv_5d": actual, "biased_rv": actual * bend,
        })
    frame = pd.DataFrame(rows)
    per = calibrate_forecast_level(
        frame, forecast_column="biased_rv", output_column="cal_rv",
        source_segment="validation", method="per_regime", min_observations=5, n_buckets=3,
    )
    global_rms = calibrate_forecast_level(
        frame, forecast_column="biased_rv", output_column="cal_rv",
        source_segment="validation", method="variance_rms", min_observations=5,
    )
    test = frame["segment"] == "test"
    actual = frame.loc[test, "future_rv_5d"].to_numpy()
    assert float(np.sqrt(np.mean((per.loc[test, "cal_rv"].to_numpy() - actual) ** 2))) < float(
        np.sqrt(np.mean((global_rms.loc[test, "cal_rv"].to_numpy() - actual) ** 2))
    )


def test_per_regime_falls_back_to_global_for_small_buckets() -> None:
    # Constant forecasts -> degenerate buckets -> per-bucket scale not estimated,
    # falls back to the asset-wide scale; must still return a finite column.
    dates = pd.date_range("2020-01-01", periods=12, freq="B")
    frame = pd.DataFrame({
        "asset": ["A"] * 12,
        "date": list(dates),
        "segment": ["validation"] * 6 + ["test"] * 6,
        "future_rv_5d": [0.10] * 12,
        "ewma_rv": [0.11] * 12,
    })
    out = calibrate_forecast_level(
        frame, forecast_column="ewma_rv", output_column="cal_rv",
        source_segment="validation", method="per_regime", min_observations=5,
    )
    assert out["cal_rv"].notna().any()
