from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.features import (
    add_ewma_volatility_baseline,
    add_ewma_volatility_candidates,
    add_historical_volatility_baseline,
    fit_har_vix_by_asset,
)


def test_ewma_formula_and_start() -> None:
    returns = np.arange(1, 26, dtype=float) / 1000
    frame = pd.DataFrame({"asset": "SPY", "date": pd.date_range("2020-01-01", periods=25), "log_return": returns})
    result = add_ewma_volatility_baseline(frame, decay=0.9, min_periods=3)
    expected_variance = np.mean(np.square(returns[:3]))
    expected_variance = 0.9 * expected_variance + 0.1 * returns[3] ** 2
    assert result["ewma_rv"].iloc[:2].isna().all()
    assert np.isclose(result["ewma_rv"].iloc[2], np.sqrt(252 * np.mean(np.square(returns[:3]))) )
    assert np.isclose(result["ewma_rv"].iloc[3], np.sqrt(252 * expected_variance))


def test_ewma_rejects_invalid_decay() -> None:
    frame = pd.DataFrame({"asset": ["SPY"], "date": pd.date_range("2020-01-01", periods=1), "log_return": [0.01]})
    with pytest.raises(ValueError, match="decay"):
        add_ewma_volatility_baseline(frame, decay=1.0)


def test_ewma_candidates_validate_and_match_single() -> None:
    frame = pd.DataFrame({
        "asset": "SPY", "date": pd.date_range("2020-01-01", periods=25),
        "log_return": np.arange(1, 26, dtype=float) / 1000,
    })
    candidates = add_ewma_volatility_candidates(frame, decays=[0.90, 0.94], min_periods=3)
    single = add_ewma_volatility_baseline(frame, decay=0.94, min_periods=3)
    assert np.allclose(candidates["ewma_rv_lambda_0.94"], single["ewma_rv"], equal_nan=True)
    with pytest.raises(ValueError, match="unique"):
        add_ewma_volatility_candidates(frame, decays=[0.94, 0.94])


def test_window_start_and_formula() -> None:
    returns = np.arange(1, 26, dtype=float) / 1000
    frame = pd.DataFrame({"asset": "SPY", "date": pd.date_range("2020-01-01", periods=25), "log_return": returns})
    result = add_historical_volatility_baseline(frame, window=21)
    assert result["historical_rv_21d"].iloc[:20].isna().all()
    expected = np.sqrt(252 * np.square(returns[:21]).mean())
    assert result["historical_rv_21d"].iloc[20] == expected


def test_assets_do_not_share_rolling_windows() -> None:
    dates = pd.date_range("2020-01-01", periods=24)
    frame = pd.concat([
        pd.DataFrame({"asset": "A", "date": dates, "log_return": 0.01}),
        pd.DataFrame({"asset": "B", "date": dates, "log_return": 0.10}),
    ], ignore_index=True)
    result = add_historical_volatility_baseline(frame, window=21)
    values = result.groupby("asset")["historical_rv_21d"].last()
    assert values["A"] == np.sqrt(252 * 0.01**2)
    assert values["B"] == np.sqrt(252 * 0.10**2)


def test_unsorted_and_sorted_inputs_match() -> None:
    frame = pd.DataFrame({
        "asset": ["B", "A"] * 25,
        "date": list(pd.date_range("2020-01-01", periods=25)) * 2,
        "log_return": np.linspace(-0.03, 0.04, 50),
    })
    sorted_frame = frame.sort_values(["asset", "date"])
    expected = add_historical_volatility_baseline(sorted_frame).reset_index(drop=True)
    actual = add_historical_volatility_baseline(frame.sample(frac=1, random_state=7)).reset_index(drop=True)
    pdt.assert_frame_equal(actual, expected)


def _har_vix_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=810)
    rows = []
    for asset, base in (("A", 0.10), ("B", 0.18)):
        for date in dates:
            daily = base / np.sqrt(252) + abs(float(rng.normal(0, 0.002)))
            vix = float(rng.normal(28.0 if asset == "A" else 20.0, 3.0))
            segment = (
                "train" if date <= pd.Timestamp("2020-11-30")
                else "validation" if date <= pd.Timestamp("2021-01-31")
                else "test"
            )
            rows.append({
                "asset": asset,
                "date": date,
                "segment": segment,
                "future_rv_5d": base + abs(float(rng.normal(0, 0.01))),
                "har_daily_rv": max(daily * daily, 1e-8),
                "har_weekly_rv": max(base ** 2 / 252 * 5, 1e-8),
                "har_monthly_rv": max(base ** 2 / 252 * 22, 1e-8),
                "log_vix": np.log(max(vix, 5.0)),
            })
    return pd.DataFrame(rows)


def test_har_vix_log_variance_model_never_clips_and_smearing_raises_level() -> None:
    frame = _har_vix_frame()
    with_smear, coefficients = fit_har_vix_by_asset(frame, smearing=True)
    without, _ = fit_har_vix_by_asset(frame, smearing=False)
    assert with_smear["har_vix_rv"].notna().all()
    assert (with_smear["har_vix_rv"] > 0).all()
    assert (with_smear["har_vix_rv"] < 5).all()
    assert "har_vix_logvar" in with_smear.columns
    assert set(coefficients["asset"]) == {"A", "B"}
    assert {"smearing_variance", "smearing_offset"}.issubset(coefficients.columns)
    assert (coefficients["below_floor_count"] == 0).all()
    assert (coefficients["nonfinite_prediction_count"] == 0).all()
    means_with = with_smear.groupby("asset")["har_vix_rv"].mean()
    means_without = without.groupby("asset")["har_vix_rv"].mean()
    assert (means_with > means_without).all()


def test_har_vix_rejects_invalid_floor_and_requires_train_rows() -> None:
    frame = _har_vix_frame()
    with pytest.raises(ValueError, match="log_floor"):
        fit_har_vix_by_asset(frame, log_floor=0.0)
    no_train = frame.loc[frame["segment"] != "train"].copy()
    with pytest.raises(ValueError, match="log VIX standard deviation"):
        fit_har_vix_by_asset(no_train)
