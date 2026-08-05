from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.labels import add_future_realized_volatility, add_range_based_future_volatility


def make_frame(returns: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "asset": "SPY",
        "date": pd.date_range("2020-01-01", periods=len(returns), freq="B"),
        "log_return": returns,
    })


def test_label_uses_exactly_t_plus_1_through_t_plus_5() -> None:
    returns = np.array([0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08])
    result = add_future_realized_volatility(make_frame(returns), horizon=5)
    expected = np.sqrt((252 / 5) * np.square(returns[1:6]).sum())
    assert result.loc[0, "future_rv_5d"] == expected

    changed_t = returns.copy(); changed_t[0] = 9.0
    changed_t6 = returns.copy(); changed_t6[6] = 9.0
    changed_t5 = returns.copy(); changed_t5[5] = 9.0
    assert add_future_realized_volatility(make_frame(changed_t)).loc[0, "future_rv_5d"] == expected
    assert add_future_realized_volatility(make_frame(changed_t6)).loc[0, "future_rv_5d"] == expected
    assert add_future_realized_volatility(make_frame(changed_t5)).loc[0, "future_rv_5d"] != expected


def test_final_five_labels_are_missing() -> None:
    result = add_future_realized_volatility(make_frame(np.linspace(-0.04, 0.05, 10)))
    assert result["future_rv_5d"].tail(5).isna().all()
    assert result["future_rv_5d"].iloc[:-5].notna().all()


def test_labels_are_isolated_by_asset() -> None:
    frame = pd.concat([make_frame(np.arange(8) / 100).assign(asset="A"), make_frame(np.arange(8) / 10).assign(asset="B")])
    result = add_future_realized_volatility(frame)
    expected_a = np.sqrt((252 / 5) * np.square(np.arange(1, 6) / 100).sum())
    expected_b = np.sqrt((252 / 5) * np.square(np.arange(1, 6) / 10).sum())
    assert result.loc[result.asset == "A", "future_rv_5d"].iloc[0] == expected_a
    assert result.loc[result.asset == "B", "future_rv_5d"].iloc[0] == expected_b


def _ohlc_frame(rows: int = 12) -> pd.DataFrame:
    close = 100.0 + np.arange(rows, dtype=float)
    return pd.DataFrame({
        "asset": "SPY",
        "date": pd.date_range("2020-01-01", periods=rows, freq="B"),
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
    })


def test_range_based_labels_formula_and_missing_tail() -> None:
    result = add_range_based_future_volatility(_ohlc_frame(12), horizon=5)
    parkinson_day = 0.5 * np.square(np.log(1.01 / 0.99))
    garman_klass_day = 0.5 * np.square(np.log(1.01 / 0.99)) - (2.0 * np.log(2.0) - 1.0) * np.square(np.log(1.0 / 0.999))
    assert result["future_rv_parkinson_5d"].iloc[0] == pytest.approx(np.sqrt(252 * parkinson_day))
    assert result["future_rv_garman_klass_5d"].iloc[0] == pytest.approx(np.sqrt(252 * garman_klass_day))
    assert result["future_rv_parkinson_5d"].tail(5).isna().all()
    assert result["future_rv_garman_klass_5d"].tail(5).isna().all()


def test_range_based_labels_are_isolated_by_asset() -> None:
    frame = pd.concat([
        _ohlc_frame(8).assign(asset="A"),
        _ohlc_frame(8).assign(asset="B"),
    ], ignore_index=True)
    frame.loc[frame["asset"] == "B", "high"] *= 1.2
    result = add_range_based_future_volatility(frame)
    a = result.loc[result["asset"] == "A", "future_rv_parkinson_5d"].iloc[0]
    b = result.loc[result["asset"] == "B", "future_rv_parkinson_5d"].iloc[0]
    assert a != b
