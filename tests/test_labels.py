from __future__ import annotations

import numpy as np
import pandas as pd

from src.labels import add_future_realized_volatility


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
