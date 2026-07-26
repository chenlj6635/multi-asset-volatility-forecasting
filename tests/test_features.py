from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from src.features import add_historical_volatility_baseline


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
