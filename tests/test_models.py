from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import fit_garch_by_asset


def _garch_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=300, freq="B")
    rows = []
    for asset, vol_base in (("A", 0.01), ("B", 0.03)):
        returns = rng.normal(0, vol_base, len(dates))
        for index, date in enumerate(dates):
            rows.append({
                "asset": asset,
                "date": date,
                "segment": "train" if index < 240 else ("validation" if index < 270 else "test"),
                "log_return": returns[index],
            })
    return pd.DataFrame(rows)


def test_garch_forecasts_are_positive_finite_and_params_reported() -> None:
    frame = _garch_frame()
    result, params = fit_garch_by_asset(frame, min_train_observations=100)
    assert set(params["asset"]) == {"A", "B"}
    assert (params["n_train"] == 240).all()
    expected_columns = {
        "asset", "omega", "alpha", "beta", "mu", "alpha_plus_beta",
        "stationary", "n_train", "convergence_flag", "likelihood", "status",
    }
    assert expected_columns.issubset(params.columns)
    assert (params["omega"] > 0).all()
    assert (params["alpha"] >= 0).all()
    assert (params["beta"] >= 0).all()
    forecasts = result["garch_rv"]
    assert forecasts.notna().all()
    assert (forecasts > 0).all()
    assert (forecasts < 5).all()
    test_means = result.loc[result["segment"] == "test"].groupby("asset")["garch_rv"].mean()
    assert test_means["B"] > test_means["A"]


def test_garch_insufficient_train_is_reported_and_forecasts_missing() -> None:
    frame = _garch_frame()
    result, params = fit_garch_by_asset(frame, min_train_observations=10_000)
    assert (params["status"] == "insufficient_train_observations").all()
    assert result["garch_rv"].isna().all()


def test_garch_rejects_invalid_orders() -> None:
    frame = _garch_frame()
    with np.testing.assert_raises(ValueError):
        fit_garch_by_asset(frame, p=0, q=0)
