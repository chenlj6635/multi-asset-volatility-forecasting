from __future__ import annotations

import numpy as np
import pandas as pd

from src.models import _fit_ridge_ls, fit_garch_by_asset, fit_lightgbm_by_asset, fit_ridge_by_asset


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


RIDGE_FEATURES = ["har_daily_rv", "market", "log_vix"]


def _ridge_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=280, freq="B")
    rows = []
    for asset, base in (("A", 0.10), ("B", 0.20)):
        for index, date in enumerate(dates):
            x1 = base + rng.normal(0, 0.01)
            x2 = rng.normal(0, 1.0)
            x3 = np.log(20.0 + rng.normal(0, 3.0))
            future = base + 0.5 * (x1 - base) + 0.02 * x2 + abs(rng.normal(0, 0.02))
            rows.append({
                "asset": asset,
                "date": date,
                "segment": "train" if index < 200 else ("validation" if index < 240 else "test"),
                "future_rv_5d": max(float(future), 0.02),
                "har_daily_rv": max(x1 * x1, 1e-6),
                "market": x2,
                "log_vix": x3,
            })
    return pd.DataFrame(rows)


def test_ridge_selects_lambda_on_validation_and_forecasts_positive() -> None:
    frame = _ridge_frame()
    grid = [0.0, 0.1, 1.0, 10.0, 100.0]
    result, params, selection = fit_ridge_by_asset(
        frame, feature_columns=RIDGE_FEATURES, penalty="ridge",
        lambda_grid=grid, min_train_observations=100, min_validation_observations=20,
    )
    assert set(params["asset"]) == {"A", "B"}
    assert set(params["penalty"]) == {"ridge"}
    assert (params["status"] == "ok").all()
    assert params["selected_lambda"].notna().all()
    assert len(selection) == 2 * len(grid)
    for asset in ("A", "B"):
        assert selection.loc[selection["asset"] == asset, "selected"].sum() == 1
    assert result["ridge_rv"].notna().all()
    assert (result["ridge_rv"] > 0).all()


def test_ridge_zero_lambda_matches_least_squares() -> None:
    frame = _ridge_frame()
    train = frame.loc[frame["segment"] == "train"]
    x = train[RIDGE_FEATURES].to_numpy(dtype=float)
    x_scaled = (x - x.mean(axis=0)) / x.std(axis=0)
    y = np.square(train["future_rv_5d"].to_numpy(dtype=float))
    beta, _ = _fit_ridge_ls(x_scaled, y, l2=0.0)
    expected = np.linalg.lstsq(x_scaled, y - y.mean(), rcond=None)[0]
    np.testing.assert_allclose(beta, expected, atol=1e-10)


def test_lasso_path_is_finite_and_positive() -> None:
    frame = _ridge_frame()
    result, params, _ = fit_ridge_by_asset(
        frame, feature_columns=RIDGE_FEATURES, penalty="lasso",
        lambda_grid=[0.0, 0.0005, 0.001, 0.005, 0.01],
        min_train_observations=100, min_validation_observations=20,
    )
    assert (params["status"] == "ok").all()
    assert result["ridge_rv"].notna().all()
    assert (result["ridge_rv"] > 0).all()


def test_ridge_insufficient_data_is_reported() -> None:
    frame = _ridge_frame()
    result, params, selection = fit_ridge_by_asset(
        frame, feature_columns=[RIDGE_FEATURES[0]], penalty="ridge",
        lambda_grid=[1.0], min_train_observations=10_000, min_validation_observations=20,
    )
    assert (params["status"] == "insufficient_observations").all()
    assert result["ridge_rv"].isna().all()
    assert selection.empty


def test_lightgbm_selects_hyperparameters_on_validation_and_forecasts_positive() -> None:
    frame = _ridge_frame()
    result, params, selection, importance = fit_lightgbm_by_asset(
        frame, feature_columns=RIDGE_FEATURES,
        num_leaves_grid=(4, 8), learning_rate_grid=(0.05, 0.1), n_estimators_grid=(5, 10),
        min_child_samples=5, min_train_observations=100, min_validation_observations=20,
    )
    assert set(params["asset"]) == {"A", "B"}
    assert (params["status"] == "ok").all()
    assert params["selected_num_leaves"].notna().all()
    assert params["selected_learning_rate"].notna().all()
    assert params["selected_n_estimators"].notna().all()
    assert len(selection) == 2 * 2 * 2 * 2
    for asset in ("A", "B"):
        assert selection.loc[selection["asset"] == asset, "selected"].sum() == 1
    assert result["lgb_rv"].notna().all()
    assert (result["lgb_rv"] > 0).all()
    test_means = result.loc[result["segment"] == "test"].groupby("asset")["lgb_rv"].mean()
    assert test_means["B"] > test_means["A"]


def test_lightgbm_importance_reported() -> None:
    frame = _ridge_frame()
    _, _, _, importance = fit_lightgbm_by_asset(
        frame, feature_columns=RIDGE_FEATURES,
        num_leaves_grid=(4,), learning_rate_grid=(0.05,), n_estimators_grid=(5,),
        min_child_samples=5, min_train_observations=100, min_validation_observations=20,
    )
    assert set(importance["asset"]) == {"A", "B"}
    assert set(importance["feature"]) == set(RIDGE_FEATURES)
    assert (importance["importance_gain"] >= 0).all()
    assert (importance["importance_split"] >= 0).all()
    per_asset_share = importance.groupby("asset")["importance_gain_share"].sum()
    np.testing.assert_allclose(per_asset_share, 1.0, atol=1e-6)


def test_lightgbm_insufficient_data_is_reported() -> None:
    frame = _ridge_frame()
    result, params, selection, importance = fit_lightgbm_by_asset(
        frame, feature_columns=RIDGE_FEATURES,
        num_leaves_grid=(4,), learning_rate_grid=(0.05,), n_estimators_grid=(5,),
        min_train_observations=10_000, min_validation_observations=20,
    )
    assert (params["status"] == "insufficient_observations").all()
    assert result["lgb_rv"].isna().all()
    assert selection.empty
    assert importance.empty
