from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.build_dataset import build
from src.data import read_asset_csv

ASSETS = ["SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "^VIX"]


def write_csv(path: Path, asset_index: int, rows: int = 55) -> None:
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    returns = 0.001 * np.sin(np.arange(rows) / 3 + asset_index) + 0.0005 * (asset_index + 1)
    adjusted = 100 * np.exp(np.cumsum(returns))
    close = adjusted * (1 + 0.0001 * asset_index)
    frame = pd.DataFrame({
        "Date": dates,
        "Open": close * 0.999,
        "High": close * 1.01,
        "Low": close * 0.99,
        "Close": close,
        "Adj Close": adjusted,
        "Volume": np.nan if asset_index == 6 else 1_000_000 + np.arange(rows),
    })
    frame.to_csv(path, index=False)


def make_config(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"; raw.mkdir()
    for index, asset in enumerate(ASSETS):
        write_csv(raw / ("VIX.csv" if asset == "^VIX" else f"{asset}.csv"), index)
    config = {
        "data": {
            "source": "synthetic local CSV",
            "raw_dir": str(raw),
            "processed_dir": str(tmp_path / "processed"),
            "quality_dir": str(tmp_path / "quality"),
            "allow_close_as_adjusted": False,
            "target_assets": {asset: f"{asset}.csv" for asset in ASSETS[:-1]},
            "context_assets": {"^VIX": "VIX.csv"},
        },
        "walk_forward": {
            "train_end": "2020-02-28",
            "validation_end": "2020-03-31",
        },
        "dm_test": {
            "enabled": True,
            "losses": ["qlike", "mae"],
            "primary_loss": "qlike",
            "hac_lag": 4,
        },
        "expanding_window": {
            "enabled": True,
            "validation_years": 2,
            "models": ["har_rv", "garch_rv", "ridge_rv", "lgb_rv"],
        },
        "calculation": {
            "annualization_factor": 252,
            "label_horizon": 5,
            "baseline_window": 21,
            "ewma_lambda": 0.94,
            "ewma_lambdas": [0.90, 0.94, 0.97, 0.99],
            "ewma_min_periods": 21,
            "qlike_epsilon": 1e-12,
            "forecast_variance_floor": 1e-4,
            "extreme_return_threshold": 0.20,
            "long_gap_days": 7,
            "garch": {
                "enabled": True,
                "p": 1,
                "q": 1,
                "min_train_observations": 10,
            },
            "ridge": {
                "enabled": True,
                "penalty": "ridge",
                "min_train_observations": 10,
                "min_validation_observations": 3,
                "lambda_grid": [0.0, 0.1, 1.0, 10.0],
                "lasso_lambda_grid": [0.0, 0.001, 0.01],
            },
            "lightgbm": {
                "enabled": True,
                "num_leaves": [4, 8],
                "learning_rate": [0.05, 0.1],
                "n_estimators": [5, 10],
                "min_child_samples": 3,
                "min_train_observations": 10,
                "min_validation_observations": 3,
                "worst_error_top_n": 5,
            },
        },
        "outputs": {
            "predictions": str(tmp_path / "processed/predictions.parquet"),
            "metrics": str(tmp_path / "outputs/metrics.csv"),
            "walk_forward_metrics": str(tmp_path / "outputs/walk_forward_metrics.csv"),
            "dm_tests": str(tmp_path / "outputs/dm_tests.csv"),
            "ewma_lambda_selection": str(tmp_path / "outputs/ewma_lambda_selection.csv"),
            "har_coefficients": str(tmp_path / "outputs/har_coefficients.csv"),
            "har_vix_coefficients": str(tmp_path / "outputs/har_vix_coefficients.csv"),
            "garch_params": str(tmp_path / "outputs/garch_params.csv"),
            "ridge_params": str(tmp_path / "outputs/ridge_params.csv"),
            "ridge_lambda_selection": str(tmp_path / "outputs/ridge_lambda_selection.csv"),
            "lightgbm_params": str(tmp_path / "outputs/lightgbm_params.csv"),
            "lightgbm_importance": str(tmp_path / "outputs/lightgbm_importance.csv"),
            "worst_error_dates": str(tmp_path / "outputs/worst_error_dates.csv"),
            "expanding_comparison": str(tmp_path / "outputs/expanding_comparison.csv"),
            "expanding_dm": str(tmp_path / "outputs/expanding_dm.csv"),
            "expanding_params": str(tmp_path / "outputs/expanding_params.csv"),
            "test_model_comparison": str(tmp_path / "outputs/test_model_comparison.csv"),
            "vix_incremental_comparison": str(tmp_path / "outputs/vix_incremental_comparison.csv"),
            "asset_robustness": str(tmp_path / "outputs/asset_robustness.csv"),
            "regime_robustness": str(tmp_path / "outputs/regime_robustness.csv"),
            "strategy_metrics": str(tmp_path / "outputs/strategy_metrics.csv"),
            "transmission_waterfall": str(tmp_path / "outputs/transmission_waterfall.csv"),
            "portfolio_metrics": str(tmp_path / "outputs/portfolio_metrics.csv"),
            "alt_label_metrics": str(tmp_path / "outputs/alt_label_metrics.csv"),
            "alt_label_dm": str(tmp_path / "outputs/alt_label_dm.csv"),
            "strategy_cost_sensitivity": str(tmp_path / "outputs/strategy_cost_sensitivity.csv"),
            "yearly_metrics": str(tmp_path / "outputs/yearly_metrics.csv"),
            "metadata": str(tmp_path / "outputs/metadata.json"),
            "figure": str(tmp_path / "outputs/figure.png"),
        },
    }
    path = tmp_path / "default.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_complete_pipeline_runs_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = make_config(tmp_path)
    monkeypatch.setattr("socket.create_connection", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network used")))
    metadata = build(config_path)
    assert metadata["offline_build"] is True
    expected = [
        tmp_path / "quality/asset_quality.csv",
        tmp_path / "quality/quality_summary.json",
        tmp_path / "processed/predictions.parquet",
        tmp_path / "outputs/metrics.csv",
        tmp_path / "outputs/walk_forward_metrics.csv",
        tmp_path / "outputs/dm_tests.csv",
        tmp_path / "outputs/ewma_lambda_selection.csv",
        tmp_path / "outputs/har_coefficients.csv",
        tmp_path / "outputs/har_vix_coefficients.csv",
        tmp_path / "outputs/garch_params.csv",
        tmp_path / "outputs/ridge_params.csv",
        tmp_path / "outputs/ridge_lambda_selection.csv",
        tmp_path / "outputs/lightgbm_params.csv",
        tmp_path / "outputs/lightgbm_importance.csv",
        tmp_path / "outputs/worst_error_dates.csv",
        tmp_path / "outputs/expanding_comparison.csv",
        tmp_path / "outputs/expanding_dm.csv",
        tmp_path / "outputs/test_model_comparison.csv",
        tmp_path / "outputs/vix_incremental_comparison.csv",
        tmp_path / "outputs/asset_robustness.csv",
        tmp_path / "outputs/regime_robustness.csv",
        tmp_path / "outputs/metadata.json",
        tmp_path / "outputs/figure.png",
        tmp_path / "outputs/strategy_metrics.csv",
        tmp_path / "outputs/transmission_waterfall.csv",
        tmp_path / "outputs/portfolio_metrics.csv",
        tmp_path / "outputs/alt_label_metrics.csv",
        tmp_path / "outputs/alt_label_dm.csv",
        tmp_path / "outputs/strategy_cost_sensitivity.csv",
        tmp_path / "outputs/yearly_metrics.csv",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)
    out = tmp_path / "outputs"
    predictions = pd.read_parquet(tmp_path / "processed/predictions.parquet")
    assert set(predictions.asset) == set(ASSETS[:-1])
    assert {"historical_rv_21d", "ewma_rv", "har_rv", "ridge_rv", "garch_rv", "lgb_rv", "har_vix_rv"}.issubset(predictions.columns)
    assert {"har_rv_exp", "garch_rv_exp", "ridge_rv_exp", "lgb_rv_exp"}.issubset(predictions.columns)
    assert predictions.groupby("asset")["future_rv_5d"].tail(5).isna().all()
    metrics = pd.read_csv(out / "metrics.csv")
    assert metrics.asset.tolist() == ["GLD", "IWM", "QQQ", "SPY", "TLT", "USO", "ALL"] * 7
    assert set(metrics.forecast) == {"historical_rv_21d", "ewma_rv", "har_rv", "ridge_rv", "garch_rv", "lgb_rv", "har_vix_rv"}
    assert np.isfinite(metrics[["mae", "rmse", "qlike"]].to_numpy()).all()
    walk_metrics = pd.read_csv(out / "walk_forward_metrics.csv")
    assert set(walk_metrics.segment) == {"train", "validation", "test"}
    assert set(walk_metrics.forecast) == {"historical_rv_21d", "ewma_rv", "har_rv", "ridge_rv", "garch_rv", "lgb_rv", "har_vix_rv"}
    dm = pd.read_csv(out / "dm_tests.csv")
    assert set(dm.segment) == {"train", "validation", "test"}
    assert set(dm.asset) == set(ASSETS[:-1]) | {"ALL"}
    assert {"n_dates", "paired_rows"}.issubset(dm.columns)
    assert set(dm.loss) == {"qlike", "mae"}
    assert len(dm) == 330
    assert {"lgb_rv"}.issubset(set(dm.loc[dm["loss"] == "qlike", "model_a"]))
    selection = pd.read_csv(out / "ewma_lambda_selection.csv")
    assert len(selection) == 4
    assert selection.selected.sum() == 1
    assert selection.selection_segment.iloc[0] == "validation"
    coefficients = pd.read_csv(out / "har_coefficients.csv")
    vix_coefficients = pd.read_csv(out / "har_vix_coefficients.csv")
    assert set(vix_coefficients.asset) == set(ASSETS[:-1])
    garch_params = pd.read_csv(out / "garch_params.csv")
    assert set(garch_params.asset) == set(ASSETS[:-1])
    ridge_params = pd.read_csv(out / "ridge_params.csv")
    ridge_selection = pd.read_csv(out / "ridge_lambda_selection.csv")
    assert set(ridge_params.asset) == set(ASSETS[:-1])
    assert set(ridge_selection["penalty"]) == {"ridge"}
    lightgbm_params = pd.read_csv(out / "lightgbm_params.csv")
    assert set(lightgbm_params.asset) == set(ASSETS[:-1])
    assert (lightgbm_params["status"] == "ok").all()
    assert lightgbm_params["selected_num_leaves"].notna().all()
    importance = pd.read_csv(out / "lightgbm_importance.csv")
    assert {"asset", "feature", "importance_gain", "importance_gain_share"}.issubset(importance.columns)
    assert set(importance["asset"]) == set(ASSETS[:-1])
    assert (importance["importance_gain"] >= 0).all()
    worst_errors = pd.read_csv(out / "worst_error_dates.csv")
    assert {"asset", "date", "lgb_rv", "garch_rv", "abs_err_lgb_rv"}.issubset(worst_errors.columns)
    expanding_comparison = pd.read_csv(out / "expanding_comparison.csv")
    assert set(expanding_comparison["protocol"]) == {"locked", "expanding"}
    assert set(expanding_comparison["model"]) == {"har_rv", "garch_rv", "ridge_rv", "lgb_rv"}
    assert set(expanding_comparison["asset"]) == set(ASSETS[:-1]) | {"ALL"}
    assert {"protocol", "mae", "rmse", "qlike"}.issubset(expanding_comparison.columns)
    expanding_dm = pd.read_csv(out / "expanding_dm.csv")
    assert {"har_rv_exp", "garch_rv_exp", "ridge_rv_exp", "lgb_rv_exp"}.issubset(set(expanding_dm["model_a"]))
    assert {"har_rv_exp", "garch_rv_exp"}.issubset(set(expanding_dm["model_b"]))
    assert "historical_rv_21d" in set(expanding_dm["model_b"])
    comparison = pd.read_csv(out / "test_model_comparison.csv")
    assert {"lgb_rv"}.issubset(set(comparison["forecast"]))
    vix_incremental = pd.read_csv(out / "vix_incremental_comparison.csv")
    assert set(vix_incremental["model_a"]) == {"har_vix_rv"}
    assert set(vix_incremental["model_b"]) == {"har_rv"}
    saved_metadata = json.loads((out / "metadata.json").read_text())
    assert saved_metadata["qlike_scale"] == "variance"
    assert saved_metadata["dm_test"]["primary_loss"] == "qlike"
    assert saved_metadata["dm_test"]["losses"] == ["qlike", "mae"]
    assert saved_metadata["ewma"]["test_evaluation_locked"] is True
    assert saved_metadata["walk_forward"]["forecast_state"] == "computed continuously across segment boundaries"
    assert saved_metadata["lightgbm"]["forecast_column"] == "lgb_rv"
    assert saved_metadata["lightgbm"]["parameters_locked"] is True
    assert saved_metadata["expanding_window"]["first_eval_year_equals_locked"] is True
    assert saved_metadata["expanding_window"]["models"]
    strategy_metrics = pd.read_csv(out / "strategy_metrics.csv")
    assert {"historical_rv_21d", "fixed_100pct"}.issubset(set(strategy_metrics["forecast"]))
    assert "lgb_rv" in set(strategy_metrics["forecast"])
    transmission = pd.read_csv(out / "transmission_waterfall.csv")
    assert set(transmission["stage"]) == {
        "no cap / no lag / no cost", "leverage cap", "+ one-day lag", "+ transaction cost",
    }
    portfolio = pd.read_csv(out / "portfolio_metrics.csv")
    assert set(portfolio["scheme"]) == {"equal", "inverse_historical", "inverse_forecast"}
    alt_label_metrics = pd.read_csv(out / "alt_label_metrics.csv")
    assert set(alt_label_metrics["label"]) == {"future_rv_parkinson_5d", "future_rv_garman_klass_5d"}
    assert "lgb_rv" in set(alt_label_metrics["forecast"])
    alt_label_dm = pd.read_csv(out / "alt_label_dm.csv")
    assert {"garch_rv", "ridge_rv", "lgb_rv"}.issubset(set(alt_label_dm["model_a"]))
    cost_sensitivity = pd.read_csv(out / "strategy_cost_sensitivity.csv")
    assert set(cost_sensitivity["cost_bps"]) == {10.0, 20.0}
    yearly_metrics = pd.read_csv(out / "yearly_metrics.csv")
    assert {"year", "segment", "forecast", "qlike_vs_historical"}.issubset(yearly_metrics.columns)
    assert "lgb_rv" in set(yearly_metrics["forecast"])
    assert saved_metadata["strategy"]["evaluation_segment"] == "test"


def test_missing_column_and_duplicate_date_fail_clearly(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Date": ["2020-01-01"], "Close": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing Adj Close"):
        read_asset_csv(path, "SPY")

    duplicate = pd.DataFrame({
        "Date": ["2020-01-01", "2020-01-01"], "Open": [1, 1], "High": [2, 2],
        "Low": [0.5, 0.5], "Close": [1, 1], "Adj Close": [1, 1], "Volume": [1, 1],
    })
    duplicate.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate dates"):
        read_asset_csv(path, "SPY")
