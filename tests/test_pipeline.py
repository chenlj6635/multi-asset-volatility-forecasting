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
    predictions = pd.read_parquet(expected[2])
    assert set(predictions.asset) == set(ASSETS[:-1])
    assert {"historical_rv_21d", "ewma_rv", "har_rv", "ridge_rv", "garch_rv", "har_vix_rv"}.issubset(predictions.columns)
    assert predictions.groupby("asset")["future_rv_5d"].tail(5).isna().all()
    metrics = pd.read_csv(expected[3])
    assert metrics.asset.tolist() == ["GLD", "IWM", "QQQ", "SPY", "TLT", "USO", "ALL"] * 6
    assert set(metrics.forecast) == {"historical_rv_21d", "ewma_rv", "har_rv", "ridge_rv", "garch_rv", "har_vix_rv"}
    assert np.isfinite(metrics[["mae", "rmse", "qlike"]].to_numpy()).all()
    walk_metrics = pd.read_csv(expected[4])
    assert set(walk_metrics.segment) == {"train", "validation", "test"}
    assert set(walk_metrics.forecast) == {"historical_rv_21d", "ewma_rv", "har_rv", "ridge_rv", "garch_rv", "har_vix_rv"}
    dm = pd.read_csv(expected[5])
    assert set(dm.segment) == {"train", "validation", "test"}
    assert set(dm.asset) == set(ASSETS[:-1]) | {"ALL"}
    assert {"n_dates", "paired_rows"}.issubset(dm.columns)
    assert set(dm.loss) == {"qlike", "mae"}
    assert len(dm) == 240
    selection = pd.read_csv(expected[6])
    assert len(selection) == 4
    assert selection.selected.sum() == 1
    assert selection.selection_segment.iloc[0] == "validation"
    coefficients = pd.read_csv(expected[7])
    vix_coefficients = pd.read_csv(expected[8])
    assert set(vix_coefficients.asset) == set(ASSETS[:-1])
    garch_params = pd.read_csv(expected[9])
    assert set(garch_params.asset) == set(ASSETS[:-1])
    ridge_params = pd.read_csv(expected[10])
    ridge_selection = pd.read_csv(expected[11])
    assert set(ridge_params.asset) == set(ASSETS[:-1])
    assert set(ridge_selection["penalty"]) == {"ridge"}
    comparison = pd.read_csv(expected[12])
    vix_incremental = pd.read_csv(expected[13])
    assert set(vix_incremental["model_a"]) == {"har_vix_rv"}
    assert set(vix_incremental["model_b"]) == {"har_rv"}
    saved_metadata = json.loads(expected[16].read_text())
    assert saved_metadata["qlike_scale"] == "variance"
    assert saved_metadata["dm_test"]["primary_loss"] == "qlike"
    assert saved_metadata["dm_test"]["losses"] == ["qlike", "mae"]
    assert saved_metadata["ewma"]["test_evaluation_locked"] is True
    assert saved_metadata["walk_forward"]["forecast_state"] == "computed continuously across segment boundaries"
    strategy_metrics = pd.read_csv(expected[-7])
    assert {"historical_rv_21d", "fixed_100pct"}.issubset(set(strategy_metrics["forecast"]))
    transmission = pd.read_csv(expected[-6])
    assert set(transmission["stage"]) == {
        "no cap / no lag / no cost", "leverage cap", "+ one-day lag", "+ transaction cost",
    }
    portfolio = pd.read_csv(expected[-5])
    assert set(portfolio["scheme"]) == {"equal", "inverse_historical", "inverse_forecast"}
    alt_label_metrics = pd.read_csv(expected[-4])
    assert set(alt_label_metrics["label"]) == {"future_rv_parkinson_5d", "future_rv_garman_klass_5d"}
    alt_label_dm = pd.read_csv(expected[-3])
    assert {"garch_rv", "ridge_rv"}.issubset(set(alt_label_dm["model_a"]))
    cost_sensitivity = pd.read_csv(expected[-2])
    assert set(cost_sensitivity["cost_bps"]) == {10.0, 20.0}
    yearly_metrics = pd.read_csv(expected[-1])
    assert {"year", "segment", "forecast", "qlike_vs_historical"}.issubset(yearly_metrics.columns)
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
