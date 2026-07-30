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
        "calculation": {
            "annualization_factor": 252,
            "label_horizon": 5,
            "baseline_window": 21,
            "ewma_lambda": 0.94,
            "ewma_min_periods": 21,
            "qlike_epsilon": 1e-12,
            "extreme_return_threshold": 0.20,
            "long_gap_days": 7,
        },
        "outputs": {
            "predictions": str(tmp_path / "processed/predictions.parquet"),
            "metrics": str(tmp_path / "outputs/metrics.csv"),
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
        tmp_path / "outputs/metadata.json",
        tmp_path / "outputs/figure.png",
    ]
    assert all(path.exists() and path.stat().st_size > 0 for path in expected)
    predictions = pd.read_parquet(expected[2])
    assert set(predictions.asset) == set(ASSETS[:-1])
    assert {"historical_rv_21d", "ewma_rv"}.issubset(predictions.columns)
    assert predictions.groupby("asset")["future_rv_5d"].tail(5).isna().all()
    metrics = pd.read_csv(expected[3])
    assert metrics.asset.tolist() == ["GLD", "IWM", "QQQ", "SPY", "TLT", "USO", "ALL"] * 2
    assert set(metrics.forecast) == {"historical_rv_21d", "ewma_rv"}
    assert np.isfinite(metrics[["mae", "rmse", "qlike"]].to_numpy()).all()
    saved_metadata = json.loads(expected[4].read_text())
    assert saved_metadata["qlike_scale"] == "variance"


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
