#!/usr/bin/env python3
"""Build baseline outputs exclusively from local raw CSV files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_assets
from src.features import add_ewma_volatility_baseline, add_historical_volatility_baseline
from src.labels import add_future_realized_volatility, add_log_returns
from src.metrics import metrics_by_asset
from src.reporting import build_quality_report, plot_spy_comparison, write_quality_report, write_results


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    root = config_path.parent.parent
    data_config = config["data"]
    calculation = config["calculation"]
    outputs = config["outputs"]

    target_map = data_config["target_assets"]
    context_map = data_config.get("context_assets", {})
    file_map = {**target_map, **context_map}
    raw_dir = resolve(root, data_config["raw_dir"])

    market_data, warnings = load_raw_assets(
        raw_dir,
        file_map,
        allow_close_as_adjusted=bool(data_config.get("allow_close_as_adjusted", False)),
    )
    quality_table, quality_summary = build_quality_report(
        market_data,
        warnings=warnings,
        extreme_return_threshold=float(calculation["extreme_return_threshold"]),
        long_gap_days=int(calculation["long_gap_days"]),
    )
    write_quality_report(quality_table, quality_summary, resolve(root, data_config["quality_dir"]))

    target_data = market_data.loc[market_data["asset"].isin(target_map)].copy()
    target_data = add_log_returns(target_data)
    target_data = add_future_realized_volatility(
        target_data,
        horizon=int(calculation["label_horizon"]),
        annualization_factor=float(calculation["annualization_factor"]),
        output_column="future_rv_5d",
    )
    target_data = add_historical_volatility_baseline(
        target_data,
        window=int(calculation["baseline_window"]),
        annualization_factor=float(calculation["annualization_factor"]),
        output_column="historical_rv_21d",
    )
    target_data = add_ewma_volatility_baseline(
        target_data,
        decay=float(calculation["ewma_lambda"]),
        min_periods=int(calculation["ewma_min_periods"]),
        annualization_factor=float(calculation["annualization_factor"]),
        output_column="ewma_rv",
    )
    prediction_columns = [
        "asset", "date", "adj_close", "log_return", "future_rv_5d", "historical_rv_21d", "ewma_rv"
    ]
    predictions = target_data[prediction_columns].reset_index(drop=True)
    metrics = metrics_by_asset(
        predictions,
        forecast_columns=("historical_rv_21d", "ewma_rv"),
        epsilon=float(calculation["qlike_epsilon"]),
    )

    raw_files = {
        asset: {
            "path": str(raw_dir / filename),
            "sha256": file_sha256(raw_dir / filename),
        }
        for asset, filename in file_map.items()
    }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "offline_build": True,
        "network_access": False,
        "target_assets": list(target_map),
        "context_assets": list(context_map),
        "annualization_factor": calculation["annualization_factor"],
        "label_horizon": calculation["label_horizon"],
        "label_returns": "strictly t+1 through t+5",
        "baseline_window": calculation["baseline_window"],
        "ewma": {
            "enabled": True,
            "lambda": calculation["ewma_lambda"],
            "min_periods": calculation["ewma_min_periods"],
            "output_column": "ewma_rv",
        },
        "forecast_columns": ["historical_rv_21d", "ewma_rv"],
        "qlike_scale": "variance",
        "raw_files": raw_files,
        "quality_status": quality_summary["status"],
        "prediction_rows": int(len(predictions)),
        "valid_evaluation_rows": {
            column: int(predictions[["future_rv_5d", column]].notna().all(axis=1).sum())
            for column in ["historical_rv_21d", "ewma_rv"]
        },
    }
    write_results(
        predictions,
        metrics,
        metadata,
        predictions_path=resolve(root, outputs["predictions"]),
        metrics_path=resolve(root, outputs["metrics"]),
        metadata_path=resolve(root, outputs["metadata"]),
    )
    plot_spy_comparison(predictions, resolve(root, outputs["figure"]))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    args = parser.parse_args()
    metadata = build(args.config)
    print(json.dumps({"status": "ok", **metadata}, indent=2))


if __name__ == "__main__":
    main()
