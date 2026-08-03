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
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_assets
from src.evaluation import (
    assign_walk_forward_segments,
    dm_by_segment,
    exclude_cross_segment_labels,
    select_ewma_lambda,
    fit_har_by_asset,
    assign_test_volatility_regimes,
    regime_robustness_summary,
    asset_robustness_summary,
    test_model_comparison,
    walk_forward_metrics,
)
from src.features import add_ewma_volatility_candidates, add_har_features, add_historical_volatility_baseline, add_vix_level, fit_har_vix_by_asset
from src.labels import add_future_realized_volatility, add_log_returns
from src.metrics import metrics_by_asset
from src.reporting import build_quality_report, plot_spy_comparison, write_metrics, write_quality_report, write_results


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
    target_data = add_ewma_volatility_candidates(
        target_data,
        decays=tuple(float(value) for value in calculation["ewma_lambdas"]),
        min_periods=int(calculation["ewma_min_periods"]),
        annualization_factor=float(calculation["annualization_factor"]),
    )
    target_data = add_har_features(
        target_data,
        annualization_factor=float(calculation["annualization_factor"]),
    )
    target_data = add_vix_level(target_data, market_data.loc[market_data["asset"] == "^VIX"])
    candidate_lambdas = tuple(float(value) for value in calculation["ewma_lambdas"])
    candidate_columns = [f"ewma_rv_lambda_{value:g}" for value in candidate_lambdas]
    prediction_columns = [
        "asset", "date", "adj_close", "log_return", "future_rv_5d", "historical_rv_21d", *candidate_columns,
        "har_daily_rv", "har_weekly_rv", "har_monthly_rv", "log_vix",
    ]
    predictions = target_data[prediction_columns].reset_index(drop=True)
    walk_config = config["walk_forward"]
    segmented = assign_walk_forward_segments(
        predictions,
        train_end=walk_config["train_end"],
        validation_end=walk_config["validation_end"],
    )
    segmented, excluded_rows = exclude_cross_segment_labels(
        segmented,
        horizon=int(calculation["label_horizon"]),
    )
    selected_lambda, lambda_selection = select_ewma_lambda(
        segmented,
        lambdas=tuple(float(value) for value in calculation["ewma_lambdas"]),
        epsilon=float(calculation["qlike_epsilon"]),
    )
    selected_column = f"ewma_rv_lambda_{selected_lambda:g}"
    segmented["ewma_rv"] = segmented[selected_column]
    predictions["ewma_rv"] = predictions[selected_column]
    segmented, har_coefficients = fit_har_by_asset(segmented)
    predictions["har_rv"] = segmented.set_index(["asset", "date"]).reindex(predictions.set_index(["asset", "date"]).index)["har_rv"].to_numpy()
    segmented, har_vix_coefficients = fit_har_vix_by_asset(segmented, variance_floor=float(calculation["forecast_variance_floor"]))
    predictions["har_vix_rv"] = segmented.set_index(["asset", "date"]).reindex(predictions.set_index(["asset", "date"]).index)["har_vix_rv"].to_numpy()
    predictions["log_vix_z"] = segmented.set_index(["asset", "date"]).reindex(predictions.set_index(["asset", "date"]).index)["log_vix_z"].to_numpy()
    predictions["har_vix_variance_raw"] = segmented.set_index(["asset", "date"]).reindex(predictions.set_index(["asset", "date"]).index)["har_vix_variance_raw"].to_numpy()
    metrics = metrics_by_asset(
        predictions,
        forecast_columns=("historical_rv_21d", "ewma_rv", "har_rv", "har_vix_rv"),
        epsilon=float(calculation["qlike_epsilon"]),
    )
    walk_metrics = walk_forward_metrics(
        segmented,
        forecast_columns=("historical_rv_21d", "ewma_rv", "har_rv", "har_vix_rv"),
        epsilon=float(calculation["qlike_epsilon"]),
    )
    comparison = pd.concat([
        test_model_comparison(segmented, asset=asset)
        for asset in [*sorted(segmented["asset"].unique()), "ALL"]
    ], ignore_index=True)
    robustness = asset_robustness_summary(segmented)
    regime_data, regime_thresholds = assign_test_volatility_regimes(segmented)
    regime_robustness = regime_robustness_summary(regime_data)
    dm_config = config["dm_test"]
    dm_results = pd.concat([
        dm_by_segment(segmented, max_lag=int(dm_config["hac_lag"]), epsilon=float(calculation["qlike_epsilon"]), losses=tuple(dm_config.get("losses", [dm_config.get("loss", "qlike")])), model_a_column="ewma_rv", model_b_column="historical_rv_21d"),
        dm_by_segment(segmented, max_lag=int(dm_config["hac_lag"]), epsilon=float(calculation["qlike_epsilon"]), losses=tuple(dm_config.get("losses", [dm_config.get("loss", "qlike")])), model_a_column="har_rv", model_b_column="historical_rv_21d"),
        dm_by_segment(segmented, max_lag=int(dm_config["hac_lag"]), epsilon=float(calculation["qlike_epsilon"]), losses=tuple(dm_config.get("losses", [dm_config.get("loss", "qlike")])), model_a_column="har_vix_rv", model_b_column="har_rv"),
    ], ignore_index=True) if bool(dm_config.get("enabled", True)) else pd.DataFrame()
    vix_incremental = dm_results.loc[(dm_results["model_a"] == "har_vix_rv") & (dm_results["model_b"] == "har_rv")].copy()
    vix_diagnostics = segmented.groupby("segment").agg(vix_nonmissing=("log_vix", "count"), vix_missing=("log_vix", lambda values: int(values.isna().sum())), log_vix_z_nonmissing=("log_vix_z", "count"), log_vix_z_missing=("log_vix_z", lambda values: int(values.isna().sum()))).reset_index()
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
            "candidate_lambdas": list(candidate_lambdas),
            "selected_lambda": selected_lambda,
            "selection_segment": "validation",
            "selection_metric": "pooled_qlike",
            "tie_break": "smallest lambda among equal validation QLIKE",
            "test_evaluation_locked": True,
            "min_periods": calculation["ewma_min_periods"],
            "output_column": "ewma_rv",
        },
        "har": {
            "enabled": True,
            "target": "future_rv_5d_squared",
            "features": ["har_daily_rv", "har_weekly_rv", "har_monthly_rv"],
            "training_segment": "train",
            "parameters_locked": True,
            "coefficients_output": outputs["har_coefficients"],
            "coefficient_assets": int(len(har_coefficients)),
            "coefficient_columns": list(har_coefficients.columns),
            "forecast_column": "har_rv",
        },
        "regime_robustness": {
            "definition": "test future_rv_5d pooled tertiles",
            "thresholds": regime_thresholds,
            "models": ["historical_rv_21d", "ewma_rv", "har_rv"],
            "output": outputs["regime_robustness"],
        },
        "vix_incremental": {
            "comparison": "har_vix_rv_minus_har_rv",
            "output": outputs["vix_incremental_comparison"],
            "diagnostics": vix_diagnostics.to_dict(orient="records"),
            "experimental_only": True,
        },
        "har_vix": {
            "enabled": True,
            "alignment": "exact_date_left_join",
            "source_asset": "^VIX",
            "source_column": "adj_close",
            "feature_column": "log_vix_z",
            "raw_feature_column": "log_vix",
            "standardization": "train_global_mean_std",
            "variance_floor": calculation["forecast_variance_floor"],
            "missing_policy": "retain_nan_no_future_fill",
            "training_segment": "train",
            "parameters_locked": True,
            "coefficients_output": outputs["har_vix_coefficients"],
            "coefficient_assets": int(len(har_vix_coefficients)),
            "forecast_column": "har_vix_rv",
        },
        "forecast_columns": ["historical_rv_21d", "ewma_rv", "har_rv", "har_vix_rv"],
        "qlike_scale": "variance",
        "raw_files": raw_files,
        "quality_status": quality_summary["status"],
        "prediction_rows": int(len(predictions)),
        "valid_evaluation_rows": {
            column: int(predictions[["future_rv_5d", column]].notna().all(axis=1).sum())
            for column in ["historical_rv_21d", "ewma_rv", "har_rv", "har_vix_rv"]
        },
        "ewma_lambda_selection": lambda_selection.to_dict(orient="records"),
        "dm_test": {
            "enabled": bool(dm_config.get("enabled", True)),
            "losses": list(dm_config.get("losses", [dm_config.get("loss", "qlike")])),
            "primary_loss": dm_config.get("primary_loss", "qlike"),
            "comparisons": [
                {"model_a": "ewma_rv", "model_b": "historical_rv_21d"},
                {"model_a": "har_rv", "model_b": "historical_rv_21d"},
                {"model_a": "har_rv", "model_b": "ewma_rv"},
                {"model_a": "har_vix_rv", "model_b": "har_rv"},
            ],
            "pooled_rule": "cross-sectional mean by date before HAC",
            "model_order": "model_a_minus_model_b",
            "hac_method": "Bartlett HAC",
            "hac_lag": int(dm_config["hac_lag"]),
            "asset_robustness_output": outputs["asset_robustness"],
            "regime_robustness_output": outputs["regime_robustness"],
        },
        "walk_forward": {
            "train_end": str(walk_config["train_end"]),
            "validation_end": str(walk_config["validation_end"]),
            "segment_rule": "segment assigned by forecast date t",
            "cross_segment_label_rows_excluded": excluded_rows,
            "forecast_state": "computed continuously across segment boundaries",
            "valid_rows": {
                segment: {
                    forecast: int(walk_metrics.loc[
                        (walk_metrics["segment"] == segment)
                        & (walk_metrics["forecast"] == forecast)
                        & (walk_metrics["asset"] == "ALL"),
                        "n_obs",
                    ].iloc[0])
                    for forecast in ["historical_rv_21d", "ewma_rv", "har_rv"]
                }
                for segment in ["train", "validation", "test"]
            },
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
    write_metrics(walk_metrics, resolve(root, outputs["walk_forward_metrics"]))
    write_metrics(lambda_selection, resolve(root, outputs["ewma_lambda_selection"]))
    write_metrics(har_coefficients, resolve(root, outputs["har_coefficients"]))
    write_metrics(har_vix_coefficients, resolve(root, outputs["har_vix_coefficients"]))
    write_metrics(comparison, resolve(root, outputs["test_model_comparison"]))
    write_metrics(robustness, resolve(root, outputs["asset_robustness"]))
    write_metrics(regime_robustness, resolve(root, outputs["regime_robustness"]))
    write_metrics(vix_incremental, resolve(root, outputs["vix_incremental_comparison"]))
    if bool(dm_config.get("enabled", True)):
        write_metrics(dm_results, resolve(root, outputs["dm_tests"]))
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
