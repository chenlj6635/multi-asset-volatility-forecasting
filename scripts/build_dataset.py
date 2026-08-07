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
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_raw_assets
from src.evaluation import (
    assign_walk_forward_segments,
    dm_by_segment,
    exclude_cross_segment_labels,
    expanding_window_forecasts,
    select_ewma_lambda,
    fit_har_by_asset,
    assign_test_volatility_regimes,
    regime_robustness_summary,
    asset_robustness_summary,
    test_model_comparison,
    walk_forward_metrics,
    worst_error_summary,
)
from src.features import (
    add_ewma_volatility_candidates,
    add_har_features,
    add_historical_volatility_baseline,
    add_market_state_features,
    add_range_features,
    add_vix_level,
    fit_har_vix_by_asset,
)
from src.labels import add_future_realized_volatility, add_log_returns, add_range_based_future_volatility
from src.metrics import evaluate_forecast, metrics_by_asset
from src.mcs import build_pooled_loss_matrix, mcs_summary_frame, model_confidence_set
from src.models import fit_garch_by_asset, fit_lightgbm_by_asset, fit_ridge_by_asset
from src.reporting import build_quality_report, plot_spy_comparison, write_metrics, write_quality_report, write_results
from src.strategy import calibrate_forecast_level, portfolio_metrics, transmission_waterfall, vol_targeting_metrics

# All prediction columns including the experimental HAR-VIX variant.
ALL_FORECAST_COLUMNS = ("historical_rv_21d", "ewma_rv", "har_rv", "ridge_rv", "garch_rv", "lgb_rv", "har_vix_rv")
# Formal models used in comparisons, yearly metrics, and alternative labels.
FORMAL_FORECAST_COLUMNS = ("historical_rv_21d", "ewma_rv", "har_rv", "ridge_rv", "garch_rv", "lgb_rv")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve(root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else root / path


def redirect_paths(paths: dict[str, str], prefix: str) -> dict[str, str]:
    """Redirect configured output paths under a prefix (for robustness runs)."""
    prefix = Path(prefix)
    return {key: str(prefix / Path(value)) for key, value in paths.items()}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(
    config_path: str | Path,
    *,
    exclude_years: tuple[int, ...] = (),
    output_prefix: str | None = None,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    root = config_path.parent.parent
    data_config = dict(config["data"])
    calculation = config["calculation"]
    outputs = dict(config["outputs"])
    if output_prefix:
        outputs = redirect_paths(outputs, output_prefix)
        data_config["quality_dir"] = redirect_paths(
            {"quality_dir": data_config["quality_dir"]}, output_prefix
        )["quality_dir"]

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
    if exclude_years:
        years = {int(year) for year in exclude_years}
        target_data = target_data.loc[~target_data["date"].dt.year.isin(years)].copy()
    target_data = add_log_returns(target_data)
    target_data = add_future_realized_volatility(
        target_data,
        horizon=int(calculation["label_horizon"]),
        annualization_factor=float(calculation["annualization_factor"]),
        output_column="future_rv_5d",
    )
    target_data = add_range_based_future_volatility(
        target_data,
        horizon=int(calculation["label_horizon"]),
        annualization_factor=float(calculation["annualization_factor"]),
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
    target_data = add_range_features(target_data, annualization_factor=float(calculation["annualization_factor"]))
    target_data = add_market_state_features(target_data)
    candidate_lambdas = tuple(float(value) for value in calculation["ewma_lambdas"])
    candidate_columns = [f"ewma_rv_lambda_{value:g}" for value in candidate_lambdas]
    ridge_config = calculation.get("ridge", {}) or {}
    ridge_feature_columns = [
        "har_daily_rv", "har_weekly_rv", "har_monthly_rv",
        "parkinson_5d", "parkinson_22d", "garman_klass_22d",
        "rel_return_5d", "rel_return_21d", "downside_frac_21d",
        "close_to_ma_21d", "drawdown_21d", "volume_ratio_21d",
        "log_vix", "vix_change_5d",
    ]
    prediction_columns = list(dict.fromkeys([
        "asset", "date", "adj_close", "log_return", "future_rv_5d", "future_rv_parkinson_5d", "future_rv_garman_klass_5d",
        "historical_rv_21d", *candidate_columns,
        "har_daily_rv", "har_weekly_rv", "har_monthly_rv", "log_vix", *ridge_feature_columns,
    ]))
    predictions = target_data[prediction_columns].reset_index(drop=True)
    walk_config = config["walk_forward"]
    segmented = assign_walk_forward_segments(
        predictions,
        train_end=walk_config["train_end"],
        validation_end=walk_config["validation_end"],
    )
    segmented_full = segmented.copy()
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
    har_vix_config = calculation.get("har_vix", {}) or {}
    segmented, har_vix_coefficients = fit_har_vix_by_asset(
        segmented,
        log_floor=float(har_vix_config.get("log_floor", 1.0e-12)),
        smearing=bool(har_vix_config.get("smearing", True)),
    )
    predictions["har_vix_rv"] = segmented.set_index(["asset", "date"]).reindex(predictions.set_index(["asset", "date"]).index)["har_vix_rv"].to_numpy()
    predictions["log_vix_z"] = segmented.set_index(["asset", "date"]).reindex(predictions.set_index(["asset", "date"]).index)["log_vix_z"].to_numpy()
    predictions["har_vix_logvar"] = segmented.set_index(["asset", "date"]).reindex(predictions.set_index(["asset", "date"]).index)["har_vix_logvar"].to_numpy()
    garch_config = calculation.get("garch", {}) or {}
    segmented, garch_params = fit_garch_by_asset(
        segmented,
        horizon=int(calculation["label_horizon"]),
        annualization_factor=float(calculation["annualization_factor"]),
        p=int(garch_config.get("p", 1)),
        q=int(garch_config.get("q", 1)),
        min_train_observations=int(garch_config.get("min_train_observations", 120)),
    )
    predictions["garch_rv"] = segmented.set_index(["asset", "date"]).reindex(predictions.set_index(["asset", "date"]).index)["garch_rv"].to_numpy()
    ridge_penalty = str(ridge_config.get("penalty", "ridge"))
    ridge_grid = [float(value) for value in ridge_config.get(
        "lasso_lambda_grid" if ridge_penalty == "lasso" else "lambda_grid",
        [0.0, 0.01, 0.1, 1.0, 10.0, 100.0] if ridge_penalty == "ridge" else [0.0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
    )]
    segmented, ridge_params, ridge_selection = fit_ridge_by_asset(
        segmented,
        feature_columns=ridge_feature_columns,
        penalty=ridge_penalty,
        lambda_grid=ridge_grid,
        variance_floor=float(calculation["forecast_variance_floor"]),
        epsilon=float(calculation["qlike_epsilon"]),
        min_train_observations=int(ridge_config.get("min_train_observations", 120)),
        min_validation_observations=int(ridge_config.get("min_validation_observations", 20)),
    )
    predictions["ridge_rv"] = segmented.set_index(["asset", "date"]).reindex(predictions.set_index(["asset", "date"]).index)["ridge_rv"].to_numpy()
    lightgbm_config = calculation.get("lightgbm", {}) or {}
    lgb_feature_columns = [str(value) for value in lightgbm_config.get("feature_columns", [])] or ridge_feature_columns
    segmented, lightgbm_params, lightgbm_selection, lightgbm_importance = fit_lightgbm_by_asset(
        segmented,
        feature_columns=lgb_feature_columns,
        num_leaves_grid=tuple(int(value) for value in lightgbm_config.get("num_leaves", [8, 31])),
        learning_rate_grid=tuple(float(value) for value in lightgbm_config.get("learning_rate", [0.05, 0.1])),
        n_estimators_grid=tuple(int(value) for value in lightgbm_config.get("n_estimators", [100, 300])),
        min_child_samples=int(lightgbm_config.get("min_child_samples", 20)),
        subsample=float(lightgbm_config.get("subsample", 0.8)),
        colsample_bytree=float(lightgbm_config.get("colsample_bytree", 0.8)),
        reg_lambda=float(lightgbm_config.get("reg_lambda", 1.0)),
        random_state=int(lightgbm_config.get("random_state", 42)),
        variance_floor=float(calculation["forecast_variance_floor"]),
        epsilon=float(calculation["qlike_epsilon"]),
        min_train_observations=int(lightgbm_config.get("min_train_observations", 120)),
        min_validation_observations=int(lightgbm_config.get("min_validation_observations", 20)),
    )
    predictions["lgb_rv"] = segmented.set_index(["asset", "date"]).reindex(predictions.set_index(["asset", "date"]).index)["lgb_rv"].to_numpy()
    exp_config = config.get("expanding_window", {}) or {}
    exp_validation_years = int(exp_config.get("validation_years", 2))
    exp_models: list[str] = []
    exp_eval_years: tuple[int, ...] = ()
    if exp_config.get("enabled", True):
        exp_models = [str(value) for value in exp_config.get("models", ["har_rv", "garch_rv", "ridge_rv", "lgb_rv"])]
        exp_horizon = int(calculation["label_horizon"])
        exp_eval_years = tuple(sorted(
            int(year) for year in set(segmented["date"].dt.year)
            if segmented.loc[segmented["date"].dt.year == year, "segment"].eq("test").any()
        ))
        exp_function_map = {
            "har_rv": fit_har_by_asset,
            "garch_rv": fit_garch_by_asset,
            "ridge_rv": fit_ridge_by_asset,
            "lgb_rv": fit_lightgbm_by_asset,
        }
        exp_fit_kwargs = {
            "har_rv": {"train_segment": "train", "actual_column": "future_rv_5d"},
            "garch_rv": {
                "train_segment": "train", "horizon": exp_horizon,
                "annualization_factor": float(calculation["annualization_factor"]),
                "p": int(garch_config.get("p", 1)), "q": int(garch_config.get("q", 1)),
                "min_train_observations": int(garch_config.get("min_train_observations", 120)),
            },
            "ridge_rv": {
                "train_segment": "train", "validation_segment": "validation",
                "feature_columns": ridge_feature_columns, "actual_column": "future_rv_5d",
                "penalty": ridge_penalty, "lambda_grid": ridge_grid,
                "variance_floor": float(calculation["forecast_variance_floor"]),
                "epsilon": float(calculation["qlike_epsilon"]),
                "min_train_observations": int(ridge_config.get("min_train_observations", 120)),
                "min_validation_observations": int(ridge_config.get("min_validation_observations", 20)),
            },
            "lgb_rv": {
                "train_segment": "train", "validation_segment": "validation",
                "feature_columns": lgb_feature_columns, "actual_column": "future_rv_5d",
                "num_leaves_grid": tuple(int(value) for value in lightgbm_config.get("num_leaves", [8, 31])),
                "learning_rate_grid": tuple(float(value) for value in lightgbm_config.get("learning_rate", [0.05, 0.1])),
                "n_estimators_grid": tuple(int(value) for value in lightgbm_config.get("n_estimators", [100, 300])),
                "min_child_samples": int(lightgbm_config.get("min_child_samples", 20)),
                "subsample": float(lightgbm_config.get("subsample", 0.8)),
                "colsample_bytree": float(lightgbm_config.get("colsample_bytree", 0.8)),
                "reg_lambda": float(lightgbm_config.get("reg_lambda", 1.0)),
                "random_state": int(lightgbm_config.get("random_state", 42)),
                "variance_floor": float(calculation["forecast_variance_floor"]),
                "epsilon": float(calculation["qlike_epsilon"]),
                "min_train_observations": int(lightgbm_config.get("min_train_observations", 120)),
                "min_validation_observations": int(lightgbm_config.get("min_validation_observations", 20)),
            },
        }
        exp_param_parts: list[pd.DataFrame] = []
        for exp_model in exp_models:
            if exp_model not in exp_function_map:
                continue
            segmented_full, exp_params = expanding_window_forecasts(
                segmented_full,
                fit_function=exp_function_map[exp_model],
                output_column=f"{exp_model}_exp",
                eval_years=exp_eval_years,
                validation_years=exp_validation_years,
                horizon=exp_horizon,
                fit_kwargs={**exp_fit_kwargs[exp_model], "output_column": f"{exp_model}_exp"},
            )
            segmented[f"{exp_model}_exp"] = segmented_full[f"{exp_model}_exp"].reindex(segmented.index)
            predictions[f"{exp_model}_exp"] = segmented_full.set_index(["asset", "date"]).reindex(predictions.set_index(["asset", "date"]).index)[f"{exp_model}_exp"].to_numpy()
            if not exp_params.empty:
                exp_param_parts.append(exp_params)
        expanding_params = pd.concat(exp_param_parts, ignore_index=True) if exp_param_parts else pd.DataFrame()
        exp_cols = tuple(f"{model}_exp" for model in exp_models)
        locked_metrics = walk_forward_metrics(
            segmented, forecast_columns=tuple(exp_models), epsilon=float(calculation["qlike_epsilon"]),
        )
        expanding_metrics = walk_forward_metrics(
            segmented, forecast_columns=exp_cols, epsilon=float(calculation["qlike_epsilon"]),
        )
        comparison_part = pd.concat([locked_metrics, expanding_metrics], ignore_index=True)
        comparison_part = comparison_part.loc[comparison_part["segment"] == "test"].copy()
        comparison_part["protocol"] = np.where(
            comparison_part["forecast"].str.endswith("_exp"), "expanding", "locked"
        )
        comparison_part["model"] = np.where(
            comparison_part["protocol"] == "expanding",
            comparison_part["forecast"].str[:-4],
            comparison_part["forecast"],
        )
        expanding_comparison = comparison_part[
            ["segment", "asset", "model", "protocol", "n_obs", "mae", "rmse", "qlike", "variance_floor_count"]
        ]
        dm_parts: list[pd.DataFrame] = []
        expanding_dm_config = config.get("dm_test", {})
        dm_args = {
            "max_lag": int(expanding_dm_config["hac_lag"]),
            "epsilon": float(calculation["qlike_epsilon"]),
            "losses": tuple(expanding_dm_config.get("losses", [expanding_dm_config.get("loss", "qlike")])),
        }
        for exp_model in exp_models:
            dm_parts.append(dm_by_segment(segmented, model_a_column=f"{exp_model}_exp", model_b_column="historical_rv_21d", **dm_args))
        for model_a, model_b in (
            ("garch_rv_exp", "har_rv_exp"),
            ("ridge_rv_exp", "garch_rv_exp"),
            ("lgb_rv_exp", "garch_rv_exp"),
            ("lgb_rv_exp", "har_rv_exp"),
        ):
            dm_parts.append(dm_by_segment(segmented, model_a_column=model_a, model_b_column=model_b, **dm_args))
        for exp_model in exp_models:
            dm_parts.append(dm_by_segment(segmented, model_a_column=f"{exp_model}_exp", model_b_column=exp_model, **dm_args))
        expanding_dm = pd.concat(dm_parts, ignore_index=True)
    else:
        expanding_params = pd.DataFrame()
        expanding_comparison = pd.DataFrame()
        expanding_dm = pd.DataFrame()
        exp_cols: tuple[str, ...] = ()
    calib_config = config.get("calibration", {}) or {}
    calib_specs: list[dict[str, object]] = []
    if calib_config.get("enabled", True):
        calib_source = str(calib_config.get("source_segment", "validation"))
        specs = calib_config.get("specs")
        if specs:
            calib_specs = []
            for spec in specs:
                calib_model = str(spec["model"])
                calib_specs.append({
                    "model": calib_model,
                    "method": str(spec.get("method", "variance_rms")),
                    "column": f"{calib_model}_{spec.get('suffix', 'cal')}",
                    "n_buckets": int(spec.get("n_buckets", 3)),
                })
        else:  # legacy single-method config
            calib_method = str(calib_config.get("method", "variance_rms"))
            calib_specs = [
                {"model": str(model), "method": calib_method, "column": f"{model}_cal", "n_buckets": 3}
                for model in calib_config.get("models", ["lgb_rv"])
            ]
        for spec in calib_specs:
            segmented = calibrate_forecast_level(
                segmented,
                forecast_column=spec["model"],
                output_column=spec["column"],
                source_segment=calib_source,
                method=spec["method"],
                n_buckets=spec["n_buckets"],
            )
            predictions[spec["column"]] = segmented.set_index(["asset", "date"]).reindex(predictions.set_index(["asset", "date"]).index)[spec["column"]].to_numpy()
    metrics = metrics_by_asset(
        predictions,
        forecast_columns=ALL_FORECAST_COLUMNS,
        epsilon=float(calculation["qlike_epsilon"]),
    )
    walk_metrics = walk_forward_metrics(
        segmented,
        forecast_columns=ALL_FORECAST_COLUMNS,
        epsilon=float(calculation["qlike_epsilon"]),
    )
    yearly_rows: list[dict[str, float | int | str]] = []
    yearly_forecasts = FORMAL_FORECAST_COLUMNS
    yearly_frame = segmented.assign(year=segmented["date"].dt.year)
    for (year, segment), group in yearly_frame.groupby(["year", "segment"]):
        baseline_qlike = evaluate_forecast(
            group["future_rv_5d"], group["historical_rv_21d"], epsilon=float(calculation["qlike_epsilon"])
        ).qlike
        for forecast in yearly_forecasts:
            result = evaluate_forecast(
                group["future_rv_5d"], group[forecast], epsilon=float(calculation["qlike_epsilon"])
            )
            yearly_rows.append({
                "year": int(year),
                "segment": segment,
                "forecast": forecast,
                "n_obs": result.n_obs,
                "mae": result.mae,
                "rmse": result.rmse,
                "qlike": result.qlike,
                "qlike_vs_historical": (result.qlike - baseline_qlike) if np.isfinite(result.qlike) and np.isfinite(baseline_qlike) else np.nan,
            })
    yearly_metrics = pd.DataFrame(yearly_rows)
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
        dm_by_segment(segmented, max_lag=int(dm_config["hac_lag"]), epsilon=float(calculation["qlike_epsilon"]), losses=tuple(dm_config.get("losses", [dm_config.get("loss", "qlike")])), model_a_column="garch_rv", model_b_column="historical_rv_21d"),
        dm_by_segment(segmented, max_lag=int(dm_config["hac_lag"]), epsilon=float(calculation["qlike_epsilon"]), losses=tuple(dm_config.get("losses", [dm_config.get("loss", "qlike")])), model_a_column="garch_rv", model_b_column="har_rv"),
        dm_by_segment(segmented, max_lag=int(dm_config["hac_lag"]), epsilon=float(calculation["qlike_epsilon"]), losses=tuple(dm_config.get("losses", [dm_config.get("loss", "qlike")])), model_a_column="ridge_rv", model_b_column="historical_rv_21d"),
        dm_by_segment(segmented, max_lag=int(dm_config["hac_lag"]), epsilon=float(calculation["qlike_epsilon"]), losses=tuple(dm_config.get("losses", [dm_config.get("loss", "qlike")])), model_a_column="ridge_rv", model_b_column="har_rv"),
        dm_by_segment(segmented, max_lag=int(dm_config["hac_lag"]), epsilon=float(calculation["qlike_epsilon"]), losses=tuple(dm_config.get("losses", [dm_config.get("loss", "qlike")])), model_a_column="ridge_rv", model_b_column="garch_rv"),
        dm_by_segment(segmented, max_lag=int(dm_config["hac_lag"]), epsilon=float(calculation["qlike_epsilon"]), losses=tuple(dm_config.get("losses", [dm_config.get("loss", "qlike")])), model_a_column="lgb_rv", model_b_column="historical_rv_21d"),
        dm_by_segment(segmented, max_lag=int(dm_config["hac_lag"]), epsilon=float(calculation["qlike_epsilon"]), losses=tuple(dm_config.get("losses", [dm_config.get("loss", "qlike")])), model_a_column="lgb_rv", model_b_column="har_rv"),
        dm_by_segment(segmented, max_lag=int(dm_config["hac_lag"]), epsilon=float(calculation["qlike_epsilon"]), losses=tuple(dm_config.get("losses", [dm_config.get("loss", "qlike")])), model_a_column="lgb_rv", model_b_column="garch_rv"),
        dm_by_segment(segmented, max_lag=int(dm_config["hac_lag"]), epsilon=float(calculation["qlike_epsilon"]), losses=tuple(dm_config.get("losses", [dm_config.get("loss", "qlike")])), model_a_column="har_vix_rv", model_b_column="har_rv"),
    ], ignore_index=True) if bool(dm_config.get("enabled", True)) else pd.DataFrame()
    vix_incremental = dm_results.loc[(dm_results["model_a"] == "har_vix_rv") & (dm_results["model_b"] == "har_rv")].copy()
    mcs_config = config.get("mcs", {}) or {}
    mcs_parts: list[pd.DataFrame] = []
    if mcs_config.get("enabled", True):
        mcs_losses = tuple(str(value) for value in mcs_config.get("losses", ["qlike", "mae"]))
        mcs_alpha = float(mcs_config.get("alpha", 0.10))
        mcs_bootstrap = int(mcs_config.get("bootstrap", 10000))
        mcs_hac_lag = int((config.get("dm_test", {}) or {}).get("hac_lag", 4))
        test_frame = segmented.loc[segmented["segment"] == "test"]

        locked_mcs_models = [str(value) for value in mcs_config.get("lock_models", FORMAL_FORECAST_COLUMNS)]
        locked_present = [model for model in locked_mcs_models if model in test_frame.columns]
        for loss in mcs_losses:
            table, names = build_pooled_loss_matrix(
                test_frame, locked_present, loss=loss, epsilon=float(calculation["qlike_epsilon"])
            )
            if table.shape[0] >= 2 and table.shape[1] >= 2:
                result = model_confidence_set(
                    table.to_numpy(dtype=float), names=names, alpha=mcs_alpha,
                    max_lag=mcs_hac_lag, n_bootstrap=mcs_bootstrap, seed=42,
                )
                mcs_parts.append(mcs_summary_frame(result, loss=loss, protocol="locked"))

        exp_present = [model for model in exp_cols if model in test_frame.columns]
        if exp_present:
            for loss in mcs_losses:
                table, names = build_pooled_loss_matrix(
                    test_frame, exp_present, loss=loss, epsilon=float(calculation["qlike_epsilon"])
                )
                if table.shape[0] >= 2 and table.shape[1] >= 2:
                    result = model_confidence_set(
                        table.to_numpy(dtype=float), names=names, alpha=mcs_alpha,
                        max_lag=mcs_hac_lag, n_bootstrap=mcs_bootstrap, seed=42,
                    )
                    mcs_parts.append(mcs_summary_frame(result, loss=loss, protocol="expanding"))
    mcs = pd.concat(mcs_parts, ignore_index=True) if mcs_parts else pd.DataFrame()
    vix_diagnostics = segmented.groupby("segment").agg(vix_nonmissing=("log_vix", "count"), vix_missing=("log_vix", lambda values: int(values.isna().sum())), log_vix_z_nonmissing=("log_vix_z", "count"), log_vix_z_missing=("log_vix_z", lambda values: int(values.isna().sum()))).reset_index()
    strategy_config = config.get("strategy", {}) or {}
    strategy_segment = str(strategy_config.get("evaluation_segment", "test"))
    strategy_target_vol = float(strategy_config.get("target_vol", 0.10))
    strategy_max_leverage = float(strategy_config.get("max_leverage", 1.5))
    strategy_cost_bps = float(strategy_config.get("cost_bps", 10.0))
    strategy_rebalance_every = int(strategy_config.get("rebalance_every", 5))
    strategy_forecasters = [str(value) for value in strategy_config.get(
        "forecasting_models", ["historical_rv_21d", "ewma_rv", "har_rv", "ridge_rv", "garch_rv", "lgb_rv"]
    )]
    strategy_parts = []
    for forecast in strategy_forecasters:
        model_metrics = vol_targeting_metrics(
            segmented,
            segment=strategy_segment,
            target_vol=strategy_target_vol,
            max_leverage=strategy_max_leverage,
            cost_bps=strategy_cost_bps,
            forecast_column=forecast,
        )
        model_metrics["forecast"] = forecast
        strategy_parts.append(model_metrics)
    fixed_metrics = vol_targeting_metrics(
        segmented,
        segment=strategy_segment,
        target_vol=strategy_target_vol,
        max_leverage=None,
        cost_bps=0.0,
        fixed_weight=1.0,
    )
    fixed_metrics["forecast"] = "fixed_100pct"
    strategy_parts.append(fixed_metrics)
    strategy_metrics = pd.concat(strategy_parts, ignore_index=True)
    transmission_table = transmission_waterfall(
        segmented,
        forecast_columns=tuple(strategy_forecasters),
        segment=strategy_segment,
        target_vol=strategy_target_vol,
        max_leverage=strategy_max_leverage,
        cost_bps=strategy_cost_bps,
    )
    portfolio_rows = []
    for scheme in ("equal", "inverse_historical", "inverse_forecast"):
        portfolio_rows.append(portfolio_metrics(
            segmented,
            segment=strategy_segment,
            weight_scheme=scheme,
            forecast_column="garch_rv" if scheme == "inverse_forecast" else None,
            rebalance_every=strategy_rebalance_every,
        ))
    portfolio_table = pd.DataFrame(portfolio_rows)
    robustness_config = config.get("robustness", {}) or {}
    alt_label_columns = [
        column for column in ("future_rv_parkinson_5d", "future_rv_garman_klass_5d")
        if column in segmented.columns
    ]
    alt_label_metrics_parts: list[pd.DataFrame] = []
    alt_label_dm_parts: list[pd.DataFrame] = []
    for label in alt_label_columns:
        label_metrics = walk_forward_metrics(
            segmented,
            forecast_columns=FORMAL_FORECAST_COLUMNS,
            actual_column=label,
            epsilon=float(calculation["qlike_epsilon"]),
        )
        label_metrics["label"] = label
        alt_label_metrics_parts.append(label_metrics)
        for model_a in ("garch_rv", "ridge_rv", "lgb_rv"):
            dm_part = dm_by_segment(
                segmented,
                max_lag=int(dm_config["hac_lag"]),
                epsilon=float(calculation["qlike_epsilon"]),
                losses=("qlike", "mae"),
                model_a_column=model_a,
                model_b_column="har_rv",
                actual_column=label,
            )
            dm_part["label"] = label
            alt_label_dm_parts.append(dm_part)
    alt_label_metrics = pd.concat(alt_label_metrics_parts, ignore_index=True) if alt_label_metrics_parts else pd.DataFrame()
    alt_label_dm = pd.concat(alt_label_dm_parts, ignore_index=True) if alt_label_dm_parts else pd.DataFrame()
    cost_doubling_bps = float(robustness_config.get("cost_doubling_bps", 20.0))
    cost_rows: list[dict[str, float | int | str]] = []
    for forecast in strategy_forecasters:
        for cost_bps in (strategy_cost_bps, cost_doubling_bps):
            stage = vol_targeting_metrics(
                segmented,
                segment=strategy_segment,
                target_vol=strategy_target_vol,
                max_leverage=strategy_max_leverage,
                extra_lag_days=1,
                cost_bps=cost_bps,
                forecast_column=forecast,
            )
            pooled = stage.loc[stage["asset"] == "ALL"].iloc[0]
            cost_rows.append({
                "forecast": forecast,
                "cost_bps": cost_bps,
                "net_annual_return": pooled["net_annual_return"],
                "sharpe": pooled["sharpe"],
                "realized_vol": pooled["realized_vol"],
                "turnover": pooled["turnover"],
                "total_cost": pooled["total_cost"],
            })
    cost_sensitivity = pd.DataFrame(cost_rows)
    worst_errors = worst_error_summary(
        segmented,
        segment=strategy_segment,
        forecast_columns=("lgb_rv", "garch_rv", "har_rv", "ridge_rv"),
        top_n=int(lightgbm_config.get("worst_error_top_n", 15)),
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
        "garch": {
            "enabled": bool(garch_config.get("enabled", True)),
            "model": f"GARCH({int(garch_config.get('p', 1))},{int(garch_config.get('q', 1))})",
            "mean": "Constant",
            "distribution": "normal",
            "target": "five-day iterated conditional variance",
            "parameter_lock": "train_segment_mle",
            "forecast_rule": "fixed coefficients, recursive filter, iterated 5-step",
            "stationary_assets": int((garch_params["stationary"] == True).sum()) if not garch_params.empty else 0,
            "parameter_output": outputs["garch_params"],
            "parameter_assets": int(len(garch_params)),
            "forecast_column": "garch_rv",
        },
        "ridge": {
            "enabled": bool(ridge_config.get("enabled", True)),
            "penalty": ridge_penalty,
            "target": "log variance of future_rv_5d (exponential recovery)",
            "standardization": "per_asset_train_zscore",
            "lambda_selection": "per asset, validation pooled QLIKE",
            "tie_break": "smallest lambda",
            "feature_columns": ridge_feature_columns,
            "n_features": len(ridge_feature_columns),
            "min_train_observations": int(ridge_config.get("min_train_observations", 120)),
            "min_validation_observations": int(ridge_config.get("min_validation_observations", 20)),
            "variance_floor": float(calculation["forecast_variance_floor"]),
            "parameters_locked": True,
            "parameter_output": outputs["ridge_params"],
            "selection_output": outputs["ridge_lambda_selection"],
            "parameter_assets": int(len(ridge_params)),
            "forecast_column": "ridge_rv",
        },
        "lightgbm": {
            "enabled": bool(lightgbm_config.get("enabled", True)),
            "boosting": "gbdt",
            "target": "log variance of future_rv_5d (exponential recovery + smearing)",
            "feature_columns": lgb_feature_columns,
            "n_features": len(lgb_feature_columns),
            "hyperparameter_selection": "per asset, validation pooled QLIKE over num_leaves x learning_rate x n_estimators",
            "grid": {
                "num_leaves": [int(value) for value in lightgbm_config.get("num_leaves", [8, 31])],
                "learning_rate": [float(value) for value in lightgbm_config.get("learning_rate", [0.05, 0.1])],
                "n_estimators": [int(value) for value in lightgbm_config.get("n_estimators", [100, 300])],
            },
            "min_child_samples": int(lightgbm_config.get("min_child_samples", 20)),
            "random_state": int(lightgbm_config.get("random_state", 42)),
            "deterministic": True,
            "variance_floor": float(calculation["forecast_variance_floor"]),
            "parameters_locked": True,
            "parameter_output": outputs["lightgbm_params"],
            "importance_output": outputs["lightgbm_importance"],
            "worst_error_output": outputs["worst_error_dates"],
            "parameter_assets": int(len(lightgbm_params)),
            "forecast_column": "lgb_rv",
        },
        "expanding_window": {
            "enabled": bool(exp_config.get("enabled", True)),
            "protocol": "annual refit on expanding window (train through eval_year-1-validation_years, trailing validation_years for selection), forecast rows of eval_year only",
            "validation_years": exp_validation_years,
            "models": [f"{model}_exp" for model in exp_models] if exp_config.get("enabled", True) else [],
            "eval_years": list(exp_eval_years) if exp_config.get("enabled", True) else [],
            "label_exclusion": "same exclude_cross_segment_labels rule as locked protocol",
            "first_eval_year_equals_locked": True,
            "comparison_output": outputs.get("expanding_comparison"),
            "dm_output": outputs.get("expanding_dm"),
        },
        "calibration": {
            "enabled": bool(calib_config.get("enabled", True)),
            "source_segment": calib_source,
            "specs": [
                {"model": spec["model"], "method": spec["method"], "column": spec["column"], "n_buckets": spec["n_buckets"]}
                for spec in calib_specs
            ],
            "leakage_rule": "scale estimated on source segment only, applied to all rows",
        },
        "regime_robustness": {
            "definition": "test future_rv_5d pooled tertiles",
            "thresholds": regime_thresholds,
            "models": list(FORMAL_FORECAST_COLUMNS),
            "output": outputs["regime_robustness"],
        },
        "vix_incremental": {
            "comparison": "har_vix_rv_minus_har_rv",
            "output": outputs["vix_incremental_comparison"],
            "diagnostics": vix_diagnostics.to_dict(orient="records"),
            "experimental_only": True,
        },
        "har_vix": {
            "enabled": bool(har_vix_config.get("enabled", True)),
            "alignment": "exact_date_left_join",
            "source_asset": "^VIX",
            "source_column": "adj_close",
            "feature_column": "log_vix_z",
            "raw_feature_column": "log_vix",
            "standardization": "train_global_mean_std",
            "target": "log variance of future_rv_5d",
            "log_floor": float(har_vix_config.get("log_floor", 1.0e-12)),
            "smearing_correction": bool(har_vix_config.get("smearing", True)),
            "missing_policy": "retain_nan_no_future_fill",
            "training_segment": "train",
            "parameters_locked": True,
            "coefficients_output": outputs["har_vix_coefficients"],
            "coefficient_assets": int(len(har_vix_coefficients)),
            "forecast_column": "har_vix_rv",
        },
        "forecast_columns": list(ALL_FORECAST_COLUMNS),
        "qlike_scale": "variance",
        "raw_files": raw_files,
        "quality_status": quality_summary["status"],
        "prediction_rows": int(len(predictions)),
        "valid_evaluation_rows": {
            column: int(predictions[["future_rv_5d", column]].notna().all(axis=1).sum())
            for column in ALL_FORECAST_COLUMNS
        },
        "ewma_lambda_selection": lambda_selection.to_dict(orient="records"),
        "dm_test": {
            "enabled": bool(dm_config.get("enabled", True)),
            "losses": list(dm_config.get("losses", [dm_config.get("loss", "qlike")])),
            "primary_loss": dm_config.get("primary_loss", "qlike"),
            "comparisons": [
                {"model_a": "ewma_rv", "model_b": "historical_rv_21d"},
                {"model_a": "har_rv", "model_b": "historical_rv_21d"},
                {"model_a": "garch_rv", "model_b": "historical_rv_21d"},
                {"model_a": "garch_rv", "model_b": "har_rv"},
                {"model_a": "ridge_rv", "model_b": "historical_rv_21d"},
                {"model_a": "ridge_rv", "model_b": "har_rv"},
                {"model_a": "ridge_rv", "model_b": "garch_rv"},
                {"model_a": "lgb_rv", "model_b": "historical_rv_21d"},
                {"model_a": "lgb_rv", "model_b": "har_rv"},
                {"model_a": "lgb_rv", "model_b": "garch_rv"},
                {"model_a": "har_vix_rv", "model_b": "har_rv"},
            ],
            "pooled_rule": "cross-sectional mean by date before HAC",
            "model_order": "model_a_minus_model_b",
            "hac_method": "Bartlett HAC",
            "hac_lag": int(dm_config["hac_lag"]),
            "asset_robustness_output": outputs["asset_robustness"],
            "regime_robustness_output": outputs["regime_robustness"],
        },
        "strategy": {
            "enabled": bool(strategy_config.get("enabled", True)),
            "target_vol": strategy_target_vol,
            "max_leverage": strategy_max_leverage,
            "evaluation_segment": strategy_segment,
            "cost_bps": strategy_cost_bps,
            "position_rule": "close-of-t forecast, position held from t+1, no shorting",
            "evaluation_rule": "test segment only; forecasters are parameter-locked",
            "rebalance_every": strategy_rebalance_every,
            "forecasters": strategy_forecasters,
            "fixed_reference": "fixed_100pct",
            "portfolio_schemes": ["equal", "inverse_historical", "inverse_forecast_garch_rv"],
            "strategy_metrics_output": outputs["strategy_metrics"],
            "transmission_output": outputs["transmission_waterfall"],
            "portfolio_output": outputs["portfolio_metrics"],
        },
        "robustness": {
            "excluded_years": [int(year) for year in exclude_years],
            "alternative_labels": alt_label_columns,
            "alternative_label_test_all_qlike": {
                label: dict(zip(sub["forecast"].to_list(), sub["qlike"].to_list()))
                for label, sub in alt_label_metrics.loc[
                    (alt_label_metrics["asset"] == "ALL") & (alt_label_metrics["segment"] == "test")
                ].groupby("label")
            } if not alt_label_metrics.empty else {},
            "cost_doubling_bps": cost_doubling_bps,
            "alt_label_metrics_output": outputs["alt_label_metrics"],
            "alt_label_dm_output": outputs["alt_label_dm"],
            "strategy_cost_sensitivity_output": outputs["strategy_cost_sensitivity"],
        },
        "yearly_metrics_output": outputs["yearly_metrics"],
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
                    for forecast in FORMAL_FORECAST_COLUMNS
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
    write_metrics(garch_params, resolve(root, outputs["garch_params"]))
    write_metrics(ridge_params, resolve(root, outputs["ridge_params"]))
    write_metrics(ridge_selection, resolve(root, outputs["ridge_lambda_selection"]))
    write_metrics(lightgbm_params, resolve(root, outputs["lightgbm_params"]))
    write_metrics(lightgbm_importance, resolve(root, outputs["lightgbm_importance"]))
    write_metrics(worst_errors, resolve(root, outputs["worst_error_dates"]))
    if not expanding_comparison.empty:
        write_metrics(expanding_comparison, resolve(root, outputs["expanding_comparison"]))
    if not expanding_dm.empty:
        write_metrics(expanding_dm, resolve(root, outputs["expanding_dm"]))
    if not expanding_params.empty:
        write_metrics(expanding_params, resolve(root, outputs["expanding_params"]))
    write_metrics(comparison, resolve(root, outputs["test_model_comparison"]))
    write_metrics(robustness, resolve(root, outputs["asset_robustness"]))
    write_metrics(regime_robustness, resolve(root, outputs["regime_robustness"]))
    write_metrics(vix_incremental, resolve(root, outputs["vix_incremental_comparison"]))
    write_metrics(strategy_metrics, resolve(root, outputs["strategy_metrics"]))
    write_metrics(transmission_table, resolve(root, outputs["transmission_waterfall"]))
    write_metrics(portfolio_table, resolve(root, outputs["portfolio_metrics"]))
    if not alt_label_metrics.empty:
        write_metrics(alt_label_metrics, resolve(root, outputs["alt_label_metrics"]))
    if not alt_label_dm.empty:
        write_metrics(alt_label_dm, resolve(root, outputs["alt_label_dm"]))
    write_metrics(cost_sensitivity, resolve(root, outputs["strategy_cost_sensitivity"]))
    write_metrics(yearly_metrics, resolve(root, outputs["yearly_metrics"]))
    if bool(dm_config.get("enabled", True)):
        write_metrics(dm_results, resolve(root, outputs["dm_tests"]))
    if not mcs.empty:
        write_metrics(mcs, resolve(root, outputs["mcs"]))
    plot_spy_comparison(predictions, resolve(root, outputs["figure"]))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--exclude-year", type=int, action="append", default=[], help="drop rows from this calendar year (repeatable)")
    parser.add_argument("--output-prefix", type=str, default=None, help="redirect all outputs under this prefix (robustness runs)")
    args = parser.parse_args()
    metadata = build(args.config, exclude_years=tuple(args.exclude_year), output_prefix=args.output_prefix)
    print(json.dumps({"status": "ok", **metadata}, indent=2))


if __name__ == "__main__":
    main()
