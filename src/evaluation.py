"""Time-segmented evaluation for leakage-controlled volatility forecasts."""

from __future__ import annotations

from dataclasses import asdict
from math import erfc, sqrt

import numpy as np
import pandas as pd

from src.metrics import evaluate_forecast, metrics_by_asset

SEGMENTS = ("train", "validation", "test")


def assign_walk_forward_segments(
    frame: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Assign train, validation, and test segments from each forecast date."""
    train_boundary = pd.Timestamp(train_end)
    validation_boundary = pd.Timestamp(validation_end)
    if train_boundary >= validation_boundary:
        raise ValueError("train_end must be earlier than validation_end")

    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    dates = pd.to_datetime(data["date"])
    data["segment"] = pd.cut(
        dates,
        bins=[pd.Timestamp.min, train_boundary, validation_boundary, pd.Timestamp.max],
        labels=SEGMENTS,
        include_lowest=True,
        right=True,
    ).astype("object")
    return data


def exclude_cross_segment_labels(
    frame: pd.DataFrame,
    *,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove train/validation rows whose future label window crosses a boundary."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    excluded = pd.Series(False, index=data.index)
    counts: dict[str, int] = {}
    for segment in ("train", "validation"):
        indices = (
            data.loc[data["segment"] == segment]
            .groupby("asset", sort=False, group_keys=False)
            .tail(horizon)
            .index
        )
        excluded.loc[indices] = True
        counts[segment] = int(len(indices))
    counts["test"] = 0
    return data.loc[~excluded].copy(), counts


def forecast_losses(
    frame: pd.DataFrame,
    *,
    actual_column: str = "future_rv_5d",
    historical_column: str = "historical_rv_21d",
    ewma_column: str = "ewma_rv",
    epsilon: float = 1.0e-12,
) -> pd.DataFrame:
    """Add paired QLIKE losses and EWMA-minus-historical loss differences."""
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    data = frame.copy()
    actual = pd.to_numeric(data[actual_column], errors="coerce")
    historical = pd.to_numeric(data[historical_column], errors="coerce")
    ewma = pd.to_numeric(data[ewma_column], errors="coerce")
    valid = np.isfinite(actual) & np.isfinite(historical) & np.isfinite(ewma) & (actual >= 0)
    historical_variance = np.maximum(np.square(historical), epsilon)
    ewma_variance = np.maximum(np.square(ewma), epsilon)
    actual_variance = np.square(actual)
    data["qlike_historical"] = np.where(
        valid, np.log(historical_variance) + actual_variance / historical_variance, np.nan
    )
    data["qlike_ewma"] = np.where(
        valid, np.log(ewma_variance) + actual_variance / ewma_variance, np.nan
    )
    data["qlike_loss_diff"] = data["qlike_ewma"] - data["qlike_historical"]
    data["mae_historical"] = np.where(valid, np.abs(historical - actual), np.nan)
    data["mae_ewma"] = np.where(valid, np.abs(ewma - actual), np.nan)
    data["mae_loss_diff"] = data["mae_ewma"] - data["mae_historical"]
    return data


def fit_har_by_asset(frame: pd.DataFrame, *, train_segment: str = "train", actual_column: str = "future_rv_5d", output_column: str = "har_rv") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit HAR OLS on train rows and predict all rows."""
    features = ["har_daily_rv", "har_weekly_rv", "har_monthly_rv"]
    data = frame.copy(); data[output_column] = np.nan; coefficient_rows = []
    for asset, group in data.groupby("asset", sort=True):
        train = group.loc[(group["segment"] == train_segment) & group[[actual_column, *features]].notna().all(axis=1)]
        if len(train) < 4: raise ValueError(f"insufficient HAR training observations for {asset}")
        x = np.column_stack([np.ones(len(train)), train[features].to_numpy()]); y = np.square(train[actual_column].to_numpy())
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        all_valid = group[features].notna().all(axis=1)
        predictions = np.maximum(np.column_stack([np.ones(int(all_valid.sum())), group.loc[all_valid, features].to_numpy()]).dot(beta), 0.0)
        data.loc[group.index[all_valid], output_column] = np.sqrt(predictions)
        coefficient_rows.append({"asset": asset, "intercept": beta[0], "daily": beta[1], "weekly": beta[2], "monthly": beta[3], "n_train": len(train)})
    return data, pd.DataFrame(coefficient_rows)
def select_ewma_lambda(
    frame: pd.DataFrame,
    *,
    lambdas: tuple[float, ...] | list[float],
    validation_segment: str = "validation",
    actual_column: str = "future_rv_5d",
    candidate_prefix: str = "ewma_rv_lambda_",
    epsilon: float = 1.0e-12,
) -> tuple[float, pd.DataFrame]:
    """Select the smallest validation pooled QLIKE candidate deterministically."""
    values = [float(value) for value in lambdas]
    if not values or len(set(values)) != len(values) or not all(np.isfinite(values)):
        raise ValueError("lambdas must be a non-empty list of unique finite values")
    if not all(0 < value < 1 for value in values):
        raise ValueError("each lambda must be between 0 and 1")
    validation = frame.loc[frame["segment"] == validation_segment]
    rows: list[dict[str, float | int | bool | str]] = []
    for value in sorted(values):
        column = f"{candidate_prefix}{value:g}"
        if column not in validation:
            raise KeyError(f"missing candidate column: {column}")
        result = evaluate_forecast(validation[actual_column], validation[column], epsilon=epsilon)
        rows.append({
            "lambda": value,
            "candidate_column": column,
            "selection_segment": validation_segment,
            "selection_metric": "pooled_qlike",
            "n_obs": result.n_obs,
            "validation_qlike": result.qlike,
            "variance_floor_count": result.variance_floor_count,
        })
    table = pd.DataFrame(rows)
    valid = table[table["n_obs"] > 0].copy()
    if valid.empty:
        raise ValueError("no EWMA candidate has valid validation observations")
    best = valid.sort_values(["validation_qlike", "lambda"], kind="stable").iloc[0]["lambda"]
    table["selected"] = table["lambda"] == best
    table["tie_break"] = "smallest lambda among equal validation QLIKE"
    return float(best), table


def diebold_mariano_test(loss_difference: pd.Series | np.ndarray, *, max_lag: int = 4) -> dict[str, float | int]:
    """Compute a two-sided DM statistic using Bartlett HAC variance."""
    if max_lag < 0:
        raise ValueError("max_lag must be non-negative")
    values = np.asarray(loss_difference, dtype=float)
    values = values[np.isfinite(values)]
    n_obs = int(values.size)
    result: dict[str, float | int] = {
        "n_obs": n_obs,
        "mean_loss_diff": np.nan,
        "hac_standard_error": np.nan,
        "long_run_variance": np.nan,
        "dm_statistic": np.nan,
        "p_value": np.nan,
        "hac_lag": int(min(max_lag, max(n_obs - 1, 0))),
    }
    if n_obs == 0:
        return result
    mean_diff = float(values.mean())
    centered = values - mean_diff
    lag = int(min(max_lag, n_obs - 1))
    long_run_variance = float(np.mean(centered * centered))
    for k in range(1, lag + 1):
        autocovariance = float(np.mean(centered[k:] * centered[:-k]))
        weight = 1.0 - k / (lag + 1.0)
        long_run_variance += 2.0 * weight * autocovariance
    result["mean_loss_diff"] = mean_diff
    result["long_run_variance"] = long_run_variance
    if n_obs <= 1 or long_run_variance <= 0:
        return result
    standard_error = float(np.sqrt(long_run_variance / n_obs))
    result["hac_standard_error"] = standard_error
    result["dm_statistic"] = mean_diff / standard_error
    result["p_value"] = float(erfc(abs(result["dm_statistic"]) / sqrt(2.0)))
    return result


def dm_by_segment(
    frame: pd.DataFrame,
    *,
    max_lag: int = 4,
    actual_column: str = "future_rv_5d",
    historical_column: str = "historical_rv_21d",
    ewma_column: str = "ewma_rv",
    epsilon: float = 1.0e-12,
    losses: tuple[str, ...] = ("qlike",),
    model_a_column: str | None = None,
    model_b_column: str | None = None,
) -> pd.DataFrame:
    """Run paired DM tests by segment, asset, and pooled ALL."""
    model_a_column = model_a_column or ewma_column
    model_b_column = model_b_column or historical_column
    data = frame.copy()
    actual = pd.to_numeric(data[actual_column], errors="coerce")
    model_a = pd.to_numeric(data[model_a_column], errors="coerce")
    model_b = pd.to_numeric(data[model_b_column], errors="coerce")
    valid = np.isfinite(actual) & np.isfinite(model_a) & np.isfinite(model_b) & (actual >= 0)
    a_var = np.maximum(np.square(model_a), epsilon)
    b_var = np.maximum(np.square(model_b), epsilon)
    actual_var = np.square(actual)
    data["_qlike_diff"] = np.where(valid, np.log(a_var) + actual_var / a_var - np.log(b_var) - actual_var / b_var, np.nan)
    data["_mae_diff"] = np.where(valid, np.abs(model_a - actual) - np.abs(model_b - actual), np.nan)
    loss_columns = {"qlike": "_qlike_diff", "mae": "_mae_diff"}
    unknown = set(losses) - set(loss_columns)
    if unknown: raise ValueError(f"unsupported DM losses: {sorted(unknown)}")
    rows = []
    for segment in SEGMENTS:
        segment_frame = data.loc[data["segment"] == segment]
        for loss_name in losses:
            column = loss_columns[loss_name]
            for asset, group in segment_frame.groupby("asset", sort=True):
                valid_group = group.loc[np.isfinite(group[column])].sort_values("date")
                result = diebold_mariano_test(valid_group[column], max_lag=max_lag)
                rows.append({"segment": segment, "asset": asset, "loss": loss_name, "model_a": model_a_column, "model_b": model_b_column, "n_dates": len(valid_group), "paired_rows": len(valid_group), **result})
            valid_group = segment_frame.loc[np.isfinite(segment_frame[column])]
            daily = valid_group.groupby("date", as_index=False)[column].agg(daily_loss_diff="mean", paired_rows="count").sort_values("date")
            result = diebold_mariano_test(daily["daily_loss_diff"], max_lag=max_lag)
            rows.append({"segment": segment, "asset": "ALL", "loss": loss_name, "model_a": model_a_column, "model_b": model_b_column, "n_dates": len(daily), "paired_rows": int(daily["paired_rows"].sum()), **result})
    return pd.DataFrame(rows)[["segment", "asset", "loss", "model_a", "model_b", "n_obs", "n_dates", "paired_rows", "mean_loss_diff", "hac_lag", "long_run_variance", "hac_standard_error", "dm_statistic", "p_value"]]


def test_model_comparison(
    frame: pd.DataFrame,
    *,
    segment: str = "test",
    asset: str | None = None,
    forecast_columns: tuple[str, ...] = (
        "historical_rv_21d", "ewma_rv", "har_rv", "ridge_rv", "garch_rv", "lgb_rv",
    ),
) -> pd.DataFrame:
    """Return test-segment model metrics with per-metric ranks."""
    data = frame.loc[frame["segment"] == segment]
    metrics = walk_forward_metrics(data, forecast_columns=forecast_columns)
    metrics = metrics.loc[(metrics["segment"] == segment) & (metrics["asset"] == (asset or "ALL"))].copy()
    for metric in ("mae", "rmse"):
        metrics[f"{metric}_rank"] = metrics[metric].rank(method="min", ascending=True).astype("Int64")
    metrics["qlike_rank"] = metrics["qlike"].rank(method="min", ascending=True).astype("Int64")
    return metrics


def worst_error_summary(
    frame: pd.DataFrame,
    *,
    segment: str = "test",
    actual_column: str = "future_rv_5d",
    forecast_columns: tuple[str, ...] = ("lgb_rv", "garch_rv", "har_rv", "ridge_rv"),
    top_n: int = 15,
) -> pd.DataFrame:
    """Return the worst per-asset forecast dates in a segment with model errors.

    For each asset the ``top_n`` test rows with the largest absolute error of the
    first ``forecast_columns`` entry (the focus model) are reported together with
    the other models' predictions and absolute errors, so the dates where the
    focus model fails hardest can be inspected against competing models.
    """
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    focus_column = forecast_columns[0]
    data = frame.loc[frame["segment"] == segment].copy()
    rows: list[dict[str, float | int | str]] = []
    for asset, group in data.groupby("asset", sort=True):
        valid = group.loc[group[actual_column].notna() & group[focus_column].notna()].copy()
        valid["_abs_err_focus"] = (valid[focus_column] - valid[actual_column]).abs()
        top = valid.sort_values("_abs_err_focus", ascending=False).head(top_n)
        for _, row in top.iterrows():
            entry: dict[str, float | int | str] = {
                "asset": asset, "date": row["date"],
                actual_column: row[actual_column],
            }
            for column in forecast_columns:
                entry[column] = row[column]
                entry[f"abs_err_{column}"] = abs(float(row[column]) - float(row[actual_column]))
            rows.append(entry)
    columns = ["asset", "date", actual_column]
    for column in forecast_columns:
        columns.append(column)
        columns.append(f"abs_err_{column}")
    return pd.DataFrame(rows, columns=columns)


def assign_test_volatility_regimes(frame: pd.DataFrame, *, actual_column: str = "future_rv_5d") -> tuple[pd.DataFrame, dict[str, float]]:
    """Assign low/medium/high regimes from pooled test realized volatility tertiles."""
    data = frame.copy(); data["regime"] = pd.NA
    test = data.loc[(data["segment"] == "test") & data[actual_column].notna(), actual_column]
    if test.empty:
        data["regime"] = pd.NA
        return data, {"low_threshold": np.nan, "high_threshold": np.nan}
    low, high = float(test.quantile(1 / 3)), float(test.quantile(2 / 3))
    values = data.loc[data["segment"] == "test", actual_column]
    data.loc[data["segment"] == "test", "regime"] = pd.cut(values, bins=[-np.inf, low, high, np.inf], labels=["low", "medium", "high"], include_lowest=True).astype("object")
    return data, {"low_threshold": low, "high_threshold": high}


def regime_robustness_summary(frame: pd.DataFrame, *, models: tuple[str, ...] = ("historical_rv_21d", "ewma_rv", "har_rv", "ridge_rv", "garch_rv", "lgb_rv")) -> pd.DataFrame:
    """Evaluate formal models by test volatility regime, asset, and pooled ALL."""
    rows = []
    for regime in ("low", "medium", "high"):
        subset = frame.loc[frame["regime"] == regime]
        metrics = metrics_by_asset(subset, forecast_columns=models)
        metrics["regime"] = regime
        metrics = metrics.rename(columns={"asset": "asset"})
        rows.append(metrics)
    result = pd.concat(rows, ignore_index=True)
    for metric in ("mae", "rmse", "qlike"):
        result[f"{metric}_rank"] = result.groupby(["regime", "asset"])[metric].rank(method="min", ascending=True).astype("Int64")
    return result[["regime", "asset", "forecast", "n_obs", "mae", "rmse", "qlike", "variance_floor_count", "mae_rank", "rmse_rank", "qlike_rank"]]


def asset_robustness_summary(frame: pd.DataFrame, *, segment: str = "test", models: tuple[str, ...] = ("historical_rv_21d", "ewma_rv", "har_rv", "ridge_rv", "garch_rv", "lgb_rv")) -> pd.DataFrame:
    metrics = walk_forward_metrics(frame, forecast_columns=models)
    metrics = metrics.loc[(metrics["segment"] == segment) & (metrics["asset"] != "ALL")].copy()
    for metric in ("mae", "rmse", "qlike"):
        metrics[f"{metric}_rank"] = metrics.groupby("asset")[metric].rank(method="min", ascending=True).astype("Int64")
        metrics[f"{metric}_winner"] = metrics[f"{metric}_rank"] == 1
    return metrics


def walk_forward_metrics(
    frame: pd.DataFrame,
    *,
    forecast_columns: tuple[str, ...],
    actual_column: str = "future_rv_5d",
    epsilon: float = 1.0e-12,
) -> pd.DataFrame:
    """Evaluate each forecast by segment, asset, and direct pooled ALL row."""
    rows: list[dict[str, float | int | str]] = []
    assets = sorted(frame["asset"].unique())
    for segment in SEGMENTS:
        segment_frame = frame.loc[frame["segment"] == segment]
        for forecast in forecast_columns:
            for asset in assets:
                group = segment_frame.loc[segment_frame["asset"] == asset]
                rows.append({
                    "segment": segment,
                    "asset": asset,
                    "forecast": forecast,
                    **asdict(evaluate_forecast(group[actual_column], group[forecast], epsilon=epsilon)),
                })
            rows.append({
                "segment": segment,
                "asset": "ALL",
                "forecast": forecast,
                **asdict(evaluate_forecast(segment_frame[actual_column], segment_frame[forecast], epsilon=epsilon)),
            })
    return pd.DataFrame(rows)[
        ["segment", "asset", "forecast", "n_obs", "mae", "rmse", "qlike", "variance_floor_count"]
    ]
