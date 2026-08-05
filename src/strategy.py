"""Volatility-targeting strategies, transmission decomposition, and portfolio layer.

Leakage control: a position is decided at the close of day ``t`` from a forecast
available at ``t`` and first earns the return of day ``t+1``. All strategy and
portfolio results are evaluated exclusively on the ``test`` segment, where the
locked-parameter forecasters (HAR, GARCH, Ridge) are genuinely out of sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION = 252.0


def _simple_returns(log_return: pd.Series | np.ndarray) -> np.ndarray:
    return np.expm1(np.asarray(log_return, dtype=float))


def vol_target_position(
    frame: pd.DataFrame,
    *,
    forecast_column: str,
    target_vol: float,
    max_leverage: float | None,
    extra_lag_days: int = 0,
) -> pd.DataFrame:
    """Return per-date weight and the held position.

    ``weight_t = clip(target_vol / forecast_t, 0, max_leverage)`` and the
    position held during day ``t+1`` (``t+1+extra_lag_days``) is ``weight_t``.
    Rows with a non-finite forecast get no position.
    """
    if target_vol <= 0:
        raise ValueError("target_vol must be positive")
    if extra_lag_days < 0:
        raise ValueError("extra_lag_days must be non-negative")
    if max_leverage is not None and max_leverage <= 0:
        raise ValueError("max_leverage must be positive or None")
    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    forecast = pd.to_numeric(data[forecast_column], errors="coerce")
    weights = (target_vol / forecast).clip(lower=0.0)
    if max_leverage is not None:
        weights = weights.clip(upper=float(max_leverage))
    data["weight"] = weights.where(np.isfinite(weights))
    data["position"] = data.groupby("asset", sort=False)["weight"].shift(1 + int(extra_lag_days))
    data["position_change"] = data.groupby("asset", sort=False)["position"].diff().abs()
    return data


def _asset_metrics(
    asset: str,
    group: pd.DataFrame,
    *,
    target_vol: float | None,
) -> dict[str, float | int | str]:
    group = group.dropna(subset=["gross"])
    gross = group["gross"].to_numpy(dtype=float)
    net = group["net"].to_numpy(dtype=float)

    def stats(values: np.ndarray) -> tuple[float, float, float, float]:
        if values.size < 2 or not np.all(np.isfinite(values)):
            return np.nan, np.nan, np.nan, np.nan
        realized_vol = float(np.std(values, ddof=1) * np.sqrt(ANNUALIZATION))
        annual_return = float(np.mean(values) * ANNUALIZATION)
        sharpe = annual_return / realized_vol if realized_vol > 0 else np.nan
        cumulative = np.cumprod(1.0 + values)
        drawdown = float((cumulative / np.maximum.accumulate(cumulative) - 1.0).min())
        return realized_vol, annual_return, sharpe, drawdown

    gross_vol, gross_return, gross_sharpe, gross_mdd = stats(gross)
    _, net_return, _, net_mdd = stats(net)
    turnover = float(group["position_change"].mean()) if group["position_change"].notna().any() else 0.0
    total_cost = float(group["cost"].sum()) if group["cost"].notna().any() else 0.0
    return {
        "asset": asset,
        "n_obs": int(gross.size),
        "realized_vol": gross_vol,
        "target_deviation": abs(gross_vol - target_vol) if (target_vol is not None and np.isfinite(gross_vol)) else np.nan,
        "annual_return": gross_return,
        "net_annual_return": net_return,
        "sharpe": gross_sharpe,
        "max_drawdown": gross_mdd,
        "net_max_drawdown": net_mdd,
        "turnover": turnover,
        "total_cost": total_cost,
    }


def vol_targeting_metrics(
    frame: pd.DataFrame,
    *,
    segment: str = "test",
    target_vol: float = 0.10,
    max_leverage: float | None = 1.5,
    extra_lag_days: int = 0,
    cost_bps: float = 0.0,
    forecast_column: str | None = None,
    fixed_weight: float | None = None,
) -> pd.DataFrame:
    """Evaluate a vol-targeting scheme by asset and pooled ALL on one segment."""
    cost_rate = float(cost_bps) / 1e4
    if fixed_weight is not None:
        data = frame.sort_values(["asset", "date"], kind="stable").copy()
        data["position"] = float(fixed_weight)
        data["position_change"] = 0.0
    else:
        if forecast_column is None:
            raise ValueError("forecast_column or fixed_weight is required")
        data = vol_target_position(
            frame,
            forecast_column=forecast_column,
            target_vol=target_vol,
            max_leverage=max_leverage,
            extra_lag_days=extra_lag_days,
        )
        data["position_change"] = data["position_change"].fillna(0.0)
    simple = _simple_returns(data["log_return"])
    data["gross"] = float(1.0) * data["position"].to_numpy(dtype=float) * simple
    data["cost"] = cost_rate * data["position_change"].to_numpy(dtype=float)
    data["net"] = data["gross"] - data["cost"]
    evaluated = data.loc[data["segment"] == segment].copy()
    rows: list[dict[str, float | int | str]] = []
    for asset, group in evaluated.groupby("asset", sort=True):
        rows.append(_asset_metrics(asset, group, target_vol=target_vol))
    daily = evaluated.groupby("date")[["gross", "net"]].mean().reset_index()
    daily["asset"] = "ALL"
    daily["position_change"] = evaluated.groupby("date")["position_change"].mean().to_numpy()
    daily["cost"] = evaluated.groupby("date")["cost"].mean().to_numpy()
    rows.append(_asset_metrics("ALL", daily, target_vol=target_vol))
    return pd.DataFrame(rows)


def calibrate_forecast_level(
    frame: pd.DataFrame,
    *,
    forecast_column: str,
    output_column: str,
    actual_column: str = "future_rv_5d",
    source_segment: str = "validation",
    method: str = "variance_rms",
    min_observations: int = 30,
) -> pd.DataFrame:
    """Apply a per-asset level calibration to a volatility forecast.

    A multiplicative scale factor is estimated on ``source_segment`` rows and
    applied to every row, so the calibration uses only data available before the
    evaluation segment and cannot leak the test outcome.

    - ``method="multiplicative"`` scales by ``mean(actual) / mean(forecast)``,
      centering the average level.
    - ``method="variance_rms"`` scales by ``sqrt(mean((actual/forecast)^2))``,
      which centers the volatility-targeting implied level: a target-tracking
      position of ``target/forecast`` then realizes roughly ``target``.
    - ``method="loglinear"`` fits ``log(actual) = a + b*log(forecast)`` and can
      restore variance (``b > 1``) as well as the level.

    Assets with too few source rows get a NaN calibrated column.
    """
    if method not in ("multiplicative", "variance_rms", "loglinear"):
        raise ValueError(f"unknown calibration method: {method}")
    if min_observations <= 0:
        raise ValueError("min_observations must be positive")
    data = frame.copy()
    data[output_column] = np.nan
    for asset, group in data.groupby("asset", sort=True):
        source = group.loc[
            (group["segment"] == source_segment)
            & group[[forecast_column, actual_column]].notna().all(axis=1)
        ]
        if len(source) < min_observations:
            continue
        forecast = source[forecast_column].to_numpy(dtype=float)
        actual = source[actual_column].to_numpy(dtype=float)
        valid = np.isfinite(forecast) & np.isfinite(actual) & (actual > 0)
        forecast, actual = forecast[valid], actual[valid]
        if forecast.size == 0:
            continue
        if method == "multiplicative":
            scale = float(actual.mean() / forecast.mean())
            data.loc[group.index, output_column] = data.loc[group.index, forecast_column].to_numpy(dtype=float) * scale
        elif method == "variance_rms":
            scale = float(np.sqrt(np.mean(np.square(actual / forecast))))
            data.loc[group.index, output_column] = data.loc[group.index, forecast_column].to_numpy(dtype=float) * scale
        else:  # loglinear
            slope, intercept = np.polyfit(np.log(forecast), np.log(actual), 1)
            data.loc[group.index, output_column] = (
                np.exp(intercept) * np.power(data.loc[group.index, forecast_column].to_numpy(dtype=float), slope)
            )
    return data


def transmission_waterfall(
    frame: pd.DataFrame,
    *,
    forecast_columns: tuple[str, ...],
    segment: str = "test",
    target_vol: float = 0.10,
    max_leverage: float = 1.5,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """Decompose frictions between ideal vol targeting and the real strategy.

    Stages add, in order: the leverage cap, one extra day of execution lag, and
    transaction costs. Each forecaster is scored with the same stage sequence;
    metrics are the pooled-ALL rows.
    """
    stages = [
        {"stage": "no cap / no lag / no cost", "max_leverage": None, "extra_lag_days": 0, "cost_bps": 0.0},
        {"stage": "leverage cap", "max_leverage": float(max_leverage), "extra_lag_days": 0, "cost_bps": 0.0},
        {"stage": "+ one-day lag", "max_leverage": float(max_leverage), "extra_lag_days": 1, "cost_bps": 0.0},
        {"stage": "+ transaction cost", "max_leverage": float(max_leverage), "extra_lag_days": 1, "cost_bps": float(cost_bps)},
    ]
    rows: list[dict[str, float | int | str]] = []
    for forecast in forecast_columns:
        for stage in stages:
            metrics = vol_targeting_metrics(
                frame,
                segment=segment,
                target_vol=target_vol,
                max_leverage=stage["max_leverage"],
                extra_lag_days=stage["extra_lag_days"],
                cost_bps=stage["cost_bps"],
                forecast_column=forecast,
            )
            pooled = metrics.loc[metrics["asset"] == "ALL"].iloc[0]
            asset_rows = metrics.loc[metrics["asset"] != "ALL"]
            rows.append({
                "forecast": forecast,
                "stage": stage["stage"],
                "max_leverage": "unlimited" if stage["max_leverage"] is None else stage["max_leverage"],
                "extra_lag_days": stage["extra_lag_days"],
                "cost_bps": stage["cost_bps"],
                "n_obs": int(pooled["n_obs"]),
                "realized_vol": pooled["realized_vol"],
                "target_deviation": pooled["target_deviation"],
                "mean_asset_deviation": float(asset_rows["target_deviation"].mean()) if len(asset_rows) else np.nan,
                "annual_return": pooled["annual_return"],
                "net_annual_return": pooled["net_annual_return"],
                "sharpe": pooled["sharpe"],
                "max_drawdown": pooled["max_drawdown"],
                "turnover": pooled["turnover"],
                "total_cost": pooled["total_cost"],
            })
    return pd.DataFrame(rows)


def inverse_vol_weights(volumes: pd.Series) -> np.ndarray:
    """Inverse-volatility portfolio weights, renormalized and non-negative."""
    values = pd.to_numeric(volumes, errors="coerce")
    inverse = np.where(values.to_numpy() > 0, 1.0 / values.to_numpy(), 0.0)
    inverse = np.where(np.isfinite(inverse), inverse, 0.0)
    total = float(inverse.sum())
    if total <= 0:
        return np.full(len(volumes), np.nan)
    return inverse / total


def portfolio_metrics(
    frame: pd.DataFrame,
    *,
    segment: str = "test",
    weight_scheme: str = "equal",
    forecast_column: str | None = None,
    rebalance_every: int = 5,
    annualization_factor: float = ANNUALIZATION,
) -> dict[str, float | int | str]:
    """Weekly-rebalanced multi-asset portfolio of target assets on one segment.

    Schemes: ``equal``, ``inverse_historical`` (inverse of 21-day historical
    volatility), and ``inverse_forecast`` (inverse of a model forecast). The
    common calendar is the intersection of the assets' trading dates; positions
    start at the first weekly rebalance. Pure risk allocation: no return
    prediction is used anywhere.
    """
    if weight_scheme not in ("equal", "inverse_historical", "inverse_forecast"):
        raise ValueError("weight_scheme must be equal, inverse_historical, or inverse_forecast")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be positive")
    data = frame.loc[frame["segment"] == segment].copy()
    assets = sorted(data["asset"].unique())
    if not assets:
        return {"scheme": weight_scheme, "n_days": 0, "realized_vol": np.nan}
    common_dates = sorted(set(data.loc[data["asset"] == assets[0], "date"]))
    for asset in assets[1:]:
        common_dates = sorted(set(common_dates) & set(data.loc[data["asset"] == asset, "date"]))
    pivot_dates = pd.to_datetime(common_dates)
    rebalance_dates = pivot_dates[:: rebalance_every]

    vol_source = "historical_rv_21d" if weight_scheme == "inverse_historical" else forecast_column
    pivot = data.pivot_table(index="date", columns="asset", values="log_return", aggfunc="first")
    pivot = pivot.loc[pivot_dates]
    pivot_simple = np.expm1(pivot.to_numpy(dtype=float))
    n_assets = len(assets)

    weight_matrix = np.full(pivot.shape, np.nan)
    held_index = 0
    for date_position, date in enumerate(pivot_dates):
        if date < rebalance_dates[0]:
            continue
        while held_index + 1 < len(rebalance_dates) and date >= rebalance_dates[held_index + 1]:
            held_index += 1
        rebalance_date = rebalance_dates[held_index]
        if weight_scheme == "equal":
            weights = np.full(n_assets, 1.0 / n_assets)
        else:
            volumes = data.loc[
                (data["date"] == rebalance_date) & data["asset"].isin(assets)
            ].set_index("asset")[vol_source].reindex(pivot.columns)
            weights = inverse_vol_weights(volumes)
        weight_matrix[date_position, :] = weights

    valid_rows = np.isfinite(weight_matrix).all(axis=1)
    values = np.sum(weight_matrix[valid_rows] * pivot_simple[valid_rows], axis=1)
    if values.size < 2:
        return {"scheme": weight_scheme, "n_days": int(values.size), "realized_vol": np.nan}
    weights_held = pd.DataFrame(weight_matrix[valid_rows], columns=pivot.columns)
    turnover = float(weights_held.diff().abs().sum(axis=1).mean())
    realized_vol = float(np.std(values, ddof=1) * np.sqrt(annualization_factor))
    annual_return = float(np.mean(values) * annualization_factor)
    sharpe = annual_return / realized_vol if realized_vol > 0 else np.nan
    cumulative = np.cumprod(1.0 + values)
    max_drawdown = float((cumulative / np.maximum.accumulate(cumulative) - 1.0).min())
    rolling = pd.Series(values).rolling(20, min_periods=10).std() * np.sqrt(annualization_factor)
    return {
        "scheme": weight_scheme,
        "n_days": int(values.size),
        "n_rebalances": int(len(rebalance_dates)),
        "realized_vol": realized_vol,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "turnover": turnover,
        "vol_volatility": float(rolling.std()) if rolling.notna().sum() > 1 else np.nan,
    }
