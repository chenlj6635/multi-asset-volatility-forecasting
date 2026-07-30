"""Data-quality, result, metadata, and figure outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_quality_report(
    frame: pd.DataFrame,
    *,
    warnings: Mapping[str, list[str]] | None = None,
    extreme_return_threshold: float = 0.20,
    long_gap_days: int = 7,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    warnings = warnings or {}
    for asset, group in frame.sort_values("date").groupby("asset", sort=True):
        returns = np.log(group["adj_close"]).diff()
        ohlc_anomaly = (
            (group["high"] < group[["open", "close", "low"]].max(axis=1))
            | (group["low"] > group[["open", "close", "high"]].min(axis=1))
        )
        gaps = group["date"].diff().dt.days
        asset_warnings = list(warnings.get(asset, []))
        if ohlc_anomaly.any():
            asset_warnings.append("OHLC relationship anomalies detected")
        if (returns.abs() > extreme_return_threshold).any():
            asset_warnings.append("extreme adjusted-close returns detected; retained unchanged")
        rows.append(
            {
                "asset": asset,
                "row_count": int(len(group)),
                "start_date": group["date"].min().date().isoformat(),
                "end_date": group["date"].max().date().isoformat(),
                "missing_open": int(group["open"].isna().sum()),
                "missing_high": int(group["high"].isna().sum()),
                "missing_low": int(group["low"].isna().sum()),
                "missing_close": int(group["close"].isna().sum()),
                "missing_adj_close": int(group["adj_close"].isna().sum()),
                "missing_volume": int(group["volume"].isna().sum()),
                "ohlc_anomaly_count": int(ohlc_anomaly.sum()),
                "extreme_return_count": int((returns.abs() > extreme_return_threshold).sum()),
                "long_gap_count": int((gaps > long_gap_days).sum()),
                "warnings": " | ".join(dict.fromkeys(asset_warnings)),
            }
        )
    table = pd.DataFrame(rows)
    summary = {
        "status": "warning" if table["warnings"].astype(bool).any() else "ok",
        "asset_count": int(len(table)),
        "total_rows": int(table["row_count"].sum()),
        "assets_with_warnings": table.loc[table["warnings"].astype(bool), "asset"].tolist(),
    }
    return table, summary


def write_quality_report(table: pd.DataFrame, summary: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "asset_quality.csv", index=False)
    (output / "quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_results(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    predictions_path: str | Path,
    metrics_path: str | Path,
    metadata_path: str | Path,
) -> None:
    predictions_path = Path(predictions_path)
    metrics_path = Path(metrics_path)
    metadata_path = Path(metadata_path)
    for path in [predictions_path, metrics_path, metadata_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def plot_spy_comparison(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    forecast_columns: tuple[str, ...] = ("historical_rv_21d", "ewma_rv"),
) -> None:
    """Save a readable SPY comparison for actual and forecast volatility."""
    columns = ["date", "future_rv_5d", *forecast_columns]
    spy = frame.loc[frame["asset"] == "SPY", columns].dropna()
    if spy.empty:
        raise ValueError("SPY has no valid observations to plot")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    fig.patch.set_facecolor("#FCFCFB")
    ax.set_facecolor("#FCFCFB")
    ax.plot(spy["date"], spy["future_rv_5d"], color="#0072B2", linewidth=1.6, label="Future 5-day realized volatility")
    styles = {
        "historical_rv_21d": ("#D55E00", "--", "Historical 21-day baseline"),
        "ewma_rv": ("#009E73", "-.", "EWMA baseline"),
    }
    for column in forecast_columns:
        color, linestyle, label = styles.get(column, ("#CC79A7", ":", column))
        ax.plot(spy["date"], spy[column], color=color, linewidth=1.6, linestyle=linestyle, label=label)
    ax.set_title("SPY volatility: future realized outcome vs baselines", loc="left", weight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualized volatility")
    ax.grid(axis="y", color="#D9D9D6", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.savefig(output_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
