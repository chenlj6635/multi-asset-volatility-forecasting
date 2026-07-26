"""Raw market-data loading, normalization, and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = ["asset", "date", "open", "high", "low", "close", "adj_close", "volume"]
_REQUIRED_PRICE_COLUMNS = ["open", "high", "low", "close", "adj_close"]


def _canonical_name(name: object) -> str:
    text = str(name).strip().lower().replace("_", " ").replace("-", " ")
    text = " ".join(text.split())
    aliases = {
        "date": "date",
        "datetime": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "adj_close",
        "adjusted close": "adj_close",
        "adjclose": "adj_close",
        "volume": "volume",
    }
    return aliases.get(text, text.replace(" ", "_"))


def normalize_market_data(
    frame: pd.DataFrame,
    asset: str,
    *,
    allow_close_as_adjusted: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Normalize one asset to the canonical contract and return warnings."""
    if frame.empty:
        raise ValueError(f"{asset}: raw data is empty")

    data = frame.copy()
    if isinstance(data.columns, pd.MultiIndex):
        # yfinance commonly returns (field, ticker) even for a single ticker.
        ticker_level = next(
            (
                level
                for level in range(data.columns.nlevels)
                if asset in data.columns.get_level_values(level)
            ),
            None,
        )
        if ticker_level is not None:
            data = data.xs(asset, axis=1, level=ticker_level, drop_level=True)
        else:
            data.columns = data.columns.get_level_values(0)
    data = data.rename(columns={column: _canonical_name(column) for column in data.columns})
    if "date" not in data.columns:
        if data.index.name is not None:
            data = data.reset_index().rename(columns={data.index.name: "date"})
        elif isinstance(data.index, pd.DatetimeIndex):
            data = data.reset_index().rename(columns={data.index.name or "index": "date"})

    if "adj_close" not in data.columns:
        if allow_close_as_adjusted and "close" in data.columns:
            data["adj_close"] = data["close"]
        else:
            raise ValueError(
                f"{asset}: missing Adj Close; set allow_close_as_adjusted only after confirming adjusted prices"
            )

    missing = [column for column in ["date", *_REQUIRED_PRICE_COLUMNS] if column not in data.columns]
    if missing:
        raise ValueError(f"{asset}: missing required columns: {', '.join(missing)}")
    if "volume" not in data.columns:
        data["volume"] = np.nan

    parsed = pd.to_datetime(data["date"], errors="coerce", utc=True)
    if parsed.isna().any():
        raise ValueError(f"{asset}: {int(parsed.isna().sum())} dates could not be parsed")
    data["date"] = parsed.dt.tz_convert(None).dt.normalize()

    for column in [*_REQUIRED_PRICE_COLUMNS, "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data.duplicated(["date"]).any():
        examples = data.loc[data.duplicated(["date"], keep=False), "date"].dt.strftime("%Y-%m-%d").head(3)
        raise ValueError(f"{asset}: duplicate dates detected: {', '.join(examples)}")
    if (data["adj_close"].dropna() <= 0).any():
        raise ValueError(f"{asset}: adjusted close contains non-positive values")
    if data["adj_close"].isna().any():
        raise ValueError(f"{asset}: adjusted close contains missing values")

    warnings: list[str] = []
    if not data["date"].is_monotonic_increasing:
        warnings.append("input dates were not sorted; rows were sorted safely")
    if asset == "^VIX" and data["volume"].isna().any():
        warnings.append("VIX volume is missing; retained as a non-fatal warning")

    data["asset"] = asset
    return data[CANONICAL_COLUMNS].sort_values("date", kind="stable").reset_index(drop=True), warnings


def read_asset_csv(
    path: str | Path,
    asset: str,
    *,
    allow_close_as_adjusted: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{asset}: raw CSV not found: {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{asset}: raw CSV is empty: {path}") from exc
    return normalize_market_data(frame, asset, allow_close_as_adjusted=allow_close_as_adjusted)


def load_raw_assets(
    raw_dir: str | Path,
    file_map: Mapping[str, str],
    *,
    allow_close_as_adjusted: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Load assets independently without forcing a common calendar."""
    frames: list[pd.DataFrame] = []
    warnings: dict[str, list[str]] = {}
    for asset, filename in file_map.items():
        frame, asset_warnings = read_asset_csv(
            Path(raw_dir) / filename,
            asset,
            allow_close_as_adjusted=allow_close_as_adjusted,
        )
        frames.append(frame)
        warnings[asset] = asset_warnings
    return pd.concat(frames, ignore_index=True), warnings


def require_assets(frame: pd.DataFrame, expected: Iterable[str]) -> None:
    missing = sorted(set(expected) - set(frame["asset"].unique()))
    if missing:
        raise ValueError(f"missing required assets: {', '.join(missing)}")
