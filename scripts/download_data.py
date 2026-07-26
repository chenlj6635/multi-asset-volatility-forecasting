#!/usr/bin/env python3
"""Download raw Yahoo Finance files; this is deliberately separate from offline build."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import normalize_market_data


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def download(config_path: Path, *, retries: int = 2, retry_delay: float = 2.0) -> dict[str, Any]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is required only for download_data.py; install project dependencies first") from exc

    config_path = config_path.resolve()
    config = load_config(config_path)
    root = config_path.parent.parent
    cfg = config["data"]
    raw_dir = Path(cfg["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_map = {**cfg["target_assets"], **cfg.get("context_assets", {})}
    manifest: dict[str, Any] = {
        "source": cfg["source"],
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "start_date": cfg["start_date"],
        "end_date": cfg.get("end_date"),
        "auto_adjust": False,
        "price_return_field": "Adj Close",
        "assets": {},
    }

    for ticker, filename in file_map.items():
        record: dict[str, Any] = {"filename": filename, "status": "failed"}
        for attempt in range(retries + 1):
            try:
                downloaded = yf.download(
                    ticker,
                    start=cfg["start_date"],
                    end=cfg.get("end_date"),
                    auto_adjust=False,
                    progress=False,
                    actions=False,
                    threads=False,
                )
                normalized, warnings = normalize_market_data(downloaded, ticker)
                output = normalized.drop(columns="asset").rename(
                    columns={
                        "date": "Date", "open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "adj_close": "Adj Close", "volume": "Volume",
                    }
                )
                output.to_csv(raw_dir / filename, index=False)
                record.update(
                    status="ok",
                    row_count=int(len(output)),
                    start_date=output["Date"].min().date().isoformat(),
                    end_date=output["Date"].max().date().isoformat(),
                    warnings=warnings,
                )
                break
            except Exception as exc:  # preserve per-asset failure in the manifest
                record["error"] = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    time.sleep(retry_delay * (attempt + 1))
        manifest["assets"][ticker] = record
        (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/default.yaml")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    args = parser.parse_args()
    manifest = download(args.config, retries=args.retries, retry_delay=args.retry_delay)
    print(json.dumps(manifest, indent=2))
    if any(record["status"] != "ok" for record in manifest["assets"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
