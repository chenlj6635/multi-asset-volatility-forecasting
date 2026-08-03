from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    assign_walk_forward_segments,
    diebold_mariano_test,
    dm_by_segment,
    exclude_cross_segment_labels,
    forecast_losses,
    walk_forward_metrics,
)


def sample_frame() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    return pd.DataFrame({
        "asset": ["A"] * 12,
        "date": dates,
        "future_rv_5d": np.arange(1, 13, dtype=float),
        "historical_rv_21d": np.arange(1, 13, dtype=float),
        "ewma_rv": np.arange(1, 13, dtype=float),
    })


def test_qlike_loss_difference_and_dm() -> None:
    frame = sample_frame()
    frame["ewma_rv"] = frame["future_rv_5d"]
    losses = forecast_losses(frame)
    assert np.allclose(losses["qlike_ewma"], np.log(frame["future_rv_5d"] ** 2) + 1.0)
    result = diebold_mariano_test(np.array([-1.0, -1.0, -1.0]), max_lag=4)
    assert result["n_obs"] == 3
    assert result["hac_lag"] == 2
    assert np.isnan(result["dm_statistic"])


def test_dm_by_segment_outputs_asset_and_all_rows() -> None:
    frame = assign_walk_forward_segments(
        sample_frame(), train_end="2020-01-04", validation_end="2020-01-08"
    )
    result = dm_by_segment(frame, max_lag=1, losses=("qlike", "mae"))
    assert set(result["loss"]) == {"qlike", "mae"}
    assert set(result["segment"]) == {"train", "validation", "test"}
    assert (result["model_a"] == "ewma_rv").all()
    assert (result["model_b"] == "historical_rv_21d").all()


def test_pooled_all_averages_by_date_before_hac() -> None:
    dates = pd.date_range("2020-01-01", periods=3)
    frame = pd.DataFrame({
        "asset": ["A", "B", "A", "B", "A"],
        "date": [dates[0], dates[0], dates[1], dates[1], dates[2]],
        "segment": ["test"] * 5,
        "future_rv_5d": [1.0] * 5,
        "historical_rv_21d": [1.0] * 5,
        "ewma_rv": [np.exp(0.5), np.exp(1.5), np.exp(2.5), np.exp(4.5), np.exp(6.0)],
    })
    result = dm_by_segment(frame, max_lag=0)
    pooled = result.loc[(result["segment"] == "test") & (result["asset"] == "ALL")].iloc[0]
    expected_daily = (
        forecast_losses(frame)
        .groupby("date")["qlike_loss_diff"]
        .mean()
        .to_numpy()
    )
    assert pooled["n_obs"] == 3
    assert pooled["n_dates"] == 3
    assert pooled["paired_rows"] == 5
    assert pooled["mean_loss_diff"] == pytest.approx(expected_daily.mean())

    shuffled = dm_by_segment(frame.sample(frac=1, random_state=17), max_lag=0)
    shuffled_pooled = shuffled.loc[(shuffled["segment"] == "test") & (shuffled["asset"] == "ALL")].iloc[0]
    assert shuffled_pooled["mean_loss_diff"] == pytest.approx(pooled["mean_loss_diff"])
    result = assign_walk_forward_segments(
        sample_frame(), train_end="2020-01-04", validation_end="2020-01-08"
    )
    assert result.groupby("segment").size().to_dict() == {"train": 4, "validation": 4, "test": 4}
    with pytest.raises(ValueError, match="train_end"):
        assign_walk_forward_segments(sample_frame(), train_end="2020-01-08", validation_end="2020-01-04")


def test_cross_segment_labels_are_removed() -> None:
    segmented = assign_walk_forward_segments(
        sample_frame(), train_end="2020-01-04", validation_end="2020-01-08"
    )
    result, counts = exclude_cross_segment_labels(segmented, horizon=2)
    assert counts == {"train": 2, "validation": 2, "test": 0}
    assert result.groupby("segment").size().to_dict() == {"train": 2, "validation": 2, "test": 4}


def test_walk_forward_metrics_have_each_segment_and_forecast() -> None:
    segmented = assign_walk_forward_segments(
        sample_frame(), train_end="2020-01-04", validation_end="2020-01-08"
    )
    result = walk_forward_metrics(
        segmented, forecast_columns=("historical_rv_21d", "ewma_rv")
    )
    assert set(result["segment"]) == {"train", "validation", "test"}
    assert set(result["forecast"]) == {"historical_rv_21d", "ewma_rv"}
    assert set(result["asset"]) == {"A", "ALL"}
    assert len(result) == 12
