from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    assign_walk_forward_segments,
    assign_expanding_segments,
    diebold_mariano_test,
    dm_by_segment,
    exclude_cross_segment_labels,
    expanding_window_forecasts,
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


def _multi_year_frame(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2010-01-01", periods=6 * 250, freq="B")
    base = np.linspace(0.010, 0.030, len(dates))
    frame = pd.DataFrame({"asset": "A", "date": dates})
    frame["har_daily_rv"] = np.maximum((base + rng.normal(0, 0.002, len(dates))) ** 2, 1e-8)
    frame["har_weekly_rv"] = np.maximum((base + rng.normal(0, 0.003, len(dates))) ** 2, 1e-8)
    frame["har_monthly_rv"] = np.maximum((base + rng.normal(0, 0.004, len(dates))) ** 2, 1e-8)
    frame["future_rv_5d"] = np.maximum(
        np.sqrt(np.maximum(frame["har_daily_rv"], 1e-8)) * 1.1 + rng.normal(0, 0.005, len(dates)), 0.01
    )
    return frame


def test_expanding_segments_are_non_overlapping() -> None:
    frame = assign_expanding_segments(_multi_year_frame(), eval_year=2014, validation_years=2)
    by_year = frame.groupby(frame["date"].dt.year)["segment"].agg(lambda values: set(values))
    assert by_year[2011] == {"train"}
    assert by_year[2012] == {"validation"}
    assert by_year[2013] == {"validation"}
    assert by_year[2014] == {"test"}


def test_expanding_forecasts_first_eval_year_equals_locked() -> None:
    from src.evaluation import fit_har_by_asset

    frame = _multi_year_frame()
    expanding, _ = expanding_window_forecasts(
        frame,
        fit_function=fit_har_by_asset,
        output_column="har_rv_exp",
        eval_years=(2014, 2015),
        validation_years=2,
        horizon=5,
        fit_kwargs={"train_segment": "train", "actual_column": "future_rv_5d", "output_column": "har_rv_exp"},
    )
    assert expanding["har_rv_exp"].notna().sum() > 0
    for year in (2014, 2015):
        values = expanding.loc[expanding["date"].dt.year == year, "har_rv_exp"]
        assert values.notna().all()
        assert (values > 0).all()
    assert expanding.loc[expanding["date"].dt.year.isin([2010, 2012]), "har_rv_exp"].isna().all()

    locked = assign_walk_forward_segments(frame, train_end="2011-12-31", validation_end="2013-12-31")
    locked, _ = exclude_cross_segment_labels(locked, horizon=5)
    locked_fit, _ = fit_har_by_asset(locked, train_segment="train", actual_column="future_rv_5d", output_column="har_rv")
    exp_2014 = expanding.loc[expanding["date"].dt.year == 2014].sort_values("date")["har_rv_exp"].to_numpy()
    locked_2014 = locked_fit.loc[locked_fit["date"].dt.year == 2014].sort_values("date")["har_rv"].to_numpy()
    np.testing.assert_allclose(exp_2014, locked_2014, atol=1e-12)
