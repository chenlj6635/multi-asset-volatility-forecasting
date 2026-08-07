from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import diebold_mariano_test
from src.mcs import (
    build_pooled_loss_matrix,
    model_confidence_set,
    _pairwise_statistics,
)


def _losses(mu: float, noise_scale: float, n_dates: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return mu + noise_scale * rng.normal(size=n_dates)


def _two_col(mu0: float, mu1: float, n_dates: int = 200) -> np.ndarray:
    """Two independent columns: model0 ~ mu0, model1 ~ mu1 (lower is better)."""
    cols = []
    for mu in (mu0, mu1):
        rng = np.random.default_rng(abs(int(mu * 1000)) + 7)
        cols.append(mu + rng.normal(0, 1.0, size=n_dates))
    return np.column_stack(cols)


def test_clear_dominance_keeps_only_best() -> None:
    losses = _two_col(0.0, 8.0, n_dates=300)
    result = model_confidence_set(
        losses, names=["best", "worst"], n_bootstrap=800, seed=1
    )
    assert result.survivor_names == ["best"]
    assert result.eliminated == ["worst"]
    assert result.rounds[0].eliminated_at == 1


def test_identical_models_all_survive() -> None:
    base = _losses(3.0, 1.0, n_dates=200)
    losses = np.column_stack([base, base.copy(), base.copy()])
    result = model_confidence_set(
        losses, names=["a", "b", "c"], n_bootstrap=500, seed=2
    )
    assert set(result.survivor_names) == {"a", "b", "c"}
    assert result.eliminated == []
    assert result.p_value >= result.alpha


def test_tied_best_both_survive() -> None:
    rng = np.random.default_rng(0)
    base = rng.normal(0.3, 1.0, size=300)
    losses = np.column_stack([base, base + rng.normal(0, 0.5, size=300)])
    result = model_confidence_set(
        losses, names=["a", "b"], alpha=0.10, n_bootstrap=800, seed=3
    )
    assert result.survivor_names == ["a", "b"]
    assert result.p_value >= result.alpha


def test_worst_eliminated_first() -> None:
    losses = _two_col(0.5, 2.0, n_dates=400)
    rng = np.random.default_rng(5)
    third = 0.55 + rng.normal(0, 1.0, size=400)
    losses = np.column_stack([losses[:, 0], losses[:, 1], third])
    result = model_confidence_set(
        losses, names=["best", "clearly_worst", "near_best"], alpha=0.15, n_bootstrap=900, seed=4
    )
    assert result.eliminated[0] == "clearly_worst"
    assert set(result.survivor_names) <= {"best", "near_best"}


def test_pairwise_t_matches_dm_statistic() -> None:
    losses = _two_col(1.0, 1.6, n_dates=400)
    t_matrix, _ = _pairwise_statistics(losses, np.array([0, 1]), max_lag=4)
    diff = losses[:, 0] - losses[:, 1]
    dm = diebold_mariano_test(diff, max_lag=4)
    assert t_matrix[0, 1] == pytest.approx(dm["dm_statistic"], rel=1e-9)
    assert t_matrix[1, 0] == pytest.approx(-dm["dm_statistic"], rel=1e-9)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="row"):
        model_confidence_set(np.ones((1, 3)), names=["a", "b", "c"])
    with pytest.raises(ValueError, match="names"):
        model_confidence_set(np.ones((20, 3)), names=["a", "b"])
    with pytest.raises(ValueError, match="alpha"):
        model_confidence_set(np.ones((20, 2)), names=["a", "b"], alpha=1.5)


def test_build_pooled_loss_matrix_pools_by_date_and_drops_missing() -> None:
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    frame = pd.DataFrame({
        "asset": ["A", "B", "A", "B", "A"],
        "date": [dates[0], dates[0], dates[1], dates[1], dates[2]],
        "future_rv_5d": [0.10, 0.10, 0.10, 0.10, 0.10],
        "har_rv": [0.10, 0.10, 0.10, 0.10, 0.10],
        "lgb_rv": [0.10, np.nan, 0.20, 0.20, 0.20],
    })
    daily, names = build_pooled_loss_matrix(
        frame, ["har_rv", "lgb_rv"], actual_column="future_rv_5d", epsilon=1e-12
    )
    assert names == ["har_rv", "lgb_rv"]
    assert len(daily) == 3

    # The date-0 'lgb_rv' row for asset B is NaN -> pooled har uses both assets,
    # pooled lgb only asset A on that row. Fine: each date still has >=1 obs.
    qlike_har = frame.loc[frame["har_rv"].notna()].groupby("date").apply(
        lambda g: float(np.log(0.10 ** 2) + (0.10 ** 2) / (0.10 ** 2)), include_groups=False
    )
    assert np.allclose(daily["har_rv"], qlike_har.reindex(daily.index).to_numpy(), equal_nan=True)

    eps = 1e-12
    qlike = lambda v: float(np.log(eps + v ** 2) + (0.10 ** 2) / max(v ** 2, eps))
    assert np.allclose(daily.loc[dates[2], "har_rv"], qlike(0.10))
    assert np.allclose(daily.loc[dates[2], "lgb_rv"], qlike(0.20))


def test_full_pipeline_loss_matrix_feeds_mcs() -> None:
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    frame = pd.DataFrame({
        "asset": ["A", "B"] * 6,
        "date": list(dates) * 2,
        "future_rv_5d": [0.10] * 12,
        "garch_rv": [0.10] * 12,
        "har_rv": [0.10, 0.09] * 6,
    })
    daily, names = build_pooled_loss_matrix(frame, ["garch_rv", "har_rv"])
    result = model_confidence_set(daily.to_numpy(), names=names, n_bootstrap=200, seed=0)
    assert len(result.survivor_names) >= 1
    assert set(result.survivor_names) <= set(names)
