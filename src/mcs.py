"""Model Confidence Set (Hansen-Lunde-Nason 2011) via circular block bootstrap.

The conventional pipeline reports *pairwise* Diebold-Mariano p-values. MCS goes
one step further: it asks which *subset* of models survives a joint test that
they all have equal expected loss, and reports that surviving set as the answer
to "which models are statistically indistinguishable at the chosen level".

This implementation uses the studentized range statistic

    T(M) = max_{i,j in M} | t_ij | ,   t_ij = (mean_i - mean_j) / se_ij,

with ``se_ij`` derived from the same Bartlett HAC long-run variance that
``diebold_mariano_test`` uses (consistent variance estimation across the
project). The null distribution of T(M) is obtained from a *circular* block
bootstrap applied to the per-model mean-centered loss matrix; the block length
is set to reflect the autocorrelation of overlapping multi-day labels.

Inference is sequential: while the current candidate set M is rejected
(p_value < alpha), the model that contributes the largest pairwise violations
(sum of squared studentized t against the rest) is eliminated, and the test is
re-run on the remaining models. The models that are never eliminated form the
final MCS. A joint p-value equal to alpha means we could not reject that every
member of the reported set ties for best; models not in the set are, at this
level, measurably worse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_MAX_LAG = 4     # match walk-forward HAC lag used by the DM tests
DEFAULT_N_BOOTSTRAP = 10_000
DEFAULT_BLOCK_RATIO = 1 / 3.0  # block length ~ n ** (1/3), Politis-Romano


def bartlett_hac_variance(values: np.ndarray, max_lag: int) -> float:
    """Bartlett/Newey-West long-run variance, identical convention to the DM test."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n_obs = int(values.size)
    if n_obs == 0:
        return np.nan
    mean = float(values.mean())
    centered = values - mean
    lag = int(min(max_lag, n_obs - 1))
    long_run_variance = float(np.mean(centered * centered))
    for k in range(1, lag + 1):
        autocovariance = float(np.mean(centered[k:] * centered[:-k]))
        weight = 1.0 - k / (lag + 1.0)
        long_run_variance += 2.0 * weight * autocovariance
    return max(long_run_variance, 0.0)


def circular_block_bootstrap_indices(n_dates: int, block_length: int, *, seed: np.random.Generator) -> np.ndarray:
    """Return an index array of length ``n_dates`` drawn by the circular block bootstrap."""
    starts = seed.integers(0, n_dates, size=int(np.ceil(n_dates / block_length)) + 1)
    pooled = np.concatenate([(np.arange(block_length, dtype=int) + start) % n_dates for start in starts])
    return pooled[:n_dates]


def _pairwise_statistics(loss_matrix: np.ndarray, indices: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_ij, se_ij) matrices over members ``indices``.

    t_ij = (mean_i - mean_j) / se_ij, the pairwise studentized loss difference;
    se_ij is the Bartlett-HAC standard error of the mean difference, identical
    to the convention used by ``diebold_mariano_test``.
    """
    sub = loss_matrix[:, indices]          # T x m
    means = sub.mean(axis=0)               # per-member mean loss (lower is better)
    m = sub.shape[1]
    t_matrix = np.zeros((m, m))
    se_matrix = np.zeros((m, m))
    for a in range(m):
        for b in range(m):
            if a == b:
                continue
            diff = sub[:, a] - sub[:, b]
            lrv = bartlett_hac_variance(diff, max_lag)
            n_obs = sub.shape[0]
            se = float(np.sqrt(lrv / n_obs)) if lrv > 0 and n_obs > 0 else np.nan
            se_matrix[a, b] = se
            t_matrix[a, b] = (means[a] - means[b]) / se if (se and np.isfinite(se)) else np.nan
    return t_matrix, se_matrix


@dataclass(frozen=True)
class MCSRound:
    surviving_index: int              # column index into the original loss matrix
    model_name: str
    round_number: int
    mcs_p_value: float                # p-value of the joint test on the survived set
    eliminated_at: int | None         # round at which this model was removed (None = survived)
    max_t_statistic: float            # |T(M)| for the set that included it, when eliminated


@dataclass(frozen=True)
class MCSResult:
    survivor_names: list[str]                    # never eliminated
    eliminated: list[str]                        # in elimination order, worst-last removed first
    rounds: list[MCSRound]
    p_value: float                               # joint p-value of the final surviving set
    alpha: float
    max_lag: int
    block_length: int
    n_bootstrap: int


def model_confidence_set(
    loss_matrix: np.ndarray,
    *,
    names: list[str] | tuple[str, ...] | None = None,
    alpha: float = 0.10,
    max_lag: int = DEFAULT_MAX_LAG,
    block_length: int | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int | None = 42,
) -> MCSResult:
    """Run the MCS procedure on a ``n_dates x n_models`` loss matrix (lower is better)."""
    loss_matrix = np.asarray(loss_matrix, dtype=float)
    if loss_matrix.ndim != 2 or loss_matrix.shape[0] < 2 or loss_matrix.shape[1] < 2:
        raise ValueError("loss_matrix must be a matrix with at least 2 rows and 2 columns")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    n_dates, n_models = loss_matrix.shape
    if names is None:
        names = [f"model{i}" for i in range(n_models)]
    if len(names) != n_models:
        raise ValueError("number of names must match number of columns")

    if block_length is None:
        block_length = max(2, int(round(n_dates ** DEFAULT_BLOCK_RATIO)))
    block_length = int(block_length)
    if not (1 <= block_length <= n_dates):
        raise ValueError("block_length must be within [1, n_dates]")

    # Center each model once (the bootstrap re-samples the mean-zero fluctuations).
    centered = loss_matrix - loss_matrix.mean(axis=0, keepdims=True)

    rng = np.random.default_rng(seed)
    members = list(range(n_models))            # candidate column indices, worst-removal order
    rounds: list[MCSRound] = []
    stats_cache: dict[tuple[int, ...], np.ndarray] = {}

    while len(members) > 1:
        key = tuple(members)
        if key in stats_cache:
            t_matrix, se_matrix = stats_cache[key]
        else:
            t_matrix, se_matrix = _pairwise_statistics(loss_matrix, np.asarray(members, dtype=int), max_lag)
            stats_cache[key] = (t_matrix, se_matrix)
        finite = t_matrix[np.isfinite(t_matrix)]
        T_observed = float(np.max(np.abs(finite))) if finite.size else np.nan
        if not np.isfinite(T_observed):
            # Degenerate (e.g. a zero-variance member defeats the long-run variance).
            eliminated_index = members[int(np.argmax(loss_matrix[:, members].mean(axis=0)))]
            members.remove(eliminated_index)
            rounds.append(MCSRound(
                surviving_index=eliminated_index, model_name=names[eliminated_index],
                round_number=len(rounds) + 1, mcs_p_value=np.nan,
                eliminated_at=len(rounds) + 1, max_t_statistic=np.nan,
            ))
            continue

        # Bootstrap the null distribution of the studentized range statistic.
        # Resample mean-zero fluctuations by circular blocks, then studentize
        # each pair with its observed (fixed) standard error -- the same se used
        # for the in-sample statistic, so the bootstrap is a null-validated
        # permutation of the sample statistic.
        t_star_max = np.empty(n_bootstrap)
        sub_centered = centered[:, members]
        for b in range(n_bootstrap):
            idx = circular_block_bootstrap_indices(n_dates, block_length, seed=rng)
            boot_means = sub_centered[idx, :].mean(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (boot_means[:, None] - boot_means[None, :]) / se_matrix
            t_star_max[b] = float(np.nanmax(np.abs(ratio))) if np.isfinite(ratio).any() else 0.0
        p_value = float(np.mean(t_star_max >= T_observed))

        if p_value >= alpha:
            for member in members:
                rounds.append(MCSRound(
                    surviving_index=member, model_name=names[member],
                    round_number=len(rounds) + 1, mcs_p_value=p_value,
                    eliminated_at=None, max_t_statistic=np.nan,
                ))
            survivors = [names[m] for m in members]
            return MCSResult(
                survivor_names=survivors, eliminated=[r.model_name for r in rounds if r.eliminated_at is not None],
                rounds=rounds, p_value=float(p_value), alpha=alpha,
                max_lag=max_lag, block_length=block_length, n_bootstrap=n_bootstrap,
            )

        # Reject H0 on the current set -> remove the model with the largest
        # pairwise studentized violation (sum of squared t against the rest);
        # tie-break toward the higher mean loss so equal-t two-member sets drop
        # the worse performer rather than the better one.
        sq = np.nansum(t_matrix ** 2, axis=1)
        sq[~np.isfinite(t_matrix).all(axis=1)] = -np.inf
        member_means = loss_matrix[:, members].mean(axis=0)
        candidates = np.flatnonzero(np.isclose(sq, sq[np.nanargmax(sq)]))
        worst_local = candidates[int(np.argmax(member_means[candidates]))]
        eliminated_index = members[worst_local]
        members.remove(eliminated_index)
        rounds.append(MCSRound(
            surviving_index=eliminated_index, model_name=names[eliminated_index],
            round_number=len(rounds) + 1, mcs_p_value=p_value,
            eliminated_at=len(rounds) + 1, max_t_statistic=T_observed,
        ))

    # Exactly one model left: it is the sole survivor.
    survivor = members[0]
    p_value = rounds[-1].mcs_p_value if rounds else np.nan
    rounds.append(MCSRound(
        surviving_index=survivor, model_name=names[survivor],
        round_number=len(rounds) + 1, mcs_p_value=p_value,
        eliminated_at=None, max_t_statistic=np.nan,
    ))
    return MCSResult(
        survivor_names=[names[survivor]], eliminated=[r.model_name for r in rounds if r.eliminated_at is not None],
        rounds=rounds, p_value=float(p_value) if np.isfinite(p_value) else np.nan,
        alpha=alpha, max_lag=max_lag, block_length=block_length, n_bootstrap=n_bootstrap,
    )


def build_pooled_loss_matrix(
    frame: pd.DataFrame,
    model_columns: list[str] | tuple[str, ...],
    *,
    actual_column: str = "future_rv_5d",
    loss: str = "qlike",
    epsilon: float = 1.0e-12,
    date_column: str = "date",
) -> tuple[pd.DataFrame, list[str]]:
    """Per-date pooled-ALL loss matrix (QLIKE or MAE), mirroring the DM pooling.

    Each date row is the cross-sectional mean of per-asset loss over that date's
    valid assets (each model's row is dropped where its forecast or the actual
    is missing/extraordinary). Columns are the model names.
    """
    if loss not in ("qlike", "mae"):
        raise ValueError(f"unsupported MCS loss: {loss!r}")
    frame = frame.copy()
    finite = frame[actual_column].notna() & frame[actual_column].ge(0)
    for column in model_columns:
        finite &= frame[column].notna()
    frame = frame.loc[finite].copy()

    expected_cols = [date_column, actual_column, *model_columns]
    present = [c for c in expected_cols if c in frame.columns]
    frame = frame[present]

    loss_cols = {}
    if loss == "mae":
        for column in model_columns:
            loss_cols[column] = (frame[column] - frame[actual_column]).abs()
    else:
        for column in model_columns:
            forecast_var = np.maximum(np.square(frame[column].to_numpy(dtype=float)), epsilon)
            actual_var = np.square(frame[actual_column].to_numpy(dtype=float))
            loss_cols[column] = np.log(forecast_var) + actual_var / forecast_var
    frame = frame.assign(**loss_cols)

    daily = frame.groupby(date_column)[model_columns].mean()
    daily = daily.sort_index()
    return daily, list(model_columns)


def mcs_summary_frame(result: MCSResult, *, loss: str = "qlike", protocol: str = "locked") -> pd.DataFrame:
    """Tidy per-model summary that the pipeline writes out for logging/reporting."""
    rows = []
    for round_ in result.rounds:
        rows.append({
            "protocol": protocol,
            "loss": loss,
            "model": round_.model_name,
            "survivor": round_.eliminated_at is None,
            "eliminated_at": round_.eliminated_at,
            "round_number": round_.round_number,
            "mcs_p_value": round_.mcs_p_value,
            "max_t_statistic": round_.max_t_statistic,
        })
    frame = pd.DataFrame(rows)
    frame["alpha"] = float(result.alpha)
    frame["n_bootstrap"] = int(result.n_bootstrap)
    frame["block_length"] = int(result.block_length)
    frame["max_hac_lag"] = int(result.max_lag)
    return frame
