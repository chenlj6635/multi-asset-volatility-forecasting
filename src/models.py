"""GARCH volatility forecasts with parameters locked from the train segment."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _garch_recursive_forecast(
    returns: np.ndarray,
    *,
    omega: float,
    alpha: float,
    beta: float,
    mu: float,
    horizon: int,
    annualization_factor: float,
    initial_variance: float,
) -> np.ndarray:
    """Return the ``horizon``-day annualized volatility forecast at each date.

    Standard GARCH timing is used: the conditional variance for the return at
    date ``i`` is ``sigma2_i = omega + alpha * eps_{i-1}^2 + beta * sigma2_{i-1}``
    with ``eps = r - mu``. The forecast window for date ``t`` starts at ``t+1``:
    the first step variance is ``h1 = omega + alpha * eps_t^2 + beta * sigma2_t``
    and later steps iterate ``h_k = omega + (alpha + beta) * h_{k-1}``. Dates
    with a non-finite return get a NaN forecast and leave the variance state
    unchanged, so the path never looks ahead.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    if not all(np.isfinite([omega, alpha, beta, mu])):
        raise ValueError("GARCH parameters must be finite")
    n = len(returns)
    forecast = np.full(n, np.nan)
    sigma2_prev = float(initial_variance)
    dampening = alpha + beta
    for index in range(n):
        value = returns[index]
        if not np.isfinite(value):
            forecast[index] = np.nan
            continue
        eps = value - mu
        sigma2_i = sigma2_prev
        first_step = omega + alpha * eps * eps + beta * sigma2_i
        total = first_step
        next_step = first_step
        for _ in range(1, horizon):
            next_step = omega + dampening * next_step
            total += next_step
        forecast[index] = np.sqrt(annualization_factor / horizon * total)
        sigma2_prev = omega + alpha * eps * eps + beta * sigma2_i
    return forecast


def fit_garch_by_asset(
    frame: pd.DataFrame,
    *,
    return_column: str = "log_return",
    output_column: str = "garch_rv",
    horizon: int = 5,
    annualization_factor: float = 252.0,
    train_segment: str = "train",
    p: int = 1,
    q: int = 1,
    min_train_observations: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit GARCH(p,q) once per asset on the train segment and forecast forward.

    Coefficients are estimated by maximum likelihood on the train segment only
    and then held fixed while the conditional-variance recursion is applied to
    the full ordered history, so every forecast uses only information available
    through its own forecast date. Assets whose Gaussian fit fails, lacks enough
    training observations, or produces non-finite parameters are reported in the
    parameter table with a NaN forecast column.
    """
    if p < 0 or q < 0 or p + q == 0:
        raise ValueError("GARCH orders p and q must be non-negative and not both zero")
    if min_train_observations <= 0:
        raise ValueError("min_train_observations must be positive")
    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    data[output_column] = np.nan
    rows: list[dict[str, float | int | bool | str]] = []
    for asset, group in data.groupby("asset", sort=True):
        train = group.loc[group["segment"] == train_segment, return_column].to_numpy(dtype=float)
        train = train[np.isfinite(train)]
        if len(train) < min_train_observations:
            rows.append({
                "asset": asset, "omega": np.nan, "alpha": np.nan, "beta": np.nan,
                "mu": np.nan, "alpha_plus_beta": np.nan, "stationary": False,
                "n_train": int(len(train)), "convergence_flag": np.nan,
                "likelihood": np.nan, "status": "insufficient_train_observations",
            })
            continue
        try:
            from arch import arch_model

            # Scale returns by the training-sample standard deviation before MLE:
            # daily returns are far below arch's preferred unit scale, which can
            # hurt optimizer convergence. omega and mu are rescaled back afterward;
            # alpha and beta are scale-invariant.
            scale = float(np.std(train, ddof=1))
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("training return standard deviation must be finite and positive")
            scaled = train / scale
            model = arch_model(
                scaled, mean="Constant", vol="GARCH", p=p, q=q, dist="normal"
            )
            result = model.fit(disp="off", show_warning=False)
            omega = float(result.params["omega"]) * scale * scale
            alpha = float(result.params["alpha[1]"])
            beta = float(result.params["beta[1]"])
            mu = float(result.params["mu"]) * scale
            convergence_flag = int(getattr(result, "convergence_flag", 0))
            likelihood = float(result.loglikelihood)
        except Exception as exc:  # pragma: no cover - optimizer path depends on data
            rows.append({
                "asset": asset, "omega": np.nan, "alpha": np.nan, "beta": np.nan,
                "mu": np.nan, "alpha_plus_beta": np.nan, "stationary": False,
                "n_train": int(len(train)), "convergence_flag": np.nan,
                "likelihood": np.nan, "status": f"fit_failed: {str(exc)[:120]}",
            })
            continue
        if not np.isfinite([omega, alpha, beta, mu]).all():
            rows.append({
                "asset": asset, "omega": omega, "alpha": alpha, "beta": beta,
                "mu": mu, "alpha_plus_beta": float(alpha + beta),
                "stationary": bool(alpha + beta < 1),
                "n_train": int(len(train)), "convergence_flag": convergence_flag,
                "likelihood": likelihood, "status": "nonfinite_parameters",
            })
            continue
        unconditional_variance = omega / (1.0 - alpha - beta) if alpha + beta < 1 else float(np.var(train, ddof=1))
        forecast = _garch_recursive_forecast(
            group[return_column].to_numpy(dtype=float),
            omega=omega,
            alpha=alpha,
            beta=beta,
            mu=mu,
            horizon=horizon,
            annualization_factor=annualization_factor,
            initial_variance=unconditional_variance,
        )
        data.loc[group.index, output_column] = forecast
        rows.append({
            "asset": asset, "omega": omega, "alpha": alpha, "beta": beta,
            "mu": mu, "alpha_plus_beta": float(alpha + beta),
            "stationary": bool(alpha + beta < 1),
            "n_train": int(len(train)), "convergence_flag": convergence_flag,
            "likelihood": likelihood,
            "status": "ok" if convergence_flag == 0 else "convergence_warning",
            "nonfinite_forecast_count": int(np.sum(~np.isfinite(forecast))),
        })
    return data, pd.DataFrame(rows)
