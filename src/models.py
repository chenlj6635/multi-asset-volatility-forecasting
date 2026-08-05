"""GARCH volatility forecasts with parameters locked from the train segment."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from src.metrics import evaluate_forecast


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


def _soft_threshold(value: np.ndarray | float, threshold: float) -> np.ndarray | float:
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)


def _fit_ridge_ls(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l2: float,
) -> tuple[np.ndarray, float]:
    """Ridge least squares on standardized columns; intercept unpenalized."""
    y_centered = y - float(y.mean())
    if l2 <= 0:
        beta, *_ = np.linalg.lstsq(x, y_centered, rcond=None)
    else:
        beta = np.linalg.solve(x.T @ x + l2 * np.eye(x.shape[1]), x.T @ y_centered)
    return np.asarray(beta, dtype=float), float(y.mean())


def _fit_lasso_cd(
    x: np.ndarray,
    y: np.ndarray,
    *,
    l1: float,
    max_iter: int = 5000,
    tolerance: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Coordinate-descent LASSO on standardized columns; intercept unpenalized."""
    observations, predictors = x.shape
    beta = np.zeros(predictors)
    y_centered = y - float(y.mean())
    residual = y_centered - x @ beta
    for _ in range(max_iter):
        beta_old = beta.copy()
        for j in range(predictors):
            residual = residual + beta[j] * x[:, j]
            rho = float(x[:, j] @ residual)
            scale = float(x[:, j] @ x[:, j])
            if scale <= 0:
                continue
            beta[j] = float(_soft_threshold(rho, l1)) / scale
            residual = residual - beta[j] * x[:, j]
        if np.max(np.abs(beta - beta_old)) < tolerance:
            break
    return beta, float(y.mean())


def fit_ridge_by_asset(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...],
    actual_column: str = "future_rv_5d",
    output_column: str = "ridge_rv",
    penalty: str = "ridge",
    lambda_grid: list[float] | tuple[float, ...],
    train_segment: str = "train",
    validation_segment: str = "validation",
    target_mode: str = "log_variance",
    log_floor: float = 1.0e-12,
    variance_floor: float = 1.0e-4,
    epsilon: float = 1.0e-12,
    min_train_observations: int = 120,
    min_validation_observations: int = 20,
    lasso_max_iter: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a penalized linear model per asset with validation-selected penalty.

    Features are z-scored using the train segment per asset; the penalty lambda
    is selected per asset on the validation segment by pooled QLIKE (ties resolve
    to the smallest lambda, i.e. the least regularized model); and coefficients
    are refit on the train segment with the chosen lambda before scoring any
    segment. With ``target_mode="log_variance"`` (default) the model fits
    ``log(future_rv_5d^2)`` and forecasts are recovered exponentially, so
    variance forecasts are always strictly positive; with ``target_mode="variance"``
    raw variance forecasts below ``variance_floor`` are clipped and counted.
    """
    if penalty not in ("ridge", "lasso"):
        raise ValueError("penalty must be 'ridge' or 'lasso'")
    if target_mode not in ("log_variance", "variance"):
        raise ValueError("target_mode must be 'log_variance' or 'variance'")
    if log_floor <= 0:
        raise ValueError("log_floor must be positive")
    values = [float(value) for value in lambda_grid]
    if not values or not all(np.isfinite(values)):
        raise ValueError("lambda_grid must be a non-empty list of finite values")
    if min_train_observations <= 0 or min_validation_observations <= 0:
        raise ValueError("minimum observation counts must be positive")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")

    feature_columns = list(feature_columns)
    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    data[output_column] = np.nan
    parameter_rows: list[dict[str, float | int | str]] = []
    selection_rows: list[dict[str, float | int | str | bool]] = []

    for asset, group in data.groupby("asset", sort=True):
        train = group.loc[
            (group["segment"] == train_segment)
            & group[[actual_column, *feature_columns]].notna().all(axis=1)
        ]
        validation = group.loc[
            (group["segment"] == validation_segment)
            & group[[actual_column, *feature_columns]].notna().all(axis=1)
        ]
        if len(train) < min_train_observations or len(validation) < min_validation_observations:
            parameter_rows.append({
                "asset": asset, "penalty": penalty, "selected_lambda": np.nan,
                "intercept": np.nan, "n_train": int(len(train)),
                "n_validation": int(len(validation)), "validation_qlike": np.nan,
                "status": "insufficient_observations",
                **{name: np.nan for name in feature_columns},
            })
            continue

        train_x = train[feature_columns].to_numpy(dtype=float)
        train_actual = train[actual_column].to_numpy(dtype=float)
        if target_mode == "log_variance":
            train_y = np.log(np.maximum(np.square(train_actual), log_floor))
        else:
            train_y = np.square(train_actual)
        feature_mean = np.nanmean(train_x, axis=0)
        feature_std = np.nanstd(train_x, axis=0)
        feature_std[~np.isfinite(feature_std)] = 1.0
        feature_std[feature_std <= 1e-12] = 1.0

        def scale(values: pd.DataFrame | np.ndarray) -> np.ndarray:
            return (np.asarray(values, dtype=float) - feature_mean) / feature_std

        train_x_scaled = scale(train_x)
        validation_x = validation[feature_columns].to_numpy(dtype=float)
        validation_x_scaled = scale(validation_x)
        validation_actual_vol = validation[actual_column].to_numpy(dtype=float)

        best_lambda: float | None = None
        best_qlike = np.inf
        for candidate in values:
            if penalty == "ridge":
                beta, intercept = _fit_ridge_ls(train_x_scaled, train_y, l2=candidate)
            else:
                beta, intercept = _fit_lasso_cd(
                    train_x_scaled, train_y, l1=candidate, max_iter=lasso_max_iter
                )
            smearing = 0.0
            if target_mode == "log_variance":
                fitted_train = intercept + train_x_scaled @ beta
                smearing = 0.5 * float(np.mean(np.square(train_y - fitted_train)))
            raw_validation = intercept + validation_x_scaled @ beta + smearing
            if target_mode == "log_variance":
                raw_validation = np.exp(raw_validation)
            forecast_vol = np.sqrt(np.maximum(raw_validation, variance_floor))
            qlike = evaluate_forecast(
                validation_actual_vol, forecast_vol, epsilon=epsilon
            ).qlike
            selection_rows.append({
                "asset": asset, "penalty": penalty, "lambda": candidate,
                "validation_qlike": qlike, "n_validation": int(len(validation)),
                "selected": False,
            })
            if np.isfinite(qlike) and (best_lambda is None or qlike < best_qlike):
                best_qlike = float(qlike)
                best_lambda = candidate
        if best_lambda is None:
            parameter_rows.append({
                "asset": asset, "penalty": penalty, "selected_lambda": np.nan,
                "intercept": np.nan, "n_train": int(len(train)),
                "n_validation": int(len(validation)), "validation_qlike": np.nan,
                "status": "no_valid_candidate",
                **{name: np.nan for name in feature_columns},
            })
            continue
        # mark exactly one lambda as selected (ties already resolved to first minimum)
        for row in selection_rows:
            if row["asset"] == asset and row["penalty"] == penalty and row["lambda"] == best_lambda:
                row["selected"] = True

        if penalty == "ridge":
            beta, intercept = _fit_ridge_ls(train_x_scaled, train_y, l2=best_lambda)
        else:
            beta, intercept = _fit_lasso_cd(
                train_x_scaled, train_y, l1=best_lambda, max_iter=lasso_max_iter
            )
        smearing_offset = 0.0
        if target_mode == "log_variance":
            fitted_train = intercept + train_x_scaled @ beta
            smearing_offset = 0.5 * float(np.mean(np.square(train_y - fitted_train)))
        valid = group[feature_columns].notna().all(axis=1)
        raw_all = intercept + scale(group.loc[valid, feature_columns].to_numpy(dtype=float)) @ beta
        if target_mode == "log_variance":
            raw_all = np.exp(raw_all + smearing_offset)
        variance_clipped = np.maximum(raw_all, variance_floor)
        data.loc[group.index[valid], output_column] = np.sqrt(variance_clipped)
        parameter_rows.append({
            "asset": asset, "penalty": penalty, "selected_lambda": best_lambda,
            "intercept": intercept, "smearing_offset": smearing_offset,
            "n_train": int(len(train)),
            "n_validation": int(len(validation)), "validation_qlike": best_qlike,
            "status": "ok",
            **{name: float(coef) for name, coef in zip(feature_columns, beta)},
        })

    selection = pd.DataFrame(selection_rows)
    params = pd.DataFrame(parameter_rows)
    column_order = [
        "asset", "penalty", "selected_lambda", "intercept", "smearing_offset",
        "n_train", "n_validation", "validation_qlike", "status", *feature_columns,
    ]
    return data, params[[column for column in column_order if column in params.columns]], selection


def fit_lightgbm_by_asset(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str] | tuple[str, ...],
    actual_column: str = "future_rv_5d",
    output_column: str = "lgb_rv",
    num_leaves_grid: tuple[int, ...] = (8, 31),
    learning_rate_grid: tuple[float, ...] = (0.05, 0.1),
    n_estimators_grid: tuple[int, ...] = (100, 300),
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    n_jobs: int = 1,
    train_segment: str = "train",
    validation_segment: str = "validation",
    log_floor: float = 1.0e-12,
    variance_floor: float = 1.0e-4,
    epsilon: float = 1.0e-12,
    min_train_observations: int = 120,
    min_validation_observations: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit a LightGBM regressor per asset on log variance with locked hyperparameters.

    The model regresses ``log(future_rv_5d^2)`` on the given features using only
    train rows (all features and the target finite, the same row filter as
    :func:`fit_ridge_by_asset`). The Cartesian grid over ``num_leaves``,
    ``learning_rate``, and ``n_estimators`` is scored per asset on the validation
    segment by pooled QLIKE (ties resolve to the first configuration), the best
    configuration is refit on the train segment, and forecasts are recovered with
    an exponential plus the lognormal smearing offset (``0.5 * residual variance``)
    so they are always strictly positive and never need variance clipping.

    Returns ``(data, params, selection, importance)``:
    - ``params``: per-asset selected hyperparameters, validation QLIKE, status.
    - ``selection``: per-asset per-configuration validation QLIKE and selected flag.
    - ``importance``: per-asset gain and split feature importance from the refit
      model, with the within-asset gain share.
    Assets whose fit fails or that lack training or validation rows are reported in
    ``params`` with a NaN forecast column, mirroring the GARCH and Ridge pattern.
    """
    if log_floor <= 0:
        raise ValueError("log_floor must be positive")
    if variance_floor <= 0:
        raise ValueError("variance_floor must be positive")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if min_train_observations <= 0 or min_validation_observations <= 0:
        raise ValueError("minimum observation counts must be positive")
    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("feature_columns must be non-empty")
    num_leaves_values = [int(value) for value in num_leaves_grid]
    learning_rate_values = [float(value) for value in learning_rate_grid]
    estimator_values = [int(value) for value in n_estimators_grid]
    if not (num_leaves_values and learning_rate_values and estimator_values):
        raise ValueError("hyperparameter grids must be non-empty")

    data = frame.sort_values(["asset", "date"], kind="stable").copy()
    data[output_column] = np.nan
    parameter_rows: list[dict[str, float | int | str]] = []
    selection_rows: list[dict[str, float | int | str | bool]] = []
    importance_rows: list[dict[str, float | int | str]] = []

    for asset, group in data.groupby("asset", sort=True):
        train = group.loc[
            (group["segment"] == train_segment)
            & group[[actual_column, *feature_columns]].notna().all(axis=1)
        ]
        validation = group.loc[
            (group["segment"] == validation_segment)
            & group[[actual_column, *feature_columns]].notna().all(axis=1)
        ]
        if len(train) < min_train_observations or len(validation) < min_validation_observations:
            parameter_rows.append({
                "asset": asset, "selected_num_leaves": np.nan,
                "selected_learning_rate": np.nan, "selected_n_estimators": np.nan,
                "n_train": int(len(train)), "n_validation": int(len(validation)),
                "validation_qlike": np.nan, "smearing_offset": np.nan,
                "status": "insufficient_observations",
            })
            continue

        train_x = train[feature_columns].to_numpy(dtype=float)
        train_y = np.log(np.maximum(np.square(train[actual_column].to_numpy(dtype=float)), log_floor))
        validation_x = validation[feature_columns].to_numpy(dtype=float)
        validation_actual_vol = validation[actual_column].to_numpy(dtype=float)

        best_config: tuple[int, float, int] | None = None
        best_qlike = np.inf
        for num_leaves, learning_rate, estimators in product(
            num_leaves_values, learning_rate_values, estimator_values
        ):
            try:
                model = _fit_lgb_regressor(
                    train_x,
                    train_y,
                    num_leaves=num_leaves,
                    learning_rate=learning_rate,
                    n_estimators=estimators,
                    min_child_samples=min_child_samples,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    reg_lambda=reg_lambda,
                    random_state=random_state,
                    n_jobs=n_jobs,
                )
                fitted_train = model.predict(train_x)
                smearing = 0.5 * float(np.mean(np.square(train_y - fitted_train)))
                raw_validation = np.exp(model.predict(validation_x) + smearing)
            except Exception as exc:  # pragma: no cover - native training can fail
                selection_rows.append({
                    "asset": asset, "num_leaves": num_leaves,
                    "learning_rate": learning_rate, "n_estimators": estimators,
                    "validation_qlike": np.nan, "n_validation": int(len(validation)),
                    "status": f"fit_failed: {str(exc)[:120]}", "selected": False,
                })
                continue
            forecast_vol = np.sqrt(np.maximum(raw_validation, variance_floor))
            qlike = evaluate_forecast(validation_actual_vol, forecast_vol, epsilon=epsilon).qlike
            selection_rows.append({
                "asset": asset, "num_leaves": num_leaves,
                "learning_rate": learning_rate, "n_estimators": estimators,
                "validation_qlike": qlike, "n_validation": int(len(validation)),
                "status": "ok", "selected": False,
            })
            if np.isfinite(qlike) and (best_config is None or qlike < best_qlike):
                best_qlike = float(qlike)
                best_config = (num_leaves, learning_rate, estimators)
        if best_config is None:
            parameter_rows.append({
                "asset": asset, "selected_num_leaves": np.nan,
                "selected_learning_rate": np.nan, "selected_n_estimators": np.nan,
                "n_train": int(len(train)), "n_validation": int(len(validation)),
                "validation_qlike": np.nan, "smearing_offset": np.nan,
                "status": "no_valid_candidate",
            })
            continue
        for row in selection_rows:
            if (
                row["asset"] == asset
                and row["num_leaves"] == best_config[0]
                and row["learning_rate"] == best_config[1]
                and row["n_estimators"] == best_config[2]
            ):
                row["selected"] = True

        num_leaves, learning_rate, estimators = best_config
        model = _fit_lgb_regressor(
            train_x,
            train_y,
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=estimators,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        fitted_train = model.predict(train_x)
        smearing_offset = 0.5 * float(np.mean(np.square(train_y - fitted_train)))
        valid = group[feature_columns].notna().all(axis=1)
        raw_all = np.exp(model.predict(group.loc[valid, feature_columns].to_numpy(dtype=float)) + smearing_offset)
        data.loc[group.index[valid], output_column] = np.sqrt(np.maximum(raw_all, variance_floor))

        gains = np.asarray(model.booster_.feature_importance(importance_type="gain"), dtype=float)
        splits = np.asarray(model.booster_.feature_importance(importance_type="split"), dtype=int)
        total_gain = float(gains.sum())
        for index, name in enumerate(feature_columns):
            share = float(gains[index] / total_gain) if total_gain > 0 else np.nan
            importance_rows.append({
                "asset": asset, "feature": name,
                "importance_gain": float(gains[index]),
                "importance_split": int(splits[index]),
                "importance_gain_share": share,
            })
        parameter_rows.append({
            "asset": asset,
            "selected_num_leaves": int(num_leaves),
            "selected_learning_rate": float(learning_rate),
            "selected_n_estimators": int(estimators),
            "n_train": int(len(train)),
            "n_validation": int(len(validation)),
            "validation_qlike": best_qlike,
            "smearing_offset": smearing_offset,
            "status": "ok",
        })

    selection = pd.DataFrame(selection_rows)
    params = pd.DataFrame(parameter_rows)
    importance = pd.DataFrame(importance_rows)
    params_order = [
        "asset", "selected_num_leaves", "selected_learning_rate",
        "selected_n_estimators", "n_train", "n_validation",
        "validation_qlike", "smearing_offset", "status",
    ]
    return data, params[params_order], selection, importance


def _fit_lgb_regressor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    num_leaves: int,
    learning_rate: float,
    n_estimators: int,
    min_child_samples: int,
    subsample: float,
    colsample_bytree: float,
    reg_lambda: float,
    random_state: int,
    n_jobs: int,
):
    """Fit a deterministic LightGBM regressor on ``log(future_rv_5d^2)``."""
    from lightgbm import LGBMRegressor

    model = LGBMRegressor(
        objective="regression",
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=n_jobs,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )
    model.fit(train_x, train_y)
    return model
