"""Ordinary least squares estimation for named regression designs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from twfeiw.design import DesignMatrix


@dataclass(frozen=True)
class OLSResult:
    """Point-estimation output from an ordinary least squares regression."""

    params: pd.Series
    fitted_values: pd.Series
    residuals: pd.Series
    nobs: int
    rank: int
    df_resid: int
    rss: float
    tss: float
    r_squared: float
    adj_r_squared: float
    regressor_cols: list[str]

# * says that everything after * must be passed by keywords, not positions
def fit_ols(design: DesignMatrix, *, rcond: float | None = None) -> OLSResult:
    """Estimate OLS coefficients for a prepared regression design."""

    if not isinstance(design, DesignMatrix):
        raise TypeError("design must be a DesignMatrix")

    y = design.outcome.to_numpy(dtype=float)
    X = design.regressors.to_numpy(dtype=float)

    nobs, nregressors = X.shape
    if len(y) != nobs:
        raise ValueError("outcome and regressors must have the same number of rows")

    # solves the least squares problem and returnst theta and rank
    theta_hat, _, rank, _ = np.linalg.lstsq(X, y, rcond=rcond)
    rank = int(rank)

    if rank < nregressors:
        raise ValueError(
            "regressor matrix is rank deficient; one or more coefficients are "
            "not identified"
        )

    fitted = X @ theta_hat
    residuals = y - fitted
    rss = float(residuals @ residuals)
    y_centered = y - y.mean()
    tss = float(y_centered @ y_centered)
    df_resid = int(nobs - rank)

    if tss == 0.0:
        r_squared = np.nan
        adj_r_squared = np.nan
    else:
        r_squared = 1.0 - rss / tss
        if df_resid <= 0 or nobs <= 1:
            adj_r_squared = np.nan
        else:
            adj_r_squared = 1.0 - (rss / df_resid) / (tss / (nobs - 1))

    return OLSResult(
        params=pd.Series(theta_hat, index=design.regressor_cols, name="coef"),
        fitted_values=pd.Series(
            fitted,
            index=design.outcome.index,
            name="fitted_values",
        ),
        residuals=pd.Series(
            residuals,
            index=design.outcome.index,
            name="residuals",
        ),
        nobs=int(nobs),
        rank=rank,
        df_resid=df_resid,
        rss=rss,
        tss=tss,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        regressor_cols=list(design.regressor_cols),
    )
