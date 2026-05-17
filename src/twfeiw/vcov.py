"""Variance-covariance estimators for fitted OLS designs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from twfeiw.design import DesignMatrix
from twfeiw.ols import OLSResult

"""
For now, we specify 3 types of vcov specifications:
homoscedastic
heteroscedastic without covariances
clustered
"""
# Stores the covariance matrix, standard errors, and metadata for one vcov method.
@dataclass(frozen=True)
class VCovResult:
    """Variance-covariance output for a fitted regression design."""

    vcov: pd.DataFrame
    standard_errors: pd.Series
    method: str
    nobs: int
    rank: int
    df_resid: int
    small_sample: bool
    n_clusters: int | None = None
    cluster_name: str | None = None


# Dispatches to the requested variance-covariance estimator.
def compute_vcov(
    design: DesignMatrix,
    result: OLSResult,
    *,
    method: str = "classical",
    clusters: pd.Series | np.ndarray | list[object] | None = None,
    small_sample: bool = True,
) -> VCovResult:
    """Compute a variance-covariance matrix for a fitted OLS result."""

    if not isinstance(method, str):
        raise TypeError("method must be a string")

    normalized_method = method.casefold()
    if normalized_method == "classical":
        return compute_classical_vcov(design, result)
    if normalized_method == "hc1":
        return compute_heteroskedastic_vcov(design, result, kind="HC1")
    if normalized_method == "cluster":
        if clusters is None:
            raise ValueError("clusters are required when method is 'cluster'")
        return compute_clustered_vcov(
            design,
            result,
            clusters=clusters,
            small_sample=small_sample,
        )

    raise ValueError("unsupported vcov method")


# Computes the homoskedastic OLS covariance matrix sigma^2 * (X'X)^-1.
def compute_classical_vcov(
    design: DesignMatrix,
    result: OLSResult,
) -> VCovResult:
    """Compute classical homoskedastic OLS standard errors."""

    X, _, bread_inv = _validate_and_prepare_inputs(design, result)

    sigma2 = result.rss / result.df_resid
    vcov = sigma2 * bread_inv

    return _build_vcov_result(
        vcov,
        columns=design.regressor_cols,
        method="classical",
        result=result,
        small_sample=True,
    )


# Computes the HC1 heteroskedasticity-robust sandwich covariance matrix.
# HC1 refers to normalizing the covariance matrix estimate with the degrees of freedom
def compute_heteroskedastic_vcov(
    design: DesignMatrix,
    result: OLSResult,
    *,
    kind: str = "HC1",
) -> VCovResult:
    """Compute a heteroskedasticity-consistent variance-covariance matrix."""

    if kind != "HC1":
        raise ValueError("unsupported heteroskedastic vcov kind")

    X, residuals, bread_inv = _validate_and_prepare_inputs(design, result)

    # below gives X'Omeaga^X
    weighted_X = X * (residuals**2)[:, None]
    meat = X.T @ weighted_X

    # this gives [(X'X)^-1 @ X'(Omeaga^)X @ (X'X)^-1] *n/(n-k)
    vcov = bread_inv @ meat @ bread_inv
    vcov *= result.nobs / result.df_resid

    return _build_vcov_result(
        vcov,
        columns=design.regressor_cols,
        method="HC1",
        result=result,
        small_sample=True,
    )


# Computes one-way cluster-robust covariance from cluster-level score sums.
def compute_clustered_vcov(
    design: DesignMatrix,
    result: OLSResult,
    *,
    clusters: pd.Series | np.ndarray | list[object],
    small_sample: bool = True,
) -> VCovResult:
    """Compute one-way cluster-robust standard errors."""

    X, residuals, bread_inv = _validate_and_prepare_inputs(design, result)
    cluster_values, cluster_name = _validate_clusters(clusters, result.nobs)

    meat = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    for cluster in cluster_values.unique():
        mask = (cluster_values == cluster).to_numpy()
        score = X[mask].T @ residuals[mask]
        meat += np.outer(score, score)

    vcov = bread_inv @ meat @ bread_inv

    n_clusters = int(cluster_values.nunique())
    if small_sample:
        correction = (n_clusters / (n_clusters - 1)) * (
            (result.nobs - 1) / result.df_resid
        )
        vcov *= correction

    return _build_vcov_result(
        vcov,
        columns=design.regressor_cols,
        method="cluster",
        result=result,
        small_sample=small_sample,
        n_clusters=n_clusters,
        cluster_name=cluster_name,
    )


# Checks matrix/result consistency and returns X, residuals, and (X'X)^-1.
def _validate_and_prepare_inputs(
    design: DesignMatrix,
    result: OLSResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(design, DesignMatrix):
        raise TypeError("design must be a DesignMatrix")
    if not isinstance(result, OLSResult):
        raise TypeError("result must be an OLSResult")

    X = design.regressors.to_numpy(dtype=float)
    residuals = result.residuals.to_numpy(dtype=float)

    if X.shape[0] != result.nobs:
        raise ValueError("regressors and OLS result must have the same nobs")
    if len(residuals) != result.nobs:
        raise ValueError("residuals and OLS result must have the same nobs")
    if X.shape[1] != len(result.regressor_cols):
        raise ValueError("regressor column metadata does not match regressors")
    if result.regressor_cols != design.regressor_cols:
        raise ValueError("result regressor columns must match design regressor columns")
    if result.rank != X.shape[1]:
        raise ValueError("regressor matrix must be full rank for vcov estimation")
    if result.df_resid <= 0:
        raise ValueError("residual degrees of freedom must be positive")

    try:
        bread_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError as exc:
        raise ValueError("regressor matrix is not invertible") from exc

    return X, residuals, bread_inv # bread_inv is (X'X)^-1


# Validates one cluster label per observation and preserves the cluster name.
def _validate_clusters(
    clusters: pd.Series | np.ndarray | list[object],
    nobs: int,
) -> tuple[pd.Series, str | None]:
    cluster_name = clusters.name if isinstance(clusters, pd.Series) else None
    cluster_values = pd.Series(clusters).reset_index(drop=True)

    if len(cluster_values) != nobs:
        raise ValueError("clusters must have the same length as the regression data")
    if cluster_values.isna().any():
        raise ValueError("clusters cannot contain missing values")

    n_clusters = int(cluster_values.nunique())
    if n_clusters < 2:
        raise ValueError("clustered vcov requires at least two clusters")

    return cluster_values, cluster_name


# Converts a raw NumPy covariance matrix into labeled pandas outputs.
def _build_vcov_result(
    vcov: np.ndarray,
    *,
    columns: list[str],
    method: str,
    result: OLSResult,
    small_sample: bool,
    n_clusters: int | None = None,
    cluster_name: str | None = None,
) -> VCovResult:
    vcov = vcov.copy()
    diag = np.diag(vcov).copy()
    negative = diag < -1e-12
    if negative.any():
        raise ValueError("variance-covariance matrix has negative diagonal entries")

    diag = np.clip(diag, 0.0, None)  # converts < 0 elements to 0.
    np.fill_diagonal(vcov, diag)
    vcov_df = pd.DataFrame(vcov, index=columns, columns=columns)
    standard_errors = pd.Series(
        np.sqrt(diag),
        index=columns,
        name="std_err",
    )

    return VCovResult(
        vcov=vcov_df,
        standard_errors=standard_errors,
        method=method,
        nobs=result.nobs,
        rank=result.rank,
        df_resid=result.df_resid,
        small_sample=small_sample,
        n_clusters=n_clusters,
        cluster_name=cluster_name,
    )
