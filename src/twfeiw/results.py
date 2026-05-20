"""Regression result assembly and coefficient-level inference."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from scipy import stats

from twfeiw.design import DesignMatrix, SunAbrahamDesign
from twfeiw.ols import OLSResult
from twfeiw.vcov import VCovResult


@dataclass(frozen=True)
class RegressionResult:
    """User-facing regression output assembled from OLS and vcov results."""

    params: pd.Series
    vcov: pd.DataFrame
    standard_errors: pd.Series
    t_stats: pd.Series
    p_values: pd.Series
    conf_int: pd.DataFrame
    summary: pd.DataFrame
    rss: float
    tss: float
    r_squared: float
    adj_r_squared: float
    model: str
    outcome_variable: str
    effect_cols: list[str]
    nobs: int
    rank: int
    df_resid: int
    inference_df: int
    vcov_method: str
    small_sample: bool
    n_clusters: int | None
    cluster_name: str | None
    alpha: float


@dataclass(frozen=True)
class SunAbrahamAggregation:
    """Sun-Abraham cohort-event estimates and interaction-weighted averages."""

    cohort_event_table: pd.DataFrame
    event_table: pd.DataFrame
    weights: pd.DataFrame


def build_regression_result(
    design: DesignMatrix,
    ols_result: OLSResult,
    vcov_result: VCovResult,
    *,
    alpha: float = 0.05,
) -> RegressionResult:
    """Combine point estimates and standard errors into inference output."""

    _validate_inputs(design, ols_result, vcov_result)
    alpha = _validate_alpha(alpha)
    inference_df = _inference_degrees_of_freedom(ols_result, vcov_result)

    params = ols_result.params.copy()
    vcov = vcov_result.vcov.copy()
    standard_errors = vcov_result.standard_errors.copy()

    t_stats = (params / standard_errors).rename("t")
    p_values = pd.Series(
        2.0 * stats.t.sf(np.abs(t_stats.to_numpy(dtype=float)), df=inference_df),
        index=params.index,
        name="p_value",
    )

    critical = stats.t.ppf(1.0 - alpha / 2.0, df=inference_df)
    ci_lower = (params - critical * standard_errors).rename("ci_lower")
    ci_upper = (params + critical * standard_errors).rename("ci_upper")
    conf_int = pd.DataFrame(
        {
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        },
        index=params.index,
    )

    summary = pd.DataFrame(
        {
            "coef": params,
            "std_err": standard_errors,
            "t": t_stats,
            "p_value": p_values,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        },
        index=params.index,
    )

    return RegressionResult(
        params=params,
        vcov=vcov,
        standard_errors=standard_errors,
        t_stats=t_stats,
        p_values=p_values,
        conf_int=conf_int,
        summary=summary,
        rss=ols_result.rss,
        tss=ols_result.tss,
        r_squared=ols_result.r_squared,
        adj_r_squared=ols_result.adj_r_squared,
        model=design.model,
        outcome_variable=design.outcome_variable,
        effect_cols=list(design.effect_cols),
        nobs=ols_result.nobs,
        rank=ols_result.rank,
        df_resid=ols_result.df_resid,
        inference_df=inference_df,
        vcov_method=vcov_result.method,
        small_sample=vcov_result.small_sample,
        n_clusters=vcov_result.n_clusters,
        cluster_name=vcov_result.cluster_name,
        alpha=alpha,
    )


def build_sun_abraham_aggregation(
    sun_design: SunAbrahamDesign,
    regression: RegressionResult,
) -> SunAbrahamAggregation:
    """Aggregate Sun-Abraham cohort-event coefficients by event time."""

    _validate_sun_abraham_inputs(sun_design, regression)

    cohort_event_table = _build_cohort_event_table(sun_design, regression)
    event_table = _build_interaction_weighted_event_table(
        cohort_event_table,
        regression,
    )
    weights = cohort_event_table[["cell_count", "weight"]].copy()

    return SunAbrahamAggregation(
        cohort_event_table=cohort_event_table,
        event_table=event_table,
        weights=weights,
    )


def _validate_inputs(
    design: DesignMatrix,
    ols_result: OLSResult,
    vcov_result: VCovResult,
) -> None:
    """Check that design, OLS, and vcov results are type-safe and aligned."""

    if not isinstance(design, DesignMatrix):
        raise TypeError("design must be a DesignMatrix")
    if not isinstance(ols_result, OLSResult):
        raise TypeError("ols_result must be an OLSResult")
    if not isinstance(vcov_result, VCovResult):
        raise TypeError("vcov_result must be a VCovResult")

    expected = list(design.regressor_cols)
    if ols_result.regressor_cols != expected:
        raise ValueError("regressor names must match across results")
    if ols_result.params.index.tolist() != expected:
        raise ValueError("regressor names must match across results")
    if vcov_result.vcov.index.tolist() != expected:
        raise ValueError("regressor names must match across results")
    if vcov_result.vcov.columns.tolist() != expected:
        raise ValueError("regressor names must match across results")
    if vcov_result.standard_errors.index.tolist() != expected:
        raise ValueError("regressor names must match across results")

    if ols_result.nobs != vcov_result.nobs:
        raise ValueError("OLS and vcov nobs metadata must match")
    if ols_result.rank != vcov_result.rank:
        raise ValueError("OLS and vcov rank metadata must match")
    if ols_result.df_resid != vcov_result.df_resid:
        raise ValueError("OLS and vcov df_resid metadata must match")


def _validate_sun_abraham_inputs(
    sun_design: SunAbrahamDesign,
    regression: RegressionResult,
) -> None:
    """Check Sun-Abraham design metadata and regression output alignment."""

    if not isinstance(sun_design, SunAbrahamDesign):
        raise TypeError("sun_design must be a SunAbrahamDesign")
    if not isinstance(regression, RegressionResult):
        raise TypeError("regression must be a RegressionResult")
    if sun_design.design.model != "sun_abraham" or regression.model != "sun_abraham":
        raise ValueError("Sun-Abraham aggregation requires a sun_abraham model")
    if regression.effect_cols != sun_design.design.effect_cols:
        raise ValueError("Sun-Abraham effect columns must match regression output")
    if set(sun_design.cohort_event_by_column) != set(regression.effect_cols):
        raise ValueError("cohort-event metadata must match effect columns")


def _build_cohort_event_table(
    sun_design: SunAbrahamDesign,
    regression: RegressionResult,
) -> pd.DataFrame:
    """Build inference output indexed by cohort and event time."""

    rows: list[dict[str, object]] = []
    index: list[tuple[int, int]] = []

    for column in regression.effect_cols:
        cohort, event_time = sun_design.cohort_event_by_column[column]
        summary_row = regression.summary.loc[column]
        cell_count = int(sun_design.cell_counts.loc[(cohort, event_time)])

        rows.append(
            {
                "coef": float(summary_row["coef"]),
                "std_err": float(summary_row["std_err"]),
                "t": float(summary_row["t"]),
                "p_value": float(summary_row["p_value"]),
                "ci_lower": float(summary_row["ci_lower"]),
                "ci_upper": float(summary_row["ci_upper"]),
                "column": column,
                "cell_count": cell_count,
            }
        )
        index.append((cohort, event_time))

    table = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["cohort", "event_time"]),
    )
    event_totals = table.groupby(level="event_time")["cell_count"].transform("sum")
    table["weight"] = table["cell_count"] / event_totals
    return table[
        [
            "coef",
            "std_err",
            "t",
            "p_value",
            "ci_lower",
            "ci_upper",
            "column",
            "cell_count",
            "weight",
        ]
    ]


def _build_interaction_weighted_event_table(
    cohort_event_table: pd.DataFrame,
    regression: RegressionResult,
) -> pd.DataFrame:
    """Build one interaction-weighted estimate for each event time."""

    rows: list[dict[str, float | int]] = []
    index: list[int] = []

    critical = stats.t.ppf(1.0 - regression.alpha / 2.0, df=regression.inference_df)

    for event_time, event_rows in cohort_event_table.groupby(
        level="event_time",
        sort=True,
    ):
        columns = event_rows["column"].tolist()
        weights = event_rows["weight"].to_numpy(dtype=float)
        params = regression.params.loc[columns].to_numpy(dtype=float)
        vcov = regression.vcov.loc[columns, columns].to_numpy(dtype=float)

        coef = float(weights @ params)
        variance = float(weights @ vcov @ weights)
        std_err = float(np.sqrt(max(variance, 0.0)))
        if std_err == 0.0:
            t_stat = float(np.sign(coef) * np.inf) if coef != 0.0 else np.nan
        else:
            t_stat = coef / std_err
        p_value = float(2.0 * stats.t.sf(np.abs(t_stat), df=regression.inference_df))
        ci_lower = float(coef - critical * std_err)
        ci_upper = float(coef + critical * std_err)

        rows.append(
            {
                "coef": coef,
                "std_err": std_err,
                "t": t_stat,
                "p_value": p_value,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_cohorts": int(len(event_rows)),
                "cell_count": int(event_rows["cell_count"].sum()),
            }
        )
        index.append(int(event_time))

    table = pd.DataFrame(rows, index=pd.Index(index, name="event_time"))
    return table[
        [
            "coef",
            "std_err",
            "t",
            "p_value",
            "ci_lower",
            "ci_upper",
            "n_cohorts",
            "cell_count",
        ]
    ]


def _validate_alpha(alpha: float) -> float:
    """Normalize alpha after confirming it is strictly between zero and one."""

    if not isinstance(alpha, Real):
        raise ValueError("alpha must be strictly between 0 and 1")

    alpha_float = float(alpha)
    if not np.isfinite(alpha_float) or not 0.0 < alpha_float < 1.0:
        raise ValueError("alpha must be strictly between 0 and 1")

    return alpha_float


def _inference_degrees_of_freedom(
    ols_result: OLSResult,
    vcov_result: VCovResult,
) -> int:
    """Select the t-distribution degrees of freedom for inference."""

    method = vcov_result.method.casefold()
    if method == "cluster":
        if vcov_result.n_clusters is None:
            raise ValueError("clustered inference requires n_clusters")
        inference_df = int(vcov_result.n_clusters - 1)
    elif method in {"classical", "hc1"}:
        inference_df = int(ols_result.df_resid)
    else:
        raise ValueError("unsupported vcov method")

    if inference_df <= 0:
        raise ValueError("inference degrees of freedom must be positive")

    return inference_df
