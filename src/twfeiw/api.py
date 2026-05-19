"""Public API for user-facing TWFE estimators."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from twfeiw.data import PreparedPanel, prepare_panel
from twfeiw.design import (
    DesignMatrix,
    EventStudyDesign,
    build_twfe_design,
    build_twfe_event_study_design,
)
from twfeiw.ols import OLSResult, fit_ols
from twfeiw.results import RegressionResult, build_regression_result
from twfeiw.vcov import VCovResult, compute_vcov


@dataclass(frozen=True)
class TWFEResult:
    """User-facing result from a standard two-way fixed effects regression.

    Attributes
    ----------
    regression
        Comprehensive regression result with coefficients, standard errors,
        p-values, confidence intervals, fit statistics, and summary output.
    panel
        Prepared panel data used for estimation.
    design
        Regression design matrix used for OLS.
    ols
        Point-estimation output.
    vcov
        Variance-covariance output.
    """

    regression: RegressionResult
    panel: PreparedPanel
    design: DesignMatrix
    ols: OLSResult
    vcov: VCovResult

    @property
    def params(self) -> pd.Series:
        """Estimated regression coefficients."""

        return self.regression.params

    @property
    def vcov_matrix(self) -> pd.DataFrame:
        """Estimated variance-covariance matrix."""

        return self.regression.vcov

    @property
    def standard_errors(self) -> pd.Series:
        """Coefficient standard errors."""

        return self.regression.standard_errors

    @property
    def t_stats(self) -> pd.Series:
        """Coefficient t-statistics."""

        return self.regression.t_stats

    @property
    def p_values(self) -> pd.Series:
        """Coefficient p-values."""

        return self.regression.p_values

    @property
    def conf_int(self) -> pd.DataFrame:
        """Coefficient confidence intervals."""

        return self.regression.conf_int

    @property
    def r_squared(self) -> float:
        """Regression R-squared."""

        return self.regression.r_squared

    @property
    def adj_r_squared(self) -> float:
        """Regression adjusted R-squared."""

        return self.regression.adj_r_squared

    @property
    def nobs(self) -> int:
        """Number of observations used in the regression."""

        return self.regression.nobs

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom from the OLS fit."""

        return self.regression.df_resid

    @property
    def inference_df(self) -> int:
        """Degrees of freedom used for p-values and confidence intervals."""

        return self.regression.inference_df

    @property
    def vcov_method(self) -> str:
        """Variance-covariance method used for inference."""

        return self.regression.vcov_method

    @property
    def effect_name(self) -> str:
        """Name of the single TWFE treatment effect column."""

        return self._single_effect_name()

    @property
    def effect(self) -> float:
        """Estimated coefficient on the TWFE treatment variable."""

        return float(self.params[self.effect_name])

    @property
    def effect_se(self) -> float:
        """Standard error for the TWFE treatment effect."""

        return float(self.standard_errors[self.effect_name])

    @property
    def effect_p_value(self) -> float:
        """P-value for the TWFE treatment effect."""

        return float(self.p_values[self.effect_name])

    @property
    def effect_conf_int(self) -> pd.Series:
        """Confidence interval row for the TWFE treatment effect."""

        return self.conf_int.loc[self.effect_name].copy()

    def summary(self) -> pd.DataFrame:
        """Return a copy of the coefficient summary table."""

        return self.regression.summary.copy()

    def _single_effect_name(self) -> str:
        if len(self.regression.effect_cols) != 1:
            raise ValueError("effect accessors require exactly one effect column")
        return self.regression.effect_cols[0]


@dataclass(frozen=True)
class EventStudyResult:
    """User-facing result from a conventional TWFE event-study regression.

    Attributes
    ----------
    regression
        Comprehensive regression result with event-time coefficients, standard
        errors, p-values, confidence intervals, fit statistics, and summary
        output.
    panel
        Prepared panel data used for estimation.
    event_design
        Event-study design matrix and event-time metadata.
    ols
        Point-estimation output.
    vcov
        Variance-covariance output.
    """

    regression: RegressionResult
    panel: PreparedPanel
    event_design: EventStudyDesign
    ols: OLSResult
    vcov: VCovResult

    @property
    def design(self) -> DesignMatrix:
        """Regression design matrix used for OLS."""

        return self.event_design.design

    @property
    def params(self) -> pd.Series:
        """Estimated regression coefficients."""

        return self.regression.params

    @property
    def vcov_matrix(self) -> pd.DataFrame:
        """Estimated variance-covariance matrix."""

        return self.regression.vcov

    @property
    def standard_errors(self) -> pd.Series:
        """Coefficient standard errors."""

        return self.regression.standard_errors

    @property
    def t_stats(self) -> pd.Series:
        """Coefficient t-statistics."""

        return self.regression.t_stats

    @property
    def p_values(self) -> pd.Series:
        """Coefficient p-values."""

        return self.regression.p_values

    @property
    def conf_int(self) -> pd.DataFrame:
        """Coefficient confidence intervals."""

        return self.regression.conf_int

    @property
    def r_squared(self) -> float:
        """Regression R-squared."""

        return self.regression.r_squared

    @property
    def adj_r_squared(self) -> float:
        """Regression adjusted R-squared."""

        return self.regression.adj_r_squared

    @property
    def nobs(self) -> int:
        """Number of observations used in the regression."""

        return self.regression.nobs

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom from the OLS fit."""

        return self.regression.df_resid

    @property
    def inference_df(self) -> int:
        """Degrees of freedom used for p-values and confidence intervals."""

        return self.regression.inference_df

    @property
    def vcov_method(self) -> str:
        """Variance-covariance method used for inference."""

        return self.regression.vcov_method

    @property
    def reference_event_time(self) -> int:
        """Omitted event time used as the event-study reference period."""

        return self.event_design.reference_event_time

    @property
    def event_times(self) -> list[int]:
        """Event times included as regression coefficients."""

        return self.event_design.included_event_times

    @property
    def event_columns(self) -> list[str]:
        """Regression column names for event-time coefficients."""

        return list(self.event_design.event_time_by_column)

    def summary(self) -> pd.DataFrame:
        """Return a copy of the full coefficient summary table."""

        return self.regression.summary.copy()

    def event_table(self) -> pd.DataFrame:
        """Return event-time coefficients indexed by integer event time."""

        event_rows = self.regression.summary.loc[self.event_columns].copy()
        event_rows["event_time"] = [
            self.event_design.event_time_by_column[column]
            for column in event_rows.index
        ]
        event_rows = event_rows.set_index("event_time").sort_index()
        event_rows.index.name = "event_time"
        return event_rows[
            ["coef", "std_err", "t", "p_value", "ci_lower", "ci_upper"]
        ]


def twfe(
    data: pd.DataFrame,
    *,
    unit: str = "unit_id",
    time: str = "time",
    outcome: str = "y",
    treatment: str = "treatment",
    vcov: str = "classical",
    alpha: float = 0.05,
    small_sample: bool = True,
    rcond: float | None = None,
) -> TWFEResult:
    """Estimate a standard two-way fixed effects regression.

    Parameters
    ----------
    data
        Input panel data. The first version expects one row per observed
        unit-time pair, integer-like time periods, a numeric outcome, and a
        binary absorbing treatment column.
    unit
        Column identifying the panel unit. Defaults to ``"unit_id"``.
    time
        Integer-like column identifying the time period. Defaults to
        ``"time"``.
    outcome
        Numeric outcome column. Defaults to ``"y"``.
    treatment
        Binary absorbing treatment column. Defaults to ``"treatment"``.
    vcov
        Variance-covariance method. Supported values are ``"classical"``,
        ``"HC1"``, and ``"cluster"``. If ``"cluster"``, standard errors are
        clustered by the panel unit.
    alpha
        Significance level used for confidence intervals. Defaults to ``0.05``.
    small_sample
        Whether to apply the small-sample correction for clustered standard
        errors. Defaults to ``True``.
    rcond
        Cutoff passed to ``numpy.linalg.lstsq`` for determining rank.

    Returns
    -------
    TWFEResult
        Fitted TWFE result with coefficient estimates, standard errors,
        p-values, confidence intervals, fit statistics, and summary output.
    """

    if not isinstance(vcov, str):
        raise TypeError("vcov must be a string")

    panel = prepare_panel(
        data,
        unit=unit,
        time=time,
        outcome=outcome,
        treatment=treatment,
    )
    design = build_twfe_design(panel)
    ols_result = fit_ols(design, rcond=rcond)

    if vcov.casefold() == "cluster":
        vcov_result = compute_vcov(
            design,
            ols_result,
            method="cluster",
            clusters=panel.data[panel.unit_code_col],
            small_sample=small_sample,
        )
    else:
        vcov_result = compute_vcov(
            design,
            ols_result,
            method=vcov,
            small_sample=small_sample,
        )

    regression = build_regression_result(
        design,
        ols_result,
        vcov_result,
        alpha=alpha,
    )

    return TWFEResult(
        regression=regression,
        panel=panel,
        design=design,
        ols=ols_result,
        vcov=vcov_result,
    )


def event_study(
    data: pd.DataFrame,
    *,
    unit: str = "unit_id",
    time: str = "time",
    outcome: str = "y",
    treatment: str = "treatment",
    min_event_time: int | None = None,
    max_event_time: int | None = None,
    reference_event_time: int = -1,
    vcov: str = "classical",
    alpha: float = 0.05,
    small_sample: bool = True,
    rcond: float | None = None,
) -> EventStudyResult:
    """Estimate a conventional TWFE event-study regression.

    Parameters
    ----------
    data
        Input panel data. The first version expects one row per observed
        unit-time pair, integer-like time periods, a numeric outcome, and a
        binary absorbing treatment column.
    unit
        Column identifying the panel unit. Defaults to ``"unit_id"``.
    time
        Integer-like column identifying the time period. Defaults to
        ``"time"``.
    outcome
        Numeric outcome column. Defaults to ``"y"``.
    treatment
        Binary absorbing treatment column. Defaults to ``"treatment"``.
    min_event_time
        Smallest event time to include as a coefficient. If ``None``, no lower
        event-time bound is applied.
    max_event_time
        Largest event time to include as a coefficient. If ``None``, no upper
        event-time bound is applied.
    reference_event_time
        Event time omitted from the regression as the reference period.
        Defaults to ``-1``.
    vcov
        Variance-covariance method. Supported values are ``"classical"``,
        ``"HC1"``, and ``"cluster"``. If ``"cluster"``, standard errors are
        clustered by the panel unit.
    alpha
        Significance level used for confidence intervals. Defaults to ``0.05``.
    small_sample
        Whether to apply the small-sample correction for clustered standard
        errors. Defaults to ``True``.
    rcond
        Cutoff passed to ``numpy.linalg.lstsq`` for determining rank.

    Returns
    -------
    EventStudyResult
        Fitted TWFE event-study result with event-time coefficients, standard
        errors, p-values, confidence intervals, fit statistics, and event table
        output.

    Notes
    -----
    This is the conventional TWFE event-study regression with event-time
    indicators and unit/time fixed effects. Under staggered adoption and
    heterogeneous treatment effects, these coefficients can reflect problematic
    comparisons; Sun-Abraham interaction-weighted event studies are a separate
    planned estimator.
    """

    if not isinstance(vcov, str):
        raise TypeError("vcov must be a string")

    panel = prepare_panel(
        data,
        unit=unit,
        time=time,
        outcome=outcome,
        treatment=treatment,
    )
    event_design = build_twfe_event_study_design(
        panel,
        min_event_time=min_event_time,
        max_event_time=max_event_time,
        reference_event_time=reference_event_time,
    )
    design = event_design.design
    ols_result = fit_ols(design, rcond=rcond)

    if vcov.casefold() == "cluster":
        vcov_result = compute_vcov(
            design,
            ols_result,
            method="cluster",
            clusters=panel.data[panel.unit_code_col],
            small_sample=small_sample,
        )
    else:
        vcov_result = compute_vcov(
            design,
            ols_result,
            method=vcov,
            small_sample=small_sample,
        )

    regression = build_regression_result(
        design,
        ols_result,
        vcov_result,
        alpha=alpha,
    )

    return EventStudyResult(
        regression=regression,
        panel=panel,
        event_design=event_design,
        ols=ols_result,
        vcov=vcov_result,
    )
