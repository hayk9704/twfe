"""Regression design construction for twfeiw estimators."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from twfeiw.data import PreparedPanel


@dataclass(frozen=True)
class DesignMatrix:
    """Named regression inputs plus metadata for later estimation results."""

    outcome: pd.Series
    regressors: pd.DataFrame
    model: str
    outcome_variable: str
    effect_cols: list[str]
    fixed_effect_cols: list[str]
    regressor_cols: list[str]
    reference_unit: object
    reference_time: int


def build_twfe_design(panel: PreparedPanel) -> DesignMatrix:
    """Build an explicit dummy-variable design for the standard TWFE model."""

    if not isinstance(panel, PreparedPanel):
        raise TypeError("panel must be a PreparedPanel")

    data = panel.data.reset_index(drop=True)

    outcome = _build_outcome(data, panel.outcome)
    intercept = _build_intercept(len(data))
    treatment = _build_treatment(data, panel.treatment)

    unit_dummies, reference_unit = _build_unit_fe_dummies(data, panel)
    time_dummies, reference_time = _build_time_fe_dummies(data, panel.time)

    regressors = pd.concat(
        [intercept, treatment, unit_dummies, time_dummies],
        axis=1,
    )

    fixed_effect_cols = list(unit_dummies.columns) + list(time_dummies.columns)

    return DesignMatrix(
        outcome=outcome,
        regressors=regressors,
        model="twfe",
        outcome_variable=panel.outcome,
        effect_cols=[panel.treatment],
        fixed_effect_cols=fixed_effect_cols,
        regressor_cols=list(regressors.columns),
        reference_unit=reference_unit,
        reference_time=reference_time,
    )


def _build_outcome(data: pd.DataFrame, outcome: str) -> pd.Series:
    return data[outcome].astype(float).rename(outcome)


def _build_intercept(nobs: int) -> pd.DataFrame:
    return pd.DataFrame({"const": [1.0] * nobs})


def _build_treatment(data: pd.DataFrame, treatment: str) -> pd.DataFrame:
    return data[[treatment]].astype(float)


def _build_unit_fe_dummies(
    data: pd.DataFrame,
    panel: PreparedPanel,
) -> tuple[pd.DataFrame, object]:
    unit_codes = sorted(data[panel.unit_code_col].unique())
    reference_unit_code = unit_codes[0]
    reference_unit = data.loc[
        data[panel.unit_code_col] == reference_unit_code,
        panel.unit,
    ].iloc[0]

    dummies = _build_dummies(
        data[panel.unit_code_col],
        categories=unit_codes,
        prefix="unit_fe",
    )
    return dummies, reference_unit


def _build_time_fe_dummies(
    data: pd.DataFrame,
    time: str,
) -> tuple[pd.DataFrame, int]:
    time_periods = sorted(data[time].unique())
    reference_time = int(time_periods[0])

    dummies = _build_dummies(
        data[time],
        categories=time_periods,
        prefix="time_fe",
    )
    return dummies, reference_time


def _build_dummies(
    values: pd.Series,
    *,
    categories: list[object],
    prefix: str,
) -> pd.DataFrame:
    categorical = pd.Categorical(values, categories=categories, ordered=True)
    return pd.get_dummies(
        categorical,
        prefix=prefix,
        drop_first=True,
        dtype=float,
    )
