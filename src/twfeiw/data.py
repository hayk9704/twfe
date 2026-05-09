"""Data validation and preparation for twfeiw estimators."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype


UNIT_CODE_COL = "_twfeiw_unit_code"
FIRST_TREAT_COL = "_twfeiw_first_treat_time"
EVENT_TIME_COL = "_twfeiw_event_time"
EVER_TREATED_COL = "_twfeiw_ever_treated"


@dataclass(frozen=True)
class PreparedPanel:
    """Validated panel data plus internal columns needed by estimators."""

    data: pd.DataFrame
    unit: str
    time: str
    outcome: str
    treatment: str
    unit_code_col: str
    first_treat_col: str
    event_time_col: str
    ever_treated_col: str


def prepare_panel(
    data: pd.DataFrame,
    *,
    unit: str = "unit_id",
    time: str = "time",
    outcome: str = "y",
    treatment: str = "treatment",
) -> PreparedPanel:
    """Validate raw panel data and add treatment-timing variables.

    The first version expects a staggered-adoption panel with integer-like time
    periods, one row per unit-time pair, numeric outcomes, and binary absorbing
    treatment.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    required = [unit, time, outcome, treatment]
    _check_required_columns(data, required)
    _check_internal_columns_available(data)
    _check_no_missing_required(data, required)
    _check_numeric_outcome(data, outcome)
    _check_integer_time(data, time)
    _check_unique_unit_time(data, unit, time)
    _check_binary_treatment(data, treatment)

    prepared = data.copy()
    prepared[treatment] = prepared[treatment].astype(int)
    prepared = _add_unit_codes(prepared, unit)
    prepared = prepared.sort_values([UNIT_CODE_COL, time]).reset_index(drop=True)

    _check_absorbing_treatment(prepared, treatment)
    prepared = _add_treatment_timing(prepared, time, treatment)

    return PreparedPanel(
        data=prepared,
        unit=unit,
        time=time,
        outcome=outcome,
        treatment=treatment,
        unit_code_col=UNIT_CODE_COL,
        first_treat_col=FIRST_TREAT_COL,
        event_time_col=EVENT_TIME_COL,
        ever_treated_col=EVER_TREATED_COL,
    )

# -> None means the function is expected not to return anything

# Checks if all required cols are present
def _check_required_columns(data: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")

# This is to make sure that the initial df doesn't already have the extra cols it is supposed to make later
def _check_internal_columns_available(data: pd.DataFrame) -> None:
    internal_columns = [
        UNIT_CODE_COL,
        FIRST_TREAT_COL,
        EVENT_TIME_COL,
        EVER_TREATED_COL,
    ]
    collisions = [column for column in internal_columns if column in data.columns]
    if collisions:
        raise ValueError(f"data contains reserved internal columns: {collisions}")

# data shouldn't contain any missing values
def _check_no_missing_required(data: pd.DataFrame, required: list[str]) -> None:
    if data[required].isna().any().any():
        raise ValueError("required columns cannot contain missing values")

# outcome variable numeric
def _check_numeric_outcome(data: pd.DataFrame, outcome: str) -> None:
    if not is_numeric_dtype(data[outcome]):
        raise ValueError("outcome column must be numeric")

# make sure time is integer
def _check_integer_time(data: pd.DataFrame, time: str) -> None:
    if not is_integer_dtype(data[time]):
        raise ValueError("time column must contain integer-like periods")

# there shouldn't be a duplicated unit - time pair
def _check_unique_unit_time(data: pd.DataFrame, unit: str, time: str) -> None:
    if data.duplicated([unit, time]).any():
        raise ValueError("data must contain at most one row per unit-time pair")

# treatment must be in {0,1}
def _check_binary_treatment(data: pd.DataFrame, treatment: str) -> None:
    values = set(data[treatment].dropna().unique())
    if not values.issubset({0, 1}):
        raise ValueError("treatment must be binary with values 0 and 1")

# make unit int codes - transflorms any int or str of unit_id to numeric codes
def _add_unit_codes(data: pd.DataFrame, unit: str) -> pd.DataFrame:
    prepared = data.copy()
    prepared[UNIT_CODE_COL] = pd.factorize(prepared[unit])[0]
    return prepared

# makes sure if treatment is starting from 1, it also continues to be 1 for the unit
def _check_absorbing_treatment(data: pd.DataFrame, treatment: str) -> None:
    diff = data.groupby(UNIT_CODE_COL, sort=False)[treatment].diff()
    if (diff == -1).any():
        raise ValueError("treatment must be absorbing within each unit")


# adds the columns of event time and boolean column of ever_treated
def _add_treatment_timing(
    data: pd.DataFrame,
    time: str,
    treatment: str,
) -> pd.DataFrame:
    prepared = data.copy()

    prepared[EVER_TREATED_COL] = (
        prepared.groupby(UNIT_CODE_COL, sort=False)[treatment]
        .transform("max")
        .astype(bool)
    )

    treated_rows = prepared[prepared[treatment] == 1]
    first_treat = treated_rows.groupby(UNIT_CODE_COL, sort=False)[time].min()

    prepared[FIRST_TREAT_COL] = (
        prepared[UNIT_CODE_COL].map(first_treat).astype("Int64")
    )
    prepared[EVENT_TIME_COL] = (
        prepared[time] - prepared[FIRST_TREAT_COL]
    ).astype("Int64")

    return prepared


# for future - add check on unit_id