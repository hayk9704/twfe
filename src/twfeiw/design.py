"""Regression design construction for twfeiw estimators."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

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


@dataclass(frozen=True)
class EventStudyDesign:
    """TWFE event-study design plus event-time metadata for reporting."""

    design: DesignMatrix
    event_time_by_column: dict[str, int]
    reference_event_time: int
    included_event_times: list[int]


@dataclass(frozen=True)
class SunAbrahamDesign:
    """Sun-Abraham design plus cohort/event-time metadata for reporting."""

    design: DesignMatrix
    cohort_event_by_column: dict[str, tuple[int, int]]
    reference_event_time: int
    included_event_times: list[int]
    cohorts: list[int]
    cell_counts: pd.Series
    control_group: str


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


def build_twfe_event_study_design(
    panel: PreparedPanel,
    *,
    min_event_time: int | None = None,
    max_event_time: int | None = None,
    reference_event_time: int = -1,
) -> EventStudyDesign:
    """Build an explicit dummy-variable design for a TWFE event study."""

    if not isinstance(panel, PreparedPanel):
        raise TypeError("panel must be a PreparedPanel")

    min_event_time = _validate_optional_event_time(
        min_event_time,
        "min_event_time",
    )
    max_event_time = _validate_optional_event_time(
        max_event_time,
        "max_event_time",
    )
    reference_event_time = _validate_event_time(
        reference_event_time,
        "reference_event_time",
    )
    if (
        min_event_time is not None
        and max_event_time is not None
        and min_event_time > max_event_time
    ):
        raise ValueError("min_event_time must be less than or equal to max_event_time")

    data = panel.data.reset_index(drop=True)
    observed_event_times = _observed_event_times(data, panel)
    if reference_event_time not in observed_event_times:
        raise ValueError(
            "reference_event_time must be observed among treated event times"
        )
    if (
        min_event_time is not None
        and reference_event_time < min_event_time
        or max_event_time is not None
        and reference_event_time > max_event_time
    ):
        raise ValueError(
            "reference_event_time must be inside the selected event-time window"
        )

    included_event_times = [
        event_time
        for event_time in observed_event_times
        if event_time != reference_event_time
        and (min_event_time is None or event_time >= min_event_time)
        and (max_event_time is None or event_time <= max_event_time)
    ]
    if not included_event_times:
        raise ValueError("event-study design must include event-time coefficients")

    outcome = _build_outcome(data, panel.outcome)
    intercept = _build_intercept(len(data))
    event_dummies, event_time_by_column = _build_event_time_dummies(
        data,
        panel,
        included_event_times,
    )

    unit_dummies, reference_unit = _build_unit_fe_dummies(data, panel)
    time_dummies, reference_time = _build_time_fe_dummies(data, panel.time)

    regressors = pd.concat(
        [intercept, event_dummies, unit_dummies, time_dummies],
        axis=1,
    )
    fixed_effect_cols = list(unit_dummies.columns) + list(time_dummies.columns)

    design = DesignMatrix(
        outcome=outcome,
        regressors=regressors,
        model="twfe_event_study",
        outcome_variable=panel.outcome,
        effect_cols=list(event_dummies.columns),
        fixed_effect_cols=fixed_effect_cols,
        regressor_cols=list(regressors.columns),
        reference_unit=reference_unit,
        reference_time=reference_time,
    )

    return EventStudyDesign(
        design=design,
        event_time_by_column=event_time_by_column,
        reference_event_time=reference_event_time,
        included_event_times=included_event_times,
    )


def build_sun_abraham_design(
    panel: PreparedPanel,
    *,
    min_event_time: int | None = None,
    max_event_time: int | None = None,
    reference_event_time: int = -1,
    control_group: str = "never_treated",
) -> SunAbrahamDesign:
    """Build an explicit dummy-variable design for Sun-Abraham estimation."""

    if not isinstance(panel, PreparedPanel):
        raise TypeError("panel must be a PreparedPanel")
    if control_group != "never_treated":
        raise ValueError('control_group must be "never_treated"')

    min_event_time = _validate_optional_event_time(
        min_event_time,
        "min_event_time",
    )
    max_event_time = _validate_optional_event_time(
        max_event_time,
        "max_event_time",
    )
    reference_event_time = _validate_event_time(
        reference_event_time,
        "reference_event_time",
    )
    if (
        min_event_time is not None
        and max_event_time is not None
        and min_event_time > max_event_time
    ):
        raise ValueError("min_event_time must be less than or equal to max_event_time")
    if (
        min_event_time is not None
        and reference_event_time < min_event_time
        or max_event_time is not None
        and reference_event_time > max_event_time
    ):
        raise ValueError(
            "reference_event_time must be inside the selected event-time window"
        )

    data = panel.data.reset_index(drop=True)
    if data[panel.ever_treated_col].all():
        raise ValueError(
            'control_group="never_treated" requires at least one never-treated unit'
        )

    cohorts = _observed_treated_cohorts(data, panel)
    if not cohorts:
        raise ValueError("Sun-Abraham design requires at least one treated cohort")

    observed_event_times = _observed_event_times(data, panel)
    if reference_event_time not in observed_event_times:
        raise ValueError(
            "reference_event_time must be observed among treated event times"
        )
    _validate_reference_observed_by_cohort(
        data,
        panel,
        cohorts,
        reference_event_time,
    )

    outcome = _build_outcome(data, panel.outcome)
    intercept = _build_intercept(len(data))
    (
        cohort_event_dummies,
        cohort_event_by_column,
        cell_counts,
    ) = _build_cohort_event_dummies(
        data,
        panel,
        cohorts,
        min_event_time=min_event_time,
        max_event_time=max_event_time,
        reference_event_time=reference_event_time,
    )
    if cohort_event_dummies.empty:
        raise ValueError("Sun-Abraham design must include cohort-event coefficients")

    unit_dummies, reference_unit = _build_unit_fe_dummies(data, panel)
    time_dummies, reference_time = _build_time_fe_dummies(data, panel.time)

    regressors = pd.concat(
        [intercept, cohort_event_dummies, unit_dummies, time_dummies],
        axis=1,
    )
    fixed_effect_cols = list(unit_dummies.columns) + list(time_dummies.columns)

    design = DesignMatrix(
        outcome=outcome,
        regressors=regressors,
        model="sun_abraham",
        outcome_variable=panel.outcome,
        effect_cols=list(cohort_event_dummies.columns),
        fixed_effect_cols=fixed_effect_cols,
        regressor_cols=list(regressors.columns),
        reference_unit=reference_unit,
        reference_time=reference_time,
    )

    included_event_times = sorted(
        {event_time for _, event_time in cohort_event_by_column.values()}
    )

    return SunAbrahamDesign(
        design=design,
        cohort_event_by_column=cohort_event_by_column,
        reference_event_time=reference_event_time,
        included_event_times=included_event_times,
        cohorts=cohorts,
        cell_counts=cell_counts,
        control_group=control_group,
    )


def _build_outcome(data: pd.DataFrame, outcome: str) -> pd.Series:
    return data[outcome].astype(float).rename(outcome)


def _build_intercept(nobs: int) -> pd.DataFrame:
    return pd.DataFrame({"const": [1.0] * nobs})


def _build_treatment(data: pd.DataFrame, treatment: str) -> pd.DataFrame:
    return data[[treatment]].astype(float)


def _build_event_time_dummies(
    data: pd.DataFrame,
    panel: PreparedPanel,
    included_event_times: list[int],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build one event-time indicator column for each included event time."""

    event_time = data[panel.event_time_col]
    dummies: dict[str, pd.Series] = {}
    event_time_by_column: dict[str, int] = {}

    for value in included_event_times:
        column = _format_event_time_column(value)
        dummies[column] = event_time.eq(value).fillna(False).astype(float)
        event_time_by_column[column] = value

    return pd.DataFrame(dummies, index=data.index), event_time_by_column


def _build_cohort_event_dummies(
    data: pd.DataFrame,
    panel: PreparedPanel,
    cohorts: list[int],
    *,
    min_event_time: int | None,
    max_event_time: int | None,
    reference_event_time: int,
) -> tuple[pd.DataFrame, dict[str, tuple[int, int]], pd.Series]:
    """Build cohort-by-event-time indicators for treated cohorts."""

    dummies: dict[str, pd.Series] = {}
    cohort_event_by_column: dict[str, tuple[int, int]] = {}
    cell_count_values: dict[tuple[int, int], int] = {}

    first_treat = data[panel.first_treat_col]
    event_time = data[panel.event_time_col]

    for cohort in cohorts:
        cohort_mask = first_treat.eq(cohort).fillna(False)
        cohort_event_times = event_time.loc[cohort_mask].dropna()
        for value in sorted(int(item) for item in cohort_event_times.unique()):
            if value == reference_event_time:
                continue
            if min_event_time is not None and value < min_event_time:
                continue
            if max_event_time is not None and value > max_event_time:
                continue

            column = _format_cohort_event_column(cohort, value)
            cell_mask = cohort_mask & event_time.eq(value)
            dummies[column] = cell_mask.fillna(False).astype(float)
            cohort_event_by_column[column] = (cohort, value)
            cell_count_values[(cohort, value)] = int(cell_mask.sum())

    cell_counts = pd.Series(cell_count_values, name="cell_count", dtype="int64")
    if not cell_counts.empty:
        cell_counts.index = pd.MultiIndex.from_tuples(
            cell_counts.index,
            names=["cohort", "event_time"],
        )

    return pd.DataFrame(dummies, index=data.index), cohort_event_by_column, cell_counts


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


def _observed_event_times(data: pd.DataFrame, panel: PreparedPanel) -> list[int]:
    """Return sorted event times observed among ever-treated units."""

    event_times = data.loc[
        data[panel.ever_treated_col],
        panel.event_time_col,
    ].dropna()
    return sorted(int(value) for value in event_times.unique())


def _observed_treated_cohorts(data: pd.DataFrame, panel: PreparedPanel) -> list[int]:
    """Return sorted first-treatment times among ever-treated units."""

    cohorts = data.loc[
        data[panel.ever_treated_col],
        panel.first_treat_col,
    ].dropna()
    return sorted(int(value) for value in cohorts.unique())


def _validate_reference_observed_by_cohort(
    data: pd.DataFrame,
    panel: PreparedPanel,
    cohorts: list[int],
    reference_event_time: int,
) -> None:
    """Confirm every treated cohort has the requested reference event time."""

    first_treat = data[panel.first_treat_col]
    event_time = data[panel.event_time_col]
    missing = [
        cohort
        for cohort in cohorts
        if not (
            first_treat.eq(cohort).fillna(False) & event_time.eq(reference_event_time)
        ).any()
    ]
    if missing:
        raise ValueError(
            "reference_event_time must be observed for every treated cohort"
        )


def _format_event_time_column(event_time: int) -> str:
    """Format an integer event time as a stable regression column name."""

    if event_time < 0:
        return f"event_time_m{abs(event_time)}"
    if event_time > 0:
        return f"event_time_p{event_time}"
    return "event_time_0"


def _format_cohort_event_column(cohort: int, event_time: int) -> str:
    """Format a cohort/event-time pair as a stable regression column name."""

    return f"cohort_{cohort}_{_format_event_time_column(event_time)}"


def _validate_optional_event_time(value: int | None, name: str) -> int | None:
    """Validate an optional event-time boundary."""

    if value is None:
        return None
    return _validate_event_time(value, name)


def _validate_event_time(value: int, name: str) -> int:
    """Validate and normalize an event-time value."""

    if not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)
