import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from twfeiw.api import SunAbrahamResult, sun_abraham
from twfeiw.data import prepare_panel
from twfeiw.design import SunAbrahamDesign, build_sun_abraham_design
from twfeiw.ols import fit_ols
from twfeiw.results import (
    SunAbrahamAggregation,
    build_regression_result,
    build_sun_abraham_aggregation,
)
from twfeiw.vcov import compute_vcov


def _valid_data() -> pd.DataFrame:
    """Create a balanced panel with two treated cohorts and never-treated units."""

    rows = []
    treatment_start = {
        "A": 2002,
        "B": 2002,
        "C": 2003,
        "D": 2003,
        "E": None,
        "F": None,
    }
    unit_base = {
        "A": 0.0,
        "B": 0.4,
        "C": 1.1,
        "D": 1.6,
        "E": 0.8,
        "F": 1.3,
    }
    for unit, first_treat in treatment_start.items():
        for year in range(2000, 2005):
            treated = int(first_treat is not None and year >= first_treat)
            event_time = None if first_treat is None else year - first_treat
            effect = 0.0
            if event_time is not None and event_time >= 0:
                effect = 0.35 + 0.2 * event_time
            y = unit_base[unit] + 0.25 * (year - 2000) + effect
            rows.append(
                {
                    "unit_id": unit,
                    "time": year,
                    "y": y,
                    "treatment": treated,
                }
            )
    return pd.DataFrame(rows)


def _low_level_sun_abraham_result(
    data: pd.DataFrame,
    *,
    vcov: str = "classical",
):
    """Build the lower-level Sun-Abraham pipeline for comparison."""

    panel = prepare_panel(data)
    sun_design = build_sun_abraham_design(panel)
    design = sun_design.design
    ols_result = fit_ols(design)

    if vcov.casefold() == "cluster":
        vcov_result = compute_vcov(
            design,
            ols_result,
            method="cluster",
            clusters=panel.data[panel.unit_code_col],
        )
    else:
        vcov_result = compute_vcov(design, ols_result, method=vcov)

    regression = build_regression_result(design, ols_result, vcov_result)
    aggregation = build_sun_abraham_aggregation(sun_design, regression)
    return panel, sun_design, ols_result, vcov_result, regression, aggregation


def test_build_sun_abraham_design_returns_sun_abraham_design():
    panel = prepare_panel(_valid_data())

    sun_design = build_sun_abraham_design(panel)

    assert isinstance(sun_design, SunAbrahamDesign)
    assert sun_design.design.model == "sun_abraham"


def test_sun_abraham_effect_columns_are_sorted_cohort_event_interactions():
    panel = prepare_panel(_valid_data())

    sun_design = build_sun_abraham_design(panel)

    assert sun_design.design.effect_cols == [
        "cohort_2002_event_time_m2",
        "cohort_2002_event_time_0",
        "cohort_2002_event_time_p1",
        "cohort_2002_event_time_p2",
        "cohort_2003_event_time_m3",
        "cohort_2003_event_time_m2",
        "cohort_2003_event_time_0",
        "cohort_2003_event_time_p1",
    ]
    assert sun_design.cohort_event_by_column[
        "cohort_2003_event_time_m2"
    ] == (2003, -2)
    assert sun_design.included_event_times == [-3, -2, 0, 1, 2]


def test_never_treated_rows_have_zero_sun_abraham_effect_columns():
    panel = prepare_panel(_valid_data())
    sun_design = build_sun_abraham_design(panel)
    never_treated = panel.data[panel.unit].isin(["E", "F"])

    values = sun_design.design.regressors.loc[
        never_treated,
        sun_design.design.effect_cols,
    ]

    assert (values == 0.0).all().all()


def test_treated_pre_periods_activate_non_reference_leads():
    panel = prepare_panel(_valid_data())
    sun_design = build_sun_abraham_design(panel)
    data = panel.data
    row = (data[panel.unit] == "C") & (data[panel.time] == 2001)

    assert data.loc[row, panel.event_time_col].iloc[0] == -2
    assert (
        sun_design.design.regressors.loc[
            row,
            "cohort_2003_event_time_m2",
        ].iloc[0]
        == 1.0
    )


def test_reference_event_time_rows_remain_with_zero_effect_columns():
    panel = prepare_panel(_valid_data())
    sun_design = build_sun_abraham_design(panel)
    data = panel.data
    row = (data[panel.unit] == "A") & (data[panel.time] == 2001)

    assert data.loc[row, panel.event_time_col].iloc[0] == -1
    assert (
        sun_design.design.regressors.loc[
            row,
            sun_design.design.effect_cols,
        ]
        == 0.0
    ).all(axis=None)


def test_invalid_control_group_raises_error():
    panel = prepare_panel(_valid_data())

    with pytest.raises(ValueError, match="control_group"):
        build_sun_abraham_design(panel, control_group="not_yet_treated")


def test_missing_never_treated_controls_raise_error():
    data = _valid_data()
    data = data[~data["unit_id"].isin(["E", "F"])].copy()

    panel = prepare_panel(data)

    with pytest.raises(ValueError, match="never-treated"):
        build_sun_abraham_design(panel)


def test_invalid_event_time_window_raises_error():
    panel = prepare_panel(_valid_data())

    with pytest.raises(ValueError, match="min_event_time"):
        build_sun_abraham_design(panel, min_event_time=2, max_event_time=-2)


def test_reference_event_time_outside_selected_window_raises_error():
    panel = prepare_panel(_valid_data())

    with pytest.raises(ValueError, match="selected event-time window"):
        build_sun_abraham_design(panel, min_event_time=0, max_event_time=2)


def test_missing_cohort_reference_period_raises_error():
    data = _valid_data()
    data = data[~((data["unit_id"].isin(["A", "B"])) & (data["time"] == 2001))]

    panel = prepare_panel(data)

    with pytest.raises(ValueError, match="every treated cohort"):
        build_sun_abraham_design(panel)


def test_sun_abraham_returns_sun_abraham_result():
    result = sun_abraham(_valid_data())

    assert isinstance(result, SunAbrahamResult)
    assert isinstance(result.aggregation, SunAbrahamAggregation)


def test_sun_abraham_matches_manual_low_level_pipeline():
    data = _valid_data()
    _, _, _, _, expected_regression, expected_aggregation = (
        _low_level_sun_abraham_result(data)
    )

    actual = sun_abraham(data)

    pd.testing.assert_series_equal(actual.params, expected_regression.params)
    pd.testing.assert_frame_equal(actual.event_table(), expected_aggregation.event_table)
    pd.testing.assert_frame_equal(
        actual.cohort_event_table(),
        expected_aggregation.cohort_event_table,
    )


def test_sun_abraham_hc1_matches_low_level_pipeline():
    data = _valid_data()
    _, _, _, _, expected_regression, expected_aggregation = (
        _low_level_sun_abraham_result(data, vcov="HC1")
    )

    actual = sun_abraham(data, vcov="HC1")

    pd.testing.assert_series_equal(
        actual.standard_errors,
        expected_regression.standard_errors,
    )
    pd.testing.assert_frame_equal(actual.event_table(), expected_aggregation.event_table)


def test_sun_abraham_cluster_defaults_to_unit_clusters():
    data = _valid_data()
    _, _, _, expected_vcov, _, expected_aggregation = (
        _low_level_sun_abraham_result(data, vcov="cluster")
    )

    actual = sun_abraham(data, vcov="cluster")

    assert actual.vcov_method == "cluster"
    assert actual.vcov.n_clusters == expected_vcov.n_clusters
    pd.testing.assert_frame_equal(actual.event_table(), expected_aggregation.event_table)


def test_cohort_event_coefficients_match_statsmodels_explicit_dummies():
    data = _valid_data()
    panel = prepare_panel(data)
    sun_design = build_sun_abraham_design(panel)
    design = sun_design.design

    stats_data = panel.data.copy()
    for column in design.effect_cols:
        stats_data[column] = design.regressors[column]

    formula = (
        "y ~ "
        + " + ".join(design.effect_cols)
        + " + C(unit_id) + C(time)"
    )
    statsmodels_fit = smf.ols(formula, data=stats_data).fit()
    result = sun_abraham(data)

    for column in design.effect_cols:
        assert result.params[column] == pytest.approx(statsmodels_fit.params[column])


def test_cohort_event_table_has_expected_index_and_columns():
    result = sun_abraham(_valid_data())

    table = result.cohort_event_table()

    assert table.index.names == ["cohort", "event_time"]
    assert table.columns.tolist() == [
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


def test_event_table_matches_manual_weighted_averages():
    result = sun_abraham(_valid_data())
    cohort_table = result.cohort_event_table()

    for event_time, event_rows in cohort_table.groupby(level="event_time"):
        columns = event_rows["column"].tolist()
        weights = event_rows["weight"].to_numpy(dtype=float)
        expected = float(weights @ result.params.loc[columns].to_numpy(dtype=float))

        assert result.event_table().loc[event_time, "coef"] == pytest.approx(expected)


def test_event_table_standard_errors_match_manual_linear_combination():
    result = sun_abraham(_valid_data())
    cohort_table = result.cohort_event_table()

    for event_time, event_rows in cohort_table.groupby(level="event_time"):
        columns = event_rows["column"].tolist()
        weights = event_rows["weight"].to_numpy(dtype=float)
        vcov = result.vcov_matrix.loc[columns, columns].to_numpy(dtype=float)
        expected = np.sqrt(float(weights @ vcov @ weights))

        assert result.event_table().loc[event_time, "std_err"] == pytest.approx(
            expected
        )


def test_weights_returns_cell_counts_and_weights():
    result = sun_abraham(_valid_data())

    weights = result.weights()

    assert weights.index.names == ["cohort", "event_time"]
    assert weights.columns.tolist() == ["cell_count", "weight"]
    assert weights.loc[(2002, 0), "cell_count"] == 2
    assert weights.loc[(2002, 0), "weight"] == pytest.approx(0.5)


def test_sun_abraham_supports_custom_column_names():
    data = _valid_data().rename(
        columns={
            "unit_id": "id",
            "time": "year",
            "y": "outcome",
            "treatment": "treated",
        }
    )

    result = sun_abraham(
        data,
        unit="id",
        time="year",
        outcome="outcome",
        treatment="treated",
    )

    assert result.panel.unit == "id"
    assert result.panel.time == "year"
    assert result.panel.outcome == "outcome"
    assert result.panel.treatment == "treated"
    assert result.event_times == [-3, -2, 0, 1, 2]


def test_sun_abraham_is_exported_from_package_root():
    from twfeiw import SunAbrahamResult as RootSunAbrahamResult
    from twfeiw import sun_abraham as root_sun_abraham

    result = root_sun_abraham(_valid_data())

    assert isinstance(result, RootSunAbrahamResult)
