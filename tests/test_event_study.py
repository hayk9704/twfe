import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from twfeiw.api import EventStudyResult, event_study
from twfeiw.data import prepare_panel
from twfeiw.design import (
    EventStudyDesign,
    build_twfe_event_study_design,
)
from twfeiw.ols import fit_ols
from twfeiw.results import build_regression_result
from twfeiw.vcov import compute_vcov


def _valid_data() -> pd.DataFrame:
    """Create a minimal balanced panel with two treated cohorts."""

    return pd.DataFrame(
        {
            "unit_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "time": [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002],
            "y": [1.0, 1.7, 2.1, 3.2, 3.4, 4.3, 2.1, 2.6, 2.9],
            "treatment": [0, 1, 1, 0, 0, 1, 0, 0, 0],
        }
    )


def _low_level_event_study_result(
    data: pd.DataFrame,
    *,
    vcov: str = "classical",
):
    """Build the lower-level event-study pipeline for comparison."""

    panel = prepare_panel(data)
    event_design = build_twfe_event_study_design(panel)
    design = event_design.design
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
    return panel, event_design, ols_result, vcov_result, regression


def test_build_twfe_event_study_design_returns_event_study_design():
    """Verify the event-study builder returns the event-study design wrapper."""

    panel = prepare_panel(_valid_data())

    event_design = build_twfe_event_study_design(panel)

    assert isinstance(event_design, EventStudyDesign)


def test_event_study_design_uses_twfe_event_study_model_name():
    """Verify the wrapped design is labeled as a TWFE event-study model."""

    panel = prepare_panel(_valid_data())

    event_design = build_twfe_event_study_design(panel)

    assert event_design.design.model == "twfe_event_study"


def test_event_study_effect_columns_exclude_reference_period():
    """Verify event-time columns include observed periods except the reference."""

    panel = prepare_panel(_valid_data())

    event_design = build_twfe_event_study_design(panel)

    assert event_design.reference_event_time == -1
    assert event_design.included_event_times == [-2, 0, 1]
    assert event_design.design.effect_cols == [
        "event_time_m2",
        "event_time_0",
        "event_time_p1",
    ]


def test_event_study_effect_columns_respect_min_and_max_event_time():
    """Verify the event-time window restricts included effect columns."""

    panel = prepare_panel(_valid_data())

    event_design = build_twfe_event_study_design(
        panel,
        min_event_time=-2,
        max_event_time=0,
    )

    assert event_design.included_event_times == [-2, 0]
    assert event_design.design.effect_cols == ["event_time_m2", "event_time_0"]


def test_event_study_event_time_mapping_is_stored():
    """Verify event-time columns can be mapped back to integer event times."""

    panel = prepare_panel(_valid_data())

    event_design = build_twfe_event_study_design(panel)

    assert event_design.event_time_by_column == {
        "event_time_m2": -2,
        "event_time_0": 0,
        "event_time_p1": 1,
    }


def test_never_treated_units_have_zero_event_time_indicators():
    """Verify never-treated rows are retained with zero event-time indicators."""

    panel = prepare_panel(_valid_data())
    event_design = build_twfe_event_study_design(panel)
    data = panel.data
    never_treated = data[panel.unit] == "C"

    values = event_design.design.regressors.loc[
        never_treated,
        event_design.design.effect_cols,
    ]

    assert (values == 0.0).all().all()


def test_event_study_design_includes_unit_and_time_fixed_effects():
    """Verify event-study designs retain the usual unit and time fixed effects."""

    panel = prepare_panel(_valid_data())

    event_design = build_twfe_event_study_design(panel)

    assert event_design.design.fixed_effect_cols == [
        "unit_fe_1",
        "unit_fe_2",
        "time_fe_2001",
        "time_fe_2002",
    ]


def test_invalid_event_time_window_raises_error():
    """Verify event-time lower bounds cannot exceed upper bounds."""

    panel = prepare_panel(_valid_data())

    with pytest.raises(ValueError, match="min_event_time"):
        build_twfe_event_study_design(
            panel,
            min_event_time=2,
            max_event_time=-2,
        )


def test_missing_reference_event_time_raises_error():
    """Verify the omitted event time must be observed among treated units."""

    panel = prepare_panel(_valid_data())

    with pytest.raises(ValueError, match="reference_event_time"):
        build_twfe_event_study_design(panel, reference_event_time=-99)


def test_reference_event_time_outside_selected_window_raises_error():
    """Verify the omitted event time must be inside the selected window."""

    panel = prepare_panel(_valid_data())

    with pytest.raises(ValueError, match="selected event-time window"):
        build_twfe_event_study_design(
            panel,
            min_event_time=0,
            max_event_time=1,
        )


def test_event_study_returns_event_study_result():
    """Verify the public event-study API returns the event-study result object."""

    result = event_study(_valid_data())

    assert isinstance(result, EventStudyResult)


def test_event_study_matches_manual_low_level_pipeline():
    """Verify public event-study output matches the explicit low-level pipeline."""

    data = _valid_data()
    _, _, _, _, expected = _low_level_event_study_result(data)

    actual = event_study(data)

    pd.testing.assert_series_equal(actual.params, expected.params)
    pd.testing.assert_series_equal(actual.standard_errors, expected.standard_errors)
    pd.testing.assert_frame_equal(actual.summary(), expected.summary)


def test_event_study_hc1_matches_low_level_pipeline():
    """Verify HC1 event-study output matches the explicit low-level workflow."""

    data = _valid_data()
    _, _, _, _, expected = _low_level_event_study_result(data, vcov="HC1")

    actual = event_study(data, vcov="HC1")

    pd.testing.assert_series_equal(actual.standard_errors, expected.standard_errors)
    pd.testing.assert_frame_equal(actual.vcov_matrix, expected.vcov)


def test_event_study_cluster_defaults_to_unit_clusters():
    """Verify clustered event-study inference clusters by unit."""

    data = _valid_data()
    _, _, _, expected_vcov, expected = _low_level_event_study_result(
        data,
        vcov="cluster",
    )

    actual = event_study(data, vcov="cluster")

    pd.testing.assert_series_equal(actual.standard_errors, expected.standard_errors)
    assert actual.vcov_method == "cluster"
    assert actual.vcov.n_clusters == expected_vcov.n_clusters
    assert actual.inference_df == expected.inference_df


def test_event_study_supports_custom_column_names():
    """Verify event-study API supports custom data column names."""

    data = _valid_data().rename(
        columns={
            "unit_id": "id",
            "time": "year",
            "y": "outcome",
            "treatment": "treated",
        }
    )

    result = event_study(
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
    assert result.event_times == [-2, 0, 1]


def test_event_table_has_event_time_index_and_expected_columns():
    """Verify event_table returns coefficient inference by event time."""

    result = event_study(_valid_data())

    table = result.event_table()

    assert table.index.name == "event_time"
    assert table.index.tolist() == [-2, 0, 1]
    assert table.columns.tolist() == [
        "coef",
        "std_err",
        "t",
        "p_value",
        "ci_lower",
        "ci_upper",
    ]


def test_event_table_is_sorted_by_event_time():
    """Verify event_table is ordered by integer event time."""

    result = event_study(_valid_data())

    table = result.event_table()

    assert table.index.tolist() == sorted(table.index.tolist())


def test_event_table_excludes_reference_event_time():
    """Verify the omitted reference period is not reported as an estimate."""

    result = event_study(_valid_data())

    assert result.reference_event_time == -1
    assert -1 not in result.event_table().index


def test_event_study_coefficients_match_statsmodels_explicit_dummies():
    """Compare event-time coefficients against a statsmodels dummy regression."""

    data = _valid_data()
    panel = prepare_panel(data)
    event_design = build_twfe_event_study_design(panel)
    design = event_design.design

    stats_data = panel.data.copy()
    for column in design.effect_cols:
        stats_data[column] = design.regressors[column]

    formula = (
        "y ~ "
        + " + ".join(design.effect_cols)
        + " + C(unit_id) + C(time)"
    )
    statsmodels_fit = smf.ols(formula, data=stats_data).fit()
    result = event_study(data)

    for column in design.effect_cols:
        assert result.params[column] == pytest.approx(statsmodels_fit.params[column])


def test_event_study_fitted_values_match_statsmodels_explicit_dummies():
    """Compare fitted values against a statsmodels dummy regression."""

    data = _valid_data()
    panel = prepare_panel(data)
    event_design = build_twfe_event_study_design(panel)
    design = event_design.design

    stats_data = panel.data.copy()
    for column in design.effect_cols:
        stats_data[column] = design.regressors[column]

    formula = (
        "y ~ "
        + " + ".join(design.effect_cols)
        + " + C(unit_id) + C(time)"
    )
    statsmodels_fit = smf.ols(formula, data=stats_data).fit()
    result = event_study(data)

    np.testing.assert_allclose(
        result.ols.fitted_values.to_numpy(),
        statsmodels_fit.fittedvalues.to_numpy(),
    )


def test_event_study_is_exported_from_package_root():
    """Verify users can import event-study API from the package root."""

    from twfeiw import EventStudyResult as RootEventStudyResult
    from twfeiw import event_study as root_event_study

    result = root_event_study(_valid_data())

    assert isinstance(result, RootEventStudyResult)
