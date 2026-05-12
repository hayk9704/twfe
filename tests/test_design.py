import pandas as pd
from pandas.api.types import is_numeric_dtype

from twfeiw.data import prepare_panel
from twfeiw.design import DesignMatrix, build_twfe_design


def _valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "time": [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002],
            "y": [1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 2.0, 2.5, 3.0],
            "treatment": [0, 1, 1, 0, 0, 1, 0, 0, 0],
        }
    )

# checks if return object is DesignMatrix and if design.model is twfe
def test_build_twfe_design_returns_design_matrix():
    panel = prepare_panel(_valid_data())

    design = build_twfe_design(panel)

    assert isinstance(design, DesignMatrix)
    assert design.model == "twfe"

# tests if build_twfe_design() returns the same outcome as prepare_panel()
def test_outcome_equals_prepared_outcome_vector():
    panel = prepare_panel(_valid_data())

    design = build_twfe_design(panel)

    expected = panel.data[panel.outcome].astype(float).reset_index(drop=True)
    pd.testing.assert_series_equal(design.outcome, expected)

# tests the names and order for regressors columns
def test_regressors_have_expected_columns_and_order():
    panel = prepare_panel(_valid_data())

    design = build_twfe_design(panel)

    assert design.regressor_cols == [
        "const",
        "treatment",
        "unit_fe_1",
        "unit_fe_2",
        "time_fe_2001",
        "time_fe_2002",
    ]
    assert design.regressor_cols == list(design.regressors.columns)

# makes sure the shape of the regressors X corresponds to the inputted panel
def test_design_matrix_has_expected_shape():
    panel = prepare_panel(_valid_data())

    design = build_twfe_design(panel)

    nobs = len(panel.data)
    nunits = panel.data[panel.unit_code_col].nunique()
    nperiods = panel.data[panel.time].nunique()
    assert design.regressors.shape == (nobs, nunits + nperiods)

# regressors first column must be for the intercept and equal to 1
def test_intercept_is_first_column_and_equals_one():
    panel = prepare_panel(_valid_data())

    design = build_twfe_design(panel)

    assert design.regressors.columns[0] == "const"
    assert design.regressors["const"].eq(1.0).all()

# treatment effect column must be names "treatment"
# the panel and design treatment columns should be the same
def test_treatment_is_effect_column():
    panel = prepare_panel(_valid_data())

    design = build_twfe_design(panel)

    assert design.effect_cols == ["treatment"]
    assert design.regressors["treatment"].tolist() == panel.data["treatment"].tolist()


# the fixed effect columns must include the elements apart from the omitted ones.
# they must not include the intercept and treatment elements
def test_fixed_effect_columns_are_unit_and_time_dummies():
    panel = prepare_panel(_valid_data())

    design = build_twfe_design(panel)

    assert design.fixed_effect_cols == [
        "unit_fe_1",
        "unit_fe_2",
        "time_fe_2001",
        "time_fe_2002",
    ]
    assert "const" not in design.fixed_effect_cols
    assert "treatment" not in design.fixed_effect_cols

# testing that 1 unit column and 1 time column are omitted.
def test_one_unit_and_one_time_category_are_omitted():
    panel = prepare_panel(_valid_data())

    design = build_twfe_design(panel)

    unit_cols = [col for col in design.regressor_cols if col.startswith("unit_fe_")]
    time_cols = [col for col in design.regressor_cols if col.startswith("time_fe_")]

    assert len(unit_cols) == panel.data[panel.unit_code_col].nunique() - 1
    assert len(time_cols) == panel.data[panel.time].nunique() - 1
    assert "unit_fe_0" not in unit_cols
    assert "time_fe_2000" not in time_cols

# the reference units and the reference times must be the first ones.
def test_reference_unit_and_time_are_stored():
    panel = prepare_panel(_valid_data())

    design = build_twfe_design(panel)

    assert design.reference_unit == "A"
    assert design.reference_time == 2000

# the regressor columns must be numeric
def test_all_regressor_columns_are_numeric():
    panel = prepare_panel(_valid_data())

    design = build_twfe_design(panel)

    assert all(is_numeric_dtype(design.regressors[col]) for col in design.regressor_cols)

# unbalance the panel and see if the design regressors matrix dims are correct
def test_unbalanced_panel_produces_valid_design_matrix():
    data = _valid_data()
    data = data[~((data["unit_id"] == "B") & (data["time"] == 2001))]
    panel = prepare_panel(data)

    design = build_twfe_design(panel)

    nobs = len(panel.data)
    nunits = panel.data[panel.unit_code_col].nunique()
    nperiods = panel.data[panel.time].nunique()
    assert design.regressors.shape == (nobs, nunits + nperiods)

# make column names different and check if design keeps them consistent
def test_custom_column_names_are_reflected_in_metadata():
    data = _valid_data().rename(
        columns={
            "unit_id": "firm_id",
            "time": "year",
            "y": "sales",
            "treatment": "treated",
        }
    )
    panel = prepare_panel(
        data,
        unit="firm_id",
        time="year",
        outcome="sales",
        treatment="treated",
    )

    design = build_twfe_design(panel)

    assert design.outcome_variable == "sales"
    assert design.effect_cols == ["treated"]
    assert "treated" in design.regressor_cols
