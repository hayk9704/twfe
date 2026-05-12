import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from twfeiw.data import prepare_panel
from twfeiw.design import DesignMatrix, build_twfe_design
from twfeiw.ols import OLSResult, fit_ols


def _valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "time": [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002],
            "y": [1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 2.0, 2.5, 3.0],
            "treatment": [0, 1, 1, 0, 0, 1, 0, 0, 0],
        }
    )


def _twfe_design() -> DesignMatrix:
    panel = prepare_panel(_valid_data())
    return build_twfe_design(panel)

# return object must be OLSResult
def test_fit_ols_returns_ols_result():
    result = fit_ols(_twfe_design())

    assert isinstance(result, OLSResult)

# in OLS result, the indices of parameter columns and regressor columns must both be 
# design.regreeeor_cols
def test_params_are_labeled_by_regressor_columns():
    design = _twfe_design()

    result = fit_ols(design)

    assert result.regressor_cols == design.regressor_cols
    assert result.params.index.tolist() == design.regressor_cols

# make X@beta^ must be very close to fitted values
def test_fitted_values_equal_regressors_times_params():
    design = _twfe_design()

    result = fit_ols(design)

    expected = design.regressors.to_numpy(dtype=float) @ result.params.to_numpy()
    np.testing.assert_allclose(result.fitted_values.to_numpy(), expected)

# y-y^ must be appx equal to epsilon^
def test_residuals_equal_outcome_minus_fitted_values():
    design = _twfe_design()

    result = fit_ols(design)

    expected = design.outcome.to_numpy() - result.fitted_values.to_numpy()
    np.testing.assert_allclose(result.residuals.to_numpy(), expected)

# epsilon^.T@epsilon^ must be close to RSS
def test_rss_is_sum_of_squared_residuals():
    result = fit_ols(_twfe_design())

    expected = float(result.residuals.to_numpy() @ result.residuals.to_numpy())
    assert result.rss == pytest.approx(expected)

# calculate the rank, residual degrees of freedom and assert equality with returned ones
def test_nobs_rank_and_df_resid_are_stored():
    design = _twfe_design()

    result = fit_ols(design)

    expected_rank = np.linalg.matrix_rank(design.regressors.to_numpy(dtype=float))
    assert result.nobs == len(design.outcome)
    assert result.rank == expected_rank
    assert result.df_resid == result.nobs - result.rank

# validate against statsmodel.ols
def test_treatment_coefficient_matches_statsmodels_formula_twfe():
    data = _valid_data()
    panel = prepare_panel(data)
    design = build_twfe_design(panel)

    result = fit_ols(design)
    statsmodels_fit = smf.ols(
        "y ~ treatment + C(unit_id) + C(time)",
        data=data,
    ).fit()

    assert result.params["treatment"] == pytest.approx(
        statsmodels_fit.params["treatment"]
    )

# make rank < n and check that we return an error.
def test_rank_deficient_design_raises_error():
    design = DesignMatrix(
        outcome=pd.Series([1.0, 2.0, 3.0], name="y"),
        regressors=pd.DataFrame(
            {
                "const": [1.0, 1.0, 1.0],
                "x": [0.0, 1.0, 2.0],
                "x_duplicate": [0.0, 1.0, 2.0],
            }
        ),
        model="test",
        outcome_variable="y",
        effect_cols=["x"],
        fixed_effect_cols=[],
        regressor_cols=["const", "x", "x_duplicate"],
        reference_unit=None,
        reference_time=0,
    )

    with pytest.raises(ValueError, match="rank deficient"):
        fit_ols(design)

# non design matrix as input must raise an error
def test_non_design_matrix_input_raises_error():
    with pytest.raises(TypeError, match="design must be a DesignMatrix"):
        fit_ols("not a design")
