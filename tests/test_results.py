from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from scipy import stats

from twfeiw.data import prepare_panel
from twfeiw.design import DesignMatrix, build_twfe_design
from twfeiw.ols import OLSResult, fit_ols
from twfeiw.results import RegressionResult, build_regression_result
from twfeiw.vcov import VCovResult, compute_vcov


def _valid_data() -> pd.DataFrame:
    """Create a minimal balanced panel used by result assembly tests."""

    return pd.DataFrame(
        {
            "unit_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "time": [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002],
            "y": [1.0, 1.7, 2.1, 3.2, 3.4, 4.3, 2.1, 2.6, 2.9],
            "treatment": [0, 1, 1, 0, 0, 1, 0, 0, 0],
        }
    )


def _twfe_parts(
    method: str = "classical",
) -> tuple[pd.DataFrame, DesignMatrix, OLSResult, VCovResult, pd.Series]:
    """Build reusable TWFE inputs and outputs for the requested vcov method."""

    data = _valid_data()
    panel = prepare_panel(data)
    design = build_twfe_design(panel)
    ols_result = fit_ols(design)
    clusters = panel.data[panel.unit_code_col]

    if method == "cluster":
        vcov_result = compute_vcov(
            design,
            ols_result,
            method="cluster",
            clusters=clusters,
        )
    else:
        vcov_result = compute_vcov(design, ols_result, method=method)

    return data, design, ols_result, vcov_result, clusters


def _statsmodels_fit(data: pd.DataFrame):
    """Fit the comparable statsmodels TWFE model for reference checks."""

    return smf.ols(
        "y ~ treatment + C(unit_id) + C(time)",
        data=data,
    ).fit()


def test_build_regression_result_returns_regression_result():
    """Verify the assembler returns the public regression result dataclass."""

    _, design, ols_result, vcov_result, _ = _twfe_parts()

    result = build_regression_result(design, ols_result, vcov_result)

    assert isinstance(result, RegressionResult)


def test_regression_result_copies_estimation_and_vcov_outputs():
    """Verify estimation outputs and model metadata are copied into the result."""

    _, design, ols_result, vcov_result, _ = _twfe_parts()

    result = build_regression_result(design, ols_result, vcov_result)

    pd.testing.assert_series_equal(result.params, ols_result.params)
    pd.testing.assert_frame_equal(result.vcov, vcov_result.vcov)
    pd.testing.assert_series_equal(
        result.standard_errors,
        vcov_result.standard_errors,
    )
    assert result.rss == ols_result.rss
    assert result.tss == ols_result.tss
    assert result.r_squared == ols_result.r_squared
    assert result.adj_r_squared == ols_result.adj_r_squared
    assert result.model == design.model
    assert result.outcome_variable == design.outcome_variable
    assert result.effect_cols == design.effect_cols
    assert result.nobs == ols_result.nobs
    assert result.rank == ols_result.rank
    assert result.df_resid == ols_result.df_resid
    assert result.vcov_method == vcov_result.method
    assert result.small_sample == vcov_result.small_sample


def test_t_stats_equal_params_divided_by_standard_errors():
    """Verify t statistics are coefficients divided by standard errors."""

    _, design, ols_result, vcov_result, _ = _twfe_parts()

    result = build_regression_result(design, ols_result, vcov_result)

    expected = (ols_result.params / vcov_result.standard_errors).rename("t")
    pd.testing.assert_series_equal(result.t_stats, expected)


def test_p_values_use_t_distribution_with_inference_df():
    """Verify p-values use the t distribution with the inference degrees of freedom."""

    _, design, ols_result, vcov_result, _ = _twfe_parts()

    result = build_regression_result(design, ols_result, vcov_result)

    expected = 2.0 * stats.t.sf(
        np.abs(result.t_stats.to_numpy(dtype=float)),
        df=result.inference_df,
    )
    np.testing.assert_allclose(result.p_values.to_numpy(), expected)


def test_confidence_intervals_use_t_critical_value():
    """Verify confidence intervals use the requested alpha and t critical value."""

    _, design, ols_result, vcov_result, _ = _twfe_parts()
    alpha = 0.1

    result = build_regression_result(
        design,
        ols_result,
        vcov_result,
        alpha=alpha,
    )

    critical = stats.t.ppf(1.0 - alpha / 2.0, df=result.inference_df)
    expected_lower = ols_result.params - critical * vcov_result.standard_errors
    expected_upper = ols_result.params + critical * vcov_result.standard_errors
    np.testing.assert_allclose(result.conf_int["ci_lower"], expected_lower)
    np.testing.assert_allclose(result.conf_int["ci_upper"], expected_upper)
    assert result.alpha == alpha


def test_summary_table_has_expected_columns_and_index():
    """Verify the summary table has the expected layout and mirrored values."""

    _, design, ols_result, vcov_result, _ = _twfe_parts()

    result = build_regression_result(design, ols_result, vcov_result)

    assert result.summary.index.tolist() == design.regressor_cols
    assert result.summary.columns.tolist() == [
        "coef",
        "std_err",
        "t",
        "p_value",
        "ci_lower",
        "ci_upper",
    ]
    np.testing.assert_allclose(result.summary["coef"], result.params)
    np.testing.assert_allclose(result.summary["std_err"], result.standard_errors)
    np.testing.assert_allclose(result.summary["t"], result.t_stats)
    np.testing.assert_allclose(result.summary["p_value"], result.p_values)


@pytest.mark.parametrize("method", ["classical", "HC1"])
def test_non_clustered_inference_df_uses_residual_degrees_of_freedom(method: str):
    """Verify non-clustered inference uses residual degrees of freedom."""

    _, design, ols_result, vcov_result, _ = _twfe_parts(method=method)

    result = build_regression_result(design, ols_result, vcov_result)

    assert result.inference_df == ols_result.df_resid


def test_clustered_inference_df_uses_number_of_clusters_minus_one():
    """Verify clustered inference uses cluster count minus one and stores metadata."""

    _, design, ols_result, vcov_result, _ = _twfe_parts(method="cluster")

    result = build_regression_result(design, ols_result, vcov_result)

    assert result.inference_df == vcov_result.n_clusters - 1
    assert result.n_clusters == vcov_result.n_clusters
    assert result.cluster_name == vcov_result.cluster_name


def test_classical_treatment_inference_matches_statsmodels():
    """Compare classical treatment inference against statsmodels output."""

    data, design, ols_result, vcov_result, _ = _twfe_parts()
    statsmodels_fit = _statsmodels_fit(data)

    result = build_regression_result(design, ols_result, vcov_result)
    statsmodels_ci = statsmodels_fit.conf_int().loc["treatment"]

    assert result.t_stats["treatment"] == pytest.approx(
        statsmodels_fit.tvalues["treatment"]
    )
    assert result.p_values["treatment"] == pytest.approx(
        statsmodels_fit.pvalues["treatment"]
    )
    assert result.conf_int.loc["treatment", "ci_lower"] == pytest.approx(
        statsmodels_ci.iloc[0]
    )
    assert result.conf_int.loc["treatment", "ci_upper"] == pytest.approx(
        statsmodels_ci.iloc[1]
    )


def test_non_design_matrix_input_raises_error():
    """Verify non-DesignMatrix inputs are rejected with a type error."""

    _, _, ols_result, vcov_result, _ = _twfe_parts()

    with pytest.raises(TypeError, match="design must be a DesignMatrix"):
        build_regression_result("not a design", ols_result, vcov_result)


def test_non_ols_result_input_raises_error():
    """Verify non-OLSResult inputs are rejected with a type error."""

    _, design, _, vcov_result, _ = _twfe_parts()

    with pytest.raises(TypeError, match="ols_result must be an OLSResult"):
        build_regression_result(design, "not an ols result", vcov_result)


def test_non_vcov_result_input_raises_error():
    """Verify non-VCovResult inputs are rejected with a type error."""

    _, design, ols_result, _, _ = _twfe_parts()

    with pytest.raises(TypeError, match="vcov_result must be a VCovResult"):
        build_regression_result(design, ols_result, "not a vcov result")


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, np.nan, "0.05"])
def test_invalid_alpha_raises_error(alpha):
    """Verify invalid confidence interval significance levels are rejected."""

    _, design, ols_result, vcov_result, _ = _twfe_parts()

    with pytest.raises(ValueError, match="alpha must be strictly between 0 and 1"):
        build_regression_result(
            design,
            ols_result,
            vcov_result,
            alpha=alpha,
        )


def test_mismatched_regressor_names_raise_error():
    """Verify inconsistent regressor labels across inputs are rejected."""

    _, design, ols_result, vcov_result, _ = _twfe_parts()
    bad_ols_result = replace(
        ols_result,
        regressor_cols=["bad"] * len(ols_result.regressor_cols),
    )

    with pytest.raises(ValueError, match="regressor names"):
        build_regression_result(design, bad_ols_result, vcov_result)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("nobs", 999, "nobs"),
        ("rank", 999, "rank"),
        ("df_resid", 999, "df_resid"),
    ],
)
def test_mismatched_metadata_raises_error(field: str, value: int, match: str):
    """Verify inconsistent OLS and vcov metadata are rejected."""

    _, design, ols_result, vcov_result, _ = _twfe_parts()
    bad_vcov_result = replace(vcov_result, **{field: value})

    with pytest.raises(ValueError, match=match):
        build_regression_result(design, ols_result, bad_vcov_result)


def test_cluster_method_without_n_clusters_raises_error():
    """Verify clustered inference requires cluster count metadata."""

    _, design, ols_result, vcov_result, _ = _twfe_parts(method="cluster")
    bad_vcov_result = replace(vcov_result, n_clusters=None)

    with pytest.raises(ValueError, match="n_clusters"):
        build_regression_result(design, ols_result, bad_vcov_result)


def test_nonpositive_inference_degrees_of_freedom_raises_error():
    """Verify inferred degrees of freedom must be positive."""

    _, design, ols_result, vcov_result, _ = _twfe_parts(method="cluster")
    bad_vcov_result = replace(vcov_result, n_clusters=1)

    with pytest.raises(ValueError, match="inference degrees of freedom"):
        build_regression_result(design, ols_result, bad_vcov_result)
