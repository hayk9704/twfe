import pandas as pd
import pytest

from twfeiw.api import TWFEResult, twfe
from twfeiw.data import prepare_panel
from twfeiw.design import build_twfe_design
from twfeiw.ols import fit_ols
from twfeiw.results import build_regression_result
from twfeiw.vcov import compute_vcov


def _valid_data() -> pd.DataFrame:
    """Create a minimal balanced panel used by API tests."""

    return pd.DataFrame(
        {
            "unit_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "time": [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002],
            "y": [1.0, 1.7, 2.1, 3.2, 3.4, 4.3, 2.1, 2.6, 2.9],
            "treatment": [0, 1, 1, 0, 0, 1, 0, 0, 0],
        }
    )


def _low_level_result(data: pd.DataFrame, *, vcov: str = "classical"):
    """Build the current lower-level TWFE pipeline for comparison."""

    panel = prepare_panel(data)
    design = build_twfe_design(panel)
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
    return panel, design, ols_result, vcov_result, regression


def test_twfe_returns_twfe_result():
    """Verify the public wrapper returns the TWFE result object."""

    result = twfe(_valid_data())

    assert isinstance(result, TWFEResult)


def test_twfe_matches_low_level_pipeline_for_classical_vcov():
    """Verify default API output matches the explicit low-level workflow."""

    data = _valid_data()
    _, _, _, _, expected = _low_level_result(data)

    actual = twfe(data)

    pd.testing.assert_series_equal(actual.params, expected.params)
    pd.testing.assert_series_equal(actual.standard_errors, expected.standard_errors)
    pd.testing.assert_frame_equal(actual.summary(), expected.summary)


def test_twfe_supports_custom_column_names():
    """Verify custom column names flow through data preparation and design."""

    data = _valid_data().rename(
        columns={
            "unit_id": "id",
            "time": "year",
            "y": "outcome",
            "treatment": "treated",
        }
    )

    result = twfe(
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
    assert result.effect_name == "treated"


def test_twfe_hc1_matches_low_level_pipeline():
    """Verify HC1 API output matches the explicit low-level workflow."""

    data = _valid_data()
    _, _, _, _, expected = _low_level_result(data, vcov="HC1")

    actual = twfe(data, vcov="HC1")

    pd.testing.assert_series_equal(actual.standard_errors, expected.standard_errors)
    pd.testing.assert_frame_equal(actual.vcov_matrix, expected.vcov)
    assert actual.vcov_method == "HC1"


def test_twfe_cluster_defaults_to_unit_clusters():
    """Verify clustered API inference clusters by the prepared unit code."""

    data = _valid_data()
    _, _, _, expected_vcov, expected = _low_level_result(data, vcov="cluster")

    actual = twfe(data, vcov="cluster")

    pd.testing.assert_series_equal(actual.standard_errors, expected.standard_errors)
    assert actual.vcov_method == "cluster"
    assert actual.vcov.n_clusters == expected_vcov.n_clusters
    assert actual.regression.inference_df == expected.inference_df


def test_invalid_vcov_method_raises_error():
    """Verify unsupported variance-covariance methods are rejected."""

    with pytest.raises(ValueError, match="unsupported vcov method"):
        twfe(_valid_data(), vcov="unsupported")


def test_non_string_vcov_raises_error():
    """Verify variance-covariance method names must be strings."""

    with pytest.raises(TypeError, match="vcov must be a string"):
        twfe(_valid_data(), vcov=123)


def test_twfe_result_exposes_common_regression_outputs():
    """Verify result convenience properties mirror the regression result."""

    result = twfe(_valid_data())

    pd.testing.assert_series_equal(result.params, result.regression.params)
    pd.testing.assert_frame_equal(result.vcov_matrix, result.regression.vcov)
    pd.testing.assert_series_equal(
        result.standard_errors,
        result.regression.standard_errors,
    )
    pd.testing.assert_series_equal(result.t_stats, result.regression.t_stats)
    pd.testing.assert_series_equal(result.p_values, result.regression.p_values)
    pd.testing.assert_frame_equal(result.conf_int, result.regression.conf_int)
    assert result.r_squared == result.regression.r_squared
    assert result.adj_r_squared == result.regression.adj_r_squared
    assert result.nobs == result.regression.nobs
    assert result.df_resid == result.regression.df_resid
    assert result.inference_df == result.regression.inference_df
    assert result.vcov_method == result.regression.vcov_method


def test_twfe_result_effect_accessors_return_treatment_outputs():
    """Verify effect shortcuts return treatment coefficient inference."""

    result = twfe(_valid_data())

    assert result.effect_name == "treatment"
    assert result.effect == result.params["treatment"]
    assert result.effect_se == result.standard_errors["treatment"]
    assert result.effect_p_value == result.p_values["treatment"]
    pd.testing.assert_series_equal(
        result.effect_conf_int,
        result.conf_int.loc["treatment"],
    )


def test_summary_method_returns_summary_copy():
    """Verify mutating a returned summary does not change stored results."""

    result = twfe(_valid_data())

    summary = result.summary()
    summary.loc["treatment", "coef"] = 999

    assert result.regression.summary.loc["treatment", "coef"] != 999


def test_twfe_is_exported_from_package_root():
    """Verify users can import the public API from the package root."""

    from twfeiw import TWFEResult as RootTWFEResult
    from twfeiw import twfe as root_twfe

    result = root_twfe(_valid_data())

    assert isinstance(result, RootTWFEResult)
