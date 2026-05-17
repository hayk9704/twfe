from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from twfeiw.data import prepare_panel
from twfeiw.design import DesignMatrix, build_twfe_design
from twfeiw.ols import OLSResult, fit_ols
from twfeiw.vcov import (
    VCovResult,
    compute_classical_vcov,
    compute_clustered_vcov,
    compute_heteroskedastic_vcov,
    compute_vcov,
)


def _valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "time": [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002],
            "y": [1.0, 1.7, 2.1, 3.2, 3.4, 4.3, 2.1, 2.6, 2.9],
            "treatment": [0, 1, 1, 0, 0, 1, 0, 0, 0],
        }
    )


def _twfe_parts() -> tuple[pd.DataFrame, DesignMatrix, OLSResult, pd.Series]:
    data = _valid_data()
    panel = prepare_panel(data)
    design = build_twfe_design(panel)
    result = fit_ols(design)
    clusters = panel.data[panel.unit_code_col]
    return data, design, result, clusters


def _statsmodels_fit(data: pd.DataFrame):
    return smf.ols(
        "y ~ treatment + C(unit_id) + C(time)",
        data=data,
    ).fit()

# returns the diagonal element of the var covar matrix for the given element of beta
# from the statsmodels ols results
def _statsmodels_vcov_value(statsmodels_result, variable: str) -> float:
    names = statsmodels_result.model.exog_names
    index = names.index(variable)
    cov = statsmodels_result.cov_params()
    if isinstance(cov, pd.DataFrame):
        return float(cov.loc[variable, variable])
    return float(cov[index, index])

# assert that the return object is a VCovResult
def test_compute_vcov_returns_vcov_result():
    _, design, result, _ = _twfe_parts()

    vcov_result = compute_vcov(design, result)

    assert isinstance(vcov_result, VCovResult)

# both vcov indices and columns must be the regressor names
def test_vcov_and_standard_errors_are_labeled_by_regressor_columns():
    _, design, result, _ = _twfe_parts()

    vcov_result = compute_vcov(design, result)

    assert vcov_result.vcov.index.tolist() == design.regressor_cols
    assert vcov_result.vcov.columns.tolist() == design.regressor_cols
    assert vcov_result.standard_errors.index.tolist() == design.regressor_cols

# standard errors must be the square root of the diagonal of the covariance matrix
def test_standard_errors_are_square_root_of_vcov_diagonal():
    _, design, result, _ = _twfe_parts()

    vcov_result = compute_vcov(design, result)

    expected = np.sqrt(np.diag(vcov_result.vcov.to_numpy()))
    np.testing.assert_allclose(vcov_result.standard_errors.to_numpy(), expected)

# check vcov metadata correctness
def test_vcov_metadata_is_stored():
    _, design, result, _ = _twfe_parts()

    vcov_result = compute_vcov(design, result)

    assert vcov_result.method == "classical"
    assert vcov_result.nobs == result.nobs
    assert vcov_result.rank == result.rank
    assert vcov_result.df_resid == result.df_resid
    assert vcov_result.small_sample is True
    assert vcov_result.n_clusters is None


def test_classical_treatment_variance_matches_statsmodels():
    data, design, result, _ = _twfe_parts()
    statsmodels_fit = _statsmodels_fit(data)

    vcov_result = compute_vcov(design, result, method="classical")

    assert vcov_result.vcov.loc["treatment", "treatment"] == pytest.approx(
        _statsmodels_vcov_value(statsmodels_fit, "treatment")
    )
    assert vcov_result.standard_errors["treatment"] == pytest.approx(
        statsmodels_fit.bse["treatment"]
    )

# test that the classical one returns the method as "classical"
def test_classical_helper_returns_classical_method():
    _, design, result, _ = _twfe_parts()

    vcov_result = compute_classical_vcov(design, result)

    assert vcov_result.method == "classical"


def test_hc1_treatment_variance_matches_statsmodels():
    data, design, result, _ = _twfe_parts()
    statsmodels_fit = _statsmodels_fit(data)
    statsmodels_robust = statsmodels_fit.get_robustcov_results(cov_type="HC1")

    vcov_result = compute_vcov(design, result, method="HC1")

    assert vcov_result.vcov.loc["treatment", "treatment"] == pytest.approx(
        _statsmodels_vcov_value(statsmodels_robust, "treatment")
    )


def test_hc1_helper_returns_hc1_method():
    _, design, result, _ = _twfe_parts()

    vcov_result = compute_heteroskedastic_vcov(design, result, kind="HC1")

    assert vcov_result.method == "HC1"


def test_unsupported_heteroskedastic_kind_raises_error():
    _, design, result, _ = _twfe_parts()

    with pytest.raises(ValueError, match="unsupported heteroskedastic vcov kind"):
        compute_heteroskedastic_vcov(design, result, kind="HC0")


def test_clustered_treatment_variance_matches_statsmodels():
    data, design, result, clusters = _twfe_parts()
    statsmodels_fit = _statsmodels_fit(data)
    statsmodels_clustered = statsmodels_fit.get_robustcov_results(
        cov_type="cluster",
        groups=clusters,
        use_correction=True,
    )

    vcov_result = compute_vcov(
        design,
        result,
        method="cluster",
        clusters=clusters,
    )

    assert vcov_result.vcov.loc["treatment", "treatment"] == pytest.approx(
        _statsmodels_vcov_value(statsmodels_clustered, "treatment")
    )
    assert vcov_result.standard_errors["treatment"] == pytest.approx(
        np.sqrt(_statsmodels_vcov_value(statsmodels_clustered, "treatment"))
    )


def test_clustered_metadata_is_stored():
    _, design, result, clusters = _twfe_parts()

    vcov_result = compute_clustered_vcov(
        design,
        result,
        clusters=clusters.rename("unit_cluster"),
    )

    assert vcov_result.method == "cluster"
    assert vcov_result.n_clusters == 3
    assert vcov_result.cluster_name == "unit_cluster"
    assert vcov_result.small_sample is True


def test_clustered_small_sample_correction_can_be_disabled():
    _, design, result, clusters = _twfe_parts()

    corrected = compute_vcov(
        design,
        result,
        method="cluster",
        clusters=clusters,
        small_sample=True,
    )
    uncorrected = compute_vcov(
        design,
        result,
        method="cluster",
        clusters=clusters,
        small_sample=False,
    )

    assert corrected.small_sample is True
    assert uncorrected.small_sample is False
    assert corrected.vcov.loc["treatment", "treatment"] != pytest.approx(
        uncorrected.vcov.loc["treatment", "treatment"]
    )


def test_non_design_matrix_input_raises_error():
    _, _, result, _ = _twfe_parts()

    with pytest.raises(TypeError, match="design must be a DesignMatrix"):
        compute_vcov("not a design", result)


def test_non_ols_result_input_raises_error():
    _, design, _, _ = _twfe_parts()

    with pytest.raises(TypeError, match="result must be an OLSResult"):
        compute_vcov(design, "not a result")


def test_unsupported_vcov_method_raises_error():
    _, design, result, _ = _twfe_parts()

    with pytest.raises(ValueError, match="unsupported vcov method"):
        compute_vcov(design, result, method="HC0")


def test_cluster_method_without_clusters_raises_error():
    _, design, result, _ = _twfe_parts()

    with pytest.raises(ValueError, match="clusters are required"):
        compute_vcov(design, result, method="cluster")


def test_cluster_length_mismatch_raises_error():
    _, design, result, _ = _twfe_parts()

    with pytest.raises(ValueError, match="same length"):
        compute_vcov(design, result, method="cluster", clusters=[0, 1])


def test_missing_cluster_labels_raise_error():
    _, design, result, clusters = _twfe_parts()
    clusters = clusters.astype("float")
    clusters.iloc[0] = np.nan

    with pytest.raises(ValueError, match="clusters cannot contain missing values"):
        compute_vcov(design, result, method="cluster", clusters=clusters)


def test_fewer_than_two_clusters_raises_error():
    _, design, result, _ = _twfe_parts()

    with pytest.raises(ValueError, match="at least two clusters"):
        compute_vcov(design, result, method="cluster", clusters=[1] * result.nobs)


def test_mismatched_regressor_columns_raise_error():
    _, design, result, _ = _twfe_parts()
    result = replace(result, regressor_cols=["bad"] * len(result.regressor_cols))

    with pytest.raises(ValueError, match="result regressor columns"):
        compute_vcov(design, result)
