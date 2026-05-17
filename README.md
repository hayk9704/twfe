# twfeiw

Transparent two-way fixed effects (TWFE) and interaction-weighted event-study
estimators for Python.

This package is currently in early development. The immediate goal is to build
the standard TWFE workflow from first principles before extending it to TWFE
event-study and Sun-Abraham interaction-weighted specifications.

## Current Status

The package currently has four working internal layers:

```text
raw DataFrame
    ->
prepare_panel()
    ->
PreparedPanel
    ->
build_twfe_design()
    ->
DesignMatrix
    ->
fit_ols()
    ->
OLSResult
    ->
compute_vcov()
    ->
VCovResult
```

These pieces are intentionally modular. Each layer has one job:

- `data.py` validates and prepares the raw panel data.
- `design.py` builds the regression outcome vector and regressor matrix.
- `ols.py` estimates OLS coefficients from a prepared design matrix.
- `vcov.py` computes variance-covariance matrices and standard errors.

The public high-level estimator function, such as `twfe(df)`, is not implemented
yet. For now, the package is used through the lower-level building blocks.

## Input Data Contract

The first version expects a pandas DataFrame with these required columns:

```text
unit_id
time
y
treatment
```

The meanings are:

- `unit_id`: unit or entity identifier. Strings and integers are allowed.
- `time`: integer-like discrete time period.
- `y`: numeric outcome variable.
- `treatment`: binary treatment indicator with values `0` and `1`.

The current treatment setup assumes staggered adoption:

```text
valid:   0, 0, 1, 1
invalid: 0, 1, 0, 1
```

That is, once a unit becomes treated, it must remain treated in later observed
periods.

Unbalanced panels are allowed. Missing unit-time pairs are fine. Duplicate
observed `unit_id x time` rows are not allowed.

## Implemented Pieces

### 1. Data Preparation

`prepare_panel()` validates the raw DataFrame and adds internal columns needed by
later estimators.

```python
from twfeiw.data import prepare_panel

panel = prepare_panel(df)
```

The returned object is a `PreparedPanel` dataclass.

Important fields include:

```python
panel.data
panel.unit
panel.time
panel.outcome
panel.treatment
panel.unit_code_col
panel.first_treat_col
panel.event_time_col
panel.ever_treated_col
```

The prepared DataFrame preserves the original `unit_id` and adds internal
columns:

```text
_twfeiw_unit_code
_twfeiw_first_treat_time
_twfeiw_event_time
_twfeiw_ever_treated
```

The internal unit code is used so later numerical routines can work with compact
integer unit identifiers without modifying the user's original IDs.

### 2. TWFE Dummy-Variable Design

`build_twfe_design()` takes a `PreparedPanel` and builds the standard TWFE
dummy-variable design.

```python
from twfeiw.design import build_twfe_design

design = build_twfe_design(panel)
```

The model represented is:

```text
y_it = alpha_i + lambda_t + beta * treatment_it + error_it
```

Internally, this is written as a dummy-variable OLS regression:

```text
y_it =
    alpha
    + beta * treatment_it
    + unit fixed-effect dummies
    + time fixed-effect dummies
    + error_it
```

The returned object is a `DesignMatrix` dataclass.

Important fields include:

```python
design.outcome
design.regressors
design.model
design.outcome_variable
design.effect_cols
design.fixed_effect_cols
design.regressor_cols
design.reference_unit
design.reference_time
```

For standard TWFE:

```python
design.effect_cols == ["treatment"]
```

The regressor matrix contains:

```text
const
treatment
unit fixed-effect dummies
time fixed-effect dummies
```

The first sorted unit code and first sorted time period are omitted as reference
categories.

### 3. OLS Point Estimation

`fit_ols()` takes a `DesignMatrix` and estimates OLS coefficients.

```python
from twfeiw.ols import fit_ols

ols_result = fit_ols(design)
```

The returned object is an `OLSResult` dataclass.

Important fields include:

```python
ols_result.params
ols_result.fitted_values
ols_result.residuals
ols_result.nobs
ols_result.rank
ols_result.df_resid
ols_result.rss
ols_result.regressor_cols
```

`params` is a labeled pandas Series. For example:

```python
ols_result.params["treatment"]
```

returns the standard TWFE treatment coefficient from the current design.

The OLS layer uses `numpy.linalg.lstsq` and checks that the regressor matrix has
full column rank. If the design is rank deficient, it raises an error instead of
silently returning unidentified coefficients.

### 4. Variance-Covariance And Standard Errors

`compute_vcov()` takes a `DesignMatrix` and an `OLSResult`, then computes a
labeled variance-covariance matrix and standard errors.

```python
from twfeiw.vcov import compute_vcov

vcov_result = compute_vcov(design, ols_result, method="classical")
```

The returned object is a `VCovResult` dataclass.

Important fields include:

```python
vcov_result.vcov
vcov_result.standard_errors
vcov_result.method
vcov_result.nobs
vcov_result.rank
vcov_result.df_resid
vcov_result.small_sample
vcov_result.n_clusters
vcov_result.cluster_name
```

The currently supported methods are:

```text
classical
HC1
cluster
```

For unit-clustered standard errors, pass one cluster label per regression row:

```python
vcov_result = compute_vcov(
    design,
    ols_result,
    method="cluster",
    clusters=panel.data[panel.unit_code_col],
)
```

Then the standard error for the TWFE treatment coefficient is:

```python
vcov_result.standard_errors["treatment"]
```

This layer only computes covariance matrices and standard errors. P-values,
confidence intervals, and formatted result tables are planned for `results.py`.

## Example Current Workflow

```python
from twfeiw.data import prepare_panel
from twfeiw.design import build_twfe_design
from twfeiw.ols import fit_ols
from twfeiw.vcov import compute_vcov

panel = prepare_panel(df)
design = build_twfe_design(panel)
ols_result = fit_ols(design)
vcov_result = compute_vcov(
    design,
    ols_result,
    method="cluster",
    clusters=panel.data[panel.unit_code_col],
)

twfe_effect = ols_result.params["treatment"]
twfe_se = vcov_result.standard_errors["treatment"]
```

This is the current low-level workflow. A future public function will wrap these
steps.

## Intended Final Workflow

The intended user-facing API will eventually look more like this:

```python
from twfeiw import twfe

result = twfe(df)
result.effect
result.summary()
```

Internally, that high-level function should still use the same modular pipeline:

```text
prepare_panel()
build_twfe_design()
fit_ols()
compute standard errors
return TWFEResult
```

## Planned Architecture

The standard TWFE implementation is expected to grow into these modules:

```text
data.py
    Validate raw panel data and compute treatment timing.

design.py
    Build regression matrices for TWFE, event-study, and Sun-Abraham designs.

ols.py
    Estimate OLS point estimates from a design matrix.

vcov.py
    Compute variance-covariance matrices and standard errors.

results.py
    Store user-facing model results, summaries, confidence intervals, and
    effect-specific outputs.

estimators.py
    Provide public functions such as twfe(), event_study(), and sun_abraham().
```

## Planned Extensions

### TWFE Event Study

The event-study specification will replace the single treatment column with
event-time indicators:

```text
y_it =
    alpha_i
    + lambda_t
    + sum_k beta_k * 1[event_time_it = k]
    + error_it
```

The existing `DesignMatrix` structure already separates:

```text
effect_cols
fixed_effect_cols
regressor_cols
```

That makes it possible for a future event-study design to use many effect
columns instead of just `treatment`.

### Sun-Abraham Interaction-Weighted Event Study

The Sun-Abraham design will use cohort-by-event-time interactions:

```text
y_it =
    alpha_i
    + lambda_t
    + sum_g sum_k theta_gk
        * 1[first_treat_time_i = g]
        * 1[event_time_it = k]
    + error_it
```

After estimating cohort-event coefficients, a later aggregation layer will
compute interaction-weighted event-study effects.

## Testing

Run all tests from the project root:

```powershell
.venv\Scripts\python.exe -m pytest
```

Run only one test file:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_ols.py
```

Run the standard-error tests:

```powershell
.venv\Scripts\python.exe -m pytest tests\test_vcov.py
```

Run Ruff checks:

```powershell
.venv\Scripts\python.exe -m ruff check src tests
```

Current tests cover:

- data validation and treatment-timing construction
- TWFE dummy-variable design construction
- OLS point estimation
- classical, HC1, and one-way clustered standard errors
- agreement of the TWFE treatment coefficient with a `statsmodels` formula
  regression using `y ~ treatment + C(unit_id) + C(time)`

## Development Notes

This package is intentionally being built in small layers. The current priority
is correctness and transparency for the standard TWFE estimator. Performance
optimizations, sparse matrices, residualization, event-study plotting, and
public result summaries are planned later.
