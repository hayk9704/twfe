import pandas as pd
import pytest

from twfeiw.data import PreparedPanel, prepare_panel

# check the existence of columns
# check the datatypes are correct
# check treatment is absorbing
# check event time is correctly calculated - plus the ever_treated column
# inclusion of columns that must be added later
# 

# a valid df
def _valid_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "time": [2000, 2001, 2002, 2000, 2001, 2002, 2000, 2001, 2002],
            "y": [1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 2.0, 2.5, 3.0],
            "treatment": [0, 1, 1, 0, 0, 1, 0, 0, 0],
        }
    )


def test_prepare_panel_returns_prepared_panel():
    panel = prepare_panel(_valid_data())

    assert isinstance(panel, PreparedPanel)         # checks whether the object is an instance of a given class
    assert panel.unit == "unit_id"
    assert panel.time == "time"
    assert panel.outcome == "y"
    assert panel.treatment == "treatment"


# making sure the prepare_panel doesn't change the original inputted df
def test_prepare_panel_does_not_mutate_original_dataframe():
    data = _valid_data()
    original = data.copy(deep=True)

    prepare_panel(data)

    pd.testing.assert_frame_equal(data, original)

# all columns are intact
def test_missing_required_column_raises_error():
    data = _valid_data().drop(columns="y")

# this structures says that when we run prepare_panel(data), it must raise the following error
    with pytest.raises(ValueError, match="data is missing required columns"):
        prepare_panel(data)

# NANs
def test_missing_required_values_raise_error():
    data = _valid_data()
    data.loc[0, "y"] = None

    with pytest.raises(ValueError, match="required columns cannot contain missing"):
        prepare_panel(data)


def test_non_numeric_outcome_raises_error():
    data = _valid_data()
    data["y"] = ["low"] * len(data)

    with pytest.raises(ValueError, match="outcome column must be numeric"):
        prepare_panel(data)


def test_non_integer_time_raises_error():
    data = _valid_data()
    data["time"] = data["time"].astype(float)

    with pytest.raises(ValueError, match="time column must contain integer-like"):
        prepare_panel(data)


def test_duplicate_unit_time_rows_raise_error():
    data = pd.concat([_valid_data(), _valid_data().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="one row per unit-time pair"):
        prepare_panel(data)


def test_non_binary_treatment_raises_error():
    data = _valid_data()
    data.loc[0, "treatment"] = 2

    with pytest.raises(ValueError, match="treatment must be binary"):
        prepare_panel(data)


def test_non_absorbing_treatment_raises_error():
    data = _valid_data()
    data.loc[data["unit_id"] == "A", "treatment"] = [0, 1, 0]

    with pytest.raises(ValueError, match="treatment must be absorbing"):
        prepare_panel(data)


def test_unbalanced_panel_is_allowed():
    data = _valid_data()
    data = data[~((data["unit_id"] == "B") & (data["time"] == 2001))]

    panel = prepare_panel(data)

    assert len(panel.data) == len(data)


def test_unit_code_is_added_while_original_unit_id_is_preserved():
    panel = prepare_panel(_valid_data())

    assert panel.unit_code_col in panel.data.columns
    assert panel.data.loc[panel.data["unit_id"] == "A", panel.unit_code_col].eq(0).all()
    assert panel.data.loc[panel.data["unit_id"] == "B", panel.unit_code_col].eq(1).all()
    assert panel.data.loc[panel.data["unit_id"] == "C", panel.unit_code_col].eq(2).all()
    assert panel.data["unit_id"].tolist() == ["A", "A", "A", "B", "B", "B", "C", "C", "C"]


def test_first_treat_time_is_computed():
    panel = prepare_panel(_valid_data())
    first_treat = panel.data.groupby("unit_id")[panel.first_treat_col].first()

    assert first_treat["A"] == 2001
    assert first_treat["B"] == 2002
    assert pd.isna(first_treat["C"])


def test_event_time_is_computed():
    panel = prepare_panel(_valid_data())

    a_event_time = panel.data.loc[
        panel.data["unit_id"] == "A", panel.event_time_col
    ].tolist()
    b_event_time = panel.data.loc[
        panel.data["unit_id"] == "B", panel.event_time_col
    ].tolist()

    assert a_event_time == [-1, 0, 1]
    assert b_event_time == [-2, -1, 0]


def test_never_treated_units_have_missing_first_treat_and_event_time():
    panel = prepare_panel(_valid_data())
    never_treated = panel.data[panel.data["unit_id"] == "C"]

    assert never_treated[panel.ever_treated_col].eq(False).all()
    assert never_treated[panel.first_treat_col].isna().all()
    assert never_treated[panel.event_time_col].isna().all()


def test_custom_column_names_are_supported():
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

    assert panel.unit == "firm_id"
    assert panel.time == "year"
    assert panel.outcome == "sales"
    assert panel.treatment == "treated"
