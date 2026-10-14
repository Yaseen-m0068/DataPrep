
import pandas as pd
import pytest

from src.transformations.duplicates import remove_duplicate_rows


def test_remove_exact_duplicate_rows():
    df = pd.DataFrame({
        "Student_ID": ["S001", "S005", "S005"],
        "Name": ["Arun", "Meera", "Meera"],
        "CGPA": [8.2, 7.8, 7.8],
    })

    result = remove_duplicate_rows(df)

    assert len(result) == 2
    assert not result.duplicated().any()


def test_keep_first_duplicate_occurrence():
    df = pd.DataFrame({
        "Student_ID": ["S005", "S005"],
        "Name": ["Arun", "Arun"],
        "CGPA": [8.2, 8.2],
    })

    result = remove_duplicate_rows(df)

    assert len(result) == 1
    assert result.iloc[0]["Name"] == "Arun"


def test_original_dataframe_is_unchanged():
    df = pd.DataFrame({
        "Student_ID": ["S001", "S001"],
        "Name": ["Arun", "Arun"],
    })
    original = df.copy(deep=True)

    remove_duplicate_rows(df)

    pd.testing.assert_frame_equal(df, original)


def test_no_duplicates_returns_all_rows():
    df = pd.DataFrame({
        "Student_ID": ["S001", "S002"],
        "Name": ["Arun", "Meera"],
    })

    result = remove_duplicate_rows(df)

    assert len(result) == 2


def test_reject_non_dataframe():
    with pytest.raises(TypeError):
        remove_duplicate_rows([1, 2, 3])


def test_reject_empty_dataframe():
    with pytest.raises(ValueError):
        remove_duplicate_rows(pd.DataFrame())
