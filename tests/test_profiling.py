
import pandas as pd
import pytest

from src.profiling.profile import profile_dataset


def test_profile_basic_dataset():
    df = pd.DataFrame({
        "Age": [20, 25, None, 30],
        "City": ["Kochi", "Kochi", "Trivandrum", "Kochi"],
    })

    profile = profile_dataset(df)

    assert profile["row_count"] == 4
    assert profile["column_count"] == 2
    assert profile["columns"]["Age"]["missing_count"] == 1
    assert profile["columns"]["Age"]["missing_percentage"] == 25.0
    assert profile["columns"]["City"]["unique_count"] == 2


def test_profile_duplicate_rows():
    df = pd.DataFrame({
        "Name": ["Arun", "Meera", "Arun"],
        "Age": [21, 22, 21],
    })

    profile = profile_dataset(df)

    assert profile["duplicate_rows"] == 1


def test_profile_rejects_non_dataframe():
    with pytest.raises(TypeError):
        profile_dataset([1, 2, 3])


def test_profile_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty dataset"):
        profile_dataset(pd.DataFrame())


def test_profile_includes_semantic_types():
    df = pd.DataFrame({
        "Age": [20, 25, 30],
        "City": ["Kochi", "Trivandrum", "Kochi"],
        "Join_Date": [
            "2025-01-10",
            "2025-02-15",
            "2025-03-20",
        ],
    })

    profile = profile_dataset(df)

    assert (
        profile["columns"]["Age"]["semantic_type"]
        == "numerical"
    )
    assert (
        profile["columns"]["City"]["semantic_type"]
        == "categorical"
    )
    assert (
        profile["columns"]["Join_Date"]["semantic_type"]
        == "datetime_candidate"
    )


def test_profile_includes_quality_issues():
    df = pd.DataFrame({
        "Age": [20, None, 25, 25],
        "City": ["Kochi", "Kochi", "Trivandrum", "Trivandrum"],
    })

    profile = profile_dataset(df)

    issue_types = {
        issue["issue_type"]
        for issue in profile["quality_issues"]
    }

    assert "missing_values" in issue_types
    assert "duplicate_rows" in issue_types
