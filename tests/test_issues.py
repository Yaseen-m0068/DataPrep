
import pandas as pd
import pytest

from src.detection.issues import (
    detect_missing_values,
    detect_duplicate_rows,
    detect_all_issues,
)


def test_detect_missing_values():
    df = pd.DataFrame({
        "Age": [20, None, 25, 30],
        "City": ["Kochi", "Kochi", "Trivandrum", "Kollam"],
        "CGPA": [8.0, 7.5, None, 9.0],
    })

    issues = detect_missing_values(df)

    assert len(issues) == 2

    age_issue = next(
        issue for issue in issues if issue["column"] == "Age"
    )
    assert age_issue["count"] == 1
    assert age_issue["percentage"] == 25.0
    assert age_issue["issue_type"] == "missing_values"


def test_no_missing_values():
    df = pd.DataFrame({"Age": [20, 25, 30]})

    assert detect_missing_values(df) == []


def test_reject_non_dataframe():
    with pytest.raises(TypeError):
        detect_missing_values([1, 2, 3])


def test_reject_empty_dataframe():
    with pytest.raises(ValueError, match="empty dataset"):
        detect_missing_values(pd.DataFrame())


def test_detect_duplicate_rows():
    df = pd.DataFrame({
        "Student_ID": ["S001", "S002", "S002"],
        "Name": ["Arun", "Meera", "Meera"],
    })

    issues = detect_duplicate_rows(df)

    assert len(issues) == 1
    assert issues[0]["issue_type"] == "duplicate_rows"
    assert issues[0]["count"] == 1
    assert issues[0]["percentage"] == 33.33


def test_no_duplicate_rows():
    df = pd.DataFrame({
        "Student_ID": ["S001", "S002", "S003"],
        "Name": ["Arun", "Meera", "Rahul"],
    })

    assert detect_duplicate_rows(df) == []


def test_duplicate_detection_rejects_non_dataframe():
    with pytest.raises(TypeError):
        detect_duplicate_rows([1, 2, 3])


def test_duplicate_detection_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty dataset"):
        detect_duplicate_rows(pd.DataFrame())


def test_detect_all_issues():
    df = pd.DataFrame({
        "Age": [20, None, 25, 25],
        "City": ["Kochi", "Kochi", "Trivandrum", "Trivandrum"],
    })

    issues = detect_all_issues(df)

    issue_types = {issue["issue_type"] for issue in issues}

    assert "missing_values" in issue_types
    assert "duplicate_rows" in issue_types


def test_detect_all_issues_on_clean_data():
    df = pd.DataFrame({
        "Age": [20, 25, 30],
        "City": ["Kochi", "Trivandrum", "Kollam"],
    })

    assert detect_all_issues(df) == []
