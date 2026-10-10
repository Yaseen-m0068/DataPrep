
import pandas as pd
import pytest

from src.detection.issues import detect_missing_values


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
