
import pandas as pd
import pytest

from src.detection.issues import (
    detect_missing_values,
    detect_duplicate_rows,
    detect_all_issues,
    detect_inconsistent_categories,
    detect_numeric_outliers,
    detect_constant_columns,
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


def test_detect_inconsistent_categories():
    df = pd.DataFrame({
        "Gender": ["Male", "male", "MALE", "Female"],
    })

    issues = detect_inconsistent_categories(df)

    assert len(issues) == 1
    assert issues[0]["issue_type"] == "inconsistent_categories"
    assert issues[0]["column"] == "Gender"
    assert issues[0]["count"] == 3
    assert issues[0]["variants"] == [["MALE", "Male", "male"]]


def test_detect_whitespace_inconsistency():
    df = pd.DataFrame({
        "City": ["Kochi", " Kochi", "Trivandrum"],
    })

    issues = detect_inconsistent_categories(df)

    assert len(issues) == 1
    assert issues[0]["variants"] == [[" Kochi", "Kochi"]]


def test_no_inconsistent_categories():
    df = pd.DataFrame({
        "City": ["Kochi", "Trivandrum", "Kollam"],
    })

    assert detect_inconsistent_categories(df) == []


def test_inconsistent_category_detection_rejects_non_dataframe():
    with pytest.raises(TypeError):
        detect_inconsistent_categories(["Male", "male"])


def test_inconsistent_category_detection_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty dataset"):
        detect_inconsistent_categories(pd.DataFrame())


def test_detect_numeric_outliers():
    df = pd.DataFrame({
        "Score": [10, 12, 13, 14, 100],
    })

    issues = detect_numeric_outliers(df)

    assert len(issues) == 1
    assert issues[0]["issue_type"] == "numeric_outliers"
    assert issues[0]["column"] == "Score"
    assert issues[0]["outlier_values"] == [100]
    assert issues[0]["method"] == "IQR"


def test_no_numeric_outliers():
    df = pd.DataFrame({
        "Score": [10, 11, 12, 13, 14],
    })

    assert detect_numeric_outliers(df) == []


def test_numeric_outlier_detection_skips_identifiers():
    df = pd.DataFrame({
        "Customer_ID": [1001, 1002, 1003, 1004, 1005],
    })

    assert detect_numeric_outliers(df) == []


def test_numeric_outlier_detection_skips_small_samples():
    df = pd.DataFrame({
        "Score": [10, 12, 13],
    })

    assert detect_numeric_outliers(df) == []


def test_numeric_outlier_detection_rejects_non_dataframe():
    with pytest.raises(TypeError):
        detect_numeric_outliers([10, 12, 13, 100])


def test_numeric_outlier_detection_rejects_empty_dataframe():
    with pytest.raises(ValueError, match="empty dataset"):
        detect_numeric_outliers(pd.DataFrame())


def test_value_on_upper_iqr_boundary_is_not_outlier():
    df = pd.DataFrame({
        "Score": [8.2, 7.8, 8.5, 8.2, 8.2, 9.1, 7.5, 3.2],
    })

    issues = detect_numeric_outliers(df)

    assert len(issues) == 1
    assert issues[0]["outlier_values"] == [3.2]


def test_detect_constant_string_column():
    df = pd.DataFrame({
        "Country": ["India", "India", "India", "India"],
    })

    issues = detect_constant_columns(df)

    assert len(issues) == 1
    assert issues[0]["column"] == "Country"
    assert issues[0]["issue_type"] == "constant_column"
    assert issues[0]["count"] == 4
    assert issues[0]["constant_value"] == "India"


def test_detect_constant_numeric_column():
    df = pd.DataFrame({
        "Score": [10, 10, 10, 10],
    })

    issues = detect_constant_columns(df)

    assert len(issues) == 1
    assert issues[0]["constant_value"] == "10"


def test_constant_column_with_missing_values():
    df = pd.DataFrame({
        "Status": ["Active", None, "Active", "Active"],
    })

    issues = detect_constant_columns(df)

    assert len(issues) == 1
    assert issues[0]["count"] == 3
    assert issues[0]["percentage"] == 75.0


def test_no_constant_columns():
    df = pd.DataFrame({
        "City": ["Kochi", "Trivandrum", "Kollam"],
        "Score": [10, 20, 30],
    })

    assert detect_constant_columns(df) == []


def test_all_missing_column_is_not_constant():
    df = pd.DataFrame({
        "Notes": [None, None, None],
    })

    assert detect_constant_columns(df) == []


def test_constant_detection_rejects_invalid_inputs():
    with pytest.raises(TypeError):
        detect_constant_columns([1, 1, 1])

    with pytest.raises(ValueError, match="empty dataset"):
        detect_constant_columns(pd.DataFrame())
