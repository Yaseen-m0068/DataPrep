
import pandas as pd

from src.detection.column_types import infer_semantic_type


def test_numeric_column():
    series = pd.Series([10, 20, 30])

    result = infer_semantic_type(series, "Age")

    assert result["semantic_type"] == "numerical"


def test_identifier_name():
    series = pd.Series(["S001", "S002", "S003"])

    result = infer_semantic_type(series, "Student_ID")

    assert result["semantic_type"] == "identifier"


def test_date_column():
    series = pd.Series([
        "2025-01-10",
        "2025-02-15",
        "2025-03-20",
    ])

    result = infer_semantic_type(series, "Join_Date")

    assert result["semantic_type"] == "datetime_candidate"


def test_boolean_like_column():
    series = pd.Series(["Yes", "No", "Yes", "No"])

    result = infer_semantic_type(series, "Internship")

    assert result["semantic_type"] == "boolean_candidate"


def test_categorical_column():
    series = pd.Series([
        "Kochi", "Trivandrum", "Kochi", "Kollam"
    ])

    result = infer_semantic_type(series, "City")

    assert result["semantic_type"] == "categorical"


def test_empty_column():
    series = pd.Series([None, None])

    result = infer_semantic_type(series, "Unknown")

    assert result["semantic_type"] == "unknown"


def test_numeric_identifier_name():
    series = pd.Series([1001, 1002, 1003])

    result = infer_semantic_type(series, "Customer_ID")

    assert result["semantic_type"] == "identifier"


def test_date_name_with_unparseable_values():
    series = pd.Series(["unknown", "not available", "pending"])

    result = infer_semantic_type(series, "Join_Date")

    assert result["semantic_type"] == "datetime_candidate"


def test_repeated_boolean_values():
    series = pd.Series(["Yes", "No", "Yes", "Yes", "No"])

    result = infer_semantic_type(series, "Active")

    assert result["semantic_type"] == "boolean_candidate"
