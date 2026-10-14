
import pandas as pd
import pytest

from src.transformations.categories import standardize_categories


def test_strip_whitespace():
    df = pd.DataFrame({"City": [" Kochi ", "Trivandrum "]})

    result = standardize_categories(df, "City")

    assert result["City"].tolist() == ["Kochi", "Trivandrum"]


def test_lowercase_categories():
    df = pd.DataFrame({"Gender": ["Male", "MALE", "Female"]})

    result = standardize_categories(
        df, "Gender", case_strategy="lower"
    )

    assert result["Gender"].tolist() == ["male", "male", "female"]


def test_uppercase_categories():
    df = pd.DataFrame({"City": ["Kochi", "trivandrum"]})

    result = standardize_categories(
        df, "City", case_strategy="upper"
    )

    assert result["City"].tolist() == ["KOCHI", "TRIVANDRUM"]


def test_preserve_case_by_default():
    df = pd.DataFrame({"City": ["Kochi", "kochi"]})

    result = standardize_categories(df, "City")

    assert result["City"].tolist() == ["Kochi", "kochi"]


def test_missing_values_remain_missing():
    df = pd.DataFrame({"City": [" Kochi ", None]})

    result = standardize_categories(
        df, "City", case_strategy="lower"
    )

    assert result["City"].iloc[0] == "kochi"
    assert pd.isna(result["City"].iloc[1])


def test_original_dataframe_is_unchanged():
    df = pd.DataFrame({"City": [" Kochi ", "KOCHI"]})
    original = df.copy(deep=True)

    standardize_categories(df, "City", case_strategy="lower")

    pd.testing.assert_frame_equal(df, original)


def test_reject_unknown_column():
    df = pd.DataFrame({"City": ["Kochi"]})

    with pytest.raises(ValueError, match="Column not found"):
        standardize_categories(df, "Gender")


def test_reject_invalid_case_strategy():
    df = pd.DataFrame({"City": ["Kochi"]})

    with pytest.raises(ValueError, match="case_strategy"):
        standardize_categories(df, "City", case_strategy="invalid")


def test_reject_non_text_column():
    df = pd.DataFrame({"Age": [21, 22]})

    with pytest.raises(ValueError, match="text or categorical"):
        standardize_categories(df, "Age")


def test_reject_non_dataframe():
    with pytest.raises(TypeError):
        standardize_categories(["Kochi"], "City")


def test_reject_empty_dataframe():
    with pytest.raises(ValueError, match="empty dataset"):
        standardize_categories(pd.DataFrame(), "City")
