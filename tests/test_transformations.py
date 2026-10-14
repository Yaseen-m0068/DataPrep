
import pandas as pd
import pytest

from src.transformations.missing_values import handle_missing_values


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Age": [20.0, None, 30.0, 40.0],
        "City": ["Kochi", None, "Kochi", "Trivandrum"],
    })


def test_fill_median(sample_df):
    result = handle_missing_values(sample_df, "Age", "fill_median")

    assert result["Age"].isna().sum() == 0
    assert result.loc[1, "Age"] == 30.0
    assert sample_df["Age"].isna().sum() == 1


def test_fill_mean(sample_df):
    result = handle_missing_values(sample_df, "Age", "fill_mean")

    assert result.loc[1, "Age"] == 30.0


def test_fill_mode(sample_df):
    result = handle_missing_values(sample_df, "City", "fill_mode")

    assert result.loc[1, "City"] == "Kochi"


def test_drop_rows(sample_df):
    result = handle_missing_values(sample_df, "Age", "drop_rows")

    assert len(result) == 3
    assert result["Age"].isna().sum() == 0
    assert len(sample_df) == 4


def test_original_dataframe_is_unchanged(sample_df):
    original = sample_df.copy(deep=True)

    handle_missing_values(sample_df, "Age", "fill_median")

    pd.testing.assert_frame_equal(sample_df, original)


def test_reject_non_numeric_mean(sample_df):
    with pytest.raises(ValueError, match="numerical column"):
        handle_missing_values(sample_df, "City", "fill_mean")


def test_reject_unknown_column(sample_df):
    with pytest.raises(ValueError, match="Column not found"):
        handle_missing_values(sample_df, "Salary", "fill_median")


def test_reject_unknown_strategy(sample_df):
    with pytest.raises(ValueError, match="Unsupported strategy"):
        handle_missing_values(sample_df, "Age", "automatic")



def test_reject_column_without_missing_values(sample_df):
    sample_df["Complete"] = [1, 2, 3, 4]

    with pytest.raises(ValueError, match="no missing values"):
        handle_missing_values(
            sample_df,
            "Complete",
            "fill_median",
        )
