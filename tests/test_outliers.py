
import pandas as pd
import pytest

from src.transformations.outliers import handle_numeric_outliers


def test_cap_iqr_limits_extreme_values():
    df = pd.DataFrame({"Score": [10, 12, 13, 14, 15, 100]})

    result = handle_numeric_outliers(df, "Score", "cap_iqr")

    assert result["Score"].max() == pytest.approx(18.5)
    assert df["Score"].max() == 100


def test_drop_rows_removes_outliers():
    df = pd.DataFrame({"Score": [10, 12, 13, 14, 15, 100]})

    result = handle_numeric_outliers(df, "Score", "drop_rows")

    assert 100 not in result["Score"].values
    assert len(result) == 5


def test_cap_iqr_preserves_missing_values():
    df = pd.DataFrame({"Score": [10, 12, 13, 14, 15, 100, None]})

    result = handle_numeric_outliers(df, "Score", "cap_iqr")

    assert result["Score"].isna().sum() == 1
    assert result["Score"].max() == pytest.approx(18.5)


def test_drop_rows_preserves_missing_values():
    df = pd.DataFrame({"Score": [10, 12, 13, 14, 15, 100, None]})

    result = handle_numeric_outliers(df, "Score", "drop_rows")

    assert result["Score"].isna().sum() == 1
    assert 100 not in result["Score"].values


def test_original_dataframe_is_unchanged():
    df = pd.DataFrame({"Score": [10, 12, 13, 14, 15, 100]})
    original = df.copy(deep=True)

    handle_numeric_outliers(df, "Score", "cap_iqr")

    pd.testing.assert_frame_equal(df, original)


def test_no_outliers_returns_unchanged_values():
    df = pd.DataFrame({"Score": [10, 11, 12, 13, 14]})

    result = handle_numeric_outliers(df, "Score", "cap_iqr")

    pd.testing.assert_frame_equal(result, df)


def test_missing_column_raises_error():
    df = pd.DataFrame({"Score": [10, 20, 30]})

    with pytest.raises(ValueError, match="Column not found"):
        handle_numeric_outliers(df, "Age", "cap_iqr")


def test_non_numeric_column_raises_error():
    df = pd.DataFrame({"Name": ["Arun", "Meera", "Rahul"]})

    with pytest.raises(ValueError, match="numerical"):
        handle_numeric_outliers(df, "Name", "cap_iqr")


def test_invalid_strategy_raises_error():
    df = pd.DataFrame({"Score": [10, 20, 30]})

    with pytest.raises(ValueError, match="Unsupported strategy"):
        handle_numeric_outliers(df, "Score", "replace_with_mean")


def test_non_dataframe_input_raises_error():
    with pytest.raises(TypeError, match="pandas DataFrame"):
        handle_numeric_outliers([10, 20, 100], "Score", "cap_iqr")


def test_empty_dataframe_raises_error():
    df = pd.DataFrame({"Score": []})

    with pytest.raises(ValueError, match="empty dataset"):
        handle_numeric_outliers(df, "Score", "cap_iqr")


def test_all_missing_numeric_column_raises_error():
    df = pd.DataFrame({"Score": [None, None, None]})

    with pytest.raises(ValueError, match="observed numeric values"):
        handle_numeric_outliers(df, "Score", "cap_iqr")
