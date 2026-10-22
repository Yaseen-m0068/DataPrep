
import pandas as pd
import pytest

from src.transformations.encoding import encode_categorical_columns


def test_onehot_encoding_creates_indicator_columns():
    df = pd.DataFrame({
        "City": ["Kochi", "Trivandrum", "Kochi"],
        "Age": [22, 25, 30],
    })

    result = encode_categorical_columns(df, ["City"])

    assert "City" not in result.columns
    assert "City__Kochi" in result.columns
    assert "City__Trivandrum" in result.columns
    assert result["Age"].tolist() == [22, 25, 30]
    assert result["City__Kochi"].tolist() == [1, 0, 1]


def test_onehot_encoding_preserves_missing_as_all_zero_by_default():
    df = pd.DataFrame({"City": ["Kochi", None, "Trivandrum"]})

    result = encode_categorical_columns(df, ["City"])

    dummy_columns = [column for column in result.columns if column.startswith("City__")]

    assert result.loc[1, dummy_columns].sum() == 0


def test_onehot_encoding_can_create_missing_indicator():
    df = pd.DataFrame({"City": ["Kochi", None, "Trivandrum"]})

    result = encode_categorical_columns(
        df, ["City"], include_missing=True
    )

    assert "City__nan" in result.columns
    assert result.loc[1, "City__nan"] == 1


def test_onehot_encoding_supports_multiple_columns():
    df = pd.DataFrame({
        "City": ["Kochi", "Trivandrum"],
        "Size": ["Small", "Large"],
    })

    result = encode_categorical_columns(df, ["City", "Size"])

    assert "City" not in result.columns
    assert "Size" not in result.columns
    assert "City__Kochi" in result.columns
    assert "Size__Small" in result.columns


def test_ordinal_encoding_uses_user_defined_order():
    df = pd.DataFrame({"Level": ["Medium", "Low", "High"]})

    result = encode_categorical_columns(
        df,
        ["Level"],
        strategy="ordinal",
        ordinal_mappings={"Level": ["Low", "Medium", "High"]},
    )

    assert result["Level"].tolist() == [1, 0, 2]
    assert str(result["Level"].dtype) == "Int64"


def test_ordinal_encoding_preserves_missing_values():
    df = pd.DataFrame({"Level": ["Low", None, "High"]})

    result = encode_categorical_columns(
        df,
        ["Level"],
        strategy="ordinal",
        ordinal_mappings={"Level": ["Low", "High"]},
    )

    assert result["Level"].iloc[0] == 0
    assert pd.isna(result["Level"].iloc[1])
    assert result["Level"].iloc[2] == 1


def test_ordinal_encoding_rejects_unknown_categories():
    df = pd.DataFrame({"Level": ["Low", "Very High"]})

    with pytest.raises(ValueError, match="missing from its ordinal mapping"):
        encode_categorical_columns(
            df,
            ["Level"],
            strategy="ordinal",
            ordinal_mappings={"Level": ["Low", "High"]},
        )


def test_encoding_rejects_missing_column():
    df = pd.DataFrame({"City": ["Kochi"]})

    with pytest.raises(ValueError, match="Column not found"):
        encode_categorical_columns(df, ["Country"])


def test_encoding_rejects_numeric_columns():
    df = pd.DataFrame({"Age": [20, 30, 40]})

    with pytest.raises(ValueError, match="categorical values"):
        encode_categorical_columns(df, ["Age"])


def test_encoding_does_not_modify_original_dataframe():
    df = pd.DataFrame({"City": ["Kochi", "Trivandrum"]})
    original = df.copy(deep=True)

    encode_categorical_columns(df, ["City"])

    pd.testing.assert_frame_equal(df, original)


def test_encoding_rejects_duplicate_column_selection():
    df = pd.DataFrame({"City": ["Kochi", "Trivandrum"]})

    with pytest.raises(ValueError, match="Duplicate column selections"):
        encode_categorical_columns(df, ["City", "City"])


def test_encoding_rejects_invalid_strategy():
    df = pd.DataFrame({"City": ["Kochi", "Trivandrum"]})

    with pytest.raises(ValueError, match="strategy"):
        encode_categorical_columns(df, ["City"], strategy="label")


def test_encoding_rejects_duplicate_ordinal_categories():
    df = pd.DataFrame({"Level": ["Low", "High"]})

    with pytest.raises(ValueError, match="duplicate categories"):
        encode_categorical_columns(
            df,
            ["Level"],
            strategy="ordinal",
            ordinal_mappings={"Level": ["Low", "Low", "High"]},
        )


def test_encoding_rejects_existing_column_name_collision():
    df = pd.DataFrame({
        "City": ["Kochi", "Trivandrum"],
        "City__Kochi": [10, 20],
    })

    with pytest.raises(ValueError, match="duplicate column names"):
        encode_categorical_columns(df, ["City"])
