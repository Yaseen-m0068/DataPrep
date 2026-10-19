import numpy as np
import pandas as pd
import pytest

from src.transformations.scaling import scale_numeric_columns


def test_standard_scaling_has_zero_mean_and_unit_population_std():
    df = pd.DataFrame({"value": [1, 2, 3], "label": ["a", "b", "c"]})
    result = scale_numeric_columns(df, ["value"], strategy="standard")
    assert result["value"].mean() == pytest.approx(0.0)
    assert result["value"].std(ddof=0) == pytest.approx(1.0)
    assert result["label"].equals(df["label"])


def test_minmax_scaling_maps_values_to_zero_one():
    df = pd.DataFrame({"value": [10, 20, 30]})
    result = scale_numeric_columns(df, ["value"], strategy="minmax")
    assert result["value"].tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_scaling_preserves_missing_values():
    df = pd.DataFrame({"value": [1.0, np.nan, 3.0]})
    result = scale_numeric_columns(df, ["value"])
    assert pd.isna(result.loc[1, "value"])
    assert result["value"].notna().sum() == 2


def test_constant_column_scales_to_zero():
    df = pd.DataFrame({"value": [5, 5, 5]})
    for strategy in ("standard", "minmax"):
        result = scale_numeric_columns(df, ["value"], strategy=strategy)
        assert result["value"].tolist() == [0.0, 0.0, 0.0]


def test_original_dataframe_is_unchanged():
    df = pd.DataFrame({"value": [1, 2, 3]})
    original = df.copy(deep=True)
    scale_numeric_columns(df, ["value"])
    pd.testing.assert_frame_equal(df, original)


def test_rejects_non_numeric_column():
    df = pd.DataFrame({"category": ["a", "b"]})
    with pytest.raises(ValueError, match="must be numerical"):
        scale_numeric_columns(df, ["category"])


@pytest.mark.parametrize("columns", [[], ["missing"]])
def test_rejects_empty_or_unknown_columns(columns):
    df = pd.DataFrame({"value": [1, 2]})
    with pytest.raises(ValueError):
        scale_numeric_columns(df, columns)


def test_rejects_invalid_strategy():
    df = pd.DataFrame({"value": [1, 2]})
    with pytest.raises(ValueError, match="strategy"):
        scale_numeric_columns(df, ["value"], strategy="unknown")


def test_rejects_infinite_values():
    df = pd.DataFrame({"value": [1.0, np.inf]})
    with pytest.raises(ValueError, match="infinite"):
        scale_numeric_columns(df, ["value"])
