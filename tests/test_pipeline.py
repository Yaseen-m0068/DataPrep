
import pytest

from src.pipeline.history import record_transformation


def test_record_first_transformation():
    history = []

    result = record_transformation(
        history,
        "handle_missing_values",
        {"column": "Age", "strategy": "fill_median"},
        rows_before=100,
        rows_after=100,
    )

    assert len(result) == 1
    assert result[0]["step"] == 1
    assert result[0]["operation"] == "handle_missing_values"
    assert result[0]["parameters"] == {
        "column": "Age",
        "strategy": "fill_median",
    }
    assert result[0]["rows_before"] == 100
    assert result[0]["rows_after"] == 100


def test_record_multiple_transformations_in_order():
    history = record_transformation(
        [],
        "remove_duplicate_rows",
        {},
        rows_before=100,
        rows_after=90,
    )

    result = record_transformation(
        history,
        "handle_numeric_outliers",
        {"column": "Age", "strategy": "cap_iqr"},
        rows_before=90,
        rows_after=90,
    )

    assert len(result) == 2
    assert result[0]["step"] == 1
    assert result[1]["step"] == 2
    assert result[0]["operation"] == "remove_duplicate_rows"
    assert result[1]["operation"] == "handle_numeric_outliers"


def test_record_transformation_does_not_modify_original_history():
    history = []

    result = record_transformation(
        history,
        "remove_duplicate_rows",
        {},
        rows_before=20,
        rows_after=18,
    )

    assert history == []
    assert len(result) == 1


def test_record_transformation_does_not_share_parameters():
    parameters = {"column": "Age", "strategy": "fill_median"}

    result = record_transformation(
        [],
        "handle_missing_values",
        parameters,
        rows_before=10,
        rows_after=10,
    )

    parameters["column"] = "Name"

    assert result[0]["parameters"]["column"] == "Age"


def test_record_transformation_rejects_non_list_history():
    with pytest.raises(TypeError, match="history must be a list"):
        record_transformation(
            None,
            "remove_duplicate_rows",
            {},
            rows_before=10,
            rows_after=9,
        )


def test_record_transformation_rejects_empty_operation():
    with pytest.raises(ValueError, match="operation"):
        record_transformation(
            [],
            "",
            {},
            rows_before=10,
            rows_after=9,
        )


def test_record_transformation_rejects_non_dict_parameters():
    with pytest.raises(TypeError, match="parameters must be a dictionary"):
        record_transformation(
            [],
            "remove_duplicate_rows",
            [],
            rows_before=10,
            rows_after=9,
        )


def test_record_transformation_rejects_negative_row_counts():
    with pytest.raises(ValueError, match="non-negative"):
        record_transformation(
            [],
            "remove_duplicate_rows",
            {},
            rows_before=-1,
            rows_after=9,
        )


def test_record_transformation_rejects_non_integer_row_counts():
    with pytest.raises(TypeError, match="integers"):
        record_transformation(
            [],
            "remove_duplicate_rows",
            {},
            rows_before=10.5,
            rows_after=9,
        )


def test_record_transformation_rejects_rows_after_exceeding_rows_before():
    with pytest.raises(ValueError, match="cannot exceed"):
        record_transformation(
            [],
            "remove_duplicate_rows",
            {},
            rows_before=10,
            rows_after=11,
        )
