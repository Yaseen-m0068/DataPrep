
from copy import deepcopy
from numbers import Integral
from typing import Any


def record_transformation(
    history: list[dict[str, Any]],
    operation: str,
    parameters: dict[str, Any],
    *,
    rows_before: int,
    rows_after: int,
) -> list[dict[str, Any]]:
    #Return a new history list containing the transformation record.
    if not isinstance(history, list):
        raise TypeError("history must be a list.")

    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("operation must be a non-empty string.")

    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a dictionary.")

    for name, count in (
        ("rows_before", rows_before),
        ("rows_after", rows_after),
    ):
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError(f"{name} must be integers.")

        if count < 0:
            raise ValueError(f"{name} must be non-negative.")

    if rows_after > rows_before:
        raise ValueError("rows_after cannot exceed rows_before.")

    result = deepcopy(history)

    record = {
        "step": len(result) + 1,
        "operation": operation.strip(),
        "parameters": deepcopy(parameters),
        "rows_before": int(rows_before),
        "rows_after": int(rows_after),
    }

    result.append(record)
    return result
