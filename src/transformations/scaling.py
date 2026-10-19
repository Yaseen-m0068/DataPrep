
import numpy as np
import pandas as pd


VALID_STRATEGIES = {"standard", "minmax"}


def scale_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    strategy: str = "standard",
) -> pd.DataFrame:
    """Scale numeric columns using standardization or min-max scaling.

    Missing values are preserved. Standard scaling uses the population
    standard deviation (ddof=0); constant columns are mapped to zero.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Cannot scale columns in an empty dataset.")
    if strategy not in VALID_STRATEGIES:
        raise ValueError("strategy must be 'standard' or 'minmax'.")
    if not isinstance(columns, (list, tuple)) or not columns:
        raise ValueError("Select at least one numerical column to scale.")
    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate column selections are not allowed.")

    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column not found: {column}")
        if (
            not pd.api.types.is_numeric_dtype(df[column].dtype)
            or pd.api.types.is_bool_dtype(df[column].dtype)
        ):
            raise ValueError(f"Column '{column}' must be numerical.")
        observed = df[column].dropna()
        if observed.empty:
            raise ValueError(f"Column '{column}' contains no observed values.")
        if not np.isfinite(observed.to_numpy()).all():
            raise ValueError(
                f"Column '{column}' contains infinite values. Handle them before scaling."
            )

    result = df.copy(deep=True)
    for column in columns:
        values = result[column].astype("float64")
        observed = values.dropna()

        if strategy == "standard":
            center = observed.mean()
            scale = observed.std(ddof=0)
            if scale != 0:
                result[column] = (values - center) / scale
            else:
                result[column] = values.where(values.isna(), 0.0)
        else:
            minimum = observed.min()
            value_range = observed.max() - minimum
            if value_range != 0:
                result[column] = (values - minimum) / value_range
            else:
                result[column] = values.where(values.isna(), 0.0)

    return result
