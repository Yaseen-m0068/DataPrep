
import pandas as pd


VALID_STRATEGIES = {"onehot", "ordinal"}


def encode_categorical_columns(
    df: pd.DataFrame,
    columns: list[str],
    *,
    strategy: str = "onehot",
    ordinal_mappings: dict[str, list] | None = None,
    drop_first: bool = False,
    include_missing: bool = False,
) -> pd.DataFrame:
    """Encode categorical columns using one-hot or ordinal encoding.

    One-hot encoding creates binary indicator columns and removes the
    original selected columns. Missing values are represented by all-zero
    indicators unless include_missing=True.

    Ordinal encoding maps each category to its zero-based position in the
    user-supplied order. Missing values remain missing.

    The input DataFrame is never modified.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot encode columns in an empty dataset.")

    if strategy not in VALID_STRATEGIES:
        raise ValueError("strategy must be 'onehot' or 'ordinal'.")

    if not isinstance(columns, (list, tuple)) or not columns:
        raise ValueError("Select at least one categorical column to encode.")

    if len(set(columns)) != len(columns):
        raise ValueError("Duplicate column selections are not allowed.")

    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column not found: {column}")

        series = df[column]
        is_categorical = (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(series.dtype)
        )

        if not is_categorical:
            raise ValueError(
                f"Column '{column}' must contain categorical values."
            )

        if series.dropna().empty:
            raise ValueError(
                f"Column '{column}' contains no observed categories."
            )

    if strategy == "ordinal":
        if not isinstance(ordinal_mappings, dict):
            raise ValueError(
                "Provide ordinal_mappings as a dictionary of column names "
                "and ordered category lists."
            )

        for column in columns:
            if column not in ordinal_mappings:
                raise ValueError(
                    f"Missing ordinal mapping for column '{column}'."
                )

            order = ordinal_mappings[column]

            if not isinstance(order, list) or not order:
                raise ValueError(
                    f"Ordinal mapping for '{column}' must be a non-empty list."
                )

            if any(pd.isna(value) for value in order):
                raise ValueError(
                    f"Ordinal mapping for '{column}' cannot contain missing values."
                )

            if len(set(order)) != len(order):
                raise ValueError(
                    f"Ordinal mapping for '{column}' contains duplicate categories."
                )

            observed = df[column].dropna()
            unknown = observed[~observed.isin(order)].unique().tolist()

            if unknown:
                raise ValueError(
                    f"Categories in '{column}' are missing from its ordinal "
                    f"mapping: {unknown}"
                )

    result = df.copy(deep=True)

    if strategy == "ordinal":
        for column in columns:
            order = ordinal_mappings[column]
            mapping = {category: index for index, category in enumerate(order)}

            result[column] = (
                result[column].map(mapping).astype("Int64")
            )

        return result

    # One-hot encoding
    encoded = pd.get_dummies(
        result[columns],
        columns=columns,
        prefix=columns,
        prefix_sep="__",
        drop_first=drop_first,
        dummy_na=include_missing,
        dtype="int64",
    )

    remaining = result.drop(columns=columns)

    # Prevent generated columns from silently overwriting existing columns.
    collisions = set(encoded.columns).intersection(remaining.columns)

    if collisions:
        raise ValueError(
            f"Encoding would create duplicate column names: "
            f"{sorted(collisions)}"
        )

    return pd.concat([remaining, encoded], axis=1)
