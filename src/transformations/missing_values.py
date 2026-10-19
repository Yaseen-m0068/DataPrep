
import pandas as pd


SUPPORTED_STRATEGIES = {
    "drop_rows",
    "fill_mean",
    "fill_median",
    "fill_mode",
}


def handle_missing_values(
    df: pd.DataFrame,
    column: str,
    strategy: str,
) -> pd.DataFrame:
    #Apply a selected missing-value strategy to a copy of the dataset.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot transform an empty dataset.")

    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    if strategy not in SUPPORTED_STRATEGIES:
        supported = ", ".join(sorted(SUPPORTED_STRATEGIES))
        raise ValueError(
            f"Unsupported strategy: {strategy}. "
            f"Supported strategies: {supported}"
        )

    if not df[column].isna().any():
        raise ValueError(
            f"Column '{column}' has no missing values."
        )

    result = df.copy()

    if strategy == "drop_rows":
        result = result.dropna(subset=[column])

    elif strategy in {"fill_mean", "fill_median"}:
        if not pd.api.types.is_numeric_dtype(df[column].dtype):
            raise ValueError(
                f"Strategy '{strategy}' requires a numerical column."
            )

        if strategy == "fill_mean":
            fill_value = df[column].mean()
        else:
            fill_value = df[column].median()

        if pd.isna(fill_value):
            raise ValueError(
                f"Cannot calculate {strategy.removeprefix('fill_')} "
                f"for a column with no observed values."
            )

        result[column] = result[column].fillna(fill_value)

    elif strategy == "fill_mode":
        modes = df[column].mode(dropna=True)

        if modes.empty:
            raise ValueError(
                "Cannot calculate mode for a column with no "
                "observed values."
            )

        result[column] = result[column].fillna(modes.iloc[0])

    return result
