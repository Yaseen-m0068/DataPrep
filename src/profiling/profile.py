
import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """Generate a basic profile of a tabular dataset."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot profile an empty dataset.")

    row_count, column_count = df.shape

    missing_counts = df.isna().sum()
    missing_percentages = (
        missing_counts.div(row_count).mul(100).round(2)
    )

    column_profiles = {}

    for column in df.columns:
        series = df[column]

        column_profiles[str(column)] = {
            "dtype": str(series.dtype),
            "missing_count": int(missing_counts[column]),
            "missing_percentage": float(
                missing_percentages[column]
            ),
            "unique_count": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            column_profiles[str(column)]["statistics"] = (
                series.describe().to_dict()
            )

    profile = {
        "row_count": int(row_count),
        "column_count": int(column_count),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_bytes": int(
            df.memory_usage(deep=True).sum()
        ),
        "columns": column_profiles,
    }

    return profile
