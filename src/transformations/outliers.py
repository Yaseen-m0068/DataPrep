
import pandas as pd

SUPPORTED_STRATEGIES = {"cap_iqr", "drop_rows"}


def handle_numeric_outliers(
    df: pd.DataFrame,
    column: str,
    strategy: str,
) -> pd.DataFrame:
    """Handle numeric outliers using IQR capping or row removal."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot handle outliers in an empty dataset.")

    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    if df[column].isna().all():
        raise ValueError(
        f"Column '{column}' has no observed numeric values.")
    
    if not pd.api.types.is_numeric_dtype(df[column].dtype):
        raise ValueError(f"Column '{column}' must be numerical.")

    if strategy not in SUPPORTED_STRATEGIES:
        supported = ", ".join(sorted(SUPPORTED_STRATEGIES))
        raise ValueError(
            f"Unsupported strategy: {strategy}. "
            f"Supported strategies: {supported}"
        )

    numeric_values = df[column].dropna()

    if numeric_values.empty:
        raise ValueError(
            f"Column '{column}' has no observed numeric values."
        )

    q1 = numeric_values.quantile(0.25)
    q3 = numeric_values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    result = df.copy(deep=True)

    if strategy == "cap_iqr":
        result[column] = result[column].clip(
            lower=lower_bound,
            upper=upper_bound,
        )
    elif strategy == "drop_rows":
        outlier_mask = (
            (result[column] < lower_bound)
            | (result[column] > upper_bound)
        )
        result = result.loc[~outlier_mask].copy()

    return result
