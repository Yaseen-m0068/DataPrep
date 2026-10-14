
import pandas as pd


def standardize_categories(
    df: pd.DataFrame,
    column: str,
    *,
    strip_whitespace: bool = True,
    case_strategy: str = "preserve",
) -> pd.DataFrame:
    #Standardize whitespace and capitalization in a text column.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot standardize categories in an empty dataset.")

    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")

    if not (
        pd.api.types.is_object_dtype(df[column].dtype)
        or pd.api.types.is_string_dtype(df[column].dtype)
        or isinstance(df[column].dtype, pd.CategoricalDtype)
    ):
        raise ValueError(
            f"Column '{column}' must contain text or categorical values."
        )

    if case_strategy not in {"preserve", "lower", "upper", "title"}:
        raise ValueError(
            "case_strategy must be 'preserve', 'lower', 'upper', or 'title'."
        )

    result = df.copy(deep=True)
    values = result[column].astype("string")

    if strip_whitespace:
        values = values.str.strip()

    if case_strategy == "lower":
        values = values.str.lower()
    elif case_strategy == "upper":
        values = values.str.upper()
    elif case_strategy == "title":
        values = values.str.title()

    result[column] = values
    return result
