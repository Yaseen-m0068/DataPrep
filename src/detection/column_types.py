
import pandas as pd


DATE_NAME_HINTS = ("date", "time", "timestamp")
ID_NAME_HINTS = ("id", "identifier", "key")

TRUE_FALSE_VALUES = {
    "true", "false", "yes", "no", "y", "n",
    "0", "1", "t", "f",
}


def infer_semantic_type(
    series: pd.Series,
    column_name: str,
) -> dict:
    #Infer a column's likely semantic type without modifying it.

    name = column_name.strip().lower()
    non_null = series.dropna()
    total = len(non_null)

    result = {
        "semantic_type": "unknown",
        "confidence": "low",
        "reason": "Insufficient evidence to infer the column type.",
    }

    if total == 0:
        return result

    # Actual Boolean dtype
    if pd.api.types.is_bool_dtype(series.dtype):
        return {
            "semantic_type": "boolean",
            "confidence": "high",
            "reason": "Column has a Boolean datatype.",
        }

    # Actual datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return {
            "semantic_type": "datetime",
            "confidence": "high",
            "reason": "Column already has a datetime datatype.",
        }

    # ID-like column names
    name_parts = name.replace("-", "_").split("_")
    if any(part in ID_NAME_HINTS for part in name_parts):
        return {
            "semantic_type": "identifier",
            "confidence": "medium",
            "reason": "Column name suggests an identifier.",
        }

    # Native numeric data
    if pd.api.types.is_numeric_dtype(series.dtype):
        return {
            "semantic_type": "numerical",
            "confidence": "high",
            "reason": "Column has a numeric datatype.",
        }

    # Date-like column names or parseable date values
    
    date_name_hint = any(
        hint in name for hint in DATE_NAME_HINTS
    )

    # First use the column name as a cheap signal.
    if date_name_hint:
        return {
            "semantic_type": "datetime_candidate",
            "confidence": "medium",
            "reason": (
                "Column name suggests dates or timestamps. "
                "Confirm the format before conversion."
            ),
        }

    # Only attempt date inference for strings that look date-like.
    string_values = non_null.astype(str).str.strip()

    looks_date_like = string_values.str.contains(
        r"[-/]\d{1,2}[-/]\d{1,4}|"
        r"\d{4}-\d{2}-\d{2}",
        regex=True,
    ).mean()

    if looks_date_like >= 0.8:
        parsed_dates = pd.to_datetime(
            string_values,
            errors="coerce",
            format="mixed",
        )
        date_ratio = parsed_dates.notna().mean()

        if date_ratio >= 0.8:
            return {
                "semantic_type": "datetime_candidate",
                "confidence": "medium",
                "reason": (
                    "Most values resemble dates and can be "
                    "parsed as dates. Confirm ambiguous formats."
                ),
            }

    # Boolean-like categorical values
    normalized = non_null.astype(str).str.strip().str.lower()
    unique_values = set(normalized.unique())

    if len(unique_values) <= 2 and unique_values.issubset(
        TRUE_FALSE_VALUES
    ):
        return {
            "semantic_type": "boolean_candidate",
            "confidence": "medium",
            "reason": "Values resemble true/false or yes/no categories.",
        }

    # Identifier-like values based on uniqueness
    uniqueness_ratio = non_null.nunique() / total

    if (
        not pd.api.types.is_numeric_dtype(series.dtype)
        and total >= 5
        and uniqueness_ratio >= 0.9
    ):
        return {
            "semantic_type": "identifier_candidate",
            "confidence": "low",
            "reason": (
                "Most non-null values are unique. "
                "Review before treating this as an identifier."
            ),
        }

    # Categorical versus free text
    
    unique_count = non_null.nunique()
    unique_ratio = unique_count / total

    if unique_count <= 20 and unique_ratio <= 0.8:
        return {
            "semantic_type": "categorical",
            "confidence": "medium",
            "reason": (
                "Column contains a limited number of distinct "
                "values relative to its non-null rows."
            ),
        }

    return {
        "semantic_type": "text",
        "confidence": "low",
        "reason": "Column contains varied text values.",
    }
