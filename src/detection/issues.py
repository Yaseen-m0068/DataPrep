
import pandas as pd
import numpy as np
from src.detection.column_types import infer_semantic_type

def detect_missing_values(df: pd.DataFrame) -> list[dict]:
    #Report columns containing missing values without modifying data.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot detect issues in an empty dataset.")

    issues = []
    row_count = len(df)

    for column in df.columns:
        missing_count = int(df[column].isna().sum())

        if missing_count > 0:
            missing_percentage = round(
                (missing_count / row_count) * 100, 2
            )

            issues.append({
                "issue_type": "missing_values",
                "column": str(column),
                "count": missing_count,
                "percentage": missing_percentage,
                "severity": (
                    "high" if missing_percentage >= 30 else "medium"
                ),
                "description": (
                    f"{missing_count} missing value(s) found in "
                    f"'{column}' ({missing_percentage}%)."
                ),
            })

    return issues


def detect_duplicate_rows(df: pd.DataFrame) -> list[dict]:
    #Report exact duplicate rows without modifying the dataset.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot detect issues in an empty dataset.")

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count == 0:
        return []

    duplicate_percentage = round(
        (duplicate_count / len(df)) * 100, 2
    )

    return [{
        "issue_type": "duplicate_rows",
        "column": None,
        "count": duplicate_count,
        "percentage": duplicate_percentage,
        "severity": "medium",
        "description": (
            f"{duplicate_count} exact duplicate row(s) found."
        ),
    }]



def detect_all_issues(df: pd.DataFrame) -> list[dict]:
    """Run all available data quality detectors."""
    issues = []

    issues.extend(detect_missing_values(df))
    issues.extend(detect_duplicate_rows(df))
    issues.extend(detect_inconsistent_categories(df))
    issues.extend(detect_numeric_outliers(df))
    issues.extend(detect_constant_columns(df))

    return issues



def detect_inconsistent_categories(df: pd.DataFrame) -> list[dict]:
    #Report likely category variations caused by case or whitespace.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot detect issues in an empty dataset.")

    issues = []

    for column in df.columns:
        series = df[column].dropna()

        # Only inspect text-like columns.
        if not (
            pd.api.types.is_object_dtype(df[column].dtype)
            or pd.api.types.is_string_dtype(df[column].dtype)
            or isinstance(df[column].dtype, pd.CategoricalDtype)
        ):
            continue

        values = series.astype(str)
        groups = {}

        for value in values.unique():
            normalized = value.strip().casefold()
            groups.setdefault(normalized, []).append(value)

        variants = [
            sorted(original_values)
            for original_values in groups.values()
            if len(original_values) > 1
        ]

        if variants:
            affected_values = {
                value for group in variants for value in group
            }
            affected_rows = int(values.isin(affected_values).sum())

            issues.append({
                "issue_type": "inconsistent_categories",
                "column": str(column),
                "count": affected_rows,
                "percentage": round(
                    affected_rows / len(df) * 100, 2
                ),
                "severity": "low",
                "variants": variants,
                "description": (
                    f"Possible case or whitespace inconsistencies "
                    f"found in '{column}'."
                ),
            })

    return issues


def detect_numeric_outliers(df: pd.DataFrame) -> list[dict]:
    #Flag potential numerical outliers using the IQR method.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot detect issues in an empty dataset.")

    issues = []

    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            continue

        semantic = infer_semantic_type(df[column], str(column))

        # Avoid treating numeric identifiers as measurements.
        if semantic["semantic_type"] == "identifier":
            continue

        values = df[column].dropna()

        # Avoid unreliable IQR estimates on very small samples.
        if len(values) < 4:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        tolerance = 1e-9

        outliers = values[
            (values < lower_bound - tolerance) | (values > upper_bound + tolerance)
        ]

        if outliers.empty:
            continue

        issues.append({
            "issue_type": "numeric_outliers",
            "column": str(column),
            "count": int(len(outliers)),
            "percentage": round(
                len(outliers) / len(values) * 100, 2
            ),
            "severity": "low",
            "outlier_values": outliers.tolist(),
            "lower_bound": round(float(lower_bound), 4),
            "upper_bound": round(float(upper_bound), 4),
            "method": "IQR",
            "description": (
                f"{len(outliers)} potential outlier(s) found "
                f"in '{column}' using the IQR method."
            ),
        })

    return issues


def detect_constant_columns(df: pd.DataFrame) -> list[dict]:
    #Report columns with one distinct non-missing value.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Cannot detect issues in an empty dataset.")

    issues = []

    for column in df.columns:
        values = df[column].dropna()

        # Ignore columns that contain no observed values.
        if values.empty:
            continue

        unique_count = int(values.nunique())

        if unique_count == 1:
            constant_value = str(values.iloc[0])
            affected_count = len(values)

            issues.append({
                "issue_type": "constant_column",
                "column": str(column),
                "count": affected_count,
                "percentage": round(
                    affected_count / len(df) * 100, 2
                ),
                "severity": "low",
                "constant_value": constant_value,
                "description": (
                    f"Column '{column}' has only one distinct "
                    f"non-missing value: {constant_value!r}."
                ),
            })

    return issues
