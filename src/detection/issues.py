
import pandas as pd

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
