
RECOMMENDATION_RULES = {
    "missing_values": {
        "title": "Missing values",
        "explanation": (
            "Some rows have no recorded value in this column."
        ),
        "actions": [
            "Review why the values are missing.",
            "Consider median imputation for numerical data "
            "or mode imputation for categorical data.",
            "Consider leaving values missing if imputation "
            "would introduce bias.",
        ],
    },
    "duplicate_rows": {
        "title": "Duplicate rows",
        "explanation": (
            "Some rows contain values identical to earlier rows."
        ),
        "actions": [
            "Inspect the duplicate rows.",
            "Remove duplicates only after confirming they "
            "are unintended.",
        ],
    },
    "inconsistent_categories": {
        "title": "Inconsistent categories",
        "explanation": (
            "Values may represent the same category but use "
            "different capitalization or spacing."
        ),
        "actions": [
            "Review the reported category variants.",
            "Standardize values using an explicit mapping.",
        ],
    },
    "numeric_outliers": {
        "title": "Potential numeric outliers",
        "explanation": (
            "Some numerical values fall outside the "
            "detected interquartile-range boundaries."
        ),
        "actions": [
            "Inspect the flagged values and their context.",
            "Check for data-entry errors.",
            "Keep valid extreme values; do not remove them "
            "solely because they are outliers.",
        ],
    },
    "constant_column": {
        "title": "Constant column",
        "explanation": (
            "The column has only one distinct non-missing value."
        ),
        "actions": [
            "Check whether the column is needed for reporting "
            "or business rules.",
            "Consider dropping it only if it provides no "
            "useful information.",
        ],
    },
}


def recommend_actions(issues: list[dict]) -> list[dict]:
    """Generate recommendations without modifying the input issues."""
    recommendations = []

    for issue in issues:
        issue_type = issue.get("issue_type")
        rule = RECOMMENDATION_RULES.get(issue_type)

        if rule is None:
            continue

        recommendation = {
            "issue_type": issue_type,
            "column": issue.get("column"),
            "title": rule["title"],
            "explanation": rule["explanation"],
            "actions": list(rule["actions"]),
            "severity": issue.get("severity", "low"),
        }

        recommendations.append(recommendation)

    return recommendations
