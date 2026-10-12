
from src.recommendations.engine import recommend_actions


def test_recommends_actions_for_missing_values():
    issues = [{
        "issue_type": "missing_values",
        "column": "Age",
        "severity": "medium",
        "count": 2,
    }]

    recommendations = recommend_actions(issues)

    assert len(recommendations) == 1
    assert recommendations[0]["column"] == "Age"
    assert recommendations[0]["issue_type"] == "missing_values"
    assert len(recommendations[0]["actions"]) > 0


def test_recommends_actions_for_multiple_issue_types():
    issues = [
        {"issue_type": "duplicate_rows", "column": None},
        {"issue_type": "constant_column", "column": "Country"},
    ]

    recommendations = recommend_actions(issues)

    assert len(recommendations) == 2
    assert recommendations[0]["title"] == "Duplicate rows"
    assert recommendations[1]["title"] == "Constant column"


def test_unknown_issue_types_are_skipped():
    issues = [{"issue_type": "unknown_issue", "column": "A"}]

    assert recommend_actions(issues) == []


def test_missing_column_and_severity_use_defaults():
    issues = [{"issue_type": "constant_column"}]

    recommendations = recommend_actions(issues)

    assert recommendations[0]["column"] is None
    assert recommendations[0]["severity"] == "low"


def test_recommendations_do_not_modify_input():
    issues = [{
        "issue_type": "duplicate_rows",
        "column": None,
        "severity": "medium",
    }]
    original = [issue.copy() for issue in issues]

    recommend_actions(issues)

    assert issues == original
