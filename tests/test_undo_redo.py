
import pandas as pd
import pytest

from src.pipeline.undo_redo import (
    undo_last_transformation,
    redo_last_transformation,
)


@pytest.fixture
def transformation_state():
    original = pd.DataFrame({"Age": [20, 30, 40]})
    after_first = pd.DataFrame({"Age": [20, 30, 35]})
    after_second = pd.DataFrame({"Age": [20, 30]})

    history = [
        {
            "step": 1,
            "operation": "handle_missing_values",
            "parameters": {"column": "Age", "strategy": "fill_median"},
            "rows_before": 3,
            "rows_after": 3,
        },
        {
            "step": 2,
            "operation": "remove_duplicate_rows",
            "parameters": {},
            "rows_before": 3,
            "rows_after": 2,
        },
    ]

    snapshots = [original, after_first]

    return {
        "original": original,
        "after_first": after_first,
        "after_second": after_second,
        "history": history,
        "snapshots": snapshots,
        "redo_history": [],
        "redo_snapshots": [],
    }


def test_undo_restores_previous_dataset(transformation_state):
    state = transformation_state

    result = undo_last_transformation(
        state["after_second"],
        state["history"],
        state["snapshots"],
        state["redo_history"],
        state["redo_snapshots"],
    )

    restored_df, history, snapshots, redo_history, redo_snapshots = result

    pd.testing.assert_frame_equal(restored_df, state["after_first"])
    assert len(history) == 1
    assert len(snapshots) == 1
    assert len(redo_history) == 1
    assert len(redo_snapshots) == 1


def test_redo_restores_undone_dataset(transformation_state):
    state = transformation_state

    undone = undo_last_transformation(
        state["after_second"],
        state["history"],
        state["snapshots"],
        [],
        [],
    )

    restored_df, history, snapshots, redo_history, redo_snapshots = undone

    redone = redo_last_transformation(
        restored_df,
        history,
        snapshots,
        redo_history,
        redo_snapshots,
    )

    redone_df, new_history, new_snapshots, new_redo_history, new_redo_snapshots = redone

    pd.testing.assert_frame_equal(redone_df, state["after_second"])
    assert len(new_history) == 2
    assert len(new_snapshots) == 2
    assert new_redo_history == []
    assert new_redo_snapshots == []


def test_undo_rejects_empty_history():
    df = pd.DataFrame({"Age": [20]})

    with pytest.raises(ValueError, match="undo"):
        undo_last_transformation(df, [], [], [], [])


def test_redo_rejects_empty_redo_stack():
    df = pd.DataFrame({"Age": [20]})

    with pytest.raises(ValueError, match="redo"):
        redo_last_transformation(df, [], [], [], [])


def test_undo_does_not_modify_input_history(transformation_state):
    state = transformation_state
    original_history = [record.copy() for record in state["history"]]

    undo_last_transformation(
        state["after_second"],
        state["history"],
        state["snapshots"],
        [],
        [],
    )

    assert state["history"] == original_history


def test_redo_does_not_modify_input_history(transformation_state):
    state = transformation_state

    undone = undo_last_transformation(
        state["after_second"],
        state["history"],
        state["snapshots"],
        [],
        [],
    )

    restored_df, history, snapshots, redo_history, redo_snapshots = undone
    original_history = [record.copy() for record in history]

    redo_last_transformation(
        restored_df,
        history,
        snapshots,
        redo_history,
        redo_snapshots,
    )

    assert history == original_history
