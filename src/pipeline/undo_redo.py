
from copy import deepcopy

import pandas as pd


def undo_last_transformation(
    current_df,
    history,
    snapshots,
    redo_history,
    redo_snapshots,
):
    """Undo the latest transformation and save its state for redo."""
    if not history:
        raise ValueError("No transformation available to undo.")

    if len(snapshots) != len(history):
        raise ValueError("Undo history and snapshots are inconsistent.")

    next_history = deepcopy(history[:-1])
    next_snapshots = [
        snapshot.copy(deep=True) for snapshot in snapshots[:-1]
    ]

    next_redo_history = deepcopy(redo_history)
    next_redo_snapshots = [
        snapshot.copy(deep=True) for snapshot in redo_snapshots
    ]

    next_redo_history.append(deepcopy(history[-1]))
    next_redo_snapshots.append(current_df.copy(deep=True))

    restored_df = snapshots[-1].copy(deep=True)

    for index, record in enumerate(next_history, start=1):
        record["step"] = index

    return (
        restored_df,
        next_history,
        next_snapshots,
        next_redo_history,
        next_redo_snapshots,
    )


def redo_last_transformation(
    current_df,
    history,
    snapshots,
    redo_history,
    redo_snapshots,
):
    """Redo the most recently undone transformation."""
    if not redo_history or not redo_snapshots:
        raise ValueError("No transformation available to redo.")

    if len(redo_history) != len(redo_snapshots):
        raise ValueError("Redo history and snapshots are inconsistent.")

    restored_df = redo_snapshots[-1].copy(deep=True)

    next_history = deepcopy(history)
    next_history.append(deepcopy(redo_history[-1]))

    next_snapshots = [
        snapshot.copy(deep=True) for snapshot in snapshots
    ]
    next_snapshots.append(current_df.copy(deep=True))

    next_redo_history = deepcopy(redo_history[:-1])
    next_redo_snapshots = [
        snapshot.copy(deep=True) for snapshot in redo_snapshots[:-1]
    ]

    for index, record in enumerate(next_history, start=1):
        record["step"] = index

    return (
        restored_df,
        next_history,
        next_snapshots,
        next_redo_history,
        next_redo_snapshots,
    )
