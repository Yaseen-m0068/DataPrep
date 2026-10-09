
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.loader import load_dataset


SAMPLE_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "sample"
    / "messy_student_data.csv"
)


def test_load_csv():
    df = load_dataset(SAMPLE_FILE)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (9, 8)
    assert "Age" in df.columns


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_dataset("does_not_exist.csv")


def test_unsupported_format(tmp_path):
    file_path = tmp_path / "data.json"
    file_path.write_text("{}")

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataset(file_path)
