
from pathlib import Path
import io
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


def test_load_uploaded_csv():
    uploaded_file = io.BytesIO(
        b"Name,Age\nAlice,21\nBob,22\n"
    )
    uploaded_file.name = "sample.csv"

    df = load_dataset(uploaded_file)

    assert df.shape == (2, 2)
    assert df["Name"].tolist() == ["Alice", "Bob"]
    assert df["Age"].tolist() == [21, 22]


def test_uploaded_file_with_unsupported_extension():
    uploaded_file = io.BytesIO(b"some,data\n1,2\n")
    uploaded_file.name = "sample.txt"

    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataset(uploaded_file)
