
from pathlib import Path
from typing import BinaryIO

import pandas as pd

SUPPORTED_FORMATS = {".csv", ".xlsx", ".parquet"}


def load_dataset(file_path: str | Path | BinaryIO) -> pd.DataFrame:
    """Load a supported tabular dataset from a path or binary file object."""
    if isinstance(file_path, (str, Path)):
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")

        suffix = path.suffix.lower()
        source = path
        source_name = path.name
    else:
        source = file_path
        source_name = getattr(file_path, "name", "")

        if not source_name:
            raise ValueError("Uploaded file must have a filename.")

        suffix = Path(source_name).suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: {supported}"
        )

    try:
        if hasattr(source, "seek"):
            source.seek(0)

        if suffix == ".csv":
            return pd.read_csv(source)
        if suffix == ".xlsx":
            return pd.read_excel(source)
        if suffix == ".parquet":
            return pd.read_parquet(source)

    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Could not parse dataset: {source_name}"
        ) from exc
    except (ImportError, OSError) as exc:
        raise ValueError(
            f"Could not load dataset {source_name}: {exc}"
        ) from exc

    raise RuntimeError("Unexpected file format handling error.")
