
from pathlib import Path

import pandas as pd

SUPPORTED_FORMATS = {".csv", ".xlsx", ".parquet"}

def load_dataset(file_path: str | Path) -> pd.DataFrame:
    #Load a supported tabular dataset into a Pandas DataFrame.
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_FORMATS))
        raise ValueError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: {supported}"
        )

    try:
        if suffix == ".csv":
            return pd.read_csv(path)

        if suffix == ".xlsx":
            return pd.read_excel(path)

        if suffix == ".parquet":
            return pd.read_parquet(path)

    except (pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Could not parse dataset: {path.name}"
        ) from exc

    except (ImportError, ValueError, OSError) as exc:
        raise ValueError(
            f"Could not load dataset {path.name}: {exc}"
        ) from exc

    raise RuntimeError("Unexpected file format handling error.")
